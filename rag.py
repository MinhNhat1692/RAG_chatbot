import os
from dotenv import load_dotenv
import openai
import faiss
import numpy as np
import mysql.connector
import pickle
import json
import re

# Load .env file
load_dotenv()

# Use environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = 'text-embedding-3-small'

def connect_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def embed_text(text):
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# ------------------- Store / Load from DB -------------------
def store_knowledge(content):
    embedding = embed_text(content)
    serialized = pickle.dumps(embedding)

    db = connect_db()
    cursor = db.cursor()

    # Check if already exists
    cursor.execute("SELECT id FROM knowledge WHERE content = %s", (content,))
    if cursor.fetchone():
        print(f"[Info] Already exists in DB: \"{content[:50]}...\"")
        cursor.close()
        db.close()
        return

    cursor.execute("INSERT INTO knowledge (content, embedding) VALUES (%s, %s)", (content, serialized))
    db.commit()
    cursor.close()
    db.close()
    print(f"[Saved] \"{content[:50]}...\"")

def load_all_embeddings():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, content, embedding FROM knowledge")
    rows = cursor.fetchall()
    cursor.close()
    db.close()

    results = []
    for row in rows:
        id, content, emb_blob = row
        embedding = pickle.loads(emb_blob)
        results.append((id, content, embedding))
    return results

def get_conversation_history(convo_id):
    db = connect_db()
    cursor = db.cursor()

    # Get linked knowledge content for this conversation
    cursor.execute("""
        SELECT k.content 
        FROM conversation_link cl
        JOIN knowledge k ON cl.knowledge_id = k.id
        WHERE cl.conversation_id = %s
        ORDER BY cl.id ASC
    """, (convo_id,))
    
    history = [row[0] for row in cursor.fetchall()]

    cursor.close()
    db.close()

    return history

expected_info = ["kích thước", "màu sắc", "số bộ", "số điện thoại", "địa chỉ giao hàng"]

def detect_missing_info(convo_texts):
    joined_history = "\n".join(convo_texts)
    prompt = f"""
Bạn là một trợ lý bán hàng. Dưới đây là lịch sử cuộc trò chuyện với khách hàng.

Khách hàng có thể đặt **nhiều đơn hàng**, mỗi đơn gồm:
- Kích thước
- Màu sắc
- Số bộ

Ngoài ra cần thu thập:
- Số điện thoại
- Địa chỉ giao hàng

Hãy phân tích đoạn hội thoại và trích xuất các thông tin trên.
Nếu thiếu thông tin nào, để giá trị là null.

Trả về kết quả dưới dạng JSON như sau:

{{
  "đơn hàng": [
    {{
      "kích thước": "60" | "70" | "80" | null,
      "màu sắc": "trắng" | "đen" | null,
      "số bộ": 1 | 2 | 3 | null
    }},
    ...
  ],
  "số điện thoại": "0123456789" | null,
  "địa chỉ giao hàng": "địa chỉ đầy đủ" | null
}}

Lịch sử hội thoại:
{joined_history}
"""

    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000,
    )

    result = response.choices[0].message.content.strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        print("❌ Lỗi parse JSON:", result)
        return {}


# ------------------- RAG Class -------------------
class RAG:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

        # Load from DB
        #entries = load_all_embeddings()
        #if entries:
        #    vecs = [e[2] for e in entries]
        #    self.index.add(np.array(vecs))
        #    self.texts = [e[1] for e in entries]
        #    print(f"[Loaded] {len(self.texts)} entries from DB.")

    def add(self, text):
        if text in self.texts:
            print(f"[Skip] Already added to FAISS: \"{text[:50]}...\"")
            return

        store_knowledge(text)  # Will skip if already in DB
        vec = embed_text(text)
        self.index.add(np.array([vec]))
        self.texts.append(text)
        print(f"[Added] To FAISS: \"{text[:50]}...\"")

    def search(self, query, texts=None, k=5):
        if texts is None:
            texts = self.texts
            vecs = self.index
            return self._search_faiss(query, texts, vecs, k)
        else:
            # Combine provided texts with self.texts
            combined_texts = self.texts + texts
            vecs = [embed_text(t) for t in combined_texts]
            q_vec = embed_text(query)
            sims = [(t, np.linalg.norm(q_vec - v)) for t, v in zip(combined_texts, vecs)]
            sims.sort(key=lambda x: x[1])
            return sims[:k]

    
    def _search_faiss(self, query, texts, vecs, k):
        q_vec = embed_text(query)
        D, I = self.index.search(np.array([q_vec]), k)
        return [(texts[i], D[0][idx]) for idx, i in enumerate(I[0]) if i != -1]

    def get_conversation_knowledge(self, convo_id):
        db = connect_db()
        cursor = db.cursor()
        cursor.execute("""
            SELECT k.content 
            FROM conversation_link cl
            JOIN knowledge k ON cl.knowledge_id = k.id
            WHERE cl.conversation_id = %s
        """, (convo_id,))
        texts = [row[0] for row in cursor.fetchall()]
        cursor.close()
        db.close()
        return texts

    def store_and_link_query(self, convo_id, text, source='user'):
        db = connect_db()
        cursor = db.cursor()

        # Step 1: Check if knowledge already exists
        cursor.execute("SELECT id FROM knowledge WHERE content = %s", (text,))
        row = cursor.fetchone()

        if row:
            knowledge_id = row[0]
            print(f"[Info] Already in knowledge DB")
        else:
            # Insert new knowledge
            embedding = embed_text(text)
            serialized = pickle.dumps(embedding)
            cursor.execute(
                "INSERT INTO knowledge (content, embedding) VALUES (%s, %s)",
                (text, serialized)
            )
            knowledge_id = cursor.lastrowid
            self.index.add(np.array([embedding]))
            self.texts.append(text)
            print(f"[Saved + FAISS] \"{text[:50]}...\"")

        # Step 2: Link to conversation with 'from' column
        cursor.execute(
            "SELECT 1 FROM conversation_link WHERE conversation_id = %s AND knowledge_id = %s",
            (convo_id, knowledge_id)
        )
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO conversation_link (conversation_id, knowledge_id, from_source) VALUES (%s, %s, %s)",
                (convo_id, knowledge_id, source)
            )
            print(f"[Linked] query to conversation_id={convo_id} from={source}")

        db.commit()
        cursor.close()
        db.close()
        return knowledge_id

def answer_question(question, contexts, next_missing=None, info_status=None):
    context_block = "\n".join(contexts)

    if next_missing is None:
        # Gửi prompt để phân loại intent
        intent_prompt = f"""
Bạn là một trợ lý bán hàng. Phân loại câu của khách vào 1 trong 3 nhóm sau (chỉ trả về đúng số):
1. Khách đang cung cấp thêm thông tin đơn hàng (số điện thoại và địa chỉ)
2. Khách xác nhận muốn đặt hàng
3. Câu nói không liên quan hoặc chưa rõ ý định

Câu của khách: "{question}"

Chỉ trả lời bằng 1, 2 hoặc 3.
"""
        intent_response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": intent_prompt}],
            max_tokens=10,
            temperature=0
        )
        intent = intent_response.choices[0].message.content.strip()

        don_hang_list = info_status.get('đơn hàng', [])

        def get_first_valid_value(key):
            for dh in don_hang_list:
                val = dh.get(key)
                if val not in (None, "", "chưa rõ"):
                    return val
            return "chưa rõ"

        # Tổng số bộ = tổng cộng 'số bộ' trong các đơn hàng, bỏ qua None hoặc giá trị không hợp lệ
        total_so_bo = 0
        for dh in don_hang_list:
            try:
                so_bo = int(dh.get("số bộ", 0) or 0)
                total_so_bo += so_bo
            except (ValueError, TypeError):
                continue
        if total_so_bo == 0:
            total_so_bo = 1  # mặc định 1 nếu không có số bộ hợp lệ

        order_info = {
            "kích thước": get_first_valid_value("kích thước"),
            "màu sắc": get_first_valid_value("màu sắc"),
            "số bộ": total_so_bo,
            "số điện thoại": info_status.get("số điện thoại") or "chưa rõ",
            "địa chỉ giao hàng": info_status.get("địa chỉ giao hàng") or "chưa rõ",
        }

        # Tính tổng tiền theo số bộ
        tong_tien = total_so_bo * (170000 if total_so_bo > 1 else 175000)

        if intent == "1":
            answer = (
                f"Dạ em đã ghi nhận đầy đủ thông tin đơn hàng của mình ạ:\n" +
                "\n".join([f"- {k.capitalize()}: {v}" for k, v in order_info.items()]) +
                f"\n👉 Tổng tiền: {tong_tien:,} VNĐ\n\n"
                f"Dạ em gửi khoảng 3-4 ngày chị nhận được, chị nhận thanh toán giúp em {tong_tien:,} VNĐ và phí ship ạ"
            )
            return {
                "order_info": order_info,
                "answer": answer,
                "question": "Chị có cần em hỗ trợ gì thêm không ạ?"
            }

        elif intent == "2":
            return {
                "order_info": order_info,
                "answer": "Dạ em cảm ơn chị nhiều ạ 💖 Em sẽ tiến hành lên đơn ngay cho mình nhé!",
                "question": "Chị có cần đổi gì thêm không ạ, ví dụ số bộ hay màu sắc?"
            }

        else:
            return {
                "order_info": order_info,
                "answer": "Chị chờ em chút ạ 🫶",
                "question": "Không biết chị muốn cung cấp thêm thông tin hay xác nhận đặt hàng ạ?"
            }

    # 🧠 Trường hợp thiếu thông tin → tiếp tục hỏi
    base_prompt = f"""
Bạn là một trợ lý bán hàng chuyên nghiệp. Hãy thực hiện 3 việc:
1. Trả lời khách một cách thân thiện.
2. Nếu thiếu thông tin, hãy hỏi tiếp khách về thông tin còn thiếu.
3. Trả về kết quả dưới dạng JSON với 2 trường: answer_only (chỉ có câu trả lời hoặc câu hỏi xác nhận), question_ask_next (câu hỏi tiếp theo cần hỏi để biết thông tin cần biết tiếp theo).

Trả lời dựa trên thông tin đơn hàng ở dạng JSON dưới đây và câu hỏi của khách hàng.

Thông tin đơn hàng:
{json.dumps(info_status, ensure_ascii=False, indent=2)}
Thông tin cần biết tiếp theo: {next_missing}

Thông tin thêm:
{context_block}

Câu của khách: {question}
Kết quả trả về (JSON):""".strip()

    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": base_prompt}],
        max_tokens=500,
        temperature=0.1
    )

    print("🔍 prompt:", base_prompt)

    raw_output = response.choices[0].message.content.strip()
    print("🔍 raw model output:", raw_output)

    try:
        result = json.loads(raw_output)
    except Exception:
        # fallback: try to parse manually if model doesn't return valid JSON
        result = False

    return result

if __name__ == '__main__':
    rag = RAG()

    query = "quần áo cỡ 70 là cho đối tượng nào"
    results = rag.search(query)
    top_texts = [text for text, _ in results]

    answer = answer_question(query, top_texts)
    print("\nAnswer:\n", answer)