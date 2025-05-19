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

Mục tiêu là lấy đủ thông tin sau, theo thứ tự:
1. Kích thước
2. Màu sắc
3. Số bộ (số lượng, số cái, thường nếu khách chọn 1 màu thì 1 cái, 2 màu là 2 cái ..)
4. Số điện thoại
5. Địa chỉ giao hàng

Hãy phân tích đoạn hội thoại và trích xuất thông tin nếu có. Nếu chưa có, để giá trị là null.

Trả về kết quả dưới dạng JSON:
{{
  "kích thước": "60" | "70" | "80" | ... | null,
  "màu sắc": "trắng" | "đen" | ... | null,
  "số bộ": 2 | 3 | ... | null,
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
        print("❌ Lỗi parse JSON từ GPT:", result)
        return {}


# ------------------- RAG Class -------------------
class RAG:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

        # Load from DB
        entries = load_all_embeddings()
        if entries:
            vecs = [e[2] for e in entries]
            self.index.add(np.array(vecs))
            self.texts = [e[1] for e in entries]
            print(f"[Loaded] {len(self.texts)} entries from DB.")

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

    # Build known info summary
    known_info = []
    info_dict = {}
    if info_status:
        for key, value in info_status.items():
            if value:  # giá trị khác rỗng
                known_info.append(key)
                info_dict[key] = value  # ✅ lấy giá trị trực tiếp từ info_status

    known_info_str = ", ".join(known_info) if known_info else "Chưa có thông tin nào"

    # ✅ Nếu đã đầy đủ thông tin → xác định intent trước
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

        if intent == "1":
            # Trường hợp khách đang cung cấp thêm thông tin → trả lại như cũ
            so_bo = info_dict.get("số bộ", 1)
            try:
                so_bo = int(so_bo)
            except ValueError:
                so_bo = 1

            if so_bo > 1:
                tong_tien = so_bo * 170000
            else:
                tong_tien = so_bo * 175000

            thong_tin_don_hang = "\n".join([
                f"- {key.capitalize()}: {info_dict.get(key, '...')}"
                for key in ["kích thước", "màu sắc", "số bộ", "số điện thoại", "địa chỉ giao hàng"]
            ])

            return (
                f"Dạ em đã ghi nhận đầy đủ thông tin đơn hàng của mình ạ:\n"
                f"{thong_tin_don_hang}\n"
                f"👉 Tổng tiền: {tong_tien:,} VNĐ\n\n"
                f"Dạ em gửi khoảng 3-4 ngày chị nhận được, chị nhận thanh toán giúp em {tong_tien:,} VNĐ và phí ship ạ"
            )

        elif intent == "2":
            return "Dạ em cảm ơn chị nhiều ạ 💖 Em sẽ tiến hành lên đơn ngay cho mình nhé!"

        else:
            return "Chị chờ em chút ạ 🫶"

    # 🧠 Trường hợp thiếu thông tin → tiếp tục hỏi
    base_prompt = f"""Bạn là một trợ lý bán hàng chuyên nghiệp. Trả lời dựa trên thông tin bên dưới, sau đó hỏi thêm thông tin tiếp theo.

Thông tin đã biết: {known_info_str}
Thông tin cần biết tiếp theo: {next_missing}

Thông tin thêm:
{context_block}

Câu của khách: {question}
Trả lời:"""

    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": base_prompt}],
        max_tokens=1000,
        temperature=0.1
    )
    print("prompt:", base_prompt)
    answer = response.choices[0].message.content.strip()
    return answer

if __name__ == '__main__':
    rag = RAG()

    query = "quần áo cỡ 70 là cho đối tượng nào"
    results = rag.search(query)
    top_texts = [text for text, _ in results]

    answer = answer_question(query, top_texts)
    print("\nAnswer:\n", answer)