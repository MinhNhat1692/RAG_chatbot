from flask import Flask, request, jsonify
from rag import RAG, answer_question, get_conversation_history, detect_missing_info
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

CHATWOOT_EMAIL = os.getenv("CHATWOOT_EMAIL")
CHATWOOT_PASSWORD = os.getenv("CHATWOOT_PASSWORD")
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL")
TOKEN_FILE = "chatwoot_token.txt"

app = Flask(__name__)
rag = RAG()
# Add your knowledge base texts here or load from file/db
# rag.add("giá sản phẩm là 175k")
# rag.add("có 4 cỡ là 60, 70 , 80, 90")
# rag.add("sản phẩm có 2 màu là đen và đỏ")

def get_saved_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return f.read().strip()
    return None

def save_token(token):
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)

def login_to_chatwoot():
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/sign_in"
    response = requests.post(url, json={
        "email": CHATWOOT_EMAIL,
        "password": CHATWOOT_PASSWORD
    })
    response.raise_for_status()
    token = response.json()["data"]["attributes"]["access_token"]
    save_token(token)
    return token

def validate_token(token):
    url = f"{CHATWOOT_BASE_URL}/api/v1/profile"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    return response.status_code == 200

def send_message_to_chatwoot(conversation_id, message, token):
    url = f"{CHATWOOT_BASE_URL}/api/v1/conversations/{conversation_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "content": message,
        "message_type": "outgoing"
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()

    # ✅ Prevent empty payloads or missing content
    if not data or "content" not in data or not data.get("content", "").strip():
        return jsonify({"error": "Invalid or empty input"}), 400

    query = data["content"].strip()
    convo_id = str(data.get("conversation", {}).get("id", ""))
    chatwoot_convo_id = convo_id

    # ✅ Optional: prevent action if conversation_id is missing (for message reply)
    if not chatwoot_convo_id:
        return jsonify({"error": "Missing conversation ID"}), 400

    # Step 1: Get valid token
    token = get_saved_token()
    if not token or not validate_token(token):
        token = login_to_chatwoot()

    # 1. Get full conversation
    convo_history = get_conversation_history(convo_id)
    # 2. Detect missing info
    info_status = detect_missing_info(convo_history + [query])
    print("Missing info status:", info_status)

    # Step 3: Determine next missing field
    next_missing = next((k for k, v in info_status.items() if v is None), None)
    print("Next missing field:", next_missing)

    must_know_context = [
        "Giá mỗi bộ là 175,000 VNĐ. Mua từ 2 set giá còn 170k",
        "trả lời ngắn gọn, lịch sự, xưng hô là em, khách hàng là chị",
        "bắt đầu câu trả lời bằng từ Dạ"
    ]

    extra_context_map = {
        "kích thước": [
            "có các cỡ 60, 70, 80, 90, 100, 110, 120, 130, 140",
            "80=>1-2tuổi,9-11kg,90cm",
            "90=>2-3tuổi,11-14kg,95cm",
            "100=>3-4tuổi,14-16kg,104cm",
            "110=>4-5tuổi,17-20kg,112cm",
            "120=>5-6tuổi,20-23kg,120cm",
            "130=>6-7tuổi,23-26kg,126cm",
            "140=>8-9tuổi,27-30kg,134cm",
        ],
        "màu sắc": [
            "có các cỡ 60, 70, 80, 90, 100, 110, 120, 130, 140",
            "80=>1-2tuổi,9-11kg,90cm",
            "90=>2-3tuổi,11-14kg,95cm",
            "100=>3-4tuổi,14-16kg,104cm",
            "110=>4-5tuổi,17-20kg,112cm",
            "120=>5-6tuổi,20-23kg,120cm",
            "130=>6-7tuổi,23-26kg,126cm",
            "140=>8-9tuổi,27-30kg,134cm",
            "câu về kích thước cần trả lời theo dạng: Dạ, bé mặc vừa size xx bên em ạ"
            "Sản phẩm có các màu: Trắng, Hồng, Xanh cốm."
        ],
        "số bộ": [
        ],
        "số điện thoại": [
            "Dùng mẫu câu: Chị cho em xin số điện thoại và địa chỉ để em gửi chị ạ"
            "Số điện thoại sẽ dùng để bên vận chuyển liên hệ khi giao hàng."
        ],
        "địa chỉ giao hàng": [
            "Em cần địa chỉ đầy đủ để giao hàng chính xác, gồm: số nhà, tên đường, phường/xã, quận/huyện, tỉnh/thành."
        ]
    }

    # Step 4: If user asked about something useful, use RAG to respond
    rag_contexts = rag.search(query,convo_history)
    context_texts = [text for text, _ in rag_contexts]

    # 👉 Gộp: must-know + extra context (nếu có) + RAG
    combined_contexts = must_know_context[:]
    if next_missing in extra_context_map:
        combined_contexts.extend(extra_context_map[next_missing])
    combined_contexts.extend(context_texts)

    # Special case: If both phone number and address are missing
    if next_missing == "số điện thoại" and info_status.get("địa chỉ giao hàng") is None:
        next_missing = "số điện thoại & địa chỉ giao hàng"

    # Step 5: Generate answer from context
    answer = answer_question(query, combined_contexts, next_missing, info_status)

    # Step 6: Store both user query and bot answer
    rag.store_and_link_query(convo_id, query, source='user')
    rag.store_and_link_query(convo_id, answer, source='bot')

    # Step 3: Send answer to Chatwoot
    if chatwoot_convo_id:
        try:
            send_message_to_chatwoot(chatwoot_convo_id, answer, token)
        except Exception as e:
            print(f"Failed to send message to Chatwoot: {e}")

    return jsonify({'answer': answer})

@app.route('/params-check', methods=['POST'])
def params_check():
    # Get both JSON body and form data
    json_data = request.get_json(silent=True)
    form_data = request.form.to_dict()
    args_data = request.args.to_dict()
    headers = dict(request.headers)

    print("JSON Data:", json_data)
    print("Form Data:", form_data)
    print("Query Params:", args_data)
    print("Headers:", headers)

    return jsonify({
        'json': json_data,
        'form': form_data,
        'args': args_data,
        'headers': headers
    })

if __name__ == '__main__':
    app.run(port=5000)
