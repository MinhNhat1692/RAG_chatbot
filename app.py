from flask import Flask, request, jsonify
from rag import RAG, answer_question, get_conversation_history, detect_missing_info
from dotenv import load_dotenv
import os
import requests
import json
import threading
import re

load_dotenv()

CHATWOOT_EMAIL = os.getenv("CHATWOOT_EMAIL")
CHATWOOT_PASSWORD = os.getenv("CHATWOOT_PASSWORD")
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL")
TOKEN_FILE = "chatwoot_token.txt"

app = Flask(__name__)

def get_saved_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return f.read().strip()
    return None

def save_token(token):
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)

def login_to_chatwoot():
    url = f"{CHATWOOT_BASE_URL}/auth/sign_in"
    response = requests.post(url, json={
        "email": CHATWOOT_EMAIL,
        "password": CHATWOOT_PASSWORD
    })
    response.raise_for_status()
    token = response.json()["data"]["access_token"]
    save_token(token)
    return token

def validate_token(token):
    url = f"{CHATWOOT_BASE_URL}/api/v1/profile"
    headers = {"api_access_token": token}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            profile_data = response.json()
            return True, profile_data.get("account_id")
        else:
            return False, None
    except Exception as e:
        print(f"[validate_token] Error: {e}")
        return False, None

def get_next_missing_field(info_status):
    order_fields = ["kích thước", "màu sắc", "số bộ"]
    global_fields = ["số điện thoại", "địa chỉ giao hàng"]

    # Ensure there's at least one order to validate
    orders = info_status.get("đơn hàng", [])
    if not orders:
        orders = [{
            "kích thước": None,
            "màu sắc": None,
            "số bộ": 1
        }]

    for i, order in enumerate(orders):
        for field in order_fields:
            if order.get(field) is None:
                return f"đơn hàng {i + 1}: {field}"

    for field in global_fields:
        if info_status.get(field) is None:
            return field

    return None  # All info present

def send_message_to_chatwoot(account_id, conversation_id, message, token):
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    headers = {
        "api_access_token": f"{token}",
        "Content-Type": "application/json"
    }
    payload = {
        "content": message,
        "message_type": "outgoing",
        "private": False,
        "content_type": "text"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[send_message_to_chatwoot] Error sending message: {e}")
        return None

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()

    # ✅ Prevent empty payloads or missing content
    if not data or "content" not in data or not data.get("content", "").strip():
        return jsonify(data), 200

    # ✅ Ignore bot's own messages (outgoing messages)
    if data.get("message_type") == "outgoing":
        return jsonify(data), 200

    query = data["content"].strip()
    convo_id = str(data.get("conversation", {}).get("id", ""))
    chatwoot_convo_id = convo_id

    if not chatwoot_convo_id:
        return jsonify({"error": "Missing conversation ID"}), 400

    # ✅ Run the heavy work in a background thread
    threading.Thread(target=handle_chatwoot_message, args=(query, chatwoot_convo_id)).start()

    # ✅ Immediately return an empty response so Chatwoot can proceed
    return jsonify(data), 200

def handle_chatwoot_message(query, convo_id):
    try:
        # ✅ Get valid token and account_id
        token = get_saved_token()
        is_valid, account_id = validate_token(token)

        if not is_valid or not account_id:
            token = login_to_chatwoot()
            is_valid, account_id = validate_token(token)
            if not is_valid or not account_id:
                print("[ask] Authentication failed")
                return

        convo_history = get_conversation_history(convo_id)
        info_status = detect_missing_info(convo_history + [query])
        print("Missing info status:", info_status)

        next_missing = get_next_missing_field(info_status)
        print("Next missing field:", next_missing)

        must_know_context = [
            "Giá mỗi bộ là 175,000 VNĐ. Mua từ 2 set giá còn 170k",
            "trả lời ngắn gọn, lịch sự, xưng hô là em, khách hàng là chị",
            "bắt đầu câu trả lời bằng từ Dạ",
            "các câu khách hỏi giá như bao nhiêu, ib, inbox, xin giá ...",
        ]

        extra_context_map = {
            "kích thước": [
                "hỏi kích thước thì hỏi chiều cao, cân nặng, độ tuổi của bé",
                "có các cỡ 80, 90, 100, 110, 120, 130, 140",
                "80=>1-2tuổi,9-11kg,90cm",
                "90=>2-3tuổi,11-14kg,95cm",
                "100=>3-4tuổi,14-16kg,104cm",
                "110=>4-5tuổi,17-20kg,112cm",
                "120=>5-6tuổi,20-23kg,120cm",
                "130=>6-7tuổi,23-26kg,126cm",
                "140=>8-9tuổi,27-30kg,134cm",
                "phải nói là bé nhà mình phù hợp với cỡ xx và xx ạ (chọn 2 cỡ phù hợp nhất gồm cỡ vừa nhất và cỡ lớn hơn 1 cỡ)"
                "sau khi khách hàng đã báo thông tin thì mới hỏi 2 cỡ phù hợp nhất cho khách hàng chọn ví dụ 90 100 chị chọn cỡ nào ạ",
                "nếu khách đã báo thông tin chiều cao cân nặng rồi thì không hỏi chiều cao cân nặng độ tuổi nữa mà chỉ hỏi confirm cỡ thôi",
                "câu hỏi thì để trong biến question_ask_next, ko để trong biến answer_only"
            ],
            "màu sắc": [
                "Sản phẩm có các màu: Trắng, Hồng, Xanh cốm."
            ],
            "số bộ": [],
            "số điện thoại": [
                "Dùng mẫu câu: Chị cho em xin số điện thoại và địa chỉ để em gửi chị ạ",
                "Số điện thoại sẽ dùng để bên vận chuyển liên hệ khi giao hàng."
            ],
            "địa chỉ giao hàng": [
                "Em cần địa chỉ đầy đủ để giao hàng chính xác, gồm: số nhà, tên đường, phường/xã, quận/huyện, tỉnh/thành."
            ]
        }

        rag = RAG()  # Initialize a fresh RAG instance
        rag_contexts = rag.search(query, convo_history)
        context_texts = [text for text, _ in rag_contexts]

        if next_missing is None:
            normalized_missing = ""
        else:
            normalized_missing = re.sub(r"^đơn hàng \d+:\s*", "", next_missing).strip()

        # Combine contexts
        combined_contexts = must_know_context[:]
        if normalized_missing in extra_context_map:
            combined_contexts.extend(extra_context_map[normalized_missing])
        combined_contexts.extend(context_texts)

        # Logic to override next_missing if needed
        if normalized_missing == "số điện thoại" and info_status.get("địa chỉ giao hàng") is None:
            next_missing = "số điện thoại & địa chỉ giao hàng"

        answer = answer_question(query, combined_contexts, next_missing, info_status)

        if answer:
            rag.store_and_link_query(convo_id, query, source='user')
            rag.store_and_link_query(convo_id, answer["answer_only"], source='bot')
            rag.store_and_link_query(convo_id, answer["question_ask_next"], source='bot')
            send_message_to_chatwoot(account_id, convo_id, answer["answer_only"], token)
            send_message_to_chatwoot(account_id, convo_id, answer["question_ask_next"], token)

        del rag  # Optional memory cleanup
    except Exception as e:
        print(f"[handle_chatwoot_message] Error: {e}")

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
