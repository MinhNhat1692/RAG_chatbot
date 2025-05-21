from flask import Flask, request, jsonify
from rag import RAG, answer_question, get_conversation_history, detect_missing_info
from dotenv import load_dotenv
import os
import requests
import json
import threading

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

    # Check each order in sequence
    for i, order in enumerate(info_status.get("đơn hàng", [])):
        for field in order_fields:
            if order.get(field) is None:
                return f"đơn hàng {i + 1}: {field}"

    # Check shared fields
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
            "trong biến answer chỉ có câu trả lời hoặc câu hỏi xác nhận, câu hỏi tiếp theo cần đặt ở biến question_ask_next"
            "Thông tin đơn hàng trong biến order_info thì để ở dạng text đẹp: Kích thước: xx /n Màu sắc: xx /n Số bộ: /n"
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
                "Sản phẩm có các màu: Trắng, Hồng, Xanh cốm.",
                "khách chọn màu gần đúng thì confirm lại. ví dụ chọn xanh thì hỏi lại là có phải chọn xanh cốm không"
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

        combined_contexts = must_know_context[:]
        if next_missing in extra_context_map:
            combined_contexts.extend(extra_context_map[next_missing])
        combined_contexts.extend(context_texts)

        if next_missing == "số điện thoại" and info_status.get("địa chỉ giao hàng") is None:
            next_missing = "số điện thoại & địa chỉ giao hàng"

        answer = answer_question(query, combined_contexts, next_missing, info_status)

        if answer:
            rag.store_and_link_query(convo_id, query, source='user')
            rag.store_and_link_query(convo_id, answer["answer"], source='bot')
            rag.store_and_link_query(convo_id, answer["question_ask_next"], source='bot')
            send_message_to_chatwoot(account_id, convo_id, answer["order_info"], token)
            send_message_to_chatwoot(account_id, convo_id, answer["answer"], token)
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
