"""
Streamlit GUI for RAG Grammar Teacher
Client for FastAPI RAG service
- Input: Student question in Vietnamese
- Output: Teacher response using RAG API
"""

import streamlit as st
from dotenv import load_dotenv
import os
import json
import requests
from pathlib import Path

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ OPENAI_API_KEY not found in environment. Please add it to .env file.")
    st.stop()

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# Check if API is running
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if not check_api_health():
    st.error("❌ API server is not running. Please start the API server with: `uvicorn api:app --reload`")
    st.stop()


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="📚 Grammar Teacher - RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR - Configuration & Info
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    
    # Model settings
    model_choice = st.selectbox(
        "Chọn mô hình",
        ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    
    temperature = st.slider(
        "Temperature (độ sáng tạo)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Thấp = đáp án chặt chẽ, Cao = đáp án sáng tạo hơn"
    )
    
    k_results = st.slider(
        "Số lượng chunks để lấy",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Số lượng đoạn văn bản liên quan được truy xuất"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Thông tin")
    st.info(
        "💡 Ứng dụng này sử dụng **RAG** (Retrieval-Augmented Generation) "
        "để trả lời câu hỏi về ngữ pháp tiếng Anh dựa trên tài liệu."
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown("# 📚 Ngữ pháp - Hệ thống Q&A")
st.markdown("---")

# ============================================================================
# API FUNCTIONS
# ============================================================================
def ask_question_api(question: str, model: str, temperature: float, k_results: int):
    """Send question to RAG API and get response"""
    payload = {
        "question": question,
        "model": model,
        "temperature": temperature,
        "k_results": k_results
    }

    try:
        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

def update_config_api(model: str = None, temperature: float = None):
    """Update API configuration"""
    payload = {}
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        response = requests.put(f"{API_BASE_URL}/config", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to update config: {str(e)}")

def get_config_api():
    """Get current API configuration"""
    try:
        response = requests.get(f"{API_BASE_URL}/config", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {"model": "gpt-4o", "temperature": 0.3}

# ============================================================================
# MAIN INPUT/OUTPUT INTERFACE
# ============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    # Input section
    st.markdown("### ❓ Hỏi câu hỏi")
    
    # Text input for question
    user_question = st.text_area(
        "Nhập câu hỏi của bạn về ngữ pháp tiếng Anh (bằng tiếng Việt):",
        placeholder="Ví dụ: Cách sử dụng thì hiện tại hoàn thành là gì?",
        height=100,
        label_visibility="collapsed"
    )
    
    # Pre-defined examples
    st.markdown("**Câu hỏi gợi ý:**")
    example_questions = [
        "sử dụng thì hiện tại hoàn thành như thế nào?",
        "thì tương lai đơn là gì?",
        "cách dùng thì quá khứ tiếp diễn"
    ]
    
    for example in example_questions:
        if st.button(f"📌 {example}", key=example):
            user_question = example

with col2:
    # Settings preview
    st.markdown("### 🔧 Cài đặt hiện tại")
    st.markdown(f"""
    - **Model**: {model_choice}
    - **Temperature**: {temperature}
    - **Top-k**: {k_results}
    """)

# ============================================================================
# PROCESS QUESTION AND DISPLAY ANSWER
# ============================================================================
if st.button("🚀 Gửi câu hỏi", type="primary", use_container_width=True):
    if not user_question.strip():
        st.warning("⚠️ Vui lòng nhập một câu hỏi.")
    else:
        # Show loading state
        with st.spinner("⏳ Đang xử lý câu hỏi..."):
            try:
                # Call API
                api_response = ask_question_api(
                    user_question,
                    model_choice,
                    temperature,
                    k_results
                )

                answer_text = api_response["answer"]
                source_docs = api_response["sources"]

                # Display answer
                st.success("✅ Đã nhận câu trả lời!")
                st.markdown("---")

                st.markdown("### 🎓 Câu trả lời")
                st.markdown(answer_text)

                # Display sources in expandable section
                with st.expander("📚 Xem nguồn tài liệu (Retrieved Context)"):
                    st.markdown(f"**Đã tìm thấy {len(source_docs)} đoạn văn bản liên quan:**")
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Nguồn {i}:**")
                        if doc.get('metadata') and 'page' in doc['metadata']:
                            st.caption(f"📄 Trang: {doc['metadata']['page']}")

                        st.text_area(
                            f"Content {i}",
                            value=doc['content'],
                            height=150,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        st.markdown("---")

                # Option to save the Q&A
                st.markdown("### 💾 Lưu Q&A")
                col_save1, col_save2 = st.columns(2)

                with col_save1:
                    if st.button("💾 Lưu vào file", use_container_width=True):
                        qa_entry = {
                            "question": user_question,
                            "answer": answer_text,
                            "model": model_choice,
                            "temperature": temperature
                        }

                        # Append to QA history file
                        qa_file = Path("qa_history.jsonl")
                        with open(qa_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(qa_entry, ensure_ascii=False) + '\n')

                        st.success(f"✅ Đã lưu vào `{qa_file}`")

            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý: {str(e)}")
                st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
    📚 RAG Grammar Teacher v2.0 | FastAPI + Streamlit Client
    </div>
    """,
    unsafe_allow_html=True
)
