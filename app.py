"""
Streamlit GUI for RAG Grammar Teacher
- Input: Student question in Vietnamese
- Output: Teacher response using RAG chain
"""

import streamlit as st
from dotenv import load_dotenv
import os
import json
import sys
import subprocess
from pathlib import Path

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ OPENAI_API_KEY not found in environment. Please add it to .env file.")
    st.stop()

vector_store_path = Path("INTENSIVE_GRAMMAR_faiss_index")
chunks_file = Path("INTENSIVE_GRAMMAR_chunks.jsonl")

if not vector_store_path.exists() or not chunks_file.exists():
    subprocess.run([sys.executable, "build_index.py"], check=True)

if not vector_store_path.exists() or not chunks_file.exists():
    st.error("❌ Failed to build vector store.")
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




if not vector_store_path.exists():
    st.error("❌ Vector store không tìm thấy. Vui lòng chạy notebook để tạo FAISS index.")
    st.stop()




if not chunks_file.exists():
    st.error("❌ Chunks file không tìm thấy. Vui lòng chạy notebook trước.")
    st.stop()

# ============================================================================
# INITIALIZE SESSION STATE (Cache for expensive operations)
# ============================================================================
@st.cache_resource
def load_rag_chain(model_name: str, temp: float):
    """Load and initialize RAG chain - cached to avoid reloading"""
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load vector store from disk
    vector_store = FAISS.load_local(
        "INTENSIVE_GRAMMAR_faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temp,
        api_key=openai_api_key
    )
    
    # Define context retrieval function
    def get_context(query, k=5):
        results = vector_store.similarity_search(query, k=k)
        texts = [doc.page_content for doc in results]
        
        # Check for example markers
        markers = ['ví dụ', 'đáp án', 'ví-dụ', 'example', 'ans:']
        def has_example(text):
            t = text.lower()
            return any(m in t for m in markers)
        
        contains_example = any(has_example(t) for t in texts)
        
        # Fallback: scan JSONL for examples if not found
        if not contains_example:
            try:
                with open('INTENSIVE_GRAMMAR_chunks.jsonl', 'r', encoding='utf-8') as f:
                    for ln in f:
                        obj = json.loads(ln)
                        txt = obj.get('text', '').lower()
                        if any(m in txt for m in markers):
                            texts.append(obj.get('text', ''))
                            contains_example = True
                            break
            except FileNotFoundError:
                pass
        
        context = "\n\n---\n\n".join(texts)
        return context, results
    
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an English grammar teacher. "
            "A Vietnamese student has asked you a question about grammar.\n\n"
            "RETRIEVED CONTEXT:\n{context}\n\n"
            "STUDENT QUESTION:\n{question}\n\n"
            "Please answer using Vietnamese language following these steps:\n"
            "1. Carefully read the CONTEXT retrieved from the database. Only use information that appears in the CONTEXT.\n"
            "2. Give a short and clear explanation of the grammar point the student is asking about. Explain the meaning, usage, and structure (if included in the context).\n"
            "3. Provide an example (use examples from the context if available). If the retrieved context contains examples, include at least one example verbatim and label it exactly as 'Ví dụ:'.\n"
            "4. Re-explain the concept using simpler Vietnamese so that a language learner can understand it easily.\n"
            "5. If the concept does not exist in the retrieved context, tell me honestly.\n\n"
            "Your response:"
        )
    )
    
    # Build RAG chain
    rag_chain = (
        {
            "context": lambda x: get_context(x["question"], k=k_results)[0],
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
    )
    
    return rag_chain, get_context

# Load RAG chain
rag_chain, get_context_func = load_rag_chain(model_choice, temperature)

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
                # Retrieve context and show sources
                context, source_docs = get_context_func(user_question, k=k_results)
                
                # Get response from RAG chain
                response = rag_chain.invoke({"question": user_question})
                
                # Extract answer text
                if hasattr(response, 'content'):
                    answer_text = response.content
                elif isinstance(response, dict) and 'content' in response:
                    answer_text = response['content']
                else:
                    answer_text = str(response)
                
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
                        if hasattr(doc, 'metadata'):
                            meta = doc.metadata
                            if 'page' in meta:
                                st.caption(f"📄 Trang: {meta['page']}")
                        
                        st.text_area(
                            f"Content {i}",
                            value=doc.page_content,
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
    📚 RAG Grammar Teacher v1.0 | Powered by OpenAI + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
