1) Tạo file `.env`
- Tạo file `.env` ở thư mục gốc của project và thêm biến môi trường OpenAI API key:

	OPENAI_API_KEY=your_key_here

	Lưu ý: thay `your_key_here` bằng khóa thực tế của bạn.

2) Tạo và kích hoạt virtual environment
- Tạo virtualenv (ví dụ dùng tên `.venv`):

	python3 -m venv .venv

- Kích hoạt môi trường ảo:

	source .venv/bin/activate

- Cập nhật pip:

	pip install --upgrade pip

4) Cài đặt dependencies từ `requirements.txt`
- Cài đặt tất cả package được liệt kê:

	pip install -r requirements.txt

5) Cài đặt Streamlit
- Cài đặt Streamlit nếu chưa có:

	pip install streamlit

6) Chạy `RAG.ipynb` để tạo file `INTENSIVE_GRAMMAR_chunks.jsonl`
- Mở notebook `RAG.ipynb` bằng VS Code (Jupyter) hoặc Jupyter Lab/Notebook.
- CHÚ Ý: chọn kernel trỏ tới `.venv` để các package đã cài được sử dụng trong notebook.
- Chạy lần lượt các cell theo thứ tự:
	- Cell tải PDF và làm sạch nội dung
	- Cell chunking (tạo `chunks`)
	- Cell tạo embeddings và lưu FAISS index

- Sau khi chạy xong cell chunking, file `INTENSIVE_GRAMMAR_chunks.jsonl` sẽ được lưu trong thư mục dự án.

7) Chạy ứng dụng Streamlit (`app.py`)
- Sau khi đã có index/chunks, chạy ứng dụng:

	streamlit run app.py

- Mở trình duyệt theo địa chỉ mà Streamlit in ra.

Ghi chú và mẹo
- Đảm bảo kernel Jupyter sử dụng đúng Python interpreter trong `.venv`. Nếu chưa có kernel, cài ipykernel và đăng ký:

	pip install ipykernel
	python -m ipykernel install --user --name=rag-grammar-venv --display-name "rag-grammar (.venv)"



- Nếu bạn gặp lỗi liên quan tới OpenAI key khi chạy cell RAG, kiểm tra lại rằng `.env` đã được load (cell đầu notebook thường gọi `load_dotenv()`).



