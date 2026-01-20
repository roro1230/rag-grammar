## SETUP & RUN INSTRUCTIONS

1. Create and activate virtual environment (recommended)

python -m venv .venv

.venv/Scripts/activate (Windows)

source .venv/bin/activate (Mac/Linux)

---

2. Install dependencies

pip install -r requirements.txt

---

3. Create environment variables

Create a file named `.env` in the project root directory.

Example `.env` content:

OPENAI_API_KEY=your_openai_api_key_here

---

4. Build the vector index (run once)

This step processes the grammar documents and builds a FAISS index.

python build_index.py

After running this step, a vector index folder will be created.

---

5. Run the Streamlit application

streamlit run app.py
