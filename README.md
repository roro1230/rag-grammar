## SETUP & RUN INSTRUCTIONS

1. Create and activate virtual environment (recommended)

python -m venv .venv
.venv\Scripts\activate (Windows)
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

1. Run the Streamlit application

streamlit run app.py
