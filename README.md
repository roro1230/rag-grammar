## Project Overview

### Problem

English learners often struggle to understand and apply grammar rules effectively. Traditional grammar materials are dense, static, and hard to navigate, making it difficult to quickly find clear explanations for specific questions.

### Solution

This project implements a Retrieval-Augmented Generation (RAG) system that provides accurate and beginner-friendly grammar explanations by combining document retrieval and large language models:

- **LangChain & FAISS**: Chunk, embed, and retrieve relevant context from grammar PDF documents.
- **OpenAI LLMs**: Generate concise, Vietnamese explanations grounded in retrieved context.
- **FastAPI Backend**: Exposes a RESTful API to handle RAG pipelines and inference logic.
- **Streamlit UI**: Interactive web interface for asking questions, selecting models, tuning parameters, inspecting retrieved context, and saving Q&A history.

### Results

- A complete end-to-end web application for English grammar Q&A.
- Clear, Vietnamese explanations tailored for learners.
- Modular backend architecture, easy to extend with new documents or models.
- CI/CD pipeline with GitHub Actions and pytest to ensure code quality.
- One-command startup using Docker Compose for reproducible deployment.

## SETUP & RUN INSTRUCTIONS

### Option 1: Run with Docker (Recommended - One Command)

1. **Create environment file**

   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

2. **Run with Docker Compose**

   ```bash
   docker-compose up --build
   ```

   **Note:** The first build may take several minutes as Docker needs to download base images and install dependencies.

   The application will be available at:
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI Docs**: http://localhost:8000/docs

### Option 2: Run Locally (Development)

1. Create and activate virtual environment (recommended)

   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # Windows
   source .venv/bin/activate  # Mac/Linux
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Create environment variables

   Create a file named `.env` in the project root directory.

   Example `.env` content:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. Build the vector index (if not already built)

   ```bash
   python build_index.py
   ```

5. Start the FastAPI server

   ```bash
   python run_api.py
   ```

   The API will be available at: http://localhost:8000

   API Documentation: http://localhost:8000/docs

6. Run the Streamlit client (in a separate terminal)

   ```bash
   streamlit run app.py
   ```

![Demo UI](assets/demo1.jpg)

![Demo UI](assets/demo2.jpg)
