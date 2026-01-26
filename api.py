"""
FastAPI application for RAG Grammar Teacher
Provides REST API endpoints for RAG functionality.
"""

import json
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import sys
from pathlib import Path
from rag_service import RAGService
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please add it to .env file.")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Grammar Teacher API",
    description="API for RAG-based English grammar Q&A system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG service instance
rag_service = None

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    model: Optional[str] = "gpt-4o"
    temperature: Optional[float] = 0.3
    k_results: Optional[int] = 5

class SourceDocument(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    model: str
    temperature: float
    k_results: int

class ConfigUpdate(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None

def initialize_rag_service():
    """Initialize RAG service if not already done"""
    global rag_service
    if rag_service is None:
        vector_store_path = Path("INTENSIVE_GRAMMAR_faiss_index")
        chunks_file = Path("INTENSIVE_GRAMMAR_chunks.jsonl")

        # Build index if not exists
        if not vector_store_path.exists() or not chunks_file.exists():
            print("Building vector store...")
            build_vector_index()

        if not vector_store_path.exists() or not chunks_file.exists():
            raise RuntimeError("Failed to build vector store.")

        rag_service = RAGService(openai_api_key)
        rag_service.initialize()

def build_vector_index():
    """Build FAISS index from PDF document"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    PDF_PATH = "INTENSIVE GRAMMAR.pdf"
    INDEX_DIR = "INTENSIVE_GRAMMAR_faiss_index"
    CHUNKS_FILE = "INTENSIVE_GRAMMAR_chunks.jsonl"

    if Path(INDEX_DIR).exists() and Path(CHUNKS_FILE).exists():
        print("Index already exists. Skipping build.")
        return

    print("Loading PDF document...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("Cleaning documents...")
    clean_docs = []
    for d in docs:
        text = d.page_content.strip()
        if len(text) > 50:
            clean_docs.append(d)

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(clean_docs)

    print("Saving chunks to JSONL...")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            json.dump({"text": c.page_content}, f, ensure_ascii=False)
            f.write("\n")

    print("Creating embeddings and FAISS index...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(INDEX_DIR)

    print("FAISS index and chunks built successfully.")

@app.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup"""
    initialize_rag_service()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG Grammar Teacher API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a grammar question and get RAG response"""
    try:
        # Update model config if different
        if rag_service.llm.model_name != request.model or rag_service.llm.temperature != request.temperature:
            rag_service.update_model(request.model, request.temperature)

        # Generate response
        answer, source_docs = rag_service.generate_response(
            request.question,
            k=request.k_results
        )

        # Format sources
        sources = [
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata if hasattr(doc, 'metadata') else None
            )
            for doc in source_docs
        ]

        return AnswerResponse(
            answer=answer,
            sources=sources,
            model=request.model,
            temperature=request.temperature,
            k_results=request.k_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.put("/config")
async def update_config(config: ConfigUpdate):
    """Update model configuration"""
    try:
        if config.model or config.temperature is not None:
            current_model = rag_service.llm.model_name if rag_service.llm else "gpt-4o"
            current_temp = rag_service.llm.temperature if rag_service.llm else 0.3

            new_model = config.model or current_model
            new_temp = config.temperature if config.temperature is not None else current_temp

            rag_service.update_model(new_model, new_temp)

        return {"message": "Configuration updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current configuration"""
    if not rag_service or not rag_service.llm:
        return {"model": "gpt-4o", "temperature": 0.3}

    return {
        "model": rag_service.llm.model_name,
        "temperature": rag_service.llm.temperature
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)