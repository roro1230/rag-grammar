from dotenv import load_dotenv
import os
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

PDF_PATH = "INTENSIVE_GRAMMAR.pdf"
INDEX_DIR = "INTENSIVE_GRAMMAR_faiss_index"
CHUNKS_FILE = "INTENSIVE_GRAMMAR_chunks.jsonl"

if Path(INDEX_DIR).exists() and Path(CHUNKS_FILE).exists():
    print("Index already exists. Skipping build.")
    exit(0)

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

clean_docs = []
for d in docs:
    text = d.page_content.strip()
    if len(text) > 50:
        clean_docs.append(d)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(clean_docs)

with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps({"text": c.page_content}, ensure_ascii=False) + "\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embedding_model)
vector_store.save_local(INDEX_DIR)

print("FAISS index and chunks built successfully.")
