#!/usr/bin/env python3
"""
Script to run the FastAPI server for RAG Grammar Teacher
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    print("🚀 Starting RAG Grammar Teacher API server...")
    print("📚 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )