import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import app
from rag_service import RAGService


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


def test_api_root(client):
    """Check root endpoint returns basic info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_api_health(client):
    """Check health endpoint is healthy"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.fixture
def mock_rag_service():
    """Mocked RAG service for testing without real API calls"""
    with patch('rag_service.ChatOpenAI') as mock_chat, \
         patch('rag_service.HuggingFaceEmbeddings') as mock_embeddings, \
         patch('rag_service.FAISS') as mock_faiss, \
         patch('rag_service.Path') as mock_path:

        # Mock paths exist
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock embeddings
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock FAISS vector store
        mock_faiss_instance = MagicMock()
        mock_faiss.load_local.return_value = mock_faiss_instance
        mock_faiss.return_value = mock_faiss_instance

        # Mock OpenAI chat model
        mock_chat_instance = MagicMock()
        mock_chat_instance.model_name = "gpt-4o"
        mock_chat_instance.temperature = 0.3
        mock_chat.return_value = mock_chat_instance

        # Initialize service
        service = RAGService("dummy_key")
        service.initialize()

        yield service


def test_rag_service_initialization(mock_rag_service):
    """Verify RAG service components are initialized"""
    service = mock_rag_service
    assert service.embedding_model is not None
    assert service.vector_store is not None
    assert service.llm is not None
    assert service.rag_chain is not None


def test_rag_service_get_context(mock_rag_service):
    """Test context retrieval from vector store"""
    service = mock_rag_service

    # Mock search results
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "This is a test context."
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Another piece of context."
    service.vector_store.similarity_search.return_value = [mock_doc1, mock_doc2]

    context, docs = service.get_context("test query", k=2)

    assert len(docs) == 2
    assert "test context" in context.lower()
    service.vector_store.similarity_search.assert_called_once_with("test query", k=2)


