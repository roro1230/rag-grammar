"""
RAG Service Module
Handles all RAG-related logic: vector store loading, context retrieval, and response generation.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


class RAGService:
    def __init__(self, openai_api_key: str, vector_store_path: str = "INTENSIVE_GRAMMAR_faiss_index",
                 chunks_file: str = "INTENSIVE_GRAMMAR_chunks.jsonl"):
        self.openai_api_key = openai_api_key
        self.vector_store_path = Path(vector_store_path)
        self.chunks_file = Path(chunks_file)
        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self.rag_chain = None
        self.prompt_template = None

    def initialize(self, model_name: str = "gpt-4o", temperature: float = 0.3):
        """Initialize the RAG service with model and temperature"""
        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load vector store from disk
        self.vector_store = FAISS.load_local(
            str(self.vector_store_path),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.openai_api_key
        )

        # Create prompt template
        self.prompt_template = PromptTemplate(
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
        self.rag_chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
        )

    def get_context(self, query: str, k: int = 5) -> Tuple[str, List[Document]]:
        """Retrieve context for a given query"""
        results = self.vector_store.similarity_search(query, k=k)
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
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        obj = json.loads(line)
                        txt = obj.get('text', '').lower()
                        if any(m in txt for m in markers):
                            texts.append(obj.get('text', ''))
                            contains_example = True
                            break
            except FileNotFoundError:
                pass

        context = "\n\n---\n\n".join(texts)
        return context, results

    def generate_response(self, question: str, k: int = 5) -> Tuple[str, List[Document]]:
        """Generate a response for the given question"""
        if not self.rag_chain:
            raise ValueError("RAG service not initialized. Call initialize() first.")

        # Get context
        context, source_docs = self.get_context(question, k)

        # Generate response
        response = self.rag_chain.invoke({
            "context": context,
            "question": question
        })

        # Extract answer text
        if hasattr(response, 'content'):
            answer_text = response.content
        elif isinstance(response, dict) and 'content' in response:
            answer_text = response['content']
        else:
            answer_text = str(response)

        return answer_text, source_docs

    def update_model(self, model_name: str, temperature: float):
        """Update the model and temperature"""
        self.initialize(model_name, temperature)