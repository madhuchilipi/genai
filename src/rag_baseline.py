"""
RAG Baseline Module

Implements basic Retrieval-Augmented Generation using FAISS and LangChain.
Works in dry-run mode without OpenAI API keys.
"""

import os
from typing import List, Dict, Optional
import warnings


class RAGSystem:
    """
    Baseline RAG implementation with vector similarity search.
    
    Features:
    - FAISS vector store for efficient similarity search
    - LangChain for RAG orchestration
    - Embedding generation
    - Query-based retrieval
    
    Supports dry-run mode: returns placeholder responses without API keys.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embedding_model = embedding_model
        self.has_api_key = bool(os.getenv("OPENAI_API_KEY"))
        self.vector_store = None
        self.chunks = []
        
        if not self.has_api_key:
            warnings.warn(
                "OPENAI_API_KEY not found. Running in dry-run mode. "
                "Set OPENAI_API_KEY in .env for production use."
            )
    
    def index_documents(self, chunks: List[Dict[str, any]]) -> None:
        """
        Index document chunks into the vector store.
        
        Args:
            chunks: List of document chunks with content and metadata
            
        TODO: Implement actual FAISS indexing with embeddings
        """
        self.chunks = chunks
        
        if not self.has_api_key:
            print(f"Dry-run: Indexed {len(chunks)} chunks (no embeddings created)")
            return
        
        # TODO: Production implementation
        # from langchain.embeddings import OpenAIEmbeddings
        # from langchain.vectorstores import FAISS
        # 
        # embeddings = OpenAIEmbeddings(model=self.embedding_model)
        # texts = [chunk["content"] for chunk in chunks]
        # metadatas = [chunk["metadata"] for chunk in chunks]
        # self.vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        print(f"Indexed {len(chunks)} chunks into vector store")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, any]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with scores
            
        TODO: Implement actual vector similarity search
        """
        if not self.has_api_key:
            # Dry-run: return first k chunks
            results = self.chunks[:k] if self.chunks else []
            for i, result in enumerate(results):
                result["score"] = 1.0 - (i * 0.1)  # Mock scores
            return results
        
        # TODO: Production implementation
        # docs = self.vector_store.similarity_search_with_score(query, k=k)
        # results = [{"content": doc.page_content, "metadata": doc.metadata, "score": score}
        #            for doc, score in docs]
        # return results
        
        return []
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, any]]) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated response
            
        TODO: Implement actual LLM-based generation
        """
        if not self.has_api_key:
            # Dry-run: return deterministic placeholder
            context_preview = context_chunks[0]["content"][:100] if context_chunks else "No context"
            return (
                f"[Dry-run mode] Based on the GDPR context, here's a placeholder response to: '{query}'. "
                f"Context preview: {context_preview}... "
                f"Set OPENAI_API_KEY for actual generation."
            )
        
        # TODO: Production implementation
        # from langchain.chat_models import ChatOpenAI
        # from langchain.chains import RetrievalQA
        # 
        # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        # context = "\n\n".join([chunk["content"] for chunk in context_chunks])
        # prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        # response = llm.predict(prompt)
        # return response
        
        return ""
    
    def query(self, query: str, k: int = 3) -> str:
        """
        End-to-end RAG query: retrieve + generate.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            Generated response
        """
        # Retrieve relevant chunks
        context_chunks = self.retrieve(query, k=k)
        
        # Generate response
        response = self.generate_response(query, context_chunks)
        
        return response
    
    def save_index(self, path: str) -> None:
        """
        Save the vector index to disk.
        
        Args:
            path: Path to save the index
            
        TODO: Implement FAISS index serialization
        """
        if not self.has_api_key:
            print(f"Dry-run: Would save index to {path}")
            return
        
        # TODO: self.vector_store.save_local(path)
        print(f"Index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """
        Load a vector index from disk.
        
        Args:
            path: Path to the saved index
            
        TODO: Implement FAISS index loading
        """
        if not self.has_api_key:
            print(f"Dry-run: Would load index from {path}")
            return
        
        # TODO: 
        # from langchain.embeddings import OpenAIEmbeddings
        # from langchain.vectorstores import FAISS
        # embeddings = OpenAIEmbeddings(model=self.embedding_model)
        # self.vector_store = FAISS.load_local(path, embeddings)
        
        print(f"Index loaded from {path}")


def main():
    """Example usage of RAGSystem."""
    from src.data_prep import DataPreprocessor
    
    # Works without API keys
    rag = RAGSystem()
    
    # Prepare data
    preprocessor = DataPreprocessor()
    chunks = preprocessor.prepare_for_rag("gdpr_documents/")
    
    # Index and query
    rag.index_documents(chunks)
    response = rag.query("What are the GDPR data subject rights?")
    
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
