"""
Baseline RAG implementation for GDPR question answering.

Provides a simple RAG pipeline: query → FAISS retrieval → prompt → LLM answer.
"""

import os
from typing import List, Dict, Any, Optional
import warnings


class BaselineRAG:
    """
    Baseline RAG system for GDPR queries.
    
    Implements a simple retrieval-augmented generation pipeline:
    1. Query the FAISS vector store for relevant chunks
    2. Format retrieved context into a prompt
    3. Send prompt to LLM for answer generation
    """
    
    def __init__(
        self,
        faiss_path: str = "faiss_index",
        openai_api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        top_k: int = 3
    ):
        """
        Initialize the baseline RAG system.
        
        Args:
            faiss_path: Path to the FAISS index directory
            openai_api_key: OpenAI API key (optional, uses env var if not provided)
            model: OpenAI model to use for generation
            top_k: Number of documents to retrieve
        """
        self.faiss_path = faiss_path
        self.model = model
        self.top_k = top_k
        
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_api_key = openai_api_key
        
        self.vectorstore = None
        self.llm = None
        
        if self.openai_api_key:
            print(f"Initializing BaselineRAG with model {model}")
            # TODO: Load actual FAISS index and initialize LLM
            # from langchain.vectorstores import FAISS
            # from langchain.embeddings import OpenAIEmbeddings
            # from langchain.chat_models import ChatOpenAI
            # self.vectorstore = FAISS.load_local(faiss_path, OpenAIEmbeddings())
            # self.llm = ChatOpenAI(model=model, openai_api_key=openai_api_key)
        else:
            print("[DRY-RUN] BaselineRAG initialized in dry-run mode (no API key)")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User question
            
        Returns:
            List of retrieved documents with content and metadata
        """
        if self.vectorstore:
            # TODO: Implement actual retrieval
            # docs = self.vectorstore.similarity_search(query, k=self.top_k)
            # return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            pass
        
        # Dry-run mode: return placeholder results
        print(f"[DRY-RUN] Retrieving top {self.top_k} docs for query: '{query[:50]}...'")
        return [
            {
                "content": "Article 1: This Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data.",
                "metadata": {"article": 1, "page": 1},
                "score": 0.92
            },
            {
                "content": "Article 4: 'personal data' means any information relating to an identified or identifiable natural person ('data subject').",
                "metadata": {"article": 4, "page": 2},
                "score": 0.88
            },
            {
                "content": "Article 5: Personal data shall be processed lawfully, fairly and in a transparent manner.",
                "metadata": {"article": 5, "page": 3},
                "score": 0.85
            }
        ]
    
    def format_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Format the prompt with retrieved context.
        
        Args:
            query: User question
            context_docs: Retrieved documents
            
        Returns:
            Formatted prompt string
        """
        context_str = "\n\n".join([
            f"[Article {doc['metadata'].get('article', 'N/A')}]: {doc['content']}"
            for doc in context_docs
        ])
        
        prompt = f"""You are a helpful assistant answering questions about GDPR (General Data Protection Regulation).
Use the following context from the GDPR regulation to answer the question. 
If the context doesn't contain enough information, say so.

Context:
{context_str}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            Generated answer
        """
        if self.llm:
            # TODO: Implement actual LLM call
            # response = self.llm.invoke(prompt)
            # return response.content
            pass
        
        # Dry-run mode: return placeholder answer
        print("[DRY-RUN] Generating answer with LLM...")
        return "Based on the GDPR regulation, personal data refers to any information relating to an identified or identifiable natural person. The regulation lays down rules for protecting natural persons with regard to the processing of personal data. [Placeholder answer - requires OpenAI API key for actual generation]"
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve, format, generate.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant documents
        docs = self.retrieve(question)
        
        # Format prompt
        prompt = self.format_prompt(question, docs)
        
        # Generate answer
        answer = self.generate_answer(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": docs,
            "num_sources": len(docs)
        }


def run_example():
    """Run example RAG query."""
    print("=== Baseline RAG Example ===\n")
    
    rag = BaselineRAG()
    
    question = "What is personal data according to GDPR?"
    result = rag.query(question)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources used: {result['num_sources']}")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. Article {source['metadata'].get('article', 'N/A')} (score: {source.get('score', 'N/A')})")


if __name__ == "__main__":
    run_example()
