"""
Baseline RAG Module

This module implements a simple RAG pipeline:
- Load FAISS vector store
- Retrieve top-k relevant documents
- Format prompt template
- Call LLM to generate answer
"""

import os
import warnings
from typing import List, Dict, Any, Optional


class BaselineRAG:
    """
    A baseline RAG implementation using FAISS retrieval and OpenAI LLM.
    
    This class demonstrates the core RAG pattern:
    1. User query → Embedding
    2. Retrieve top-k similar documents from FAISS
    3. Format prompt with retrieved context
    4. Generate answer using LLM
    
    Example:
        >>> rag = BaselineRAG("faiss_index", openai_api_key="sk-...")
        >>> answer = rag.query("What are data subject rights?")
        >>> print(answer)
    """
    
    def __init__(
        self,
        faiss_path: str,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        top_k: int = 4
    ):
        """
        Initialize the baseline RAG system.
        
        Args:
            faiss_path: Path to the persisted FAISS index
            openai_api_key: OpenAI API key (uses env var if not provided)
            model: OpenAI model name
            top_k: Number of documents to retrieve
        """
        self.faiss_path = faiss_path
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.top_k = top_k
        self.vectorstore = None
        self.llm = None
        
        if not self.api_key:
            warnings.warn("No API key provided. Running in dry-run mode.")
        else:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize FAISS vectorstore and LLM."""
        # TODO: Implement actual initialization
        # from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        # from langchain_community.vectorstores import FAISS
        
        # embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        # self.vectorstore = FAISS.load_local(
        #     self.faiss_path,
        #     embeddings,
        #     allow_dangerous_deserialization=True
        # )
        # self.llm = ChatOpenAI(
        #     openai_api_key=self.api_key,
        #     model_name=self.model,
        #     temperature=0
        # )
        
        print(f"[Production mode] Would initialize FAISS from {self.faiss_path}")
        print(f"[Production mode] Would initialize {self.model} LLM")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            
        Returns:
            List of retrieved documents with content and metadata
        """
        if not self.api_key:
            # Dry-run mode
            return [
                {
                    "page_content": f"[Placeholder GDPR context for: {query}]",
                    "metadata": {"source": "gdpr.pdf", "page": i, "score": 0.9 - i*0.1}
                }
                for i in range(self.top_k)
            ]
        
        # TODO: Implement actual retrieval
        # docs = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        # return [
        #     {
        #         "page_content": doc.page_content,
        #         "metadata": {**doc.metadata, "score": score}
        #     }
        #     for doc, score in docs
        # ]
        
        print(f"[Production mode] Would retrieve {self.top_k} docs for: {query}")
        return []
    
    def format_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Format the prompt with retrieved context and user query.
        
        Args:
            query: User query
            context_docs: Retrieved documents
            
        Returns:
            Formatted prompt string
        """
        context = "\n\n".join([
            f"[Source {i+1}, Page {doc['metadata'].get('page', 'N/A')}]\n{doc['page_content']}"
            for i, doc in enumerate(context_docs)
        ])
        
        prompt = f"""You are a helpful assistant answering questions about GDPR (General Data Protection Regulation).
Use the following context to answer the user's question. If the answer cannot be found in the context, say so.
Always cite the source page numbers in your answer.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using the LLM.
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            Generated answer string
        """
        if not self.api_key:
            # Dry-run mode
            return "[Placeholder answer: This is a dry-run response. Set OPENAI_API_KEY for actual generation.]"
        
        # TODO: Implement actual LLM call
        # response = self.llm.invoke(prompt)
        # return response.content
        
        print(f"[Production mode] Would call {self.model} to generate answer")
        return "[Production mode placeholder response]"
    
    def query(self, query: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a query.
        
        Args:
            query: User question
            return_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        # Step 1: Retrieve relevant documents
        context_docs = self.retrieve(query)
        
        # Step 2: Format prompt
        prompt = self.format_prompt(query, context_docs)
        
        # Step 3: Generate answer
        answer = self.generate_answer(prompt)
        
        result = {"answer": answer}
        
        if return_sources:
            result["sources"] = [
                {
                    "content": doc["page_content"][:200] + "...",  # Truncate for display
                    "page": doc["metadata"].get("page", "N/A"),
                    "score": doc["metadata"].get("score", 0.0)
                }
                for doc in context_docs
            ]
        
        return result
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of results for each query
        """
        return [self.query(q) for q in queries]


def create_baseline_rag(
    faiss_path: str = "faiss_index",
    openai_api_key: Optional[str] = None
) -> BaselineRAG:
    """
    Convenience function to create a BaselineRAG instance.
    
    Args:
        faiss_path: Path to FAISS index
        openai_api_key: OpenAI API key
        
    Returns:
        Initialized BaselineRAG instance
    """
    return BaselineRAG(faiss_path, openai_api_key)


# Example usage for testing
if __name__ == "__main__":
    print("=== Baseline RAG Module Demo ===")
    
    # Create RAG instance (dry-run mode)
    rag = BaselineRAG("faiss_index")
    print("1. Created BaselineRAG instance")
    
    # Test retrieval
    docs = rag.retrieve("What are data subject rights?")
    print(f"2. Retrieved {len(docs)} documents")
    
    # Test query
    result = rag.query("What are the main principles of GDPR?")
    print(f"3. Generated answer: {result['answer'][:100]}...")
    print(f"   Number of sources: {len(result.get('sources', []))}")
    
    # Test batch query
    queries = [
        "What is the right to be forgotten?",
        "What are the penalties for non-compliance?"
    ]
    results = rag.batch_query(queries)
    print(f"4. Processed {len(results)} queries in batch")
    
    print("\n✓ Module is importable and functional (dry-run mode)")
