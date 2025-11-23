"""
Baseline RAG Module

Implements a simple RAG pipeline: query → FAISS retrieval → LLM generation
with OpenAI models via LangChain.

Safe defaults: Returns placeholder responses when API keys are not provided.
"""

import os
from typing import List, Optional, Dict
from pathlib import Path


class BaselineRAG:
    """
    Baseline Retrieval-Augmented Generation system.
    
    Loads a FAISS index, retrieves relevant documents for queries,
    and generates answers using an LLM.
    """
    
    def __init__(
        self,
        faiss_path: str,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        top_k: int = 3,
        temperature: float = 0.0
    ):
        """
        Initialize the baseline RAG system.
        
        Args:
            faiss_path: Path to the FAISS index directory
            openai_api_key: OpenAI API key (optional, uses env var if not provided)
            model_name: Name of the OpenAI model to use
            top_k: Number of documents to retrieve
            temperature: LLM temperature (0 = deterministic)
        """
        self.faiss_path = Path(faiss_path)
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.top_k = top_k
        self.temperature = temperature
        self.vectorstore = None
        self.llm = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the vector store and LLM."""
        if not self.api_key:
            print("[DRY-RUN] No OpenAI API key provided, running in mock mode")
            return
        
        try:
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            from langchain.vectorstores import FAISS
            
            # Load FAISS index
            if self.faiss_path.exists():
                embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
                self.vectorstore = FAISS.load_local(
                    str(self.faiss_path),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"[INFO] Loaded FAISS index from {self.faiss_path}")
            else:
                print(f"[WARNING] FAISS index not found at {self.faiss_path}")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key
            )
            print(f"[INFO] Initialized {self.model_name} with temperature={self.temperature}")
            
        except ImportError as e:
            print(f"[WARNING] Required libraries not available: {e}")
            print("[DRY-RUN] Running in mock mode")
        except Exception as e:
            print(f"[WARNING] Failed to initialize RAG components: {e}")
            print("[DRY-RUN] Running in mock mode")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            
        Returns:
            List of retrieved documents with content and metadata
        """
        if not self.vectorstore:
            # Mock retrieval
            print(f"[DRY-RUN] Mock retrieval for query: '{query}'")
            return [
                {
                    "content": "Article 1: Subject-matter and objectives. This Regulation lays down rules relating to the protection of natural persons...",
                    "metadata": {"source": "mock", "article": 1},
                    "score": 0.95
                },
                {
                    "content": "Article 5: Principles relating to processing of personal data. Personal data shall be processed lawfully, fairly and transparently...",
                    "metadata": {"source": "mock", "article": 5},
                    "score": 0.87
                }
            ]
        
        try:
            # Actual retrieval
            results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
            
            docs = []
            for doc, score in results:
                docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            print(f"[INFO] Retrieved {len(docs)} documents for query")
            return docs
            
        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate an answer using the LLM and retrieved context.
        
        Args:
            query: User query string
            context_docs: List of retrieved documents
            
        Returns:
            Generated answer string
        """
        if not self.llm:
            # Mock generation
            print(f"[DRY-RUN] Mock answer generation for query: '{query}'")
            return (
                "According to the GDPR regulations (mock response), "
                "personal data must be processed lawfully, fairly, and transparently. "
                "The regulation establishes rules for protecting natural persons with regard to "
                "the processing of personal data."
            )
        
        try:
            # Build context from retrieved documents
            context = "\n\n".join([
                f"[Source {i+1}] {doc['content']}"
                for i, doc in enumerate(context_docs)
            ])
            
            # Create prompt
            prompt = self._format_prompt(query, context)
            
            # Generate answer
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            answer = response.content
            print(f"[INFO] Generated answer ({len(answer)} chars)")
            
            return answer
            
        except Exception as e:
            print(f"[ERROR] Answer generation failed: {e}")
            return "I apologize, but I encountered an error generating an answer."
    
    def _format_prompt(self, query: str, context: str) -> str:
        """
        Format the prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        return f"""You are a helpful assistant answering questions about GDPR regulations.
Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {query}

Answer: Provide a clear, accurate answer based on the context above. Cite specific articles when possible."""
    
    def query(self, query: str, return_sources: bool = False) -> str:
        """
        Complete RAG pipeline: retrieve and generate answer.
        
        Args:
            query: User query string
            return_sources: Whether to include source documents in response
            
        Returns:
            Generated answer (optionally with sources)
        """
        print(f"\n[QUERY] {query}")
        
        # Retrieve relevant documents
        docs = self.retrieve(query)
        
        # Generate answer
        answer = self.generate_answer(query, docs)
        
        if return_sources:
            # Format with sources
            sources_text = "\n\nSources:\n" + "\n".join([
                f"- {doc['metadata'].get('source', 'Unknown')} (score: {doc.get('score', 0):.2f})"
                for doc in docs
            ])
            return answer + sources_text
        
        return answer
    
    def batch_query(self, queries: List[str]) -> List[str]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of generated answers
        """
        return [self.query(q) for q in queries]


def create_rag_chain(faiss_path: str, openai_api_key: Optional[str] = None):
    """
    Factory function to create a RAG chain (for compatibility with LangChain patterns).
    
    Args:
        faiss_path: Path to FAISS index
        openai_api_key: OpenAI API key
        
    Returns:
        BaselineRAG instance configured as a chain
        
    TODO: Implement proper LangChain LCEL chain for better composability
    """
    return BaselineRAG(faiss_path=faiss_path, openai_api_key=openai_api_key)
