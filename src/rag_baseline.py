"""
Baseline RAG Implementation

Simple retrieval-augmented generation pipeline that:
1. Loads FAISS index
2. Retrieves top-k relevant chunks
3. Formats prompt with context
4. Calls LLM to generate answer
"""

import os
from typing import List, Dict, Optional, Any
import warnings


class BaselineRAG:
    """
    Baseline RAG system for GDPR question answering.
    
    This class implements a simple RAG pipeline:
    1. Query embedding
    2. FAISS similarity search
    3. Context formatting
    4. LLM generation
    
    Example:
        >>> rag = BaselineRAG(faiss_path="faiss_index/", openai_api_key="sk-...")
        >>> answer = rag.query("What are data subject rights?")
        >>> print(answer)
    """
    
    def __init__(
        self,
        faiss_path: str,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        top_k: int = 3
    ):
        """
        Initialize the Baseline RAG system.
        
        Args:
            faiss_path: Path to the FAISS index directory
            openai_api_key: OpenAI API key (optional for dry-run)
            model: LLM model name
            top_k: Number of chunks to retrieve
        """
        self.faiss_path = faiss_path
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.top_k = top_k
        self.vectorstore = None
        self.llm = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize FAISS vectorstore and LLM."""
        if not self.openai_api_key:
            print("[DRY-RUN] No OpenAI API key, running in placeholder mode")
            return
        
        try:
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.vectorstores import FAISS
            from langchain.chat_models import ChatOpenAI
        except ImportError:
            warnings.warn("LangChain not available, using placeholder mode")
            return
        
        # TODO: Load actual FAISS index
        # embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        # self.vectorstore = FAISS.load_local(self.faiss_path, embeddings)
        # self.llm = ChatOpenAI(
        #     openai_api_key=self.openai_api_key,
        #     model=self.model,
        #     temperature=0
        # )
        
        print(f"[DRY-RUN] Would load FAISS from {self.faiss_path}")
        print(f"[DRY-RUN] Would initialize {self.model}")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query: User question
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.vectorstore:
            return self._placeholder_retrieve(query)
        
        # TODO: Implement actual retrieval
        # results = self.vectorstore.similarity_search(query, k=self.top_k)
        # return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        
        return self._placeholder_retrieve(query)
    
    def _placeholder_retrieve(self, query: str) -> List[Dict]:
        """Placeholder retrieval for testing."""
        print(f"[DRY-RUN] Retrieving top {self.top_k} chunks for: {query}")
        
        # Return placeholder results
        return [
            {
                "content": "GDPR Article 15: The data subject shall have the right to obtain from the controller confirmation as to whether or not personal data concerning him or her are being processed...",
                "metadata": {"source": "gdpr.pdf", "page": 7, "article": 15}
            },
            {
                "content": "GDPR Article 17: The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay...",
                "metadata": {"source": "gdpr.pdf", "page": 8, "article": 17}
            },
            {
                "content": "GDPR Article 20: The data subject shall have the right to receive the personal data concerning him or her in a structured, commonly used format...",
                "metadata": {"source": "gdpr.pdf", "page": 9, "article": 20}
            }
        ]
    
    def format_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """
        Format the prompt with retrieved context.
        
        Args:
            query: User question
            context_docs: Retrieved document chunks
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([
            f"[Source: {doc['metadata'].get('source', 'unknown')}, "
            f"Page {doc['metadata'].get('page', '?')}, "
            f"Article {doc['metadata'].get('article', '?')}]\n{doc['content']}"
            for doc in context_docs
        ])
        
        prompt = f"""You are a helpful assistant answering questions about GDPR regulations.
Use the following context to answer the question. Include citations to specific articles.

Context:
{context_text}

Question: {query}

Answer (with citations):"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            Generated answer
        """
        if not self.llm:
            return self._placeholder_generate(prompt)
        
        # TODO: Implement actual LLM generation
        # response = self.llm.predict(prompt)
        # return response
        
        return self._placeholder_generate(prompt)
    
    def _placeholder_generate(self, prompt: str) -> str:
        """Placeholder generation for testing."""
        print("[DRY-RUN] Generating answer with LLM")
        
        # Return placeholder answer
        return """According to GDPR regulations, data subjects have several key rights:

1. **Right to Access (Article 15)**: You can request confirmation of whether your personal data is being processed and obtain a copy of that data.

2. **Right to Erasure (Article 17)**: Also known as the "right to be forgotten," you can request deletion of your personal data under certain circumstances.

3. **Right to Data Portability (Article 20)**: You have the right to receive your personal data in a structured, commonly used format and transmit it to another controller.

These rights empower individuals to have control over their personal data. [Citations: GDPR Articles 15, 17, 20]"""
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve, format, generate.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and metadata
        """
        # Step 1: Retrieve relevant chunks
        docs = self.retrieve(question)
        
        # Step 2: Format prompt
        prompt = self.format_prompt(question, docs)
        
        # Step 3: Generate answer
        answer = self.generate_answer(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [doc["metadata"] for doc in docs],
            "num_sources": len(docs)
        }


def evaluate_retrieval(retrieved_docs: List[Dict], ground_truth: List[int]) -> Dict[str, float]:
    """
    Evaluate retrieval quality using precision and recall.
    
    Args:
        retrieved_docs: List of retrieved documents
        ground_truth: List of relevant article numbers
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if not retrieved_docs or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    retrieved_articles = set([
        doc["metadata"].get("article") 
        for doc in retrieved_docs 
        if "article" in doc["metadata"]
    ])
    ground_truth_set = set(ground_truth)
    
    true_positives = len(retrieved_articles & ground_truth_set)
    
    precision = true_positives / len(retrieved_articles) if retrieved_articles else 0.0
    recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "retrieved_articles": list(retrieved_articles),
        "ground_truth": list(ground_truth_set)
    }
