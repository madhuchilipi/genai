"""
Agentic RAG Module

Implements multi-agent orchestration with specialized tools:
- Retriever: Fetches relevant documents
- Citation Checker: Verifies claims against sources
- Summarizer: Condenses information

Safe defaults: Returns mock responses when dependencies unavailable.
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class Tool:
    """Represents an agent tool."""
    name: str
    description: str
    function: Callable


class RetrieverTool:
    """Tool for retrieving relevant documents."""
    
    def __init__(self, vectorstore=None):
        """
        Initialize retriever tool.
        
        Args:
            vectorstore: FAISS vectorstore or similar
        """
        self.vectorstore = vectorstore
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents with content and metadata
        """
        if not self.vectorstore:
            print(f"[RETRIEVER] Mock retrieval for: {query}")
            return [
                {
                    "content": f"Mock document 1 relevant to: {query}",
                    "metadata": {"source": "mock", "id": 1},
                    "score": 0.95
                },
                {
                    "content": f"Mock document 2 relevant to: {query}",
                    "metadata": {"source": "mock", "id": 2},
                    "score": 0.87
                }
            ]
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            docs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in results
            ]
            print(f"[RETRIEVER] Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            print(f"[RETRIEVER] Error: {e}")
            return []


class CitationChecker:
    """Tool for verifying claims against source documents."""
    
    def __init__(self):
        """Initialize citation checker."""
        pass
    
    def check_citation(self, claim: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Check if a claim is supported by source documents.
        
        Args:
            claim: Claim to verify
            sources: Source documents
            
        Returns:
            Verification result with score and evidence
        """
        print(f"[CITATION_CHECKER] Checking claim: {claim[:50]}...")
        
        # Simple keyword overlap check (production would use semantic similarity)
        claim_words = set(claim.lower().split())
        
        evidence = []
        for source in sources:
            content = source.get("content", "").lower()
            content_words = set(content.split())
            
            overlap = len(claim_words & content_words)
            overlap_ratio = overlap / len(claim_words) if claim_words else 0
            
            if overlap_ratio > 0.3:  # Threshold for support
                evidence.append({
                    "source": source.get("metadata", {}).get("source", "unknown"),
                    "overlap_ratio": overlap_ratio,
                    "content_snippet": source.get("content", "")[:100]
                })
        
        is_supported = len(evidence) > 0
        confidence = sum(e["overlap_ratio"] for e in evidence) / len(evidence) if evidence else 0.0
        
        result = {
            "claim": claim,
            "is_supported": is_supported,
            "confidence": confidence,
            "evidence": evidence
        }
        
        print(f"[CITATION_CHECKER] Supported: {is_supported}, Confidence: {confidence:.2f}")
        return result
    
    def verify_answer(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Verify an entire answer against sources.
        
        Args:
            answer: Generated answer
            sources: Source documents
            
        Returns:
            Verification summary
        """
        # Split answer into sentences (simple approach)
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        verifications = []
        for sentence in sentences[:5]:  # Check first 5 sentences
            if len(sentence) > 10:  # Skip very short sentences
                result = self.check_citation(sentence, sources)
                verifications.append(result)
        
        overall_confidence = (
            sum(v["confidence"] for v in verifications) / len(verifications)
            if verifications else 0.0
        )
        
        return {
            "answer": answer,
            "overall_confidence": overall_confidence,
            "sentence_verifications": verifications,
            "sources_used": len(sources)
        }


class Summarizer:
    """Tool for summarizing and condensing information."""
    
    def __init__(self, llm=None):
        """
        Initialize summarizer.
        
        Args:
            llm: Language model for summarization (optional)
        """
        self.llm = llm
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text to a target length.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in characters
            
        Returns:
            Summarized text
        """
        if not self.llm:
            # Simple extraction summary (first sentences)
            print(f"[SUMMARIZER] Mock summarization (no LLM)")
            sentences = text.split('.')
            summary = '. '.join(sentences[:3]) + '.'
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        
        try:
            # Use LLM for summarization
            from langchain.schema import HumanMessage
            
            prompt = f"Summarize the following text in about {max_length} characters:\n\n{text}"
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            summary = response.content
            print(f"[SUMMARIZER] Generated summary ({len(summary)} chars)")
            return summary
            
        except Exception as e:
            print(f"[SUMMARIZER] Error: {e}, using fallback")
            # Fallback to simple summary
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def summarize_documents(self, docs: List[Dict], max_length: int = 500) -> str:
        """
        Summarize multiple documents.
        
        Args:
            docs: List of documents
            max_length: Maximum summary length
            
        Returns:
            Combined summary
        """
        # Concatenate documents
        combined = "\n\n".join([doc.get("content", "") for doc in docs])
        return self.summarize(combined, max_length)


class AgentRunner:
    """
    Orchestrates multiple tools to answer queries.
    
    Implements a simple agent loop:
    1. Retrieve relevant documents
    2. Generate initial answer
    3. Check citations
    4. Summarize if needed
    """
    
    def __init__(
        self,
        retriever: RetrieverTool,
        citation_checker: CitationChecker,
        summarizer: Summarizer,
        llm=None
    ):
        """
        Initialize agent runner.
        
        Args:
            retriever: RetrieverTool instance
            citation_checker: CitationChecker instance
            summarizer: Summarizer instance
            llm: Language model for generation (optional)
        """
        self.retriever = retriever
        self.citation_checker = citation_checker
        self.summarizer = summarizer
        self.llm = llm
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the agent pipeline.
        
        Args:
            query: User query
            verbose: Print intermediate steps
            
        Returns:
            Dictionary with answer, sources, and verification
        """
        if verbose:
            print(f"\n[AGENT] Processing query: {query}")
            print("=" * 60)
        
        # Step 1: Retrieve
        if verbose:
            print("\n[STEP 1] Retrieving documents...")
        docs = self.retriever.retrieve(query, top_k=3)
        
        # Step 2: Generate answer
        if verbose:
            print("\n[STEP 2] Generating answer...")
        answer = self._generate_answer(query, docs)
        
        # Step 3: Check citations
        if verbose:
            print("\n[STEP 3] Verifying citations...")
        verification = self.citation_checker.verify_answer(answer, docs)
        
        # Step 4: Summarize if answer is too long
        if len(answer) > 500:
            if verbose:
                print("\n[STEP 4] Summarizing answer...")
            summary = self.summarizer.summarize(answer, max_length=300)
        else:
            summary = answer
        
        result = {
            "query": query,
            "answer": answer,
            "summary": summary,
            "sources": docs,
            "verification": verification,
            "confidence": verification["overall_confidence"]
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"[AGENT] Confidence: {result['confidence']:.2%}")
        
        return result
    
    def _generate_answer(self, query: str, docs: List[Dict]) -> str:
        """
        Generate answer from query and documents.
        
        Args:
            query: User query
            docs: Retrieved documents
            
        Returns:
            Generated answer
        """
        if not self.llm:
            # Mock answer
            sources_text = "; ".join([d["content"][:50] for d in docs[:2]])
            return f"Based on the GDPR regulations, {query.lower()} is addressed in the following context: {sources_text}..."
        
        try:
            from langchain.schema import HumanMessage
            
            context = "\n\n".join([doc["content"] for doc in docs])
            prompt = f"""Answer the question based on the context below. Include specific citations.

Context:
{context}

Question: {query}

Answer:"""
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"[ERROR] Answer generation failed: {e}")
            return "Unable to generate answer due to an error."


def create_agent_rag(faiss_path: str = None, openai_api_key: Optional[str] = None):
    """
    Factory function to create an agentic RAG system.
    
    Args:
        faiss_path: Path to FAISS index (optional)
        openai_api_key: OpenAI API key (optional)
        
    Returns:
        AgentRunner instance
        
    TODO: Integrate with LangGraph for more sophisticated orchestration
    """
    import os
    
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    
    # Initialize components
    vectorstore = None
    llm = None
    
    if api_key and faiss_path:
        try:
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            from langchain.vectorstores import FAISS
            from pathlib import Path
            
            if Path(faiss_path).exists():
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                vectorstore = FAISS.load_local(
                    faiss_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
            print("[INFO] Initialized agent RAG with LLM and vectorstore")
        except Exception as e:
            print(f"[WARNING] Failed to initialize components: {e}")
    else:
        print("[DRY-RUN] Running agent RAG in mock mode")
    
    retriever = RetrieverTool(vectorstore)
    citation_checker = CitationChecker()
    summarizer = Summarizer(llm)
    
    return AgentRunner(retriever, citation_checker, summarizer, llm)
