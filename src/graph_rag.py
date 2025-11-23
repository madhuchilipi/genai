"""
Graph RAG Module

Implements graph-enhanced RAG with:
- Query rephrasing for better retrieval
- Anchor document retrieval
- Neighboring chunk expansion
- Answer composition with citations

Safe defaults: Works without API keys using mock implementations.
"""

from typing import List, Dict, Optional, Any, Tuple


def rephrase_query(query: str, llm=None) -> List[str]:
    """
    Rephrase a query into multiple variations for better retrieval.
    
    Args:
        query: Original user query
        llm: Language model for rephrasing (optional)
        
    Returns:
        List of rephrased queries (including original)
    """
    if not llm:
        # Simple rule-based rephrasing
        print(f"[GRAPH_RAG] Mock query rephrasing for: {query}")
        
        rephrased = [
            query,  # Original
            f"What does GDPR say about {query.lower()}?",  # Template 1
            f"According to GDPR regulations, {query.lower()}",  # Template 2
        ]
        
        print(f"[GRAPH_RAG] Generated {len(rephrased)} query variations")
        return rephrased
    
    try:
        from langchain.schema import HumanMessage
        
        prompt = f"""Rephrase the following question in 3 different ways to improve retrieval:

Original question: {query}

Provide 3 rephrased versions (one per line):"""
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        # Parse rephrased queries
        rephrased = [line.strip() for line in response.content.split('\n') if line.strip()]
        rephrased = [query] + rephrased[:3]  # Include original + top 3
        
        print(f"[GRAPH_RAG] Generated {len(rephrased)} query variations")
        return rephrased
        
    except Exception as e:
        print(f"[GRAPH_RAG] Rephrasing error: {e}, using original query")
        return [query]


def retrieve_anchors(queries: List[str], vectorstore, top_k: int = 2) -> List[Dict]:
    """
    Retrieve anchor documents for multiple query variations.
    
    Args:
        queries: List of query variations
        vectorstore: Vector store for retrieval
        top_k: Number of documents per query
        
    Returns:
        List of anchor documents with scores
    """
    if not vectorstore:
        print(f"[GRAPH_RAG] Mock anchor retrieval for {len(queries)} queries")
        
        anchors = []
        for i, query in enumerate(queries):
            anchors.append({
                "content": f"Mock anchor document for query: {query}",
                "metadata": {"source": "mock", "query_idx": i, "anchor": True},
                "score": 0.9 - (i * 0.1),
                "query": query
            })
        
        print(f"[GRAPH_RAG] Retrieved {len(anchors)} anchor documents")
        return anchors
    
    try:
        all_anchors = []
        seen_content = set()
        
        for query in queries:
            results = vectorstore.similarity_search_with_score(query, k=top_k)
            
            for doc, score in results:
                content = doc.page_content
                
                # Deduplicate
                if content not in seen_content:
                    seen_content.add(content)
                    all_anchors.append({
                        "content": content,
                        "metadata": {**doc.metadata, "anchor": True},
                        "score": float(score),
                        "query": query
                    })
        
        print(f"[GRAPH_RAG] Retrieved {len(all_anchors)} unique anchor documents")
        return all_anchors
        
    except Exception as e:
        print(f"[GRAPH_RAG] Anchor retrieval error: {e}")
        return []


def get_neighboring_chunks(anchor_docs: List[Dict], vectorstore=None, context_window: int = 1) -> List[Dict]:
    """
    Retrieve neighboring chunks around anchor documents.
    
    Args:
        anchor_docs: Anchor documents
        vectorstore: Vector store (optional, for metadata-based navigation)
        context_window: Number of neighbors to retrieve on each side
        
    Returns:
        List of neighboring documents
    """
    print(f"[GRAPH_RAG] Retrieving neighbors for {len(anchor_docs)} anchors (window={context_window})")
    
    neighbors = []
    
    for anchor in anchor_docs:
        metadata = anchor.get("metadata", {})
        source = metadata.get("source", "")
        
        # In a real implementation, this would:
        # 1. Use document metadata (page numbers, article IDs, etc.)
        # 2. Retrieve sequential chunks from the original document
        # 3. Use graph structure if available
        
        # Mock implementation: create placeholder neighbors
        for offset in range(-context_window, context_window + 1):
            if offset == 0:
                continue  # Skip the anchor itself
            
            neighbor = {
                "content": f"Neighbor chunk (offset {offset:+d}) near: {anchor['content'][:50]}...",
                "metadata": {
                    **metadata,
                    "anchor": False,
                    "neighbor_offset": offset,
                    "neighbor_of": source
                },
                "score": anchor.get("score", 0.5) * 0.8,  # Lower score than anchor
            }
            neighbors.append(neighbor)
    
    print(f"[GRAPH_RAG] Retrieved {len(neighbors)} neighboring chunks")
    return neighbors


def compose_answer_with_citations(
    query: str,
    anchor_docs: List[Dict],
    neighbor_docs: List[Dict],
    llm=None
) -> Dict[str, Any]:
    """
    Compose answer from anchors and neighbors with proper citations.
    
    Args:
        query: Original query
        anchor_docs: Anchor documents
        neighbor_docs: Neighbor documents
        llm: Language model for generation
        
    Returns:
        Dictionary with answer, citations, and confidence
    """
    print(f"[GRAPH_RAG] Composing answer with {len(anchor_docs)} anchors and {len(neighbor_docs)} neighbors")
    
    # Combine all documents
    all_docs = anchor_docs + neighbor_docs
    
    # Sort by score (anchors should naturally be higher)
    all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Take top documents
    top_docs = all_docs[:5]
    
    if not llm:
        # Mock answer with citations
        citations = [
            {
                "source": doc["metadata"].get("source", "unknown"),
                "is_anchor": doc["metadata"].get("anchor", False),
                "snippet": doc["content"][:100]
            }
            for doc in top_docs
        ]
        
        answer = f"""Based on GDPR regulations and related context:

{query}

The regulations address this through multiple provisions. [Citation 1,2] The key points include...

[Mock answer - would be generated by LLM in production]"""
        
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "num_anchors": len(anchor_docs),
            "num_neighbors": len(neighbor_docs),
            "confidence": 0.85
        }
    
    try:
        from langchain.schema import HumanMessage
        
        # Build context with citations
        context_parts = []
        for i, doc in enumerate(top_docs):
            is_anchor = doc["metadata"].get("anchor", False)
            doc_type = "ANCHOR" if is_anchor else "CONTEXT"
            context_parts.append(f"[{doc_type} {i+1}] {doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Answer the question using the provided context. Include citations in the format [ANCHOR X] or [CONTEXT X].

Context:
{context}

Question: {query}

Answer with proper citations:"""
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        answer = response.content
        
        # Extract citations
        citations = [
            {
                "source": doc["metadata"].get("source", "unknown"),
                "is_anchor": doc["metadata"].get("anchor", False),
                "snippet": doc["content"][:100],
                "score": doc.get("score", 0)
            }
            for doc in top_docs
        ]
        
        # Calculate confidence based on anchor scores
        anchor_scores = [d.get("score", 0) for d in anchor_docs]
        confidence = sum(anchor_scores) / len(anchor_scores) if anchor_scores else 0.5
        
        print(f"[GRAPH_RAG] Generated answer with {len(citations)} citations (confidence: {confidence:.2f})")
        
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "num_anchors": len(anchor_docs),
            "num_neighbors": len(neighbor_docs),
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"[GRAPH_RAG] Answer composition error: {e}")
        return {
            "query": query,
            "answer": "Error generating answer",
            "citations": [],
            "error": str(e)
        }


class GraphRAG:
    """
    Graph-enhanced RAG system.
    
    Implements the full graph RAG pipeline:
    1. Query rephrasing
    2. Anchor retrieval
    3. Neighbor expansion
    4. Answer composition with citations
    """
    
    def __init__(
        self,
        vectorstore=None,
        llm=None,
        top_k_anchors: int = 3,
        context_window: int = 1
    ):
        """
        Initialize Graph RAG system.
        
        Args:
            vectorstore: Vector store for retrieval
            llm: Language model
            top_k_anchors: Number of anchor documents per query variation
            context_window: Number of neighbors to retrieve around each anchor
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k_anchors = top_k_anchors
        self.context_window = context_window
    
    def query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Process a query through the graph RAG pipeline.
        
        Args:
            query: User query
            verbose: Print intermediate steps
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        if verbose:
            print(f"\n[GRAPH_RAG] Processing: {query}")
            print("=" * 60)
        
        # Step 1: Rephrase query
        if verbose:
            print("\n[STEP 1] Rephrasing query...")
        queries = rephrase_query(query, self.llm)
        
        # Step 2: Retrieve anchors
        if verbose:
            print("\n[STEP 2] Retrieving anchor documents...")
        anchors = retrieve_anchors(queries, self.vectorstore, top_k=self.top_k_anchors)
        
        # Step 3: Get neighboring chunks
        if verbose:
            print("\n[STEP 3] Expanding to neighboring chunks...")
        neighbors = get_neighboring_chunks(anchors, self.vectorstore, context_window=self.context_window)
        
        # Step 4: Compose answer
        if verbose:
            print("\n[STEP 4] Composing answer with citations...")
        result = compose_answer_with_citations(query, anchors, neighbors, self.llm)
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"[GRAPH_RAG] Complete. Confidence: {result.get('confidence', 0):.2%}")
        
        return result


def create_graph_rag(faiss_path: str = None, openai_api_key: Optional[str] = None):
    """
    Factory function to create a graph RAG system.
    
    Args:
        faiss_path: Path to FAISS index (optional)
        openai_api_key: OpenAI API key (optional)
        
    Returns:
        GraphRAG instance
    """
    import os
    
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    
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
            print("[INFO] Initialized Graph RAG with LLM and vectorstore")
        except Exception as e:
            print(f"[WARNING] Failed to initialize components: {e}")
    else:
        print("[DRY-RUN] Running Graph RAG in mock mode")
    
    return GraphRAG(vectorstore, llm)
