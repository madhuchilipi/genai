"""
Graph-enhanced RAG Module

This module implements graph-based retrieval strategies:
- Question rephrasing for better retrieval
- Anchor retrieval: Find initial relevant documents
- Neighbor retrieval: Explore related documents
- Citation-aware answer composition
"""

import warnings
from typing import List, Dict, Any, Optional, Set


def rephrase_question(query: str, context: Optional[str] = None) -> List[str]:
    """
    Generate multiple rephrasings of a query for better retrieval.
    
    This technique helps overcome vocabulary mismatch between
    the query and documents.
    
    Args:
        query: Original user query
        context: Optional conversation context
        
    Returns:
        List of rephrased queries (including original)
        
    Example:
        >>> variants = rephrase_question("What is the right to be forgotten?")
        >>> print(variants)
        ['What is the right to be forgotten?',
         'What is the right to erasure?',
         'GDPR data deletion rights']
    """
    # TODO: Implement LLM-based query rephrasing
    # In production, use an LLM to generate semantic variants
    
    rephrased = [query]  # Always include original
    
    # Simple rule-based rephrasing as placeholder
    query_lower = query.lower()
    
    # Add common synonyms for GDPR terms
    synonyms = {
        "right to be forgotten": "right to erasure",
        "data controller": "organization processing data",
        "data subject": "individual",
        "personal data": "personally identifiable information",
        "processing": "handling data",
    }
    
    for term, synonym in synonyms.items():
        if term in query_lower:
            rephrased.append(query.replace(term, synonym))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_rephrased = []
    for q in rephrased:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique_rephrased.append(q)
    
    return unique_rephrased


def anchor_retrieval(
    queries: List[str],
    retriever_func: callable,
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Perform anchor retrieval using multiple query variants.
    
    Retrieves documents for each query variant and combines results.
    
    Args:
        queries: List of query variants
        retriever_func: Function to retrieve documents (e.g., rag.retrieve)
        top_k: Number of documents to retrieve per query
        
    Returns:
        Deduplicated list of anchor documents with scores
        
    Example:
        >>> queries = rephrase_question("What is GDPR Article 17?")
        >>> anchors = anchor_retrieval(queries, rag.retrieve, top_k=3)
    """
    all_docs = []
    seen_content = set()
    
    for query in queries:
        docs = retriever_func(query)
        
        for doc in docs[:top_k]:
            # Deduplicate by content hash
            content = doc.get("page_content", "")
            content_hash = hash(content)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                doc["retrieval_query"] = query  # Track which query found this
                all_docs.append(doc)
    
    # Sort by score if available
    all_docs.sort(
        key=lambda x: x.get("metadata", {}).get("score", 0),
        reverse=True
    )
    
    return all_docs[:top_k * 2]  # Return up to 2x top_k for neighbor exploration


def neighbor_retrieval(
    anchor_docs: List[Dict[str, Any]],
    all_documents: Optional[List[Dict[str, Any]]] = None,
    max_neighbors: int = 2
) -> List[Dict[str, Any]]:
    """
    Retrieve neighboring documents for each anchor.
    
    Neighbors are documents that are:
    - From the same source/chapter
    - Have adjacent page numbers
    - Share similar metadata
    
    Args:
        anchor_docs: Anchor documents from initial retrieval
        all_documents: Complete document collection (optional)
        max_neighbors: Max neighbors per anchor
        
    Returns:
        List of neighboring documents
        
    Example:
        >>> anchors = anchor_retrieval(queries, rag.retrieve)
        >>> neighbors = neighbor_retrieval(anchors, max_neighbors=2)
    """
    if all_documents is None:
        # In dry-run mode, generate placeholder neighbors
        warnings.warn("No document collection provided. Generating placeholders.")
        neighbors = []
        for i, anchor in enumerate(anchor_docs):
            page = anchor.get("metadata", {}).get("page", 0)
            for offset in [-1, 1]:
                if offset == 0:
                    continue
                neighbor = {
                    "page_content": f"[Neighbor of anchor {i}, page {page + offset}]",
                    "metadata": {
                        "page": page + offset,
                        "source": anchor.get("metadata", {}).get("source", "gdpr.pdf"),
                        "is_neighbor": True,
                        "anchor_page": page
                    }
                }
                neighbors.append(neighbor)
        return neighbors[:max_neighbors * len(anchor_docs)]
    
    # TODO: Implement actual neighbor retrieval
    # In production:
    # 1. For each anchor, find documents with adjacent page numbers
    # 2. Find documents from same article/section
    # 3. Use graph structure if available (e.g., citation graph)
    
    neighbors = []
    
    for anchor in anchor_docs:
        anchor_page = anchor.get("metadata", {}).get("page")
        anchor_source = anchor.get("metadata", {}).get("source")
        
        if not all_documents or anchor_page is None:
            continue
        
        # Find adjacent pages
        for doc in all_documents:
            doc_page = doc.get("metadata", {}).get("page")
            doc_source = doc.get("metadata", {}).get("source")
            
            if doc_source == anchor_source and doc_page is not None:
                # Check if adjacent (within 2 pages)
                if abs(doc_page - anchor_page) <= 2 and doc_page != anchor_page:
                    doc_copy = doc.copy()
                    doc_copy["metadata"] = {
                        **doc.get("metadata", {}),
                        "is_neighbor": True,
                        "anchor_page": anchor_page
                    }
                    neighbors.append(doc_copy)
                    
                    if len(neighbors) >= max_neighbors * len(anchor_docs):
                        break
    
    return neighbors


def compose_answer_with_citations(
    query: str,
    anchor_docs: List[Dict[str, Any]],
    neighbor_docs: List[Dict[str, Any]],
    llm_func: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Compose an answer using both anchor and neighbor documents with citations.
    
    Args:
        query: User query
        anchor_docs: Primary retrieved documents
        neighbor_docs: Neighboring/related documents
        llm_func: Function to call LLM (optional)
        
    Returns:
        Dictionary with answer and detailed citations
        
    Example:
        >>> result = compose_answer_with_citations(
        ...     query, anchors, neighbors, llm_func=rag.generate_answer
        ... )
    """
    # Combine all documents
    all_docs = anchor_docs + neighbor_docs
    
    # Build context with clear source markers
    context_parts = []
    
    for i, doc in enumerate(anchor_docs):
        page = doc.get("metadata", {}).get("page", "N/A")
        content = doc.get("page_content", "")
        context_parts.append(
            f"[Anchor {i+1}, Page {page}]\n{content}"
        )
    
    for i, doc in enumerate(neighbor_docs):
        page = doc.get("metadata", {}).get("page", "N/A")
        anchor_page = doc.get("metadata", {}).get("anchor_page", "N/A")
        content = doc.get("page_content", "")
        context_parts.append(
            f"[Neighbor {i+1}, Page {page}, Related to Page {anchor_page}]\n{content}"
        )
    
    context = "\n\n".join(context_parts)
    
    # Format prompt
    prompt = f"""You are a GDPR expert. Answer the question using the provided context.
Include specific citations to page numbers and indicate whether information comes from anchor or neighbor documents.

Context:
{context}

Question: {query}

Provide a detailed answer with citations in the format [Anchor X, Page Y] or [Neighbor X, Page Y]:"""
    
    # Generate answer
    if llm_func is None:
        # Dry-run mode
        answer = f"[Placeholder answer for: {query}] Based on [Anchor 1, Page X] and [Neighbor 1, Page Y]."
    else:
        answer = llm_func(prompt)
    
    return {
        "answer": answer,
        "num_anchor_docs": len(anchor_docs),
        "num_neighbor_docs": len(neighbor_docs),
        "anchor_pages": [d.get("metadata", {}).get("page") for d in anchor_docs],
        "neighbor_pages": [d.get("metadata", {}).get("page") for d in neighbor_docs],
        "full_context": context,
        "retrieval_strategy": "graph_rag_with_neighbors"
    }


def graph_rag_pipeline(
    query: str,
    retriever_func: callable,
    llm_func: Optional[callable] = None,
    all_documents: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Execute the complete graph RAG pipeline.
    
    Pipeline steps:
    1. Rephrase query into variants
    2. Anchor retrieval with query variants
    3. Neighbor retrieval for each anchor
    4. Compose answer with full context and citations
    
    Args:
        query: User query
        retriever_func: Function to retrieve documents
        llm_func: Function to call LLM
        all_documents: Full document collection for neighbor finding
        
    Returns:
        Complete result with answer and retrieval metadata
        
    Example:
        >>> result = graph_rag_pipeline(
        ...     "What is the right to erasure?",
        ...     rag.retrieve,
        ...     rag.generate_answer
        ... )
    """
    # Step 1: Rephrase query
    query_variants = rephrase_question(query)
    
    # Step 2: Anchor retrieval
    anchor_docs = anchor_retrieval(query_variants, retriever_func, top_k=4)
    
    # Step 3: Neighbor retrieval
    neighbor_docs = neighbor_retrieval(
        anchor_docs,
        all_documents,
        max_neighbors=2
    )
    
    # Step 4: Compose answer
    result = compose_answer_with_citations(
        query,
        anchor_docs,
        neighbor_docs,
        llm_func
    )
    
    # Add pipeline metadata
    result["query_variants"] = query_variants
    result["pipeline_steps"] = [
        "query_rephrasing",
        "anchor_retrieval",
        "neighbor_retrieval",
        "answer_composition"
    ]
    
    return result


# Example usage for testing
if __name__ == "__main__":
    print("=== Graph RAG Module Demo ===")
    
    # Test query rephrasing
    query = "What is the right to be forgotten?"
    variants = rephrase_question(query)
    print(f"1. Query variants for '{query}':")
    for v in variants:
        print(f"   - {v}")
    
    # Mock retriever function
    def mock_retrieve(q):
        return [
            {
                "page_content": f"Content about: {q}",
                "metadata": {"page": 17, "score": 0.9}
            },
            {
                "page_content": "GDPR Article 17 defines erasure...",
                "metadata": {"page": 18, "score": 0.85}
            }
        ]
    
    # Test anchor retrieval
    anchors = anchor_retrieval(variants, mock_retrieve, top_k=2)
    print(f"\n2. Anchor retrieval found {len(anchors)} documents")
    
    # Test neighbor retrieval
    neighbors = neighbor_retrieval(anchors, max_neighbors=1)
    print(f"3. Neighbor retrieval found {len(neighbors)} neighbors")
    
    # Test full pipeline
    result = graph_rag_pipeline(query, mock_retrieve)
    print(f"\n4. Pipeline result:")
    print(f"   Query variants: {len(result['query_variants'])}")
    print(f"   Anchor docs: {result['num_anchor_docs']}")
    print(f"   Neighbor docs: {result['num_neighbor_docs']}")
    print(f"   Answer: {result['answer'][:80]}...")
    
    print("\n✓ Module is importable and functional")
