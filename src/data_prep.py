"""
Data Preparation Module

Functions for downloading GDPR PDF, parsing with LangChain loaders,
chunking strategies, embedding generation, and FAISS index building.
"""

import os
from typing import List, Optional
import warnings


def download_gdpr_pdf(save_path: str) -> str:
    """
    Download the GDPR regulation PDF from official EU sources.
    
    Args:
        save_path: Path where to save the PDF file
        
    Returns:
        Path to the downloaded PDF file
        
    Example:
        >>> path = download_gdpr_pdf("data/gdpr.pdf")
        >>> print(f"Downloaded to {path}")
    """
    # TODO: Implement actual download from EU official sources
    # URL: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # Placeholder implementation for CI/testing without network access
    print(f"[DRY-RUN] Would download GDPR PDF to {save_path}")
    print("In production, use requests library to download from:")
    print("https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679")
    
    # Create a placeholder file for testing
    with open(save_path, "w") as f:
        f.write("Placeholder GDPR PDF content for testing")
    
    return save_path


def load_and_split(path: str, strategy: str = "paragraph") -> List[dict]:
    """
    Load PDF and split into chunks using specified strategy.
    
    Args:
        path: Path to the PDF file
        strategy: Chunking strategy - "paragraph", "article", or "token"
        
    Returns:
        List of document chunks with metadata
        
    Strategies:
        - paragraph: Split by paragraphs (natural breaks)
        - article: Split by GDPR articles (semantic units)
        - token: Fixed token-size chunks with overlap
        
    Example:
        >>> docs = load_and_split("data/gdpr.pdf", strategy="paragraph")
        >>> print(f"Created {len(docs)} chunks")
    """
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        warnings.warn("LangChain not available, using placeholder implementation")
        return _placeholder_load_and_split(path, strategy)
    
    if not os.path.exists(path):
        print(f"[DRY-RUN] File {path} not found, returning placeholder documents")
        return _placeholder_load_and_split(path, strategy)
    
    # TODO: In production, implement actual PDF loading
    # loader = PyPDFLoader(path)
    # documents = loader.load()
    
    # Configure splitter based on strategy
    if strategy == "paragraph":
        # TODO: Use paragraph-aware splitter
        chunk_size = 1000
        chunk_overlap = 200
    elif strategy == "article":
        # TODO: Use GDPR article-specific splitter
        chunk_size = 2000
        chunk_overlap = 0
    elif strategy == "token":
        # TODO: Use token-based splitter
        chunk_size = 512
        chunk_overlap = 50
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # TODO: Implement actual splitting
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap
    # )
    # splits = text_splitter.split_documents(documents)
    
    return _placeholder_load_and_split(path, strategy)


def _placeholder_load_and_split(path: str, strategy: str) -> List[dict]:
    """Placeholder implementation for testing without actual PDF."""
    print(f"[DRY-RUN] Loading {path} with {strategy} strategy")
    
    # Return placeholder documents
    placeholder_docs = [
        {
            "page_content": "GDPR Article 1: This regulation lays down rules relating to the protection of natural persons...",
            "metadata": {"source": path, "page": 1, "article": 1}
        },
        {
            "page_content": "GDPR Article 5: Personal data shall be processed lawfully, fairly and in a transparent manner...",
            "metadata": {"source": path, "page": 2, "article": 5}
        },
        {
            "page_content": "GDPR Article 17: The data subject shall have the right to obtain from the controller the erasure...",
            "metadata": {"source": path, "page": 8, "article": 17}
        }
    ]
    
    print(f"[DRY-RUN] Created {len(placeholder_docs)} placeholder chunks")
    return placeholder_docs


def build_and_persist_faiss(
    docs: List[dict],
    faiss_path: str,
    openai_api_key: Optional[str] = None
) -> str:
    """
    Build FAISS index from documents and persist to disk.
    
    Args:
        docs: List of document chunks
        faiss_path: Directory path to save FAISS index
        openai_api_key: OpenAI API key for embeddings (optional for dry-run)
        
    Returns:
        Path to the persisted FAISS index
        
    Example:
        >>> docs = load_and_split("data/gdpr.pdf")
        >>> index_path = build_and_persist_faiss(docs, "faiss_index/", api_key)
        >>> print(f"Index saved to {index_path}")
    """
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("[DRY-RUN] No OpenAI API key provided, using placeholder FAISS index")
        return _placeholder_build_faiss(docs, faiss_path)
    
    try:
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import FAISS
    except ImportError:
        warnings.warn("LangChain not available, using placeholder implementation")
        return _placeholder_build_faiss(docs, faiss_path)
    
    # TODO: Implement actual FAISS index building
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # texts = [doc["page_content"] for doc in docs]
    # metadatas = [doc["metadata"] for doc in docs]
    # vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    # vectorstore.save_local(faiss_path)
    
    return _placeholder_build_faiss(docs, faiss_path)


def _placeholder_build_faiss(docs: List[dict], faiss_path: str) -> str:
    """Placeholder implementation for testing without API key."""
    os.makedirs(faiss_path, exist_ok=True)
    
    print(f"[DRY-RUN] Building FAISS index with {len(docs)} documents")
    print(f"[DRY-RUN] Index would be saved to {faiss_path}")
    print("[DRY-RUN] In production, use OpenAIEmbeddings with text-embedding-ada-002")
    
    # Create placeholder index file
    index_file = os.path.join(faiss_path, "index.faiss")
    with open(index_file, "w") as f:
        f.write("Placeholder FAISS index")
    
    return faiss_path


def get_chunking_stats(docs: List[dict]) -> dict:
    """
    Calculate statistics about document chunks.
    
    Args:
        docs: List of document chunks
        
    Returns:
        Dictionary with chunk statistics
    """
    if not docs:
        return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    
    lengths = [len(doc["page_content"]) for doc in docs]
    
    return {
        "num_chunks": len(docs),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }
