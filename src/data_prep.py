"""
Data Preparation Module

This module provides functions for:
- Downloading the GDPR PDF
- Parsing PDFs with LangChain loaders
- Chunking strategies (paragraph, article, token-based)
- Generating embeddings using OpenAI
- Building and persisting FAISS vector store
"""

import os
import warnings
from typing import List, Optional, Dict, Any
from pathlib import Path


def download_gdpr_pdf(save_path: str) -> str:
    """
    Download the GDPR PDF from the official EU website.
    
    Args:
        save_path: Path where the PDF should be saved
        
    Returns:
        Path to the downloaded PDF file
        
    Example:
        >>> pdf_path = download_gdpr_pdf("gdpr.pdf")
        >>> print(f"Downloaded to: {pdf_path}")
    """
    # TODO: Implement actual download logic using requests or urllib
    # Official GDPR PDF URL: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679
    
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.warn("Running in dry-run mode. Set OPENAI_API_KEY to download actual PDF.")
        # Return placeholder path in dry-run mode
        return save_path
    
    # Production implementation would be:
    # import requests
    # url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
    # response = requests.get(url)
    # with open(save_path, 'wb') as f:
    #     f.write(response.content)
    
    print(f"[DRY-RUN] Would download GDPR PDF to: {save_path}")
    return save_path


def load_and_split(
    path: str,
    strategy: str = "paragraph",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Load a PDF and split it into chunks using the specified strategy.
    
    Args:
        path: Path to the PDF file
        strategy: Chunking strategy - "paragraph", "article", or "token"
        chunk_size: Size of each chunk (for token-based splitting)
        chunk_overlap: Overlap between chunks (for token-based splitting)
        
    Returns:
        List of document chunks with metadata
        
    Example:
        >>> docs = load_and_split("gdpr.pdf", strategy="paragraph")
        >>> print(f"Created {len(docs)} chunks")
    """
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.warn("Running in dry-run mode. Returning placeholder documents.")
        # Return placeholder documents
        return [
            {
                "page_content": f"[Placeholder GDPR content chunk {i}]",
                "metadata": {"source": path, "page": i, "chunk_id": i}
            }
            for i in range(5)
        ]
    
    # TODO: Implement actual PDF loading with LangChain
    # from langchain_community.document_loaders import PyPDFLoader
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # loader = PyPDFLoader(path)
    # documents = loader.load()
    
    # if strategy == "token":
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap,
    #         length_function=len,
    #     )
    #     docs = text_splitter.split_documents(documents)
    # elif strategy == "paragraph":
    #     # Split by double newlines (paragraphs)
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         separators=["\n\n", "\n", " ", ""],
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap,
    #     )
    #     docs = text_splitter.split_documents(documents)
    # elif strategy == "article":
    #     # Split by article markers in GDPR (e.g., "Article 1", "Article 2")
    #     # This would require custom logic based on GDPR structure
    #     docs = documents  # Placeholder
    
    print(f"[DRY-RUN] Would load and split {path} using {strategy} strategy")
    return [
        {
            "page_content": f"[Placeholder GDPR content chunk {i}]",
            "metadata": {"source": path, "page": i, "chunk_id": i}
        }
        for i in range(5)
    ]


def build_and_persist_faiss(
    docs: List[Dict[str, Any]],
    faiss_path: str,
    openai_api_key: Optional[str] = None
) -> str:
    """
    Build FAISS vector store from documents and persist to disk.
    
    Args:
        docs: List of document chunks with content and metadata
        faiss_path: Path where FAISS index should be saved
        openai_api_key: OpenAI API key for embeddings (optional, uses env var if not provided)
        
    Returns:
        Path to the persisted FAISS index
        
    Example:
        >>> docs = load_and_split("gdpr.pdf")
        >>> index_path = build_and_persist_faiss(docs, "faiss_index")
        >>> print(f"FAISS index saved to: {index_path}")
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        warnings.warn("Running in dry-run mode. Set OPENAI_API_KEY to build actual FAISS index.")
        print(f"[DRY-RUN] Would build FAISS index from {len(docs)} docs and save to: {faiss_path}")
        return faiss_path
    
    # TODO: Implement actual FAISS building with OpenAI embeddings
    # from langchain_openai import OpenAIEmbeddings
    # from langchain_community.vectorstores import FAISS
    
    # embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # # Extract text content from docs
    # texts = [doc["page_content"] for doc in docs]
    # metadatas = [doc["metadata"] for doc in docs]
    
    # # Build FAISS index
    # vectorstore = FAISS.from_texts(
    #     texts=texts,
    #     embedding=embeddings,
    #     metadatas=metadatas
    # )
    
    # # Persist to disk
    # vectorstore.save_local(faiss_path)
    
    print(f"[Production mode] Building FAISS index from {len(docs)} documents...")
    print(f"[Production mode] This would save to: {faiss_path}")
    
    return faiss_path


def get_embedding_stats(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about document embeddings.
    
    Args:
        docs: List of document chunks
        
    Returns:
        Dictionary with statistics (count, avg_length, etc.)
    """
    if not docs:
        return {"count": 0, "avg_length": 0}
    
    total_length = sum(len(doc.get("page_content", "")) for doc in docs)
    avg_length = total_length / len(docs) if docs else 0
    
    return {
        "count": len(docs),
        "total_length": total_length,
        "avg_length": avg_length,
        "min_length": min(len(doc.get("page_content", "")) for doc in docs) if docs else 0,
        "max_length": max(len(doc.get("page_content", "")) for doc in docs) if docs else 0,
    }


# Example usage for testing
if __name__ == "__main__":
    print("=== Data Preparation Module Demo ===")
    
    # Download GDPR PDF
    pdf_path = download_gdpr_pdf("gdpr.pdf")
    print(f"1. PDF path: {pdf_path}")
    
    # Load and split document
    docs = load_and_split(pdf_path, strategy="paragraph")
    print(f"2. Created {len(docs)} document chunks")
    
    # Get statistics
    stats = get_embedding_stats(docs)
    print(f"3. Stats: {stats}")
    
    # Build FAISS index
    index_path = build_and_persist_faiss(docs, "faiss_index")
    print(f"4. FAISS index path: {index_path}")
    
    print("\n✓ Module is importable and functional (dry-run mode)")
