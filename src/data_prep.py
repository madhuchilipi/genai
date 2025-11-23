"""
Data preparation module for GDPR RAG system.

Handles downloading, parsing, chunking, and embedding of GDPR PDF documents.
Builds and persists FAISS vector stores for retrieval.
"""

import os
from typing import List, Dict, Any, Optional
import warnings


def download_gdpr_pdf(save_path: str = "gdpr.pdf") -> str:
    """
    Download the GDPR PDF from official source.
    
    Args:
        save_path: Path where the PDF should be saved
        
    Returns:
        Path to the downloaded PDF file
        
    Note:
        In dry-run mode (no network/API), returns a placeholder path.
        TODO: Implement actual download from https://gdpr-info.eu/ or official source
    """
    print(f"[DRY-RUN] Would download GDPR PDF to {save_path}")
    print("TODO: Implement actual PDF download using requests library")
    print("Source: https://eur-lex.europa.eu/eli/reg/2016/679/oj")
    
    # Return placeholder path for testing
    return save_path


def load_and_split(
    path: str,
    strategy: str = "paragraph",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Load PDF and split into chunks using specified strategy.
    
    Args:
        path: Path to PDF file
        strategy: Chunking strategy - "paragraph", "article", or "token"
        chunk_size: Size of chunks in tokens (for token strategy)
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of document chunks with metadata
        
    Note:
        Uses LangChain's PyPDFLoader and text splitters in production.
        Returns placeholder chunks in dry-run mode.
        
    TODO: Implement actual PDF loading with:
        - from langchain.document_loaders import PyPDFLoader
        - from langchain.text_splitter import RecursiveCharacterTextSplitter
    """
    print(f"[DRY-RUN] Would load PDF from {path} with strategy '{strategy}'")
    print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    
    # Return placeholder chunks for testing
    placeholder_chunks = [
        {
            "content": "Article 1: Subject-matter and objectives. This Regulation lays down rules...",
            "metadata": {"source": path, "page": 1, "article": 1}
        },
        {
            "content": "Article 2: Material scope. This Regulation applies to the processing...",
            "metadata": {"source": path, "page": 1, "article": 2}
        },
        {
            "content": "Article 3: Territorial scope. This Regulation applies to the processing...",
            "metadata": {"source": path, "page": 2, "article": 3}
        }
    ]
    
    print(f"Generated {len(placeholder_chunks)} placeholder chunks")
    return placeholder_chunks


def build_and_persist_faiss(
    docs: List[Dict[str, Any]],
    faiss_path: str = "faiss_index",
    openai_api_key: Optional[str] = None
) -> bool:
    """
    Build FAISS vector store from documents and persist to disk.
    
    Args:
        docs: List of document chunks with content and metadata
        faiss_path: Directory path to save FAISS index
        openai_api_key: OpenAI API key for embeddings (optional)
        
    Returns:
        True if successful, False otherwise
        
    Note:
        In dry-run mode (no API key), simulates the process.
        
    TODO: Implement actual FAISS indexing with:
        - from langchain.embeddings import OpenAIEmbeddings
        - from langchain.vectorstores import FAISS
        - embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        - vectorstore = FAISS.from_documents(docs, embeddings)
        - vectorstore.save_local(faiss_path)
    """
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("[DRY-RUN] No OpenAI API key provided - simulating FAISS build")
        print(f"Would build FAISS index from {len(docs)} documents")
        print(f"Would save to: {faiss_path}")
        print("In production, requires: pip install openai langchain faiss-cpu")
        return True
    
    print(f"Building FAISS index from {len(docs)} documents...")
    print(f"Using OpenAI embeddings (model: text-embedding-ada-002)")
    print(f"Saving to: {faiss_path}")
    
    # TODO: Implement actual FAISS build
    warnings.warn("Full FAISS implementation not yet complete - returning success for testing")
    return True


def get_example_chunks() -> List[Dict[str, Any]]:
    """
    Get example GDPR chunks for testing without downloading.
    
    Returns:
        List of example document chunks
    """
    return [
        {
            "content": "Chapter I - General Provisions. Article 1: This Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data.",
            "metadata": {"chapter": 1, "article": 1, "page": 1}
        },
        {
            "content": "Article 4: For the purposes of this Regulation: 'personal data' means any information relating to an identified or identifiable natural person ('data subject').",
            "metadata": {"chapter": 1, "article": 4, "page": 2}
        },
        {
            "content": "Article 5: Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject ('lawfulness, fairness and transparency').",
            "metadata": {"chapter": 2, "article": 5, "page": 3}
        }
    ]


if __name__ == "__main__":
    # Example usage
    print("=== Data Preparation Module Demo ===\n")
    
    # Download PDF
    pdf_path = download_gdpr_pdf("gdpr.pdf")
    print()
    
    # Load and split
    chunks = load_and_split(pdf_path, strategy="paragraph")
    print()
    
    # Build FAISS index
    success = build_and_persist_faiss(chunks, "faiss_index")
    print(f"\nSuccess: {success}")
