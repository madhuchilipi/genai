"""
Data Preparation Module

Handles downloading, parsing, chunking, embedding, and FAISS index creation
for the GDPR regulation document.

Safe defaults: Returns placeholder data when API keys are not provided.
"""

import os
from typing import List, Optional
from pathlib import Path


def download_gdpr_pdf(save_path: str) -> str:
    """
    Download the GDPR PDF from the official EU website.
    
    Args:
        save_path: Path where the PDF should be saved
        
    Returns:
        Path to the downloaded PDF file
        
    Note:
        In dry-run mode (no network), creates a placeholder file.
        TODO: Implement actual download from https://eur-lex.europa.eu/
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Placeholder implementation - creates a dummy file
    print(f"[DRY-RUN] Would download GDPR PDF to {save_path}")
    print("[DRY-RUN] In production, would download from EU official source")
    
    # Create placeholder file for testing
    save_path.write_text("GDPR Regulation Placeholder Content\n\nArticle 1: Subject-matter and objectives...")
    
    return str(save_path)


def load_and_split(
    path: str,
    strategy: str = "paragraph",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[dict]:
    """
    Load PDF and split into chunks using specified strategy.
    
    Args:
        path: Path to the PDF file
        strategy: Chunking strategy - "paragraph", "article", or "token"
        chunk_size: Size of chunks (for token-based splitting)
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with metadata
        
    Note:
        Uses LangChain loaders in production.
        TODO: Implement PyPDFLoader or UnstructuredPDFLoader
    """
    try:
        # Try to use LangChain if available
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        
        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create splitter based on strategy
        if strategy == "token":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        else:
            # For paragraph/article, use newline-based splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        
        # Create document and split
        doc = Document(page_content=content, metadata={"source": path, "strategy": strategy})
        docs = splitter.split_documents([doc])
        
        print(f"[INFO] Loaded and split document into {len(docs)} chunks using '{strategy}' strategy")
        
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        
    except ImportError:
        # Fallback without LangChain
        print("[DRY-RUN] LangChain not available, using simple splitting")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple paragraph-based splitting
        chunks = content.split('\n\n')
        docs = [
            {"content": chunk.strip(), "metadata": {"source": path, "strategy": strategy, "index": i}}
            for i, chunk in enumerate(chunks) if chunk.strip()
        ]
        
        print(f"[INFO] Created {len(docs)} chunks using simple splitting")
        return docs


def build_and_persist_faiss(
    docs: List[dict],
    faiss_path: str,
    openai_api_key: Optional[str] = None
) -> str:
    """
    Build FAISS index from documents with OpenAI embeddings.
    
    Args:
        docs: List of document chunks with content and metadata
        faiss_path: Directory path to save FAISS index
        openai_api_key: OpenAI API key for embeddings (optional)
        
    Returns:
        Path to the saved FAISS index
        
    Note:
        In dry-run mode (no API key), creates a mock index.
        TODO: Add support for other embedding models (e.g., sentence-transformers)
    """
    faiss_path = Path(faiss_path)
    faiss_path.mkdir(parents=True, exist_ok=True)
    
    # Check for API key
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("[DRY-RUN] No OpenAI API key provided, creating mock FAISS index")
        # Create placeholder index metadata
        index_path = faiss_path / "index.faiss"
        metadata_path = faiss_path / "metadata.txt"
        
        index_path.write_text("Mock FAISS index")
        metadata_path.write_text(f"Documents: {len(docs)}\nEmbedding: mock\n")
        
        print(f"[DRY-RUN] Mock FAISS index saved to {faiss_path}")
        return str(faiss_path)
    
    try:
        # Try to use LangChain with OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.schema import Document
        
        # Convert dict docs to LangChain Documents
        documents = [
            Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
            for doc in docs
        ]
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Build FAISS index
        print(f"[INFO] Building FAISS index from {len(documents)} documents...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Persist to disk
        vectorstore.save_local(str(faiss_path))
        print(f"[INFO] FAISS index saved to {faiss_path}")
        
        return str(faiss_path)
        
    except ImportError as e:
        print(f"[WARNING] Required libraries not available: {e}")
        print("[DRY-RUN] Creating mock FAISS index instead")
        
        index_path = faiss_path / "index.faiss"
        metadata_path = faiss_path / "metadata.txt"
        
        index_path.write_text("Mock FAISS index")
        metadata_path.write_text(f"Documents: {len(docs)}\nEmbedding: mock\n")
        
        return str(faiss_path)
    
    except Exception as e:
        print(f"[ERROR] Failed to build FAISS index: {e}")
        raise


def get_chunk_statistics(docs: List[dict]) -> dict:
    """
    Calculate statistics about the document chunks.
    
    Args:
        docs: List of document chunks
        
    Returns:
        Dictionary with statistics (count, avg length, etc.)
    """
    if not docs:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    
    lengths = [len(doc["content"]) for doc in docs]
    
    return {
        "count": len(docs),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths)
    }
