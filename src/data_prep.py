"""
Data Preparation Module

Handles loading, preprocessing, and chunking of GDPR documents for RAG.
Works in dry-run mode without API keys.
"""

import os
from typing import List, Dict, Optional
import warnings


class DataPreprocessor:
    """
    Preprocessor for GDPR documents.
    
    Handles:
    - Loading documents from various sources
    - Text cleaning and normalization
    - Chunking with overlap for better retrieval
    - Metadata extraction
    
    Works without API keys by using deterministic text processing.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the data preprocessor.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_documents(self, source_path: str) -> List[Dict[str, str]]:
        """
        Load documents from a source path.
        
        Args:
            source_path: Path to documents
            
        Returns:
            List of document dictionaries with 'content' and 'metadata'
            
        TODO: Implement actual document loading from files/URLs
        """
        # Dry-run mode: return sample GDPR content
        sample_docs = [
            {
                "content": "Article 1: Subject matter and objectives. This Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data and rules relating to the free movement of personal data.",
                "metadata": {"article": "1", "title": "Subject matter and objectives"}
            },
            {
                "content": "Article 5: Principles relating to processing of personal data. Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject ('lawfulness, fairness and transparency').",
                "metadata": {"article": "5", "title": "Principles relating to processing"}
            },
            {
                "content": "Article 17: Right to erasure ('right to be forgotten'). The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay.",
                "metadata": {"article": "17", "title": "Right to erasure"}
            }
        ]
        
        warnings.warn(
            "Running in dry-run mode with sample GDPR documents. "
            f"To load real documents, provide path: {source_path}"
        )
        return sample_docs
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Chunk documents into smaller pieces with overlap.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunks with metadata
            
        TODO: Implement more sophisticated chunking (semantic, sentence-based)
        """
        chunks = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc["content"]
            metadata = doc.get("metadata", {})
            
            # Simple character-based chunking
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk_text = content[i:i + self.chunk_size]
                
                chunk = {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "doc_id": doc_idx,
                        "chunk_id": len(chunks),
                        "start_char": i
                    }
                }
                chunks.append(chunk)
                
                if i + self.chunk_size >= len(content):
                    break
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
            
        TODO: Add more preprocessing steps (stopword removal, lemmatization)
        """
        # Basic preprocessing
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        return text
    
    def prepare_for_rag(self, source_path: str) -> List[Dict[str, any]]:
        """
        End-to-end pipeline: load, preprocess, and chunk documents.
        
        Args:
            source_path: Path to documents
            
        Returns:
            List of preprocessed chunks ready for RAG
        """
        # Load documents
        documents = self.load_documents(source_path)
        
        # Preprocess
        for doc in documents:
            doc["content"] = self.preprocess_text(doc["content"])
        
        # Chunk
        chunks = self.chunk_documents(documents)
        
        return chunks


def main():
    """Example usage of DataPreprocessor."""
    preprocessor = DataPreprocessor(chunk_size=500, chunk_overlap=100)
    
    # Works without API keys
    chunks = preprocessor.prepare_for_rag("gdpr_documents/")
    
    print(f"Prepared {len(chunks)} chunks")
    print(f"Sample chunk: {chunks[0]['content'][:100]}...")


if __name__ == "__main__":
    main()
