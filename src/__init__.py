"""
Project 3: Responsible AI-powered RAG System for GDPR

This package provides a complete scaffold for building a production-grade
Retrieval-Augmented Generation (RAG) system with responsible AI practices.

Modules:
    - data_prep: PDF processing, chunking, and vector store creation
    - rag_baseline: Basic RAG pipeline implementation
    - memory: LangGraph memory integration for chat sessions
    - guardrails: Input/output safety filters
    - agent_rag: Multi-agent orchestration with tools
    - graph_rag: Graph-enhanced retrieval strategies
    - responsible_ai: Testing and evaluation utilities
    - langsmith_integration: LangSmith tracing helpers
"""

__version__ = "0.1.0"
__author__ = "madhuchilipi"

# Import main classes and functions for convenient access
# Note: These imports are safe even without API keys
try:
    from .data_prep import download_gdpr_pdf, load_and_split, build_and_persist_faiss
    from .rag_baseline import BaselineRAG
    from .guardrails import detect_adversarial_prompt, safe_rewrite
    from .agent_rag import AgentRunner
    from .graph_rag import rephrase_question, anchor_retrieval, neighbor_retrieval
    from .responsible_ai import calculate_hallucination_score, run_robustness_test
    from .langsmith_integration import init_langsmith, export_traces
except ImportError as e:
    # Graceful degradation if dependencies are missing
    import warnings
    warnings.warn(f"Some imports failed: {e}. Install requirements.txt to use all features.")

__all__ = [
    "download_gdpr_pdf",
    "load_and_split",
    "build_and_persist_faiss",
    "BaselineRAG",
    "detect_adversarial_prompt",
    "safe_rewrite",
    "AgentRunner",
    "rephrase_question",
    "anchor_retrieval",
    "neighbor_retrieval",
    "calculate_hallucination_score",
    "run_robustness_test",
    "init_langsmith",
    "export_traces",
]
