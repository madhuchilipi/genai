"""
Course-end Project 3: Responsible AI RAG System for GDPR

This package provides a comprehensive RAG implementation with responsible AI features.

Modules:
- data_prep: Data loading and preprocessing
- rag_baseline: Basic RAG with vector search
- memory: Conversation memory management
- guardrails: Input/output validation and safety
- agent_rag: Agent-based RAG with tools
- graph_rag: Knowledge graph-based retrieval
- responsible_ai: Bias detection and ethical safeguards
- langsmith_integration: Monitoring and tracing

All modules support dry-run mode without API keys.
"""

__version__ = "0.1.0"
__author__ = "Course Project Team"

# Import main classes for convenience (optional)
# These imports are safe and will work without API keys
try:
    from .data_prep import DataPreprocessor
    from .rag_baseline import RAGSystem
    from .memory import MemoryManager
    from .guardrails import GuardrailsManager
    from .agent_rag import AgentRAG
    from .graph_rag import GraphRAG
    from .responsible_ai import ResponsibleAI
    from .langsmith_integration import LangSmithTracer
    
    __all__ = [
        'DataPreprocessor',
        'RAGSystem',
        'MemoryManager',
        'GuardrailsManager',
        'AgentRAG',
        'GraphRAG',
        'ResponsibleAI',
        'LangSmithTracer',
    ]
except ImportError as e:
    # Graceful degradation if dependencies are not installed
    print(f"Warning: Some imports failed: {e}")
    print("Install all dependencies with: pip install -r requirements.txt")
    __all__ = []
