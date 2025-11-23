"""
Responsible AI-powered RAG System for GDPR

This package provides a complete implementation of a Retrieval-Augmented Generation
system with responsible AI features, including guardrails, memory, and agentic workflows.
"""

__version__ = "0.1.0"

# Import main components for easier access
from . import data_prep
from . import rag_baseline
from . import memory
from . import guardrails
from . import agent_rag
from . import graph_rag
from . import responsible_ai
from . import langsmith_integration

__all__ = [
    "data_prep",
    "rag_baseline",
    "memory",
    "guardrails",
    "agent_rag",
    "graph_rag",
    "responsible_ai",
    "langsmith_integration",
]
