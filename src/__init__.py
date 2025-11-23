"""
Project 3: Responsible AI-powered RAG System for GDPR

A modular RAG implementation with responsible AI practices including:
- Data preparation and vector storage
- Baseline RAG pipeline
- Memory integration for conversations
- Guardrails for safety
- Agentic orchestration
- Graph-enhanced RAG
- Responsible AI testing and monitoring
"""

__version__ = "0.1.0"

# Import main modules for easy access
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
