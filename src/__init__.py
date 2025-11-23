"""
Project 3: Responsible AI-powered RAG System for GDPR

A comprehensive RAG system with:
- Data preparation and FAISS indexing
- Baseline and advanced RAG implementations
- Memory integration with LangGraph
- Guardrails for safety
- Agentic workflows
- Graph-enhanced retrieval
- Responsible AI testing and tracing
"""

__version__ = "0.1.0"

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
