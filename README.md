# Project 3: Responsible AI-powered RAG System for GDPR

## Overview

This project implements a Responsible AI-powered Retrieval-Augmented Generation (RAG) system for the General Data Protection Regulation (GDPR). The system demonstrates advanced RAG techniques including baseline retrieval, memory integration, guardrails, agentic workflows, and graph-enhanced retrieval.

## Architecture

The system consists of the following components:

1. **Data Preparation Pipeline**: Downloads GDPR PDF, parses and chunks the document, generates embeddings, and builds a FAISS vector store
2. **Baseline RAG**: Simple query → retrieval → LLM answer pipeline
3. **Memory Integration**: LangGraph-based conversational memory for multi-turn interactions
4. **Guardrails**: Input/output safety filters to detect and handle adversarial prompts
5. **Agentic RAG**: Multi-tool orchestration with Retriever, CitationChecker, and Summarizer agents
6. **Graph RAG**: Enhanced retrieval using question rephrasing, anchor retrieval, and neighboring chunk expansion
7. **Responsible AI Testing**: Adversarial testing, hallucination detection, and LangSmith trace analysis

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/madhuchilipi/genai.git
cd genai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (create a `.env` file):
```bash
# Required for full functionality
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional, for tracing

# LangSmith configuration (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=gdpr-rag-system
```

**Note**: The code is designed to work without API keys for testing and CI purposes. When API keys are not provided, modules will run in "dry-run" mode with placeholder outputs.

## Usage

### Running Notebooks

The project includes step-by-step Jupyter notebooks in the `notebooks/` directory:

1. **01_data_preparation.ipynb**: Download and prepare GDPR data
2. **02_rag_baseline.ipynb**: Build and test baseline RAG pipeline
3. **03_memory_integration.ipynb**: Add conversational memory
4. **04_guardrails.ipynb**: Implement safety filters
5. **05_agentic_rag.ipynb**: Multi-agent orchestration
6. **06_graph_rag.ipynb**: Graph-enhanced retrieval
7. **07_responsible_ai_and_tests.ipynb**: Testing and evaluation

Start Jupyter:
```bash
jupyter notebook
```

### Using Python Modules

The `src/` package provides reusable modules:

```python
from src.data_prep import download_gdpr_pdf, load_and_split, build_and_persist_faiss
from src.rag_baseline import BaselineRAG
from src.guardrails import detect_adversarial_prompt, safe_rewrite
from src.agent_rag import AgentRunner
from src.graph_rag import rephrase_question, anchor_retrieve, neighbor_retrieve
from src.responsible_ai import detect_hallucination, run_robustness_tests
```

### Running Tests

```bash
pytest tests/
```

## Project Deliverables

- ✅ Complete RAG system implementation with modular architecture
- ✅ 7 step-by-step Jupyter notebooks demonstrating each milestone
- ✅ Comprehensive Python package with documented modules
- ✅ Input/output guardrails for responsible AI
- ✅ Agentic and graph-enhanced RAG workflows
- ✅ Testing infrastructure with adversarial examples
- ✅ LangSmith integration for trace analysis
- ✅ CI/CD pipeline for automated testing
- ✅ Documentation on responsible AI considerations

## Responsible AI Considerations

This project prioritizes responsible AI practices:

1. **Safety Guardrails**: Input validation and output filtering to detect harmful content
2. **Hallucination Detection**: Automatic scoring of answers against retrieved context
3. **Citation Tracking**: All answers include source citations for transparency
4. **Adversarial Testing**: Systematic testing with edge cases and malicious inputs
5. **Observability**: LangSmith integration for full trace analysis and debugging
6. **Dry-run Mode**: Safe operation without API keys for development and testing

See [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) for detailed information.

## Development

### Project Structure

```
genai/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/           # Step-by-step Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_rag_baseline.ipynb
│   ├── 03_memory_integration.ipynb
│   ├── 04_guardrails.ipynb
│   ├── 05_agentic_rag.ipynb
│   ├── 06_graph_rag.ipynb
│   └── 07_responsible_ai_and_tests.ipynb
├── src/                 # Python package modules
│   ├── __init__.py
│   ├── data_prep.py
│   ├── rag_baseline.py
│   ├── memory.py
│   ├── guardrails.py
│   ├── agent_rag.py
│   ├── graph_rag.py
│   ├── responsible_ai.py
│   └── langsmith_integration.py
├── tests/               # Test suite
│   └── test_imports.py
├── assets/              # Assets and diagrams
│   └── diagram_placeholder.png
├── docs/                # Documentation
│   └── RESPONSIBLE_AI.md
└── .github/
    └── workflows/
        └── ci.yml       # GitHub Actions CI
```

### Contributing

When contributing, please ensure:
- All modules are importable without API keys
- Tests pass in CI without secrets
- Code includes appropriate docstrings and comments
- Notebooks include clear markdown explanations

### TODO and Future Work

The current implementation provides a solid scaffold with working demonstrations. For production deployment, consider:

- [ ] Implement production-grade error handling and retries
- [ ] Add caching layer for embeddings and LLM calls
- [ ] Expand test coverage with more edge cases
- [ ] Add support for additional document formats beyond PDF
- [ ] Implement more sophisticated chunking strategies
- [ ] Add evaluation metrics and benchmarking suite
- [ ] Deploy as a web service with REST API
- [ ] Add user authentication and rate limiting

## License

This project is part of an educational initiative focused on responsible AI development.

## Contact

For questions or issues, please open a GitHub issue in the repository.
