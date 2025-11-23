# Project 3: Responsible AI-powered RAG System for GDPR

A comprehensive Retrieval-Augmented Generation (RAG) system implementing best practices for responsible AI, focused on the General Data Protection Regulation (GDPR) document.

## Overview

This project demonstrates a production-grade RAG system with:
- **Data Preparation**: PDF parsing, chunking strategies, and FAISS vector store
- **Baseline RAG**: Simple retrieval and generation pipeline
- **Memory Integration**: Chat-style sessions using LangGraph
- **Safety Guardrails**: Input/output filtering for responsible AI
- **Agentic RAG**: Multi-tool orchestration with citation checking
- **Graph-enhanced RAG**: Advanced retrieval with question rephrasing and neighbor exploration
- **Responsible AI Testing**: Adversarial testing, hallucination detection, and LangSmith tracing

## Architecture

The system follows a modular architecture:

```
genai/
├── src/                          # Core Python modules
│   ├── data_prep.py             # PDF download, parsing, chunking, embeddings
│   ├── rag_baseline.py          # Basic RAG pipeline
│   ├── memory.py                # LangGraph memory integration
│   ├── guardrails.py            # Safety filters
│   ├── agent_rag.py             # Multi-agent orchestration
│   ├── graph_rag.py             # Graph-enhanced retrieval
│   ├── responsible_ai.py        # Testing and evaluation utilities
│   └── langsmith_integration.py # LangSmith tracing helpers
├── notebooks/                    # Step-by-step Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_rag_baseline.ipynb
│   ├── 03_memory_integration.ipynb
│   ├── 04_guardrails.ipynb
│   ├── 05_agentic_rag.ipynb
│   ├── 06_graph_rag.ipynb
│   └── 07_responsible_ai_and_tests.ipynb
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── assets/                       # Diagrams and resources
```

## Setup

### Prerequisites
- Python 3.9 or higher
- OpenAI API key (for embeddings and LLM calls)
- LangSmith API key (optional, for tracing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/madhuchilipi/genai.git
cd genai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "LANGSMITH_API_KEY=your_langsmith_api_key_here" >> .env  # Optional
```

**Important**: The code is designed to run in "dry-run" mode without API keys. All modules are importable and return placeholder values when secrets are missing. This allows CI/CD pipelines to run without exposing credentials.

## Usage

### Running Notebooks

Start Jupyter:
```bash
jupyter notebook notebooks/
```

Execute notebooks in order (01 through 07) to walk through the complete pipeline:
1. **Data Preparation**: Download GDPR PDF and build vector store
2. **RAG Baseline**: Implement basic retrieval and generation
3. **Memory Integration**: Add conversation history
4. **Guardrails**: Implement safety filters
5. **Agentic RAG**: Build multi-agent system
6. **Graph RAG**: Add advanced retrieval techniques
7. **Responsible AI**: Test and evaluate the system

### Using Python Modules

```python
from src.data_prep import download_gdpr_pdf, build_and_persist_faiss
from src.rag_baseline import BaselineRAG
from src.guardrails import detect_adversarial_prompt

# Download and prepare data (requires OPENAI_API_KEY)
download_gdpr_pdf("gdpr.pdf")
build_and_persist_faiss(docs, "faiss_index", openai_api_key)

# Create RAG pipeline
rag = BaselineRAG("faiss_index", openai_api_key)
answer = rag.query("What are the data subject rights under GDPR?")

# Apply guardrails
is_safe = detect_adversarial_prompt(user_input)
```

### Running Tests

```bash
pytest tests/
```

## Deliverables

This project includes:

- [x] Complete source code with 8 core modules
- [x] 7 comprehensive Jupyter notebooks
- [x] Unit tests for module imports
- [x] CI/CD pipeline via GitHub Actions
- [x] Comprehensive documentation
- [x] Responsible AI considerations document
- [x] Placeholder assets

## Responsible AI Considerations

This project implements responsible AI best practices:

### 1. Safety Guardrails
- **Input filtering**: Detect and reject adversarial prompts
- **Output validation**: Filter inappropriate or unsafe responses
- **Content moderation**: Flag potentially harmful content

### 2. Transparency and Explainability
- **Citation tracking**: All answers include source references
- **Confidence scoring**: Evaluate retrieval quality
- **Trace logging**: Use LangSmith for complete observability

### 3. Robustness and Evaluation
- **Adversarial testing**: Test with edge cases and attacks
- **Hallucination detection**: Verify answers against retrieved context
- **Bias assessment**: Evaluate for potential biases in responses

### 4. Privacy and Security
- **No sensitive data**: Use only public GDPR regulation
- **Secure API handling**: Environment variables for credentials
- **Data minimization**: Process only necessary information

See [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) for detailed guidelines.

## Development

### Adding New Features

1. Implement functionality in `src/` modules
2. Add corresponding tests in `tests/`
3. Create or update notebook demonstrations
4. Update documentation

### Code Quality

The project uses:
- Type hints where appropriate
- Comprehensive docstrings
- Modular, testable code
- CI/CD via GitHub Actions

### TODO for Production

Several areas are marked with TODO comments for production readiness:
- [ ] Add comprehensive error handling
- [ ] Implement retry logic for API calls
- [ ] Add logging and monitoring
- [ ] Optimize chunking strategies
- [ ] Fine-tune retrieval parameters
- [ ] Add more comprehensive tests
- [ ] Implement rate limiting
- [ ] Add caching layer
- [ ] Benchmark performance
- [ ] Add user authentication (if deploying as service)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure CI passes
5. Submit a pull request

## License

This project is provided as-is for educational purposes.

## Acknowledgments

Built using:
- [LangChain](https://python.langchain.com/) - LLM application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [OpenAI](https://openai.com/) - Embeddings and LLM
- [LangSmith](https://smith.langchain.com/) - Observability and tracing

---

For questions or issues, please open a GitHub issue in the repository.
