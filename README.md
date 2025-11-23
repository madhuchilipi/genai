# Course-end Project 3: Responsible AI Retrieval-Augmented Generation (RAG) System

## Overview

This project implements a comprehensive Responsible AI RAG system for the GDPR (General Data Protection Regulation) corpus. The system demonstrates advanced techniques in retrieval-augmented generation, including baseline RAG, memory management, guardrails, agent-based RAG, and graph-based RAG approaches.

### Key Features

- **Baseline RAG**: Vector similarity search with LangChain and FAISS
- **Memory Management**: Conversation history and context preservation
- **Guardrails**: Input/output validation, PII detection, and content filtering
- **Agent-based RAG**: Autonomous agent with tool use via LangGraph
- **Graph RAG**: Knowledge graph-based retrieval for enhanced reasoning
- **Responsible AI**: Bias detection, transparency, and ethical safeguards
- **LangSmith Integration**: Production monitoring and debugging

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                             │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Input Guardrails    │
        │  (PII, Content Filter)│
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    Agent / Router     │
        │    (LangGraph)        │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌──────────────┐
│  Vector RAG   │       │  Graph RAG   │
│  (FAISS)      │       │  (Knowledge) │
└───────┬───────┘       └──────┬───────┘
        │                      │
        └───────────┬──────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Memory Manager      │
        │   (Conversation)      │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Output Guardrails    │
        │  (Bias, Transparency) │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   LangSmith Tracing   │
        └───────────┬───────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Response    │
            └───────────────┘
```

## Setup

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- LangSmith API key (optional, for monitoring)

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

4. Configure environment variables:

Create a `.env` file in the project root:
```bash
# Required for production mode
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For LangSmith monitoring
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=project3-rag-gdpr
```

⚠️ **Important**: Never commit your `.env` file or API keys to version control!

### Dry-Run Mode

The system supports a **dry-run mode** for testing and development without API keys:

- When `OPENAI_API_KEY` is not set, modules return deterministic placeholder responses
- When `LANGSMITH_API_KEY` is not set, tracing is disabled gracefully
- All tests and CI runs work without requiring actual API credentials

This allows you to:
- Explore the codebase without incurring API costs
- Run tests in CI/CD pipelines without secrets
- Develop and debug locally before obtaining API keys

## Usage

### Running Notebooks

The project includes 7 sequential notebooks that guide you through the implementation:

1. **01_data_preparation.ipynb**: Load and preprocess GDPR documents
2. **02_rag_baseline.ipynb**: Build baseline RAG with vector search
3. **03_memory_management.ipynb**: Add conversation memory
4. **04_guardrails.ipynb**: Implement input/output guardrails
5. **05_agent_rag.ipynb**: Create autonomous agent with tools
6. **06_graph_rag.ipynb**: Build knowledge graph RAG
7. **07_responsible_ai_and_tests.ipynb**: Test responsible AI features

Start Jupyter:
```bash
jupyter notebook notebooks/
```

### Using the Python Modules

```python
from src.rag_baseline import RAGSystem
from src.guardrails import GuardrailsManager

# Initialize system (works in dry-run mode without API keys)
rag = RAGSystem()
guardrails = GuardrailsManager()

# Query the system
response = rag.query("What are the GDPR data subject rights?")
print(response)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_imports.py

# Run with verbose output
pytest -v
```

## Project Deliverables

### Code Deliverables
- ✅ **src/**: Complete Python package with 9 modules
- ✅ **notebooks/**: 7 Jupyter notebooks with documented workflow
- ✅ **tests/**: Pytest suite for imports and functionality
- ✅ **requirements.txt**: Pinned dependencies

### Documentation Deliverables
- ✅ **README.md**: This file (architecture, setup, usage)
- ✅ **docs/RESPONSIBLE_AI.md**: Responsible AI considerations
- ✅ **assets/diagram_placeholder.png**: System architecture diagram
- ✅ Inline code documentation with TODO comments

### CI/CD Deliverables
- ✅ **.github/workflows/ci.yml**: Automated testing pipeline
- ✅ **Import safety**: All modules work without API keys
- ✅ **Test coverage**: Import tests and smoke tests

## Responsible AI Considerations

This project implements several responsible AI practices:

1. **Privacy Protection**
   - PII detection and redaction in guardrails
   - Data minimization principles
   - GDPR compliance by design

2. **Transparency**
   - LangSmith tracing for explainability
   - Clear documentation of model decisions
   - Audit logs for all queries

3. **Bias Detection**
   - Output analysis for potential biases
   - Diverse testing scenarios
   - Fairness metrics

4. **Safety Guardrails**
   - Input validation and sanitization
   - Content filtering (harmful content, misinformation)
   - Output review before delivery

5. **Ethical Considerations**
   - User consent and data handling
   - Right to explanation
   - Human oversight mechanisms

See [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) for detailed discussion.

## Module Overview

### Core Modules

- **data_prep.py**: Data loading, chunking, and preprocessing for GDPR documents
- **rag_baseline.py**: Basic RAG implementation with FAISS vector store
- **memory.py**: Conversation memory and context management
- **guardrails.py**: Input/output validation, PII detection, content filtering
- **agent_rag.py**: Agent-based RAG with tool use (LangGraph)
- **graph_rag.py**: Knowledge graph construction and graph-based retrieval
- **responsible_ai.py**: Bias detection, transparency, and ethical safeguards
- **langsmith_integration.py**: LangSmith tracing and monitoring

## Development

### Adding New Features

1. Implement the feature in the appropriate `src/` module
2. Add corresponding notebook examples in `notebooks/`
3. Write tests in `tests/`
4. Update documentation
5. Run linting and tests before committing

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Troubleshooting

### API Key Issues

**Problem**: `openai.error.AuthenticationError`
**Solution**: Check that `OPENAI_API_KEY` is set correctly in `.env`

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Ensure you're running from the project root directory

### FAISS Installation Issues

**Problem**: FAISS installation fails on some platforms
**Solution**: Use `faiss-cpu` (already specified in requirements.txt)

### Dry-Run Mode Not Working

**Problem**: System requires API keys even in dry-run mode
**Solution**: Check that modules properly detect missing API keys and return placeholders

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is for educational purposes as part of a GenAI course.

## Support

For questions or issues, please open an issue on GitHub or contact the course instructors.

---

**Note**: This is a course project demonstrating advanced RAG techniques with responsible AI practices. While the code is functional, additional hardening would be needed for production deployment.
