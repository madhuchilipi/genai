# Project 3: Responsible AI-powered RAG System for GDPR

A Retrieval-Augmented Generation (RAG) system implementing responsible AI practices for querying and understanding GDPR regulations.

## Overview

This project demonstrates a complete RAG pipeline with:
- Document ingestion and preprocessing (GDPR PDF)
- Vector storage using FAISS
- Baseline RAG with OpenAI LLM
- Memory integration for conversational sessions
- Guardrails for input/output safety
- Agentic RAG with tool orchestration
- Graph-enhanced RAG for improved retrieval
- Responsible AI testing and evaluation

## Architecture

The system consists of several key components:

1. **Data Preparation**: Downloads GDPR PDF, parses with LangChain, chunks strategically, generates embeddings, and builds FAISS index
2. **Baseline RAG**: Simple query → retrieve → generate pipeline
3. **Memory Integration**: LangGraph-powered conversational memory for chat sessions
4. **Guardrails**: Input/output filters for safety and compliance
5. **Agentic RAG**: Multi-agent orchestration with Retriever, Citation Checker, and Summarizer
6. **Graph RAG**: Enhanced retrieval with query rephrasing, anchor retrieval, and neighbor expansion
7. **Responsible AI**: Hallucination detection, adversarial testing, and LangSmith tracing

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key (for embeddings and LLM)
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
# Create .env file
cp .env.example .env  # If example exists, or create new .env

# Add your API keys
echo "OPENAI_API_KEY=your-openai-key-here" >> .env
echo "LANGSMITH_API_KEY=your-langsmith-key-here" >> .env  # Optional
```

**Important**: All modules are designed to work in "dry-run" mode without API keys for CI/CD and testing purposes. When API keys are not provided, the system returns placeholder outputs instead of failing.

## Usage

### Running Notebooks

The notebooks are numbered sequentially and should be run in order:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and start with:
1. `01_data_preparation.ipynb` - Set up FAISS index
2. `02_rag_baseline.ipynb` - Test baseline RAG
3. `03_memory_integration.ipynb` - Add conversational memory
4. `04_guardrails.ipynb` - Test safety filters
5. `05_agentic_rag.ipynb` - Multi-agent orchestration
6. `06_graph_rag.ipynb` - Graph-enhanced retrieval
7. `07_responsible_ai_and_tests.ipynb` - Validation and testing

### Using the Python API

```python
from src.data_prep import download_gdpr_pdf, load_and_split, build_and_persist_faiss
from src.rag_baseline import BaselineRAG

# Prepare data (run once)
pdf_path = download_gdpr_pdf("data/gdpr.pdf")
docs = load_and_split(pdf_path, strategy="paragraph")
build_and_persist_faiss(docs, "faiss_index", openai_api_key="your-key")

# Use RAG
rag = BaselineRAG(faiss_path="faiss_index", openai_api_key="your-key")
answer = rag.query("What are the key principles of GDPR?")
print(answer)
```

### Running Tests

```bash
pytest tests/
```

Tests are designed to run without API keys by validating imports and basic functionality with mocked outputs.

## Project Deliverables

- [x] README with project overview and setup instructions
- [x] Seven Jupyter notebooks demonstrating each milestone
- [x] Complete `src/` package with all modules
- [x] Test suite with import validation
- [x] CI/CD pipeline (GitHub Actions)
- [x] Responsible AI documentation
- [x] Architecture diagram placeholder

## Responsible AI Considerations

This project implements several responsible AI practices:

### 1. Guardrails
- **Input filtering**: Detect and reject adversarial prompts
- **Output filtering**: Ensure responses are safe and appropriate
- **Prompt rewriting**: Transform potentially problematic inputs

### 2. Evaluation
- **Hallucination detection**: Compare generated responses against retrieved context
- **Citation checking**: Verify claims are grounded in source material
- **Robustness testing**: Test with adversarial and edge-case inputs

### 3. Observability
- **LangSmith integration**: Trace all LLM calls for debugging and auditing
- **Performance metrics**: Track retrieval accuracy, response quality, and latency
- **Cost monitoring**: Track token usage and API costs

### 4. Safety by Default
- All modules work in dry-run mode without API keys
- No sensitive data committed to repository
- Clear documentation on API key management
- Graceful degradation when services are unavailable

See `docs/RESPONSIBLE_AI.md` for detailed guidelines.

## Development

### Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_rag_baseline.ipynb
│   ├── 03_memory_integration.ipynb
│   ├── 04_guardrails.ipynb
│   ├── 05_agentic_rag.ipynb
│   ├── 06_graph_rag.ipynb
│   └── 07_responsible_ai_and_tests.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── rag_baseline.py
│   ├── memory.py
│   ├── guardrails.py
│   ├── agent_rag.py
│   ├── graph_rag.py
│   ├── responsible_ai.py
│   └── langsmith_integration.py
├── tests/
│   └── test_imports.py
├── docs/
│   └── RESPONSIBLE_AI.md
├── assets/
│   └── diagram_placeholder.png
└── .github/
    └── workflows/
        └── ci.yml
```

### Contributing

1. Create a feature branch
2. Make changes with clear commit messages
3. Ensure tests pass: `pytest tests/`
4. Submit a pull request

## TODO

- [ ] Implement production-grade error handling in all modules
- [ ] Add more comprehensive test coverage
- [ ] Implement actual GDPR PDF download (currently uses placeholder)
- [ ] Add support for additional document formats
- [ ] Implement cost estimation utilities
- [ ] Add multilingual support
- [ ] Create web UI for easier interaction
- [ ] Add more sophisticated chunking strategies
- [ ] Implement query analytics dashboard

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue in the repository.

## Acknowledgments

- LangChain for the RAG framework
- OpenAI for embeddings and LLM
- FAISS for vector storage
- LangGraph for agent orchestration
- LangSmith for observability
