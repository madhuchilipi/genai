# Project 3: Responsible AI-powered RAG System for GDPR

A comprehensive Retrieval-Augmented Generation (RAG) system implementing responsible AI practices for querying GDPR regulations. This project demonstrates advanced RAG techniques including baseline RAG, memory integration, guardrails, agentic workflows, and graph-enhanced retrieval.

## 🏗️ Architecture

The system is built with the following components:

1. **Data Preparation Layer**: GDPR PDF ingestion, intelligent chunking, and FAISS vector indexing
2. **Baseline RAG Pipeline**: Simple retrieval-to-generation workflow
3. **Memory Integration**: Conversational context using LangGraph
4. **Guardrails**: Input/output safety filters and adversarial detection
5. **Agentic RAG**: Multi-tool orchestration with Retriever, Citation Checker, and Summarizer
6. **Graph-Enhanced RAG**: Query rephrasing with anchor and neighboring chunk retrieval
7. **Responsible AI Testing**: Hallucination detection, robustness testing, and LangSmith tracing

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Guardrails (Input Filter)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Rephrasing (Graph RAG)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FAISS Vector Retrieval                          │
│          (Anchor + Neighboring Chunks)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Agentic Orchestration (LangGraph)                  │
│      Retriever → Citation Checker → Summarizer               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            LLM Generation with Citations                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               Guardrails (Output Filter)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Final Response                              │
│            + LangSmith Trace Export                          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Setup

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
# Create a .env file
cp .env.example .env  # If available, or create manually

# Add your API keys
echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env
echo "LANGSMITH_API_KEY=your-langsmith-api-key-here" >> .env  # Optional
echo "LANGCHAIN_TRACING_V2=true" >> .env  # Optional
echo "LANGCHAIN_PROJECT=gdpr-rag" >> .env  # Optional
```

**Note**: The code is designed to work in a "dry-run" mode without API keys. When keys are missing, it will return placeholder outputs to allow for testing and CI/CD without secrets.

## 📖 Usage

### Running Notebooks

The project includes 7 step-by-step notebooks in the `notebooks/` directory:

1. **Data Preparation** (`01_data_preparation.ipynb`): Download GDPR PDF, parse, chunk, embed, and build FAISS index
2. **Baseline RAG** (`02_rag_baseline.ipynb`): Simple retrieval and generation pipeline
3. **Memory Integration** (`03_memory_integration.ipynb`): Add conversational memory with LangGraph
4. **Guardrails** (`04_guardrails.ipynb`): Implement input/output safety filters
5. **Agentic RAG** (`05_agentic_rag.ipynb`): Multi-tool agent orchestration
6. **Graph RAG** (`06_graph_rag.ipynb`): Enhanced retrieval with graph techniques
7. **Responsible AI & Testing** (`07_responsible_ai_and_tests.ipynb`): Adversarial testing and tracing

Run notebooks with:
```bash
jupyter notebook notebooks/
```

### Using Python Modules

The `src/` package provides programmatic access:

```python
from src.data_prep import download_gdpr_pdf, load_and_split, build_and_persist_faiss
from src.rag_baseline import BaselineRAG
from src.guardrails import detect_adversarial_prompt, safe_rewrite

# Prepare data
download_gdpr_pdf("data/gdpr.pdf")
docs = load_and_split("data/gdpr.pdf", strategy="paragraph")
build_and_persist_faiss(docs, "faiss_index/", openai_api_key="your-key")

# Run baseline RAG
rag = BaselineRAG(faiss_path="faiss_index/", openai_api_key="your-key")
answer = rag.query("What are the data subject rights under GDPR?")
print(answer)
```

### Running Tests

```bash
pytest tests/
```

## 📋 Deliverables

### Code Deliverables
- ✅ Complete source code in `src/` package
- ✅ 7 Jupyter notebooks demonstrating each milestone
- ✅ Unit tests in `tests/`
- ✅ CI/CD pipeline (`.github/workflows/ci.yml`)

### Documentation
- ✅ README.md (this file)
- ✅ `docs/RESPONSIBLE_AI.md` - Detailed responsible AI practices
- ✅ Inline code documentation and docstrings
- ✅ Notebook markdown cells explaining each step

### Data & Models
- ✅ FAISS vector store (generated from notebooks)
- ✅ GDPR PDF (downloaded programmatically)
- ✅ Embedding model: OpenAI `text-embedding-ada-002`
- ✅ LLM: OpenAI `gpt-3.5-turbo` or `gpt-4`

### Evaluation & Testing
- ✅ Baseline RAG evaluation metrics
- ✅ Adversarial prompt testing
- ✅ Hallucination detection
- ✅ LangSmith trace exports

## 🛡️ Responsible AI Considerations

This project implements several responsible AI practices:

### 1. **Guardrails**
- **Input Filtering**: Detect and handle adversarial, harmful, or off-topic queries
- **Output Filtering**: Validate responses for safety and relevance
- **Safe Rewriting**: Automatically rewrite unsafe prompts to safe versions

### 2. **Transparency**
- **Citations**: All answers include source document references
- **Tracing**: Full execution traces via LangSmith
- **Explainability**: Clear reasoning chains in agentic workflows

### 3. **Robustness Testing**
- **Adversarial Examples**: Test with edge cases and attack prompts
- **Hallucination Detection**: Compare answers against retrieved context
- **Evaluation Metrics**: Precision, recall, F1 for retrieval quality

### 4. **Privacy & Security**
- **Data Handling**: No PII stored in vector databases
- **API Key Management**: Secure handling via environment variables
- **Audit Logs**: LangSmith traces for compliance review

For detailed information, see [`docs/RESPONSIBLE_AI.md`](docs/RESPONSIBLE_AI.md).

## 🔧 Development

### Project Structure

```
genai/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
├── notebooks/                         # Step-by-step tutorials
│   ├── 01_data_preparation.ipynb
│   ├── 02_rag_baseline.ipynb
│   ├── 03_memory_integration.ipynb
│   ├── 04_guardrails.ipynb
│   ├── 05_agentic_rag.ipynb
│   ├── 06_graph_rag.ipynb
│   └── 07_responsible_ai_and_tests.ipynb
├── src/                               # Python package
│   ├── __init__.py
│   ├── data_prep.py                   # Data ingestion and indexing
│   ├── rag_baseline.py                # Baseline RAG implementation
│   ├── memory.py                      # LangGraph memory helpers
│   ├── guardrails.py                  # Safety filters
│   ├── agent_rag.py                   # Agentic orchestration
│   ├── graph_rag.py                   # Graph-enhanced retrieval
│   ├── responsible_ai.py              # Testing and evaluation
│   └── langsmith_integration.py       # Tracing utilities
├── tests/                             # Unit tests
│   └── test_imports.py
├── docs/                              # Documentation
│   └── RESPONSIBLE_AI.md
├── assets/                            # Images and diagrams
│   └── diagram_placeholder.png
└── .github/
    └── workflows/
        └── ci.yml                     # CI/CD pipeline
```

### Adding New Features

1. Implement the feature in `src/`
2. Add corresponding tests in `tests/`
3. Create or update a notebook demonstrating usage
4. Update documentation
5. Run CI checks: `pytest tests/`

### CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:
- Sets up Python environment
- Installs dependencies
- Runs pytest on all tests
- Validates that modules can be imported
- **Runs without API keys** using dry-run mode

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GDPR regulation text from official EU sources
- LangChain and LangGraph frameworks
- OpenAI for embeddings and language models
- FAISS for efficient vector search

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Check existing documentation in `docs/`
- Review notebook examples in `notebooks/`

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [GDPR Official Text](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
