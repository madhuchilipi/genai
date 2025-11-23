"""
Test that all modules can be imported without errors.

These tests ensure the codebase is import-safe without API keys,
which is critical for CI/CD pipelines.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_import_data_prep():
    """Test that data_prep module can be imported."""
    from src import data_prep
    assert hasattr(data_prep, 'DataPreprocessor')


def test_import_rag_baseline():
    """Test that rag_baseline module can be imported."""
    from src import rag_baseline
    assert hasattr(rag_baseline, 'RAGSystem')


def test_import_memory():
    """Test that memory module can be imported."""
    from src import memory
    assert hasattr(memory, 'MemoryManager')


def test_import_guardrails():
    """Test that guardrails module can be imported."""
    from src import guardrails
    assert hasattr(guardrails, 'GuardrailsManager')


def test_import_agent_rag():
    """Test that agent_rag module can be imported."""
    from src import agent_rag
    assert hasattr(agent_rag, 'AgentRAG')


def test_import_graph_rag():
    """Test that graph_rag module can be imported."""
    from src import graph_rag
    assert hasattr(graph_rag, 'GraphRAG')


def test_import_responsible_ai():
    """Test that responsible_ai module can be imported."""
    from src import responsible_ai
    assert hasattr(responsible_ai, 'ResponsibleAI')


def test_import_langsmith_integration():
    """Test that langsmith_integration module can be imported."""
    from src import langsmith_integration
    assert hasattr(langsmith_integration, 'LangSmithTracer')


def test_import_src_package():
    """Test that src package can be imported."""
    import src
    assert hasattr(src, '__version__')


def test_instantiate_data_preprocessor():
    """Test that DataPreprocessor can be instantiated without API keys."""
    from src.data_prep import DataPreprocessor
    
    preprocessor = DataPreprocessor(chunk_size=500, chunk_overlap=100)
    assert preprocessor.chunk_size == 500
    assert preprocessor.chunk_overlap == 100


def test_instantiate_rag_system():
    """Test that RAGSystem can be instantiated without API keys."""
    from src.rag_baseline import RAGSystem
    
    rag = RAGSystem()
    assert rag is not None
    # Should work in dry-run mode
    assert hasattr(rag, 'query')


def test_instantiate_memory_manager():
    """Test that MemoryManager can be instantiated without API keys."""
    from src.memory import MemoryManager
    
    memory = MemoryManager(max_history=5)
    assert memory.max_history == 5
    assert len(memory.conversation_history) == 0


def test_instantiate_guardrails_manager():
    """Test that GuardrailsManager can be instantiated without API keys."""
    from src.guardrails import GuardrailsManager
    
    guardrails = GuardrailsManager(strict_mode=False)
    assert not guardrails.strict_mode
    assert hasattr(guardrails, 'validate_input')


def test_instantiate_agent_rag():
    """Test that AgentRAG can be instantiated without API keys."""
    from src.agent_rag import AgentRAG
    
    agent = AgentRAG()
    assert agent is not None
    assert len(agent.tools) > 0


def test_instantiate_graph_rag():
    """Test that GraphRAG can be instantiated without API keys."""
    from src.graph_rag import GraphRAG
    
    graph = GraphRAG()
    assert graph is not None
    # Should have sample graph in dry-run mode
    assert len(graph.graph) > 0


def test_instantiate_responsible_ai():
    """Test that ResponsibleAI can be instantiated without API keys."""
    from src.responsible_ai import ResponsibleAI
    
    rai = ResponsibleAI(enable_logging=True)
    assert rai.enable_logging
    assert len(rai.audit_log) == 0


def test_instantiate_langsmith_tracer():
    """Test that LangSmithTracer can be instantiated without API keys."""
    from src.langsmith_integration import LangSmithTracer
    
    tracer = LangSmithTracer(project_name="test-project")
    assert tracer.project_name == "test-project"
    assert len(tracer.traces) == 0


def test_dry_run_data_prep():
    """Test data preprocessing in dry-run mode."""
    from src.data_prep import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    chunks = preprocessor.prepare_for_rag("dummy_path")
    
    assert len(chunks) > 0
    assert 'content' in chunks[0]
    assert 'metadata' in chunks[0]


def test_dry_run_rag_query():
    """Test RAG query in dry-run mode."""
    from src.rag_baseline import RAGSystem
    from src.data_prep import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    chunks = preprocessor.prepare_for_rag("dummy_path")
    
    rag = RAGSystem()
    rag.index_documents(chunks)
    response = rag.query("What is GDPR?")
    
    assert isinstance(response, str)
    assert len(response) > 0


def test_dry_run_memory():
    """Test memory management in dry-run mode."""
    from src.memory import MemoryManager
    
    memory = MemoryManager()
    memory.add_interaction("What is GDPR?", "GDPR is a regulation...")
    
    assert len(memory.conversation_history) == 1
    context = memory.get_conversation_context()
    assert "What is GDPR?" in context


def test_dry_run_guardrails():
    """Test guardrails in dry-run mode."""
    from src.guardrails import GuardrailsManager
    
    guardrails = GuardrailsManager()
    
    # Valid query
    is_valid, sanitized, error = guardrails.apply_guardrails("What are GDPR rights?")
    assert is_valid
    assert error is None
    
    # Empty query
    is_valid, _, error = guardrails.apply_guardrails("")
    assert not is_valid
    assert error is not None


def test_dry_run_agent():
    """Test agent in dry-run mode."""
    from src.agent_rag import AgentRAG
    
    agent = AgentRAG()
    response = agent.query("What are GDPR penalties?")
    
    assert isinstance(response, str)
    assert len(response) > 0


def test_dry_run_graph_rag():
    """Test graph RAG in dry-run mode."""
    from src.graph_rag import GraphRAG
    
    graph = GraphRAG()
    response = graph.query("What rights does a data subject have?")
    
    assert isinstance(response, str)
    assert len(response) > 0


def test_dry_run_responsible_ai():
    """Test responsible AI in dry-run mode."""
    from src.responsible_ai import ResponsibleAI
    
    rai = ResponsibleAI()
    
    # Test bias detection
    bias_results = rai.detect_bias("The developer should ensure he tests his code.")
    assert 'bias_score' in bias_results
    
    # Test ethical compliance
    ethics = rai.check_ethical_compliance("What is GDPR?", "GDPR is a regulation.")
    assert 'compliant' in ethics


def test_dry_run_langsmith():
    """Test LangSmith integration in dry-run mode."""
    from src.langsmith_integration import LangSmithTracer
    
    tracer = LangSmithTracer()
    trace_id = tracer.trace_query("Test query", "Test response")
    
    assert isinstance(trace_id, str)
    assert len(tracer.traces) == 1


def test_pii_detection():
    """Test PII detection functionality."""
    from src.guardrails import GuardrailsManager
    
    guardrails = GuardrailsManager()
    
    text = "Contact me at test@example.com or call 555-123-4567"
    detected = guardrails.detect_pii(text)
    
    assert 'email' in detected
    assert 'phone' in detected


def test_pii_redaction():
    """Test PII redaction functionality."""
    from src.guardrails import GuardrailsManager
    
    guardrails = GuardrailsManager()
    
    text = "My email is test@example.com"
    redacted, counts = guardrails.redact_pii(text)
    
    assert "REDACTED" in redacted
    assert counts.get('email', 0) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
