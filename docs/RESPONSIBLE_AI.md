# Responsible AI Guidelines for GDPR RAG System

This document outlines the responsible AI practices implemented in the GDPR RAG system and provides guidelines for maintaining and extending these safeguards.

## Table of Contents

1. [Overview](#overview)
2. [Safety Guardrails](#safety-guardrails)
3. [Evaluation Framework](#evaluation-framework)
4. [LangSmith Usage](#langsmith-usage)
5. [Best Practices](#best-practices)
6. [Maintenance Guidelines](#maintenance-guidelines)

## Overview

The GDPR RAG system implements responsible AI principles across multiple layers:

- **Input Safety**: Detect and handle adversarial prompts
- **Output Validation**: Filter inappropriate or unsafe responses
- **Transparency**: Maintain citation tracking and explainability
- **Robustness**: Test against edge cases and attacks
- **Observability**: Use LangSmith for complete tracing

## Safety Guardrails

### Input Filtering

The system implements input validation to detect:

1. **Adversarial Patterns**
   - Prompt injection attempts ("ignore previous instructions")
   - Jailbreak attempts ("you are now in DAN mode")
   - System prompt leakage attempts

2. **Sensitive Topics**
   - Personal data breach scenarios
   - Security exploitation queries
   - Inappropriate content

3. **Input Validation**
   - Maximum length limits (2000 characters)
   - Empty input handling
   - Malformed query detection

#### Implementation

```python
from src.guardrails import validate_input, detect_adversarial_prompt

# Check input safety
validation = validate_input(user_prompt)
if not validation["is_safe"]:
    # Handle unsafe input
    if validation["recommendation"] == "reject":
        return "I cannot process this request due to safety concerns."
    elif validation["recommendation"] == "rewrite":
        safe_prompt = validation["rewritten_prompt"]
```

### Output Filtering

The system filters outputs for:

- Personal identifiable information (PII)
- API keys or credentials
- Potentially harmful content

```python
from src.guardrails import filter_output

filtered_answer, was_filtered = filter_output(generated_answer)
if was_filtered:
    log_filtering_event(original_answer, filtered_answer)
```

### SafetyGuard Wrapper

Use the `SafetyGuard` class to automatically apply safety checks:

```python
from src.guardrails import SafetyGuard
from src.rag_baseline import BaselineRAG

rag = BaselineRAG("faiss_index")
safe_rag = SafetyGuard(rag, strict_mode=True)

result = safe_rag.safe_query("User query here")
# Automatically checks input and output safety
```

## Evaluation Framework

### Hallucination Detection

The system implements hallucination detection by comparing generated answers with retrieved context:

```python
from src.responsible_ai import calculate_hallucination_score

score = calculate_hallucination_score(answer, retrieved_docs)

if score["hallucination_risk"] == "high":
    # Take corrective action
    log_warning("High hallucination risk detected")
    # Option 1: Request regeneration
    # Option 2: Add disclaimer to answer
    # Option 3: Reject answer
```

**Metrics:**
- **Overlap Score**: Word-level overlap between answer and context
- **Supported Ratio**: Percentage of sentences grounded in context
- **Unsupported Claims**: Specific sentences that lack support

### Robustness Testing

Regular robustness testing ensures system reliability:

```python
from src.responsible_ai import run_robustness_test

results = run_robustness_test(rag_instance, test_cases=[
    {
        "name": "basic_query",
        "query": "What is GDPR?",
        "expected_keywords": ["regulation", "protection"]
    },
    # Add more test cases
])

print(f"Pass rate: {results['pass_rate']:.1%}")
```

**Test Categories:**
1. Basic functionality tests
2. Edge case handling (empty, very long queries)
3. Domain-specific challenges
4. Multi-turn conversation tests
5. Ambiguous query handling

### Adversarial Testing

Test system robustness against attacks:

```python
from src.responsible_ai import run_adversarial_tests

results = run_adversarial_tests(rag_instance, guardrails_enabled=True)

if results["num_failed"] > 0:
    # Review and strengthen guardrails
    review_failed_cases(results["results"])
```

### Quality Evaluation

Comprehensive quality assessment:

```python
from src.responsible_ai import evaluate_response_quality

quality = evaluate_response_quality(query, answer, sources)

# Quality metrics
print(f"Quality score: {quality['quality_score']:.2f}")
print(f"Has citations: {quality['has_citations']}")
print(f"Relevance: {quality['relevance_score']:.2f}")
```

## LangSmith Usage

### Setup

Initialize LangSmith for observability:

```python
from src.langsmith_integration import init_langsmith, setup_langchain_tracing

# Initialize LangSmith
config = init_langsmith(project_name="gdpr-rag-production")

# Enable automatic tracing for all LangChain calls
setup_langchain_tracing()
```

### Manual Tracing

Create manual traces for custom operations:

```python
from src.langsmith_integration import trace_rag_query

trace_id = trace_rag_query(
    query="What is the right to erasure?",
    answer=generated_answer,
    sources=retrieved_sources,
    metadata={
        "model": "gpt-3.5-turbo",
        "retrieval_strategy": "graph_rag",
        "hallucination_risk": "low"
    }
)
```

### Context Manager

Use the context manager for session tracing:

```python
from src.langsmith_integration import LangSmithTracer

with LangSmithTracer("gdpr-rag-session") as tracer:
    for query in user_queries:
        result = rag.query(query)
        tracer.log_query(query, result, metadata={"user_id": "12345"})
```

### Trace Export

Export traces for analysis:

```python
from src.langsmith_integration import export_traces

result = export_traces(
    project_name="gdpr-rag-production",
    output_path="traces/monthly_export.json",
    limit=1000
)
```

### Monitoring Metrics

Key metrics to monitor in LangSmith:

1. **Latency**: Query processing time
2. **Error Rate**: Failed queries
3. **Token Usage**: Cost monitoring
4. **Retrieval Quality**: Relevance scores
5. **User Feedback**: If implemented

## Best Practices

### 1. Always Use Guardrails in Production

```python
# ❌ Don't do this in production
result = rag.query(user_input)

# ✅ Do this instead
safe_rag = SafetyGuard(rag, strict_mode=True)
result = safe_rag.safe_query(user_input)
```

### 2. Validate All Outputs

```python
from src.responsible_ai import evaluate_response_quality

result = rag.query(query)
quality = evaluate_response_quality(query, result["answer"], result["sources"])

if quality["hallucination_risk"] == "high":
    # Add disclaimer or regenerate
    result["answer"] = f"Note: This answer should be verified. {result['answer']}"
```

### 3. Log All Interactions

Use LangSmith to log all interactions for:
- Debugging issues
- Improving system performance
- Identifying edge cases
- Monitoring for misuse

### 4. Regular Testing

Run automated tests regularly:

```bash
# Run import tests
pytest tests/test_imports.py

# Run robustness tests (if implemented)
pytest tests/test_robustness.py

# Run adversarial tests (if implemented)
pytest tests/test_adversarial.py
```

### 5. Update Guardrails Based on Findings

When new attack patterns or edge cases are discovered:

1. Add to test suite
2. Update guardrail patterns
3. Document in this file
4. Re-test entire system

### 6. Monitor for Drift

LLM behavior can change over time:
- Regular evaluation on fixed test sets
- Monitor quality metrics trends
- Update prompts as needed

## Maintenance Guidelines

### Monthly Review

Conduct monthly reviews of:

1. **LangSmith Traces**
   - Review failed queries
   - Identify patterns in user queries
   - Check for new attack patterns

2. **Quality Metrics**
   - Average hallucination scores
   - Citation coverage rates
   - Response quality trends

3. **Guardrail Effectiveness**
   - False positive rate (safe queries blocked)
   - False negative rate (unsafe queries allowed)
   - Update patterns as needed

### Incident Response

If a safety issue is identified:

1. **Immediate**: Block problematic pattern in guardrails
2. **Short-term**: Investigate root cause
3. **Medium-term**: Update tests and documentation
4. **Long-term**: Enhance system architecture if needed

### Documentation Updates

Keep this document updated with:
- New attack patterns discovered
- Updated guardrail patterns
- New evaluation metrics
- Lessons learned from incidents

## Compliance Considerations

Since this system deals with GDPR content:

1. **Data Privacy**: Do not train on personal data
2. **Access Control**: Implement authentication if deployed
3. **Audit Trail**: Maintain LangSmith traces for compliance
4. **Right to Explanation**: Maintain citation tracking
5. **Data Minimization**: Process only necessary information

## Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Safety Best Practices](https://python.langchain.com/docs/security)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [AI Safety Guidelines](https://www.anthropic.com/index/claude-2-system-card)

## Contact

For questions or concerns about responsible AI practices in this system:
- Open a GitHub issue
- Review the code in `src/guardrails.py` and `src/responsible_ai.py`
- Check LangSmith traces for examples

---

**Last Updated**: 2024-11-23  
**Version**: 0.1.0
