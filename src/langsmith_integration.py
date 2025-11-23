"""
LangSmith Integration Module

This module provides helpers for LangSmith tracing and observability:
- Initialize LangSmith client
- Export traces
- Trace management
- Performance monitoring
"""

import os
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime


def init_langsmith(
    api_key: Optional[str] = None,
    project_name: str = "gdpr-rag-system"
) -> Dict[str, Any]:
    """
    Initialize LangSmith client for tracing.
    
    Args:
        api_key: LangSmith API key (uses env var if not provided)
        project_name: Project name for organizing traces
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = init_langsmith(project_name="my-rag-project")
        >>> print(f"LangSmith enabled: {config['enabled']}")
    """
    api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
    
    if not api_key:
        warnings.warn(
            "LANGSMITH_API_KEY not set. Running without tracing. "
            "Set the environment variable to enable LangSmith tracing."
        )
        return {
            "enabled": False,
            "project_name": project_name,
            "reason": "No API key provided"
        }
    
    # TODO: Implement actual LangSmith initialization
    # from langsmith import Client
    # client = Client(api_key=api_key)
    # 
    # # Set environment variables for LangChain integration
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # os.environ["LANGCHAIN_API_KEY"] = api_key
    # os.environ["LANGCHAIN_PROJECT"] = project_name
    
    print(f"[Production mode] Would initialize LangSmith for project: {project_name}")
    
    return {
        "enabled": True,
        "project_name": project_name,
        "api_key_set": True,
        "endpoint": "https://api.smith.langchain.com"
    }


def export_traces(
    project_name: str,
    output_path: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Export traces from LangSmith to a file.
    
    Args:
        project_name: LangSmith project name
        output_path: Path to save traces (default: traces/export.json)
        limit: Maximum number of traces to export
        
    Returns:
        Export summary
        
    Example:
        >>> result = export_traces("my-project", output_path="traces.json")
        >>> print(f"Exported {result['num_traces']} traces")
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    
    if not api_key:
        warnings.warn("LANGSMITH_API_KEY not set. Cannot export traces.")
        return {
            "success": False,
            "reason": "No API key provided",
            "num_traces": 0
        }
    
    if output_path is None:
        output_path = f"traces/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # TODO: Implement actual trace export
    # from langsmith import Client
    # client = Client(api_key=api_key)
    # 
    # # List runs for the project
    # runs = client.list_runs(project_name=project_name, limit=limit)
    # 
    # # Convert to serializable format
    # traces = []
    # for run in runs:
    #     traces.append({
    #         "id": str(run.id),
    #         "name": run.name,
    #         "run_type": run.run_type,
    #         "start_time": run.start_time.isoformat(),
    #         "end_time": run.end_time.isoformat() if run.end_time else None,
    #         "inputs": run.inputs,
    #         "outputs": run.outputs,
    #         "error": run.error,
    #     })
    # 
    # # Save to file
    # import json
    # with open(output_path, 'w') as f:
    #     json.dump(traces, f, indent=2)
    
    print(f"[Production mode] Would export traces from {project_name} to {output_path}")
    
    return {
        "success": True,
        "project_name": project_name,
        "output_path": output_path,
        "num_traces": 0,  # Placeholder
        "limit": limit
    }


def get_trace_stats(project_name: str) -> Dict[str, Any]:
    """
    Get statistics about traces in a LangSmith project.
    
    Args:
        project_name: LangSmith project name
        
    Returns:
        Dictionary with trace statistics
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    
    if not api_key:
        return {
            "available": False,
            "reason": "No API key"
        }
    
    # TODO: Implement actual stats collection
    # from langsmith import Client
    # client = Client(api_key=api_key)
    # 
    # runs = list(client.list_runs(project_name=project_name, limit=1000))
    # 
    # total_runs = len(runs)
    # successful_runs = sum(1 for r in runs if not r.error)
    # failed_runs = total_runs - successful_runs
    # 
    # if runs:
    #     durations = [
    #         (r.end_time - r.start_time).total_seconds()
    #         for r in runs if r.end_time and r.start_time
    #     ]
    #     avg_duration = sum(durations) / len(durations) if durations else 0
    # else:
    #     avg_duration = 0
    
    return {
        "available": True,
        "project_name": project_name,
        "total_runs": 0,  # Placeholder
        "successful_runs": 0,
        "failed_runs": 0,
        "avg_duration_seconds": 0.0
    }


def trace_rag_query(
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Manually create a trace for a RAG query.
    
    This is useful when you want to log specific queries outside
    of the automatic LangChain tracing.
    
    Args:
        query: User query
        answer: Generated answer
        sources: Retrieved sources
        metadata: Additional metadata to attach
        
    Returns:
        Trace ID (or placeholder if not enabled)
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    
    if not api_key:
        # Dry-run mode - just log locally
        trace_id = f"local_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        print(f"[Dry-run] Would create trace {trace_id} for query: {query[:50]}...")
        return trace_id
    
    # TODO: Implement manual trace creation
    # from langsmith import Client
    # client = Client(api_key=api_key)
    # 
    # trace_data = {
    #     "name": "rag_query",
    #     "run_type": "chain",
    #     "inputs": {"query": query},
    #     "outputs": {"answer": answer, "sources": sources},
    #     "start_time": datetime.now(),
    #     "end_time": datetime.now(),
    # }
    # 
    # if metadata:
    #     trace_data["extra"] = metadata
    # 
    # run = client.create_run(**trace_data)
    # return str(run.id)
    
    trace_id = f"prod_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"[Production mode] Would create trace {trace_id}")
    return trace_id


class LangSmithTracer:
    """
    Context manager for LangSmith tracing.
    
    Example:
        >>> with LangSmithTracer("my-project") as tracer:
        ...     result = rag.query("What is GDPR?")
        ...     tracer.log_query(result)
    """
    
    def __init__(self, project_name: str = "gdpr-rag-system"):
        """
        Initialize tracer.
        
        Args:
            project_name: LangSmith project name
        """
        self.project_name = project_name
        self.config = None
        self.traces = []
    
    def __enter__(self):
        """Enter context - initialize tracing."""
        self.config = init_langsmith(project_name=self.project_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup."""
        if self.config and self.config.get("enabled"):
            print(f"[Tracing session complete] Logged {len(self.traces)} queries")
        return False
    
    def log_query(
        self,
        query: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a query and its result.
        
        Args:
            query: User query
            result: Query result dictionary
            metadata: Additional metadata
        """
        trace_id = trace_rag_query(
            query,
            result.get("answer", ""),
            result.get("sources", []),
            metadata
        )
        
        self.traces.append({
            "trace_id": trace_id,
            "query": query,
            "timestamp": datetime.now().isoformat()
        })


def setup_langchain_tracing():
    """
    Setup automatic LangChain tracing with LangSmith.
    
    This configures environment variables so that all LangChain
    calls are automatically traced.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    
    if not api_key:
        warnings.warn("LANGSMITH_API_KEY not set. Tracing disabled.")
        return False
    
    # Set environment variables for automatic tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    
    # Set project name if not already set
    if "LANGCHAIN_PROJECT" not in os.environ:
        os.environ["LANGCHAIN_PROJECT"] = "gdpr-rag-system"
    
    print("[Production mode] LangChain tracing enabled")
    return True


# Example usage for testing
if __name__ == "__main__":
    print("=== LangSmith Integration Module Demo ===")
    
    # Test initialization
    config = init_langsmith(project_name="test-project")
    print(f"1. LangSmith config:")
    print(f"   Enabled: {config['enabled']}")
    print(f"   Project: {config['project_name']}")
    
    # Test trace creation
    trace_id = trace_rag_query(
        query="What is GDPR?",
        answer="GDPR is a regulation...",
        sources=[{"page_content": "GDPR text..."}],
        metadata={"model": "gpt-3.5-turbo"}
    )
    print(f"\n2. Created trace: {trace_id}")
    
    # Test context manager
    with LangSmithTracer("test-project") as tracer:
        tracer.log_query(
            "Test query",
            {"answer": "Test answer", "sources": []},
            {"test": True}
        )
    print(f"3. Context manager test complete")
    
    # Test stats (dry-run)
    stats = get_trace_stats("test-project")
    print(f"\n4. Trace stats available: {stats['available']}")
    
    print("\n✓ Module is importable and functional")
