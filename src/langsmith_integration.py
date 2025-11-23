"""
LangSmith Integration Module

Provides helpers for:
- Initializing LangSmith client
- Tracing LLM calls
- Exporting traces for analysis

Safe defaults: All functions handle missing API keys gracefully.
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime


def initialize_langsmith(
    api_key: Optional[str] = None,
    project_name: str = "gdpr-rag-project"
) -> Optional[Any]:
    """
    Initialize LangSmith client for tracing.
    
    Args:
        api_key: LangSmith API key (optional, uses env var if not provided)
        project_name: Project name for organizing traces
        
    Returns:
        LangSmith client or None if unavailable
        
    Note:
        Returns None without error when API key is not available.
    """
    api_key = api_key or os.getenv("LANGSMITH_API_KEY")
    
    if not api_key:
        print("[LANGSMITH] No API key provided, tracing disabled")
        print("[LANGSMITH] Set LANGSMITH_API_KEY environment variable to enable tracing")
        return None
    
    try:
        from langsmith import Client
        
        client = Client(api_key=api_key)
        
        # Set environment variables for LangChain integration
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGSMITH_API_KEY"] = api_key
        
        print(f"[LANGSMITH] Initialized client for project: {project_name}")
        print(f"[LANGSMITH] Tracing enabled - traces will be available at https://smith.langchain.com")
        
        return client
        
    except ImportError:
        print("[LANGSMITH] langsmith package not available")
        print("[LANGSMITH] Install with: pip install langsmith")
        return None
    
    except Exception as e:
        print(f"[LANGSMITH] Failed to initialize: {e}")
        return None


def enable_tracing(project_name: str = "gdpr-rag-project") -> bool:
    """
    Enable LangSmith tracing for the current session.
    
    Args:
        project_name: Project name for organizing traces
        
    Returns:
        True if tracing enabled, False otherwise
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    
    if not api_key:
        print("[LANGSMITH] Tracing disabled (no API key)")
        return False
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    
    print(f"[LANGSMITH] Tracing enabled for project: {project_name}")
    return True


def disable_tracing():
    """Disable LangSmith tracing."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("[LANGSMITH] Tracing disabled")


def export_traces(
    client,
    project_name: str,
    output_path: str,
    limit: int = 100
) -> bool:
    """
    Export traces from LangSmith to a local file.
    
    Args:
        client: LangSmith client
        project_name: Project name
        output_path: Path to save exported traces
        limit: Maximum number of traces to export
        
    Returns:
        True if successful, False otherwise
        
    TODO: Implement actual trace export via LangSmith API
    Note: This is a skeleton implementation
    """
    if not client:
        print("[LANGSMITH] No client available, cannot export traces")
        print("[LANGSMITH] Would export to:", output_path)
        return False
    
    try:
        print(f"[LANGSMITH] Exporting traces from project: {project_name}")
        print(f"[LANGSMITH] Limit: {limit} traces")
        print(f"[LANGSMITH] Output: {output_path}")
        
        # TODO: Implement actual trace fetching
        # runs = client.list_runs(project_name=project_name, limit=limit)
        
        # Placeholder: create mock export
        from pathlib import Path
        import json
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        mock_traces = {
            "project": project_name,
            "exported_at": datetime.now().isoformat(),
            "num_traces": 0,
            "traces": [],
            "note": "This is a placeholder. Actual traces require valid LangSmith API key and runs."
        }
        
        with open(output_path, 'w') as f:
            json.dump(mock_traces, f, indent=2)
        
        print(f"[LANGSMITH] Mock export saved to {output_path}")
        print("[LANGSMITH] To export actual traces, provide valid API key and run queries")
        
        return True
        
    except Exception as e:
        print(f"[LANGSMITH] Export failed: {e}")
        return False


def get_trace_url(run_id: str, project_name: str = "gdpr-rag-project") -> str:
    """
    Get the URL to view a specific trace in LangSmith.
    
    Args:
        run_id: Run ID from LangChain
        project_name: Project name
        
    Returns:
        URL to trace
    """
    # LangSmith trace URL format
    base_url = "https://smith.langchain.com"
    url = f"{base_url}/public/{project_name}/r/{run_id}"
    
    return url


class TracingContext:
    """
    Context manager for LangSmith tracing.
    
    Usage:
        with TracingContext("my-project"):
            # Your RAG code here
            answer = rag.query("What is GDPR?")
        # Tracing automatically disabled after context
    """
    
    def __init__(self, project_name: str = "gdpr-rag-project", enabled: bool = True):
        """
        Initialize tracing context.
        
        Args:
            project_name: Project name for traces
            enabled: Whether to enable tracing
        """
        self.project_name = project_name
        self.enabled = enabled
        self.was_enabled = False
    
    def __enter__(self):
        """Enable tracing on context entry."""
        if self.enabled:
            self.was_enabled = os.getenv("LANGCHAIN_TRACING_V2") == "true"
            enable_tracing(self.project_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore tracing state on context exit."""
        if self.enabled and not self.was_enabled:
            disable_tracing()


def log_evaluation_metrics(
    client,
    run_id: str,
    metrics: Dict[str, Any]
) -> bool:
    """
    Log evaluation metrics for a specific run.
    
    Args:
        client: LangSmith client
        run_id: Run ID to attach metrics to
        metrics: Dictionary of metrics
        
    Returns:
        True if successful, False otherwise
        
    TODO: Implement actual metric logging via LangSmith API
    """
    if not client:
        print("[LANGSMITH] No client available, cannot log metrics")
        print(f"[LANGSMITH] Would log metrics for run {run_id}:", metrics)
        return False
    
    try:
        print(f"[LANGSMITH] Logging metrics for run: {run_id}")
        print(f"[LANGSMITH] Metrics: {metrics}")
        
        # TODO: Implement actual metric logging
        # client.update_run(run_id, feedback=metrics)
        
        print("[LANGSMITH] Mock metric logging complete")
        return True
        
    except Exception as e:
        print(f"[LANGSMITH] Failed to log metrics: {e}")
        return False


def create_dataset(
    client,
    dataset_name: str,
    examples: List[Dict[str, Any]],
    description: str = ""
) -> Optional[str]:
    """
    Create a dataset in LangSmith for evaluation.
    
    Args:
        client: LangSmith client
        dataset_name: Name for the dataset
        examples: List of example inputs/outputs
        description: Dataset description
        
    Returns:
        Dataset ID if successful, None otherwise
        
    TODO: Implement actual dataset creation via LangSmith API
    """
    if not client:
        print("[LANGSMITH] No client available, cannot create dataset")
        print(f"[LANGSMITH] Would create dataset: {dataset_name}")
        print(f"[LANGSMITH] Examples: {len(examples)}")
        return None
    
    try:
        print(f"[LANGSMITH] Creating dataset: {dataset_name}")
        print(f"[LANGSMITH] Examples: {len(examples)}")
        print(f"[LANGSMITH] Description: {description}")
        
        # TODO: Implement actual dataset creation
        # dataset = client.create_dataset(
        #     dataset_name=dataset_name,
        #     description=description
        # )
        # for example in examples:
        #     client.create_example(
        #         dataset_id=dataset.id,
        #         inputs=example.get("inputs", {}),
        #         outputs=example.get("outputs", {})
        #     )
        
        mock_dataset_id = f"mock-dataset-{dataset_name}"
        print(f"[LANGSMITH] Mock dataset created: {mock_dataset_id}")
        
        return mock_dataset_id
        
    except Exception as e:
        print(f"[LANGSMITH] Failed to create dataset: {e}")
        return None


def get_client_info() -> Dict[str, Any]:
    """
    Get information about LangSmith setup.
    
    Returns:
        Dictionary with setup information
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2") == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    
    info = {
        "api_key_set": bool(api_key),
        "tracing_enabled": tracing_enabled,
        "project_name": project_name,
        "langsmith_installed": False,
        "langsmith_url": "https://smith.langchain.com"
    }
    
    try:
        import langsmith
        info["langsmith_installed"] = True
        info["langsmith_version"] = getattr(langsmith, "__version__", "unknown")
    except ImportError:
        pass
    
    return info


def print_tracing_status():
    """Print current LangSmith tracing status."""
    info = get_client_info()
    
    print("\n" + "=" * 60)
    print("LANGSMITH TRACING STATUS")
    print("=" * 60)
    print(f"API Key Set: {info['api_key_set']}")
    print(f"Tracing Enabled: {info['tracing_enabled']}")
    print(f"Project Name: {info['project_name']}")
    print(f"LangSmith Installed: {info['langsmith_installed']}")
    if info['langsmith_installed']:
        print(f"LangSmith Version: {info.get('langsmith_version', 'unknown')}")
    print(f"\nView traces at: {info['langsmith_url']}")
    print("=" * 60 + "\n")
