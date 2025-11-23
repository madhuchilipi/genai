"""
LangSmith Integration Module

Provides tracing, monitoring, and debugging capabilities using LangSmith.
Gracefully degrades when API key is not available.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings


class LangSmithTracer:
    """
    LangSmith integration for RAG system monitoring.
    
    Features:
    - Request/response tracing
    - Performance monitoring
    - Error tracking
    - Cost estimation
    
    Gracefully handles missing LANGSMITH_API_KEY.
    """
    
    def __init__(self, project_name: str = "project3-rag-gdpr"):
        """
        Initialize LangSmith tracer.
        
        Args:
            project_name: LangSmith project name
        """
        self.project_name = project_name
        self.has_langsmith_key = bool(os.getenv("LANGSMITH_API_KEY"))
        self.has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        self.traces: List[Dict[str, Any]] = []
        
        if not self.has_langsmith_key:
            warnings.warn(
                "LANGSMITH_API_KEY not found. Tracing disabled. "
                "Set LANGSMITH_API_KEY in .env for production monitoring."
            )
        
        # Configure LangSmith if available
        if self.has_langsmith_key:
            self._configure_langsmith()
    
    def _configure_langsmith(self) -> None:
        """
        Configure LangSmith environment.
        
        TODO: Import and configure LangSmith SDK
        """
        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = self.project_name
        
        # TODO: Additional LangSmith configuration
        # from langsmith import Client
        # self.client = Client()
        
        print(f"LangSmith tracing enabled for project: {self.project_name}")
    
    def trace_query(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Trace a RAG query-response pair.
        
        Args:
            query: User query
            response: System response
            metadata: Optional metadata (latency, tokens, etc.)
            
        Returns:
            Trace ID
            
        TODO: Implement actual LangSmith tracing
        """
        trace_id = f"trace_{len(self.traces)}_{datetime.now().timestamp()}"
        
        trace = {
            "trace_id": trace_id,
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name
        }
        
        self.traces.append(trace)
        
        if not self.has_langsmith_key:
            print(f"[Dry-run] Would trace to LangSmith: {trace_id}")
        else:
            # TODO: Send to LangSmith
            # self.client.create_run(...)
            pass
        
        return trace_id
    
    def trace_retrieval(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                       retrieval_time: float) -> str:
        """
        Trace retrieval step.
        
        Args:
            query: Search query
            retrieved_chunks: Retrieved document chunks
            retrieval_time: Time taken for retrieval
            
        Returns:
            Trace ID
        """
        metadata = {
            "step": "retrieval",
            "num_chunks": len(retrieved_chunks),
            "retrieval_time_ms": retrieval_time * 1000,
            "scores": [chunk.get("score", 0) for chunk in retrieved_chunks]
        }
        
        return self.trace_query(
            query=query,
            response=f"Retrieved {len(retrieved_chunks)} chunks",
            metadata=metadata
        )
    
    def trace_generation(self, query: str, context: str, response: str, 
                        generation_time: float, tokens_used: Optional[int] = None) -> str:
        """
        Trace generation step.
        
        Args:
            query: User query
            context: Context provided to LLM
            response: Generated response
            generation_time: Time taken for generation
            tokens_used: Tokens consumed
            
        Returns:
            Trace ID
        """
        metadata = {
            "step": "generation",
            "context_length": len(context),
            "response_length": len(response),
            "generation_time_ms": generation_time * 1000,
            "tokens_used": tokens_used or 0,
            "estimated_cost": self._estimate_cost(tokens_used) if tokens_used else 0
        }
        
        return self.trace_query(
            query=query,
            response=response,
            metadata=metadata
        )
    
    def _estimate_cost(self, tokens: int, model: str = "gpt-3.5-turbo") -> float:
        """
        Estimate API cost.
        
        Args:
            tokens: Number of tokens
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of late 2023 (approximate)
        pricing = {
            "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
            "gpt-4": 0.06 / 1000,  # $0.06 per 1K tokens
            "text-embedding-ada-002": 0.0001 / 1000  # $0.0001 per 1K tokens
        }
        
        rate = pricing.get(model, 0.002 / 1000)
        return tokens * rate
    
    def trace_agent_step(self, step_name: str, tool_used: str, 
                        input_data: Any, output_data: Any) -> str:
        """
        Trace an agent execution step.
        
        Args:
            step_name: Name of the step
            tool_used: Tool that was used
            input_data: Input to the tool
            output_data: Output from the tool
            
        Returns:
            Trace ID
        """
        metadata = {
            "step": "agent_execution",
            "step_name": step_name,
            "tool": tool_used,
            "input_preview": str(input_data)[:200],
            "output_preview": str(output_data)[:200]
        }
        
        return self.trace_query(
            query=f"Agent step: {step_name}",
            response=f"Tool {tool_used} executed",
            metadata=metadata
        )
    
    def get_traces(self, limit: Optional[int] = None, 
                  filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get traces from local storage.
        
        Args:
            limit: Maximum number of traces to return
            filter_by: Filter criteria
            
        Returns:
            List of traces
        """
        traces = self.traces
        
        if filter_by:
            for key, value in filter_by.items():
                traces = [t for t in traces if t.get(key) == value or 
                         t.get("metadata", {}).get(key) == value]
        
        if limit:
            traces = traces[-limit:]
        
        return traces
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from traces.
        
        Returns:
            Performance metrics
        """
        if not self.traces:
            return {"message": "No traces available"}
        
        retrieval_times = []
        generation_times = []
        total_tokens = 0
        
        for trace in self.traces:
            metadata = trace.get("metadata", {})
            
            if metadata.get("step") == "retrieval":
                retrieval_times.append(metadata.get("retrieval_time_ms", 0))
            elif metadata.get("step") == "generation":
                generation_times.append(metadata.get("generation_time_ms", 0))
                total_tokens += metadata.get("tokens_used", 0)
        
        return {
            "total_traces": len(self.traces),
            "avg_retrieval_time_ms": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
            "avg_generation_time_ms": sum(generation_times) / len(generation_times) if generation_times else 0,
            "total_tokens_used": total_tokens,
            "estimated_total_cost_usd": self._estimate_cost(total_tokens)
        }
    
    def export_traces(self, filepath: str) -> None:
        """
        Export traces to a file.
        
        Args:
            filepath: Path to export file
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.traces, f, indent=2)
        
        print(f"Exported {len(self.traces)} traces to {filepath}")
    
    def clear_traces(self) -> None:
        """Clear all local traces."""
        self.traces = []
        print("All traces cleared")


def configure_langsmith_for_langchain() -> None:
    """
    Configure LangSmith for automatic LangChain tracing.
    
    This should be called at the start of your application.
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        warnings.warn("LANGSMITH_API_KEY not set. LangChain tracing disabled.")
        return
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "project3-rag-gdpr")
    
    print("LangSmith tracing configured for LangChain")


def main():
    """Example usage of LangSmithTracer."""
    tracer = LangSmithTracer()
    
    # Trace a query (works without API keys)
    trace_id = tracer.trace_query(
        query="What are GDPR penalties?",
        response="GDPR penalties can reach up to 20 million euros...",
        metadata={"model": "gpt-3.5-turbo"}
    )
    print(f"Traced query with ID: {trace_id}")
    
    # Trace retrieval
    retrieved = [
        {"content": "Article 83...", "score": 0.95},
        {"content": "Article 84...", "score": 0.87}
    ]
    tracer.trace_retrieval("GDPR penalties", retrieved, retrieval_time=0.123)
    
    # Get metrics
    metrics = tracer.get_performance_metrics()
    print(f"\nPerformance metrics: {metrics}")


if __name__ == "__main__":
    main()
