"""
Memory Management Module

Implements conversation memory and context management for multi-turn RAG.
Works in dry-run mode without API keys.
"""

from typing import List, Dict, Optional
from datetime import datetime


class MemoryManager:
    """
    Manages conversation history and context for RAG.
    
    Features:
    - Conversation history tracking
    - Context window management
    - Summary generation for long conversations
    - Memory persistence
    
    Works without API keys using in-memory storage.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the memory manager.
        
        Args:
            max_history: Maximum number of turns to keep in memory
        """
        self.max_history = max_history
        self.conversation_history: List[Dict[str, any]] = []
        self.session_id = None
        
    def add_interaction(self, query: str, response: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a query-response pair to conversation history.
        
        Args:
            query: User query
            response: System response
            metadata: Optional metadata (timestamps, scores, etc.)
        """
        interaction = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(interaction)
        
        # Trim history if it exceeds max_history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_context(self, num_turns: Optional[int] = None) -> str:
        """
        Get formatted conversation context for prompt.
        
        Args:
            num_turns: Number of recent turns to include (None = all)
            
        Returns:
            Formatted conversation string
        """
        if num_turns is None:
            turns_to_include = self.conversation_history
        else:
            turns_to_include = self.conversation_history[-num_turns:]
        
        if not turns_to_include:
            return ""
        
        context_parts = []
        for turn in turns_to_include:
            context_parts.append(f"User: {turn['query']}")
            context_parts.append(f"Assistant: {turn['response']}")
        
        return "\n".join(context_parts)
    
    def get_recent_queries(self, n: int = 3) -> List[str]:
        """
        Get the n most recent user queries.
        
        Args:
            n: Number of queries to retrieve
            
        Returns:
            List of recent queries
        """
        return [turn["query"] for turn in self.conversation_history[-n:]]
    
    def summarize_conversation(self) -> str:
        """
        Generate a summary of the conversation.
        
        Returns:
            Conversation summary
            
        TODO: Implement LLM-based summarization for production
        """
        if not self.conversation_history:
            return "No conversation yet."
        
        # Dry-run: simple heuristic summary
        num_turns = len(self.conversation_history)
        topics = set()
        
        for turn in self.conversation_history:
            # Extract simple keywords as "topics"
            words = turn["query"].lower().split()
            topics.update([w for w in words if len(w) > 5])
        
        topics_str = ", ".join(list(topics)[:5])
        
        return (
            f"Conversation with {num_turns} turns covering topics: {topics_str}. "
            f"Started at {self.conversation_history[0]['timestamp']}."
        )
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.conversation_history = []
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save conversation history to a file.
        
        Args:
            filepath: Path to save the conversation
            
        TODO: Implement proper serialization (JSON, pickle)
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"Conversation saved to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load conversation history from a file.
        
        Args:
            filepath: Path to load the conversation from
            
        TODO: Add error handling and validation
        """
        import json
        
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)
        
        print(f"Loaded {len(self.conversation_history)} turns from {filepath}")
    
    def get_contextual_query(self, current_query: str) -> str:
        """
        Enhance current query with conversation context.
        
        Args:
            current_query: Current user query
            
        Returns:
            Query enhanced with context
            
        TODO: Implement smarter context integration
        """
        if not self.conversation_history:
            return current_query
        
        # Simple approach: prepend recent context
        recent_context = self.get_conversation_context(num_turns=2)
        
        if recent_context:
            return f"Previous context:\n{recent_context}\n\nCurrent query: {current_query}"
        
        return current_query


class ConversationBuffer:
    """
    Simple buffer for managing conversation context within token limits.
    
    TODO: Implement token counting with tiktoken
    """
    
    def __init__(self, max_tokens: int = 2000):
        """
        Initialize conversation buffer.
        
        Args:
            max_tokens: Maximum tokens to keep in buffer
        """
        self.max_tokens = max_tokens
        self.buffer: List[str] = []
    
    def add_message(self, message: str) -> None:
        """
        Add a message to the buffer.
        
        Args:
            message: Message to add
            
        TODO: Implement actual token counting
        """
        self.buffer.append(message)
        
        # Placeholder: assume ~4 chars per token
        while self._estimate_tokens() > self.max_tokens:
            self.buffer.pop(0)  # Remove oldest message
    
    def _estimate_tokens(self) -> int:
        """Estimate token count (placeholder)."""
        total_chars = sum(len(msg) for msg in self.buffer)
        return total_chars // 4  # Rough estimate
    
    def get_buffer_content(self) -> str:
        """Get all buffer content as a string."""
        return "\n".join(self.buffer)


def main():
    """Example usage of MemoryManager."""
    memory = MemoryManager(max_history=5)
    
    # Simulate conversation
    memory.add_interaction(
        query="What is GDPR?",
        response="GDPR is the General Data Protection Regulation..."
    )
    
    memory.add_interaction(
        query="What are the key principles?",
        response="The key principles include lawfulness, fairness, transparency..."
    )
    
    # Get context
    context = memory.get_conversation_context()
    print("Conversation context:")
    print(context)
    
    # Summarize
    summary = memory.summarize_conversation()
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
