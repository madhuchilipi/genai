"""
Memory Integration Module

This module provides LangGraph memory integration for chat-style sessions.
Enables maintaining conversation history across multiple interactions.
"""

import os
import warnings
from typing import List, Dict, Any, Optional


class ConversationMemory:
    """
    Simple conversation memory to track chat history.
    
    This is a lightweight implementation that stores messages in memory.
    For production, consider using LangGraph's built-in memory or persistent storage.
    
    Example:
        >>> memory = ConversationMemory()
        >>> memory.add_message("user", "What is GDPR?")
        >>> memory.add_message("assistant", "GDPR is...")
        >>> history = memory.get_history()
    """
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        self.messages.append({"role": role, "content": content})
        
        # Keep only the last max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Args:
            last_n: Number of recent messages to return (None for all)
            
        Returns:
            List of message dictionaries
        """
        if last_n is None:
            return self.messages.copy()
        return self.messages[-last_n:]
    
    def clear(self):
        """Clear all conversation history."""
        self.messages = []
    
    def format_for_prompt(self, last_n: int = 5) -> str:
        """
        Format conversation history for inclusion in a prompt.
        
        Args:
            last_n: Number of recent messages to include
            
        Returns:
            Formatted conversation history string
        """
        history = self.get_history(last_n)
        if not history:
            return "No previous conversation."
        
        formatted = []
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)


def create_memory_agent(
    rag_instance: Any,
    memory: Optional[ConversationMemory] = None
) -> "MemoryRAGAgent":
    """
    Create a RAG agent with memory integration.
    
    Args:
        rag_instance: BaselineRAG or compatible RAG instance
        memory: ConversationMemory instance (creates new if None)
        
    Returns:
        MemoryRAGAgent instance with conversation tracking
    """
    if memory is None:
        memory = ConversationMemory()
    
    return MemoryRAGAgent(rag_instance, memory)


class MemoryRAGAgent:
    """
    RAG agent with conversation memory integration.
    
    This wraps a BaselineRAG instance and adds conversation tracking,
    allowing for context-aware responses across multiple turns.
    
    Example:
        >>> from src.rag_baseline import BaselineRAG
        >>> rag = BaselineRAG("faiss_index")
        >>> agent = MemoryRAGAgent(rag)
        >>> response1 = agent.chat("What is GDPR?")
        >>> response2 = agent.chat("Can you elaborate on that?")  # Uses context
    """
    
    def __init__(
        self,
        rag_instance: Any,
        memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize memory-enabled RAG agent.
        
        Args:
            rag_instance: BaselineRAG or compatible instance
            memory: ConversationMemory instance
        """
        self.rag = rag_instance
        self.memory = memory or ConversationMemory()
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a chat message with conversation context.
        
        Args:
            user_message: User's message
            
        Returns:
            Dictionary with answer and conversation metadata
        """
        # Add user message to memory
        self.memory.add_message("user", user_message)
        
        # Get conversation context
        context = self.memory.format_for_prompt(last_n=5)
        
        # Enhance query with conversation context for better retrieval
        # TODO: Implement query reformulation based on conversation history
        enhanced_query = self._enhance_query_with_context(user_message, context)
        
        # Get answer from RAG
        result = self.rag.query(enhanced_query)
        answer = result["answer"]
        
        # Add assistant response to memory
        self.memory.add_message("assistant", answer)
        
        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "conversation_turn": len(self.memory.messages) // 2,
            "context_used": context
        }
    
    def _enhance_query_with_context(self, query: str, context: str) -> str:
        """
        Enhance the current query with conversation context.
        
        This is a simple implementation. In production, you might use:
        - LLM-based query reformulation
        - Coreference resolution
        - Intent classification
        
        Args:
            query: Current user query
            context: Formatted conversation history
            
        Returns:
            Enhanced query string
        """
        # Simple heuristic: if query is short or uses pronouns, append context
        pronouns = ["it", "that", "this", "they", "them", "these", "those"]
        
        if len(query.split()) < 5 or any(p in query.lower() for p in pronouns):
            return f"{query}\n\nPrevious conversation context:\n{context}"
        
        return query
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the current conversation.
        
        Returns:
            Summary string
        """
        history = self.memory.get_history()
        if not history:
            return "No conversation yet."
        
        num_turns = len([m for m in history if m["role"] == "user"])
        return f"Conversation with {num_turns} turns, {len(history)} total messages."
    
    def reset(self):
        """Reset the conversation memory."""
        self.memory.clear()


def attach_memory_to_langgraph(graph: Any, memory: ConversationMemory) -> Any:
    """
    Attach memory to a LangGraph agent.
    
    This is a skeleton implementation. In production, use LangGraph's
    built-in memory management capabilities.
    
    Args:
        graph: LangGraph graph instance
        memory: ConversationMemory instance
        
    Returns:
        Modified graph with memory attached
    """
    # TODO: Implement LangGraph memory integration
    # from langgraph.prebuilt import MemorySaver
    # memory_saver = MemorySaver()
    # graph.attach_memory(memory_saver)
    
    warnings.warn("LangGraph memory integration is a skeleton. Implement for production.")
    print("[TODO] Attach memory to LangGraph - see langgraph.prebuilt.MemorySaver")
    
    return graph


# Example usage for testing
if __name__ == "__main__":
    print("=== Memory Integration Module Demo ===")
    
    # Create memory instance
    memory = ConversationMemory(max_messages=10)
    print("1. Created ConversationMemory")
    
    # Simulate conversation
    memory.add_message("user", "What is GDPR?")
    memory.add_message("assistant", "GDPR is the General Data Protection Regulation.")
    memory.add_message("user", "What are the key principles?")
    memory.add_message("assistant", "The key principles include transparency, purpose limitation...")
    
    print(f"2. Conversation has {len(memory.messages)} messages")
    
    # Get formatted history
    formatted = memory.format_for_prompt(last_n=4)
    print(f"3. Formatted history:\n{formatted[:100]}...")
    
    # Test with RAG agent (dry-run)
    try:
        from src.rag_baseline import BaselineRAG
        rag = BaselineRAG("faiss_index")
        agent = MemoryRAGAgent(rag)
        
        response = agent.chat("What is the right to access?")
        print(f"4. Agent response: {response['answer'][:100]}...")
        print(f"   Conversation turn: {response['conversation_turn']}")
        
    except Exception as e:
        print(f"4. Could not test with RAG (expected in isolation): {e}")
    
    print("\n✓ Module is importable and functional (dry-run mode)")
