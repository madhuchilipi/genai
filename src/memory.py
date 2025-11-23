"""
Memory integration module using LangGraph.

Provides conversational memory for multi-turn RAG interactions.
"""

from typing import Dict, Any, List, Optional
import warnings


class ConversationMemory:
    """
    Simple conversation memory for chat-based RAG.
    
    Stores conversation history and provides context for multi-turn interactions.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of turns to keep in memory
        """
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        print(f"[Memory] Initialized with max_history={max_history}")
    
    def add_turn(self, user_message: str, assistant_message: str):
        """
        Add a conversation turn to memory.
        
        Args:
            user_message: User's message
            assistant_message: Assistant's response
        """
        self.history.append({
            "role": "user",
            "content": user_message
        })
        self.history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Trim to max_history
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get full conversation history."""
        return self.history.copy()
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        print("[Memory] History cleared")
    
    def format_for_prompt(self) -> str:
        """
        Format history for inclusion in prompt.
        
        Returns:
            Formatted conversation history string
        """
        if not self.history:
            return ""
        
        formatted = "Previous conversation:\n"
        for msg in self.history:
            role = msg["role"].capitalize()
            formatted += f"{role}: {msg['content']}\n"
        return formatted


def create_memory_agent(memory: Optional[ConversationMemory] = None) -> Dict[str, Any]:
    """
    Create a LangGraph agent with memory integration.
    
    Args:
        memory: Optional conversation memory instance
        
    Returns:
        Agent configuration dictionary
        
    Note:
        This is a skeleton implementation. Full LangGraph integration requires:
        TODO: Implement using langgraph.StateGraph and langgraph.Memory
        - from langgraph.graph import StateGraph
        - from langgraph.checkpoint import MemorySaver
    """
    if memory is None:
        memory = ConversationMemory()
    
    print("[DRY-RUN] Creating LangGraph agent with memory")
    print("TODO: Implement full LangGraph StateGraph with checkpoint")
    
    # Placeholder agent configuration
    agent_config = {
        "memory": memory,
        "type": "rag_with_memory",
        "checkpoint": "memory_saver"
    }
    
    warnings.warn("Full LangGraph agent implementation pending")
    return agent_config


def attach_memory_to_chain(chain: Any, memory: ConversationMemory) -> Any:
    """
    Attach memory to an existing LangChain chain.
    
    Args:
        chain: LangChain chain or runnable
        memory: Conversation memory instance
        
    Returns:
        Chain with memory attached
        
    Note:
        TODO: Implement using LangChain's ConversationBufferMemory or similar
        - from langchain.memory import ConversationBufferMemory
        - memory_wrapper = ConversationBufferMemory()
        - chain.memory = memory_wrapper
    """
    print("[DRY-RUN] Attaching memory to chain")
    print(f"Memory has {len(memory.history)} messages")
    
    warnings.warn("Memory attachment implementation pending")
    return chain


class MemoryEnabledRAG:
    """
    RAG system with conversation memory.
    
    Extends baseline RAG with multi-turn conversation support.
    """
    
    def __init__(self, base_rag, memory: Optional[ConversationMemory] = None):
        """
        Initialize memory-enabled RAG.
        
        Args:
            base_rag: Baseline RAG instance
            memory: Optional conversation memory
        """
        self.base_rag = base_rag
        self.memory = memory or ConversationMemory()
        print("[MemoryRAG] Initialized with conversation memory")
    
    def query(self, question: str, use_history: bool = True) -> Dict[str, Any]:
        """
        Query with conversation context.
        
        Args:
            question: User question
            use_history: Whether to include conversation history
            
        Returns:
            Response dictionary with answer and sources
        """
        # Add history context to question if enabled
        context_question = question
        if use_history and self.memory.history:
            history_str = self.memory.format_for_prompt()
            context_question = f"{history_str}\nCurrent question: {question}"
            print(f"[MemoryRAG] Using {len(self.memory.history)} previous messages")
        
        # Query base RAG
        result = self.base_rag.query(context_question)
        
        # Store in memory
        self.memory.add_turn(question, result['answer'])
        
        return result
    
    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()


def run_example():
    """Run example with memory."""
    print("=== Memory Integration Example ===\n")
    
    # Create memory
    memory = ConversationMemory(max_history=5)
    
    # Simulate conversation
    memory.add_turn(
        "What is GDPR?",
        "GDPR is the General Data Protection Regulation..."
    )
    memory.add_turn(
        "When did it come into effect?",
        "It came into effect on May 25, 2018."
    )
    
    print("Conversation history:")
    print(memory.format_for_prompt())
    
    print("\nHistory length:", len(memory.history), "messages")


if __name__ == "__main__":
    run_example()
