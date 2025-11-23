"""
Memory Integration Module

Provides LangGraph-powered conversational memory for chat-style RAG sessions.
Enables the system to maintain context across multiple turns.

Safe defaults: Returns mock memory objects when dependencies are not available.
"""

import os
from typing import List, Dict, Optional, Any


class ConversationMemory:
    """
    Simple conversation memory for storing chat history.
    
    Tracks messages and provides context for multi-turn conversations.
    """
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """
        Add a message to memory.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
        self.messages.append({"role": role, "content": content})
        
        # Trim to max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in memory."""
        return self.messages.copy()
    
    def clear(self):
        """Clear all messages from memory."""
        self.messages = []
    
    def get_context_string(self) -> str:
        """
        Get formatted context string from conversation history.
        
        Returns:
            Formatted string of conversation history
        """
        if not self.messages:
            return ""
        
        context_parts = []
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)


def create_memory_enabled_rag(base_rag, memory: Optional[ConversationMemory] = None):
    """
    Wrap a RAG system with conversational memory.
    
    Args:
        base_rag: Base RAG instance (e.g., BaselineRAG)
        memory: ConversationMemory instance (creates new if None)
        
    Returns:
        Memory-enabled RAG wrapper
        
    TODO: Integrate with LangGraph for more sophisticated state management
    """
    if memory is None:
        memory = ConversationMemory()
    
    class MemoryRAG:
        """RAG with conversational memory."""
        
        def __init__(self, base_rag, memory):
            self.base_rag = base_rag
            self.memory = memory
        
        def query(self, query: str, use_history: bool = True) -> str:
            """
            Query with conversation history.
            
            Args:
                query: User query
                use_history: Whether to include conversation history
                
            Returns:
                Generated answer
            """
            # Add user message to memory
            self.memory.add_message("user", query)
            
            # Optionally include history in query
            if use_history and len(self.memory.messages) > 1:
                context = self.memory.get_context_string()
                enhanced_query = f"Conversation history:\n{context}\n\nCurrent question: {query}"
                print(f"[INFO] Using conversation history ({len(self.memory.messages)} messages)")
            else:
                enhanced_query = query
            
            # Get answer from base RAG
            answer = self.base_rag.query(enhanced_query)
            
            # Add assistant response to memory
            self.memory.add_message("assistant", answer)
            
            return answer
        
        def clear_history(self):
            """Clear conversation history."""
            self.memory.clear()
            print("[INFO] Conversation history cleared")
        
        def get_history(self) -> List[Dict[str, str]]:
            """Get conversation history."""
            return self.memory.get_messages()
    
    return MemoryRAG(base_rag, memory)


def create_langgraph_memory(checkpointer_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a LangGraph memory configuration with checkpointing.
    
    Args:
        checkpointer_path: Path for persisting checkpoints (optional)
        
    Returns:
        Memory configuration dictionary
        
    TODO: Implement actual LangGraph MemorySaver and checkpointing
    Note: This is a skeleton - full implementation requires LangGraph setup
    """
    print("[INFO] Creating LangGraph memory configuration")
    
    try:
        # Try to use LangGraph if available
        from langgraph.checkpoint import MemorySaver
        
        if checkpointer_path:
            print(f"[INFO] Would persist checkpoints to {checkpointer_path}")
            # TODO: Implement file-based checkpointing
        
        memory = MemorySaver()
        print("[INFO] LangGraph memory created successfully")
        
        return {
            "checkpointer": memory,
            "config": {"configurable": {"thread_id": "default"}}
        }
        
    except ImportError:
        print("[DRY-RUN] LangGraph not available, using mock memory")
        return {
            "checkpointer": None,
            "config": {"configurable": {"thread_id": "default"}}
        }


def attach_memory_to_agent(agent, memory_config: Dict[str, Any]):
    """
    Attach memory configuration to a LangGraph agent.
    
    Args:
        agent: LangGraph agent/graph
        memory_config: Memory configuration from create_langgraph_memory()
        
    Returns:
        Agent with memory attached
        
    TODO: Implement proper LangGraph agent memory integration
    """
    print("[INFO] Attaching memory to agent")
    
    if memory_config.get("checkpointer"):
        print("[INFO] Memory checkpointing enabled")
    else:
        print("[DRY-RUN] Mock memory attachment")
    
    # In production, this would configure the agent's graph with the checkpointer
    # For now, return the agent as-is
    return agent


class ChatSession:
    """
    High-level chat session manager with memory.
    
    Manages a conversation session with automatic memory management.
    """
    
    def __init__(self, rag_system, session_id: str = "default"):
        """
        Initialize chat session.
        
        Args:
            rag_system: RAG system to use
            session_id: Unique session identifier
        """
        self.rag_system = rag_system
        self.session_id = session_id
        self.memory = ConversationMemory()
        self.turn_count = 0
    
    def chat(self, message: str) -> str:
        """
        Send a message and get response.
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        self.turn_count += 1
        print(f"\n[TURN {self.turn_count}] Session: {self.session_id}")
        
        # Add to memory
        self.memory.add_message("user", message)
        
        # Get response (with history if available)
        if hasattr(self.rag_system, 'query'):
            response = self.rag_system.query(message)
        else:
            response = "RAG system not properly initialized"
        
        # Add response to memory
        self.memory.add_message("assistant", response)
        
        return response
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.memory.get_messages()
    
    def reset(self):
        """Reset the conversation."""
        self.memory.clear()
        self.turn_count = 0
        print(f"[INFO] Session {self.session_id} reset")
