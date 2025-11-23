"""
Guardrails Module

Implements input and output safety filters for the RAG system.
Detects adversarial prompts, unsafe content, and provides prompt rewriting.

Safe defaults: All functions work without external dependencies.
"""

import re
from typing import Dict, List, Tuple, Optional


# Adversarial prompt patterns to detect
ADVERSARIAL_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions?",
    r"disregard\s+(previous|all)\s+instructions?",
    r"forget\s+(previous|all)\s+instructions?",
    r"you\s+are\s+now",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if|a)",
    r"roleplay",
    r"new\s+instructions?:",
    r"system\s+prompt",
    r"reveal\s+your\s+(prompt|instructions?)",
    r"what\s+(are|is)\s+your\s+(instructions?|prompt)",
]

# Unsafe content patterns
UNSAFE_PATTERNS = [
    r"(how\s+to|ways\s+to)\s+(hack|exploit|breach|bypass)",
    r"generate\s+(malicious|harmful)",
    r"(credit\s+card|social\s+security)\s+number",
]


def detect_adversarial_prompt(prompt: str) -> bool:
    """
    Detect if a prompt contains adversarial patterns.
    
    Args:
        prompt: User input prompt
        
    Returns:
        True if adversarial patterns detected, False otherwise
    """
    prompt_lower = prompt.lower()
    
    for pattern in ADVERSARIAL_PATTERNS:
        if re.search(pattern, prompt_lower):
            print(f"[GUARDRAIL] Adversarial pattern detected: {pattern}")
            return True
    
    return False


def detect_unsafe_content(text: str) -> bool:
    """
    Detect if text contains unsafe content.
    
    Args:
        text: Text to check (input or output)
        
    Returns:
        True if unsafe content detected, False otherwise
    """
    text_lower = text.lower()
    
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, text_lower):
            print(f"[GUARDRAIL] Unsafe content detected: {pattern}")
            return True
    
    return False


def safe_rewrite(prompt: str) -> str:
    """
    Rewrite a potentially problematic prompt to be safer.
    
    Args:
        prompt: Original prompt
        
    Returns:
        Rewritten safe prompt
    """
    # Remove adversarial instruction attempts
    safe_prompt = prompt
    
    # Remove common adversarial phrases
    adversarial_phrases = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard previous instructions",
        "forget all instructions",
        "you are now",
        "pretend you are",
        "act as if",
    ]
    
    for phrase in adversarial_phrases:
        safe_prompt = re.sub(
            re.escape(phrase),
            "",
            safe_prompt,
            flags=re.IGNORECASE
        )
    
    # Clean up extra whitespace
    safe_prompt = re.sub(r"\s+", " ", safe_prompt).strip()
    
    if safe_prompt != prompt:
        print(f"[GUARDRAIL] Prompt rewritten for safety")
        print(f"  Original length: {len(prompt)}")
        print(f"  Rewritten length: {len(safe_prompt)}")
    
    return safe_prompt


def filter_output(output: str, max_length: int = 2000) -> str:
    """
    Filter and sanitize output from the LLM.
    
    Args:
        output: Generated output
        max_length: Maximum allowed output length
        
    Returns:
        Filtered output
    """
    # Truncate if too long
    if len(output) > max_length:
        output = output[:max_length] + "... [truncated]"
        print(f"[GUARDRAIL] Output truncated to {max_length} characters")
    
    # Check for unsafe content
    if detect_unsafe_content(output):
        print("[GUARDRAIL] Unsafe content detected in output, returning safe response")
        return "I apologize, but I cannot provide that information as it may be unsafe or inappropriate."
    
    return output


class GuardrailsChecker:
    """
    Comprehensive guardrails checker for input and output validation.
    """
    
    def __init__(
        self,
        check_adversarial: bool = True,
        check_unsafe: bool = True,
        auto_rewrite: bool = False,
        strict_mode: bool = False
    ):
        """
        Initialize guardrails checker.
        
        Args:
            check_adversarial: Enable adversarial prompt detection
            check_unsafe: Enable unsafe content detection
            auto_rewrite: Automatically rewrite problematic prompts
            strict_mode: Reject rather than rewrite in strict mode
        """
        self.check_adversarial = check_adversarial
        self.check_unsafe = check_unsafe
        self.auto_rewrite = auto_rewrite
        self.strict_mode = strict_mode
        self.violations = []
    
    def validate_input(self, prompt: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate user input against guardrails.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Tuple of (is_valid, processed_prompt, violation_reason)
        """
        violation = None
        
        # Check for adversarial patterns
        if self.check_adversarial and detect_adversarial_prompt(prompt):
            violation = "Adversarial prompt detected"
            
            if self.strict_mode:
                return False, prompt, violation
            elif self.auto_rewrite:
                prompt = safe_rewrite(prompt)
                print("[GUARDRAIL] Prompt automatically rewritten")
        
        # Check for unsafe content
        if self.check_unsafe and detect_unsafe_content(prompt):
            violation = "Unsafe content detected"
            
            if self.strict_mode or not self.auto_rewrite:
                return False, prompt, violation
        
        # Log violation if any
        if violation:
            self.violations.append({
                "type": "input",
                "reason": violation,
                "prompt": prompt
            })
        
        return True, prompt, violation
    
    def validate_output(self, output: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate system output against guardrails.
        
        Args:
            output: System generated output
            
        Returns:
            Tuple of (is_valid, processed_output, violation_reason)
        """
        violation = None
        
        # Check for unsafe content in output
        if self.check_unsafe and detect_unsafe_content(output):
            violation = "Unsafe content in output"
            
            # Always filter unsafe output
            output = "I apologize, but I cannot provide that information as it may be unsafe or inappropriate."
            
            self.violations.append({
                "type": "output",
                "reason": violation,
                "output": output
            })
            
            return False, output, violation
        
        # Apply output filtering
        output = filter_output(output)
        
        return True, output, violation
    
    def get_violations(self) -> List[Dict]:
        """Get all recorded violations."""
        return self.violations.copy()
    
    def clear_violations(self):
        """Clear violation history."""
        self.violations = []


def wrap_with_guardrails(rag_system, guardrails_config: Optional[Dict] = None):
    """
    Wrap a RAG system with guardrails.
    
    Args:
        rag_system: Base RAG system
        guardrails_config: Configuration for guardrails (optional)
        
    Returns:
        Wrapped RAG system with guardrails
    """
    config = guardrails_config or {}
    checker = GuardrailsChecker(
        check_adversarial=config.get("check_adversarial", True),
        check_unsafe=config.get("check_unsafe", True),
        auto_rewrite=config.get("auto_rewrite", True),
        strict_mode=config.get("strict_mode", False)
    )
    
    class GuardedRAG:
        """RAG system with guardrails."""
        
        def __init__(self, base_rag, checker):
            self.base_rag = base_rag
            self.checker = checker
        
        def query(self, query: str) -> str:
            """
            Query with input/output guardrails.
            
            Args:
                query: User query
                
            Returns:
                Safe answer or rejection message
            """
            # Validate input
            is_valid, safe_query, violation = self.checker.validate_input(query)
            
            if not is_valid:
                print(f"[GUARDRAIL] Query rejected: {violation}")
                return f"I cannot process this request. Reason: {violation}"
            
            # Get answer from base RAG
            answer = self.base_rag.query(safe_query)
            
            # Validate output
            is_valid, safe_answer, violation = self.checker.validate_output(answer)
            
            if not is_valid:
                print(f"[GUARDRAIL] Output filtered: {violation}")
            
            return safe_answer
        
        def get_violations(self) -> List[Dict]:
            """Get guardrail violations."""
            return self.checker.get_violations()
    
    return GuardedRAG(rag_system, checker)
