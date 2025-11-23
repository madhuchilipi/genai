"""
Guardrails Module

Implements input/output validation, PII detection, and content filtering.
Works in dry-run mode without API keys.
"""

import re
from typing import Dict, List, Tuple, Optional
import warnings


class GuardrailsManager:
    """
    Manages safety guardrails for RAG system.
    
    Features:
    - Input validation and sanitization
    - PII detection and redaction
    - Content filtering (harmful, inappropriate)
    - Output validation
    
    Works without API keys using rule-based approaches.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize guardrails manager.
        
        Args:
            strict_mode: If True, block more aggressively
        """
        self.strict_mode = strict_mode
        self.blocked_patterns = self._load_blocked_patterns()
        self.pii_patterns = self._compile_pii_patterns()
    
    def _load_blocked_patterns(self) -> List[str]:
        """
        Load patterns for harmful/inappropriate content.
        
        Returns:
            List of blocked patterns
            
        TODO: Load from configuration file in production
        """
        return [
            r'\bhack\b',
            r'\bexploit\b',
            r'\bmalware\b',
            # Add more patterns as needed
        ]
    
    def _compile_pii_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile regex patterns for PII detection.
        
        Returns:
            Dictionary of PII type to regex pattern
        """
        return {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            # Add more PII patterns
        }
    
    def validate_input(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user input for safety.
        
        Args:
            query: User query
            
        Returns:
            (is_valid, error_message) tuple
        """
        # Check length
        if len(query.strip()) == 0:
            return False, "Query cannot be empty"
        
        if len(query) > 5000:
            return False, "Query too long (max 5000 characters)"
        
        # Check for blocked patterns
        query_lower = query.lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, query_lower):
                return False, f"Query contains blocked content: {pattern}"
        
        # Check for excessive special characters (potential injection)
        special_char_ratio = sum(1 for c in query if not c.isalnum() and not c.isspace()) / len(query)
        if special_char_ratio > 0.3:
            warnings.warn("Query contains many special characters")
            if self.strict_mode:
                return False, "Query contains too many special characters"
        
        return True, None
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect personally identifiable information in text.
        
        Args:
            text: Text to scan
            
        Returns:
            Dictionary of PII type to list of detected instances
        """
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def redact_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            
        Returns:
            (redacted_text, redaction_counts) tuple
        """
        redacted_text = text
        redaction_counts = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(redacted_text)
            count = len(matches)
            
            if count > 0:
                redaction_counts[pii_type] = count
                # Replace with placeholder
                redacted_text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted_text)
        
        return redacted_text, redaction_counts
    
    def check_output_safety(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Check if generated output is safe to return.
        
        Args:
            response: Generated response
            
        Returns:
            (is_safe, warning_message) tuple
        """
        # Check for PII in output
        detected_pii = self.detect_pii(response)
        if detected_pii:
            return False, f"Output contains PII: {list(detected_pii.keys())}"
        
        # Check for blocked patterns
        response_lower = response.lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, response_lower):
                warnings.warn(f"Output contains potentially harmful content: {pattern}")
                if self.strict_mode:
                    return False, f"Output contains blocked content: {pattern}"
        
        # Check for disclaimer/citations
        # TODO: Implement citation checking
        
        return True, None
    
    def sanitize_query(self, query: str) -> str:
        """
        Sanitize user query (remove harmful content, normalize).
        
        Args:
            query: Raw user query
            
        Returns:
            Sanitized query
        """
        # Remove leading/trailing whitespace
        query = query.strip()
        
        # Normalize whitespace
        query = " ".join(query.split())
        
        # Remove potential SQL injection patterns
        query = re.sub(r'[;\'"\\]', '', query)
        
        # TODO: Add more sanitization rules
        
        return query
    
    def apply_guardrails(self, query: str) -> Tuple[bool, str, Optional[str]]:
        """
        Apply all input guardrails to a query.
        
        Args:
            query: User query
            
        Returns:
            (is_valid, sanitized_query, error_message) tuple
        """
        # Validate
        is_valid, error_msg = self.validate_input(query)
        if not is_valid:
            return False, query, error_msg
        
        # Sanitize
        sanitized = self.sanitize_query(query)
        
        # Check for PII
        detected_pii = self.detect_pii(sanitized)
        if detected_pii:
            warnings.warn(f"Query contains PII: {list(detected_pii.keys())}")
            # Optionally redact
            if self.strict_mode:
                sanitized, _ = self.redact_pii(sanitized)
        
        return True, sanitized, None
    
    def add_citation_disclaimer(self, response: str) -> str:
        """
        Add disclaimer about AI-generated content.
        
        Args:
            response: Generated response
            
        Returns:
            Response with disclaimer
        """
        disclaimer = (
            "\n\n[Note: This response is AI-generated based on GDPR documents. "
            "Please verify critical information with official sources.]"
        )
        return response + disclaimer


class ContentFilter:
    """
    Advanced content filtering using keyword matching and heuristics.
    
    TODO: Integrate with ML-based content moderation for production
    """
    
    def __init__(self):
        self.harmful_keywords = self._load_harmful_keywords()
    
    def _load_harmful_keywords(self) -> List[str]:
        """Load list of harmful keywords."""
        # Placeholder
        return ["violence", "hate", "discrimination"]
    
    def is_harmful(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text contains harmful content.
        
        Args:
            text: Text to check
            
        Returns:
            (is_harmful, detected_keywords) tuple
        """
        text_lower = text.lower()
        detected = [kw for kw in self.harmful_keywords if kw in text_lower]
        return len(detected) > 0, detected


def main():
    """Example usage of GuardrailsManager."""
    guardrails = GuardrailsManager(strict_mode=False)
    
    # Test input validation
    test_queries = [
        "What are GDPR data subject rights?",
        "My email is test@example.com, can you help?",
        "",
        "a" * 6000  # Too long
    ]
    
    for query in test_queries:
        is_valid, sanitized, error = guardrails.apply_guardrails(query)
        print(f"Query: {query[:50]}...")
        print(f"Valid: {is_valid}, Error: {error}")
        print()
    
    # Test PII detection
    text_with_pii = "Contact me at john@example.com or 555-123-4567"
    redacted, counts = guardrails.redact_pii(text_with_pii)
    print(f"Original: {text_with_pii}")
    print(f"Redacted: {redacted}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
