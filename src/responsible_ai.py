"""
Responsible AI Module

Implements bias detection, transparency, and ethical safeguards.
Works in dry-run mode without API keys.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
import warnings
from datetime import datetime


class ResponsibleAI:
    """
    Responsible AI features for RAG system.
    
    Features:
    - Bias detection in outputs
    - Transparency and explainability
    - Fairness metrics
    - Audit logging
    - Ethical guidelines enforcement
    
    Works without API keys using rule-based approaches.
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Initialize responsible AI manager.
        
        Args:
            enable_logging: Whether to enable audit logging
        """
        self.enable_logging = enable_logging
        self.audit_log: List[Dict[str, Any]] = []
        self.bias_keywords = self._load_bias_keywords()
        self.has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    
    def _load_bias_keywords(self) -> Dict[str, List[str]]:
        """
        Load keywords that may indicate bias.
        
        Returns:
            Dictionary of bias categories to keywords
        """
        return {
            "gender": ["he", "she", "his", "her", "male", "female"],
            "age": ["young", "old", "elderly", "youth"],
            "race": ["white", "black", "asian", "hispanic"],
            "disability": ["disabled", "handicapped", "impaired"],
            # Add more categories
        }
    
    def detect_bias(self, text: str) -> Dict[str, Any]:
        """
        Detect potential bias in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Bias detection results
            
        TODO: Implement ML-based bias detection for production
        """
        detected_biases = {}
        text_lower = text.lower()
        
        for category, keywords in self.bias_keywords.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                detected_biases[category] = found
        
        # Calculate bias score (0-1)
        bias_score = min(len(detected_biases) * 0.2, 1.0)
        
        return {
            "bias_score": bias_score,
            "detected_categories": list(detected_biases.keys()),
            "details": detected_biases,
            "recommendation": "Review output for potential bias" if bias_score > 0.3 else "No significant bias detected"
        }
    
    def assess_fairness(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess fairness across multiple responses.
        
        Args:
            responses: List of responses with metadata (e.g., user demographics)
            
        Returns:
            Fairness assessment
            
        TODO: Implement statistical fairness metrics
        """
        if not responses:
            return {"error": "No responses to assess"}
        
        # Placeholder fairness metrics
        return {
            "total_responses": len(responses),
            "bias_distribution": "Uniform (placeholder)",
            "fairness_score": 0.85,
            "recommendation": "System appears fair across user groups"
        }
    
    def explain_decision(self, query: str, response: str, context: List[str]) -> Dict[str, Any]:
        """
        Provide explanation for a generated response.
        
        Args:
            query: User query
            response: Generated response
            context: Context chunks used
            
        Returns:
            Explanation with transparency information
        """
        return {
            "query": query,
            "response_length": len(response),
            "context_sources": len(context),
            "context_preview": [c[:100] + "..." for c in context[:3]],
            "model": "gpt-3.5-turbo (or placeholder in dry-run)",
            "timestamp": datetime.now().isoformat(),
            "explanation": (
                f"This response was generated based on {len(context)} relevant GDPR passages. "
                "The system retrieved context using vector similarity and synthesized a response."
            )
        }
    
    def log_interaction(self, interaction: Dict[str, Any]) -> None:
        """
        Log an interaction for audit purposes.
        
        Args:
            interaction: Interaction details
        """
        if not self.enable_logging:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            **interaction
        }
        
        self.audit_log.append(log_entry)
    
    def get_audit_log(self, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries.
        
        Args:
            filter_by: Optional filters
            
        Returns:
            Filtered log entries
        """
        if not filter_by:
            return self.audit_log
        
        # Simple filtering
        filtered = self.audit_log
        for key, value in filter_by.items():
            filtered = [entry for entry in filtered if entry.get(key) == value]
        
        return filtered
    
    def check_ethical_compliance(self, query: str, response: str) -> Dict[str, Any]:
        """
        Check if interaction complies with ethical guidelines.
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Compliance check results
        """
        issues = []
        
        # Check for harmful content
        harmful_keywords = ["harm", "violence", "discriminate", "illegal"]
        for keyword in harmful_keywords:
            if keyword in response.lower():
                issues.append(f"Potentially harmful content: contains '{keyword}'")
        
        # Check for appropriate disclaimers
        if "gdpr" in query.lower() and "Note:" not in response and "disclaimer" not in response.lower():
            issues.append("Missing appropriate disclaimer for legal advice")
        
        # Check response length (very short responses might be low quality)
        if len(response) < 50:
            issues.append("Response may be too brief")
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "recommendation": "Address issues before delivering response" if issues else "Ethically compliant"
        }
    
    def generate_transparency_report(self) -> Dict[str, Any]:
        """
        Generate transparency report for the system.
        
        Returns:
            Transparency report
        """
        total_interactions = len(self.audit_log)
        
        if total_interactions == 0:
            return {"message": "No interactions logged yet"}
        
        # Analyze audit log
        bias_detected = sum(1 for entry in self.audit_log if entry.get("bias_score", 0) > 0.3)
        ethical_issues = sum(1 for entry in self.audit_log if not entry.get("compliant", True))
        
        return {
            "total_interactions": total_interactions,
            "bias_detection_rate": bias_detected / total_interactions,
            "ethical_compliance_rate": 1 - (ethical_issues / total_interactions),
            "system_mode": "dry-run" if not self.has_api_key else "production",
            "timestamp": datetime.now().isoformat(),
            "recommendations": [
                "Continue monitoring for bias",
                "Ensure proper disclaimers",
                "Regular ethical audits"
            ]
        }
    
    def apply_responsible_ai_pipeline(self, query: str, response: str, context: List[str]) -> Dict[str, Any]:
        """
        Apply full responsible AI pipeline to an interaction.
        
        Args:
            query: User query
            response: Generated response
            context: Context used
            
        Returns:
            Complete responsible AI assessment
        """
        # Detect bias
        bias_results = self.detect_bias(response)
        
        # Check ethical compliance
        ethics_results = self.check_ethical_compliance(query, response)
        
        # Generate explanation
        explanation = self.explain_decision(query, response, context)
        
        # Log interaction
        self.log_interaction({
            "query": query,
            "response": response,
            "bias_score": bias_results["bias_score"],
            "compliant": ethics_results["compliant"],
            "context_count": len(context)
        })
        
        return {
            "bias_assessment": bias_results,
            "ethical_compliance": ethics_results,
            "explanation": explanation,
            "overall_status": "safe" if ethics_results["compliant"] and bias_results["bias_score"] < 0.5 else "needs_review"
        }


class PrivacyProtector:
    """
    Privacy protection utilities.
    
    TODO: Integrate with guardrails module for comprehensive privacy
    """
    
    def __init__(self):
        self.processed_data_log = []
    
    def anonymize_data(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Anonymize sensitive data in text.
        
        Args:
            text: Text to anonymize
            
        Returns:
            (anonymized_text, anonymization_map) tuple
        """
        # Placeholder implementation
        # In production, integrate with PII detection from guardrails
        return text, {}
    
    def ensure_data_minimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure only necessary data is kept.
        
        Args:
            data: Data to minimize
            
        Returns:
            Minimized data
        """
        # Keep only essential fields
        essential_fields = ["query", "response", "timestamp"]
        return {k: v for k, v in data.items() if k in essential_fields}


def main():
    """Example usage of ResponsibleAI."""
    rai = ResponsibleAI(enable_logging=True)
    
    # Test bias detection
    text = "The system administrator should ensure that he configures the settings properly."
    bias_results = rai.detect_bias(text)
    print(f"Bias detection: {bias_results}")
    
    # Test ethical compliance
    query = "What are GDPR fines?"
    response = "GDPR fines can be substantial. This is AI-generated advice."
    ethics = rai.check_ethical_compliance(query, response)
    print(f"\nEthical compliance: {ethics}")
    
    # Full pipeline
    context = ["GDPR Article 83 discusses fines..."]
    assessment = rai.apply_responsible_ai_pipeline(query, response, context)
    print(f"\nFull assessment: {assessment['overall_status']}")
    
    # Generate report
    report = rai.generate_transparency_report()
    print(f"\nTransparency report: {report}")


if __name__ == "__main__":
    main()
