"""
Granularity levels for document processing in Trainer Mode.

Defines how the document is chunked and processed by the LLM.
"""
from enum import Enum


class GranularityLevel(str, Enum):
    """
    Granularity level for document processing.
    
    Controls how the document is split and sent to the LLM for knowledge extraction.
    """
    
    # Process each operative clause individually (most granular, highest cost)
    OPERATIVE_CLAUSE_BY_CLAUSE = "operative_clause_by_clause"
    
    # Process clauses in batches (balanced approach)
    BATCH = "batch"
    
    # Process entire document at once (least granular, lowest cost)
    FULL_DOCUMENT = "full_document"
    
    @classmethod
    def get_description(cls, level: "GranularityLevel") -> str:
        """Get human-readable description of granularity level."""
        descriptions = {
            cls.OPERATIVE_CLAUSE_BY_CLAUSE: (
                "Process each operative clause individually. "
                "Most precise but highest API cost. "
                "Recommended for critical documents requiring maximum accuracy."
            ),
            cls.BATCH: (
                "Process clauses in batches (default: 15 per batch). "
                "Balanced approach with good accuracy and reasonable cost. "
                "Recommended for most use cases."
            ),
            cls.FULL_DOCUMENT: (
                "Process entire document in one LLM call. "
                "Fastest and cheapest but may miss fine-grained details. "
                "Recommended for quick overviews or small documents."
            )
        }
        return descriptions.get(level, "Unknown granularity level")
    
    @classmethod
    def get_cost_indicator(cls, level: "GranularityLevel") -> str:
        """Get cost indicator (💰 symbols)."""
        costs = {
            cls.OPERATIVE_CLAUSE_BY_CLAUSE: "💰💰💰",
            cls.BATCH: "💰💰",
            cls.FULL_DOCUMENT: "💰"
        }
        return costs.get(level, "")
    
    @classmethod
    def get_accuracy_indicator(cls, level: "GranularityLevel") -> str:
        """Get accuracy indicator (⭐ symbols)."""
        accuracy = {
            cls.OPERATIVE_CLAUSE_BY_CLAUSE: "⭐⭐⭐",
            cls.BATCH: "⭐⭐",
            cls.FULL_DOCUMENT: "⭐"
        }
        return accuracy.get(level, "")
