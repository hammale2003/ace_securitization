"""
Granularity levels for document processing.
"""
from enum import Enum


class GranularityLevel(Enum):
    """Processing granularity for document enrichment."""
    
    OPERATIVE_CLAUSE_BY_CLAUSE = "operative_clause_by_clause"
    FULL_DOCUMENT = "full_document"
    
    def __str__(self):
        return self.value

