"""
Trainer Mode for ACE Securitization System.

Extracts knowledge from real securitization documents (Master Framework Agreements, etc.)
and enriches the playbook with strategies, definitions, templates, pitfalls, and code snippets.
"""

from trainer.document_parser import DocumentParser, ParsedDocument, ParsedClause
from trainer.knowledge_extractor import KnowledgeExtractor, ExtractedKnowledge
from trainer.knowledge_classifier import KnowledgeClassifier, ClassifiedKnowledge
from trainer.knowledge_validator import KnowledgeValidator, ValidationResult
from trainer.playbook_enricher import PlaybookEnricher, EnrichmentResult
from trainer.trainer_pipeline import TrainerPipeline, TrainerConfig, TrainerPipelineResult

__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "ParsedClause",
    "KnowledgeExtractor",
    "ExtractedKnowledge",
    "KnowledgeClassifier",
    "ClassifiedKnowledge",
    "KnowledgeValidator",
    "ValidationResult",
    "PlaybookEnricher",
    "EnrichmentResult",
    "TrainerPipeline",
    "TrainerConfig",
    "TrainerPipelineResult",
]

