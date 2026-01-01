"""
Playbook Enricher - 3-Agent Architecture for intelligent playbook enrichment.

Architecture:
1. EXTRACTOR (extractor_agent.py) - 4 specialized prompts per section, supports clause-by-clause or full-document
2. VALIDATOR (validator_agent.py) - Batch processes items, validates and enriches pointer definitions
3. CURATOR (curator_agent.py) - Deterministic application using redundancy.py logic

Main entry point: EnrichmentPipeline
"""
from playbook_enricher.document_parser import DocumentParser, ParsedDocument, ParsedClause
from playbook_enricher.extractor_agent import ExtractorAgent, ExtractedKnowledge
from playbook_enricher.validator_agent import ValidatorAgent, ValidatorOutput
from playbook_enricher.curator_agent import CuratorAgent, CuratorOperation, CuratorResult
from playbook_enricher.enrichment_pipeline import EnrichmentPipeline, EnrichmentConfig, EnrichmentResult
from playbook_enricher.granularity import GranularityLevel

__all__ = [
    "DocumentParser",
    "ParsedDocument", 
    "ParsedClause",
    "ExtractedKnowledge",
    "ExtractorAgent",
    "ValidatorAgent",
    "ValidatorOutput",
    "CuratorAgent",
    "CuratorOperation",
    "CuratorResult",
    "EnrichmentPipeline",
    "EnrichmentConfig",
    "EnrichmentResult",
    "GranularityLevel"
]

