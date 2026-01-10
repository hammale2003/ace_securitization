"""
Enrichment Pipeline - 3-Agent Architecture for intelligent playbook enrichment.

Uses Extractor, Validator, and Curator agents to process documents.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import LLMConfig
from llm_client import create_client, LLMClient
from playbook import PlaybookManager, Playbook
from retriever import PlaybookRetriever
from playbook_enricher.document_parser import DocumentParser, ParsedDocument
from playbook_enricher.extractor_agent import ExtractorAgent, ExtractedKnowledge
from playbook_enricher.validator_agent import ValidatorAgent
from playbook_enricher.curator_agent import CuratorAgent
from playbook_enricher.granularity import GranularityLevel
from utils import logger


@dataclass
class EnrichmentConfig:
    """Configuration for enrichment pipeline."""
    granularity_level: GranularityLevel = GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE
    llm_config: Optional[LLMConfig] = None
    extraction_sections: List[str] = field(default_factory=lambda: ["strategies", "pitfalls", "definitions"])
    # Redundancy control: prevent duplicates, and upgrade existing bullets when better content arrives
    dedupe_enabled: bool = True
    dedupe_similarity_threshold: float = 0.86
    upgrade_similarity_threshold: float = 0.78
    upgrade_margin: float = 0.08
    # Batch processing: process multiple items together to reduce LLM calls
    validator_batch_size: int = 5  # Batch size for Validator agent


@dataclass
class EnrichmentResult:
    """Result from enrichment pipeline."""
    document_title: str
    document_type: str
    total_extracted: int
    total_processed: int
    total_added: int
    total_skipped: int
    added_bullet_ids: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_title": self.document_title,
            "document_type": self.document_type,
            "total_extracted": self.total_extracted,
            "total_processed": self.total_processed,
            "total_added": self.total_added,
            "total_skipped": self.total_skipped,
            "added_bullet_ids": self.added_bullet_ids,
            "timestamp": self.timestamp
        }
    
    def get_summary(self) -> str:
        lines = [
            "Playbook Enrichment Complete",
            "",
            f"Document: {self.document_title}",
            f"Type: {self.document_type}",
            "",
            "Results:",
            f"  Extracted: {self.total_extracted} items",
            f"  Processed: {self.total_processed} items",
            f"  Added: {self.total_added} bullets",
            f"  Skipped: {self.total_skipped} items (duplicates/low quality)",
            ""
        ]
        return "\n".join(lines)


class EnrichmentPipeline:
    """
    Main enrichment pipeline using 3-agent architecture.
    
    Flow:
    1. EXTRACTOR: Extract knowledge using 4 specialized prompts (one per section)
    2. VALIDATOR: Batch process items, validate (is_reusable, is_correct, is_duplicate), enrich pointer definitions
    3. CURATOR: Deterministically apply recommendations using redundancy.py logic
    """
    
    def __init__(
        self,
        config: EnrichmentConfig,
        playbook_manager: PlaybookManager,
        retriever: Optional[PlaybookRetriever] = None
    ):
        self.config = config
        self.playbook_manager = playbook_manager
        self.retriever = retriever
        
        llm_config = config.llm_config or LLMConfig()
        self.llm_client = create_client(llm_config)
        
        self.document_parser = DocumentParser()
        
        # 3-agent architecture
        self.extractor = ExtractorAgent(self.llm_client)
        self.validator = ValidatorAgent(self.llm_client, self.retriever)
        self.curator = CuratorAgent(
            dedupe_enabled=config.dedupe_enabled,
            dedupe_similarity_threshold=config.dedupe_similarity_threshold,
            upgrade_similarity_threshold=config.upgrade_similarity_threshold,
            upgrade_margin=config.upgrade_margin,
            retriever=self.retriever
        )
    
    def run_from_file(self, file_path: str) -> EnrichmentResult:
        """Run enrichment from a document file."""
        document = self.document_parser.parse_json_file(file_path)
        return self.run_from_document(document)
    
    def run_from_json_string(self, json_string: str) -> EnrichmentResult:
        """Run enrichment from a JSON string."""
        document = self.document_parser.parse_json_string(json_string)
        return self.run_from_document(document)
    
    def run_from_document(self, document: ParsedDocument) -> EnrichmentResult:
        """Run enrichment from a parsed document using 3-agent architecture."""
        logger.info(f"Starting enrichment for: {document.title}")
        
        playbook = self.playbook_manager.get_playbook()
        
        # Step 1: EXTRACTOR - Extract knowledge using specialized prompts
        logger.info(f"Step 1: EXTRACTOR - Extracting knowledge (granularity: {self.config.granularity_level.value})")
        logger.info(f"Target sections: {', '.join(self.config.extraction_sections)}")
        extracted_items = self.extractor.extract_from_document(
            document=document,
            granularity_level=self.config.granularity_level,
            allowed_sections=self.config.extraction_sections,
            playbook=playbook
        )
        
        # Filter: Keep only items that match selected sections (double-check)
        extracted_items = [item for item in extracted_items if item.section in self.config.extraction_sections]
        
        # Log breakdown by section
        section_counts = {}
        for item in extracted_items:
            section_counts[item.section] = section_counts.get(item.section, 0) + 1
        
        logger.info(f"Extracted {len(extracted_items)} knowledge items")
        if section_counts:
            logger.info(f"Section breakdown: {section_counts}")
        
        # Verify all items are in allowed sections
        for item in extracted_items:
            if item.section not in self.config.extraction_sections:
                logger.error(f"ERROR: Item with section '{item.section}' not in allowed sections {self.config.extraction_sections}")
        
        if not extracted_items:
            logger.warning("No knowledge extracted from document")
            return EnrichmentResult(
                document_title=document.title,
                document_type=document.document_type,
                total_extracted=0,
                total_processed=0,
                total_added=0,
                total_skipped=0,
                added_bullet_ids=[]
            )
        
        # Step 2: VALIDATOR - Batch process and validate items
        logger.info("Step 2: VALIDATOR - Batch processing and validating items")
        logger.info(f"Validator batch size: {self.config.validator_batch_size} items per batch")
        
        # Prepare items for validation
        validation_items = []
        for item in extracted_items:
            validation_items.append({
                "content": item.content,
                "section": item.section,
                "source_info": f"{item.document_title} - {item.source_clause_title}"
            })
        
        # Validate in batches
        validator_outputs = self.validator.validate_batch(
            items=validation_items,
            playbook=playbook,
            batch_size=self.config.validator_batch_size
        )
        
        logger.info(f"Validated {len(validator_outputs)} items")
        
        # Step 3: CURATOR - Apply recommendations deterministically
        logger.info("Step 3: CURATOR - Applying recommendations")
        curator_result = self.curator.apply_recommendations(
            items=validation_items,
            validator_outputs=validator_outputs,
            playbook=playbook
        )
        
        logger.info(f"Curator generated {len(curator_result.operations)} operations")
        logger.info(f"  - ADD: {sum(1 for op in curator_result.operations if op.type == 'ADD')}")
        logger.info(f"  - MODIFY: {sum(1 for op in curator_result.operations if op.type == 'MODIFY')}")
        logger.info(f"  - SKIP: {curator_result.skipped_count}")
        
        # Execute operations
        execution_results = self.curator.execute_operations(
            operations=curator_result.operations,
            playbook=playbook,
            retriever=self.retriever
        )
        
        # Save playbook
        self.playbook_manager.save()
        
        result = EnrichmentResult(
            document_title=document.title,
            document_type=document.document_type,
            total_extracted=len(extracted_items),
            total_processed=len(extracted_items),
            total_added=len(execution_results["added_bullet_ids"]),
            total_skipped=execution_results["skipped_count"],
            added_bullet_ids=execution_results["added_bullet_ids"]
        )
        
        logger.info("=" * 80)
        logger.info(result.get_summary())
        logger.info("=" * 80)
        
        return result
    
    def get_extraction_preview(
        self,
        document: ParsedDocument,
        max_items: int = 10
    ) -> Dict[str, Any]:
        """Get a preview of what would be extracted."""
        logger.info("Preview mode: extracting sample items")
        logger.info(f"Using extraction sections: {', '.join(self.config.extraction_sections)}")
        
        playbook = self.playbook_manager.get_playbook()
        
        extracted_items = self.extractor.extract_from_document(
            document=document,
            granularity_level=self.config.granularity_level,
            allowed_sections=self.config.extraction_sections,
            playbook=playbook
        )
        
        preview_items = []
        for item in extracted_items[:max_items]:
            preview_items.append({
                "content": item.content,
                "section": item.section,
                "source": f"{item.source_clause_title} ({item.source_clause_uid})"
            })
        
        return {
            "document_title": document.title,
            "total_extracted": len(extracted_items),
            "preview_items": preview_items,
            "extraction_sections": self.config.extraction_sections
        }

