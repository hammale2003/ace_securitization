"""
Trainer Pipeline - Main orchestrator for Trainer Mode.

Coordinates all trainer agents to extract knowledge from documents
and enrich the playbook.
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import ACEConfig, LLMConfig
from llm_client import create_client, LLMClient
from playbook import PlaybookManager, Playbook
from embeddings import create_embedding_model, EmbeddingConfig
from retriever import PlaybookRetriever

from trainer.document_parser import DocumentParser, ParsedDocument
from trainer.knowledge_extractor import KnowledgeExtractor, ExtractedKnowledge
from trainer.knowledge_classifier import KnowledgeClassifier, ClassifiedKnowledge
from trainer.knowledge_validator import KnowledgeValidator, ValidationResult
from trainer.playbook_enricher import PlaybookEnricher, EnrichmentResult
from trainer.granularity import GranularityLevel
from utils import logger


@dataclass
class TrainerConfig:
    """Configuration for Trainer Mode."""
    # Extraction settings
    extraction_types: List[str] = field(default_factory=lambda: [
        "strategies", "definitions", "templates", "pitfalls", "code_snippets"
    ])
    min_extraction_confidence: float = 0.5
    
    # Granularity settings
    granularity_level: GranularityLevel = GranularityLevel.BATCH
    
    # Performance optimization settings
    batch_size: int = 15  # Number of clauses per LLM call (10-20 recommended) - used for BATCH mode
    max_clauses_preview: int = 30  # Max clauses for preview mode (fast testing)
    max_clauses_full: Optional[int] = None  # Max clauses for full processing (None = all)
    
    # Classification settings
    duplicate_threshold: float = 0.85
    
    # Validation settings
    min_validation_score: float = 0.6
    skip_invalid: bool = True
    
    # Enrichment settings
    auto_merge_threshold: float = 0.75
    
    # LLM settings
    llm_config: Optional[LLMConfig] = None
    
    # Embedding settings
    embedding_provider: str = "simple"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 256


@dataclass
class TrainerPipelineResult:
    """Result from running the trainer pipeline."""
    document_title: str
    document_type: str
    total_clauses_parsed: int
    total_extracted: int
    total_classified: int
    total_validated: int
    enrichment_result: EnrichmentResult
    extraction_by_type: Dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_title": self.document_title,
            "document_type": self.document_type,
            "total_clauses_parsed": self.total_clauses_parsed,
            "total_extracted": self.total_extracted,
            "total_classified": self.total_classified,
            "total_validated": self.total_validated,
            "enrichment_result": self.enrichment_result.to_dict(),
            "extraction_by_type": self.extraction_by_type,
            "timestamp": self.timestamp
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = []
        lines.append(f"Trainer Mode - Document Analysis Complete")
        lines.append(f"")
        lines.append(f"Document: {self.document_title}")
        lines.append(f"   Type: {self.document_type}")
        lines.append(f"")
        lines.append(f"Processing Stats:")
        lines.append(f"   Clauses parsed: {self.total_clauses_parsed}")
        lines.append(f"   Knowledge extracted: {self.total_extracted}")
        lines.append(f"   Items classified: {self.total_classified}")
        lines.append(f"   Items validated: {self.total_validated}")
        lines.append(f"")
        lines.append(f"Extraction by Type:")
        for k_type, count in self.extraction_by_type.items():
            lines.append(f"   {k_type}: {count}")
        lines.append(f"")
        lines.append(self.enrichment_result.enrichment_summary)
        
        return "\n".join(lines)


class TrainerPipeline:
    """
    Main pipeline for Trainer Mode.
    
    Orchestrates:
    1. Document Ingestion (parsing)
    2. Knowledge Extraction
    3. Knowledge Classification
    4. Knowledge Validation
    5. Playbook Enrichment
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        playbook_manager: PlaybookManager,
        retriever: Optional[PlaybookRetriever] = None
    ):
        self.config = config
        self.playbook_manager = playbook_manager
        self.retriever = retriever
        
        # Initialize LLM client
        llm_config = config.llm_config or LLMConfig()
        self.llm_client = create_client(llm_config)
        
        # Initialize embedding model
        embed_config = EmbeddingConfig(
            provider=config.embedding_provider,
            model_name=config.embedding_model,
            embedding_dim=config.embedding_dim
        )
        self.embedding_model = create_embedding_model(embed_config)
        
        # Initialize agents
        self.document_parser = DocumentParser()
        self.knowledge_extractor = KnowledgeExtractor(self.llm_client)
        self.knowledge_classifier = KnowledgeClassifier(
            self.llm_client,
            self.embedding_model,
            config.duplicate_threshold
        )
        self.knowledge_validator = KnowledgeValidator(
            self.llm_client,
            config.min_validation_score
        )
        self.playbook_enricher = PlaybookEnricher(self.playbook_manager)
    
    def run_from_file(self, file_path: str) -> TrainerPipelineResult:
        """
        Run trainer pipeline on a document file.
        
        Args:
            file_path: Path to JSON document file
        
        Returns:
            TrainerPipelineResult
        """
        # Parse document
        document = self.document_parser.parse_json_file(file_path)
        
        return self.run_from_document(document)
    
    def run_from_json_string(self, json_string: str) -> TrainerPipelineResult:
        """
        Run trainer pipeline on a JSON string.
        
        Args:
            json_string: Minified JSON document string
        
        Returns:
            TrainerPipelineResult
        """
        # Parse document
        document = self.document_parser.parse_json_string(json_string)
        
        return self.run_from_document(document)
    
    def run_from_document(self, document: ParsedDocument, progress_callback=None) -> TrainerPipelineResult:
        """
        Run trainer pipeline on a parsed document.
        
        Args:
            document: Parsed document
            progress_callback: Optional callback function(current_step, total_steps, step_name, percentage)
        
        Returns:
            TrainerPipelineResult
        """
        logger.info(f"Starting Trainer Mode for: {document.title} (Type: {document.document_type})")
        
        total_steps = 5
        
        def update_progress(step: int, step_name: str):
            if progress_callback:
                percentage = int((step / total_steps) * 100)
                progress_callback(step, total_steps, step_name, percentage)
        
        # Get current playbook
        playbook = self.playbook_manager.get_playbook()
        
        # Step 1: Extract knowledge (with batch processing for speed)
        update_progress(1, "Extraction des connaissances")
        total_clauses = len(document.get_all_clauses_flat())
        logger.info(f"Step 1: Extracting knowledge from {total_clauses} clauses (Granularity: {self.config.granularity_level.value})")
        extracted_items = self.knowledge_extractor.extract_from_document(
            document=document,
            extraction_types=self.config.extraction_types,
            min_confidence=self.config.min_extraction_confidence,
            granularity_level=self.config.granularity_level,
            batch_size=self.config.batch_size,  # Use configured batch size
            max_clauses=self.config.max_clauses_full  # Limit if configured
        )
        logger.info(f"Extracted {len(extracted_items)} knowledge items")
        
        # Count by type
        extraction_by_type = {}
        for item in extracted_items:
            extraction_by_type[item.knowledge_type] = extraction_by_type.get(item.knowledge_type, 0) + 1
        
        # Step 2: Classify knowledge
        update_progress(2, "Classification des connaissances")
        logger.info("Step 2: Classifying extracted knowledge...")
        classified_items = self.knowledge_classifier.classify_batch(
            extracted_items=extracted_items,
            existing_playbook=playbook
        )
        logger.info(f"Classified {len(classified_items)} items")
        
        # Step 3: Validate knowledge
        update_progress(3, "Validation des connaissances")
        logger.info("Step 3: Validating classified knowledge...")
        validated_items = self.knowledge_validator.validate_batch(
            classified_items=classified_items,
            existing_playbook=playbook
        )
        
        valid_count = sum(1 for v in validated_items if v.is_valid)
        logger.info(f"Validated {len(validated_items)} items ({valid_count} valid)")
        
        # Step 4: Enrich playbook
        update_progress(4, "Enrichissement du playbook")
        logger.info("Step 4: Enriching playbook...")
        enrichment_result = self.playbook_enricher.enrich(
            validated_items=validated_items,
            auto_merge_threshold=self.config.auto_merge_threshold,
            skip_invalid=self.config.skip_invalid
        )
        logger.info(enrichment_result.enrichment_summary)
        
        # Step 5: Update retriever index if available
        update_progress(5, "Mise à jour de l'index")
        if self.retriever and enrichment_result.total_added > 0:
            logger.info("Step 5: Updating retriever index...")
            updated_playbook = self.playbook_manager.get_playbook()
            indexed_count = self.retriever.index_playbook(updated_playbook)
            logger.info(f"Indexed {indexed_count} bullets")
        
        # Create result
        result = TrainerPipelineResult(
            document_title=document.title,
            document_type=document.document_type,
            total_clauses_parsed=len(document.get_all_clauses_flat()),
            total_extracted=len(extracted_items),
            total_classified=len(classified_items),
            total_validated=len(validated_items),
            enrichment_result=enrichment_result,
            extraction_by_type=extraction_by_type
        )
        
        logger.info("="*80)
        logger.info(result.get_summary())
        logger.info("="*80)
        
        return result
    
    def get_extraction_preview(
        self,
        document: ParsedDocument,
        max_items: int = 20
    ) -> Dict[str, Any]:
        """
        Get a preview of what would be extracted without actually enriching.
        OPTIMIZED: Only processes limited clauses for speed.
        
        Args:
            document: Parsed document
            max_items: Maximum items to show in preview (default: 20)
        
        Returns:
            Preview data
        """
        max_clauses = self.config.max_clauses_preview
        logger.info(f"PREVIEW MODE: Processing only first {max_clauses} clauses (Granularity: {self.config.granularity_level.value})")
        playbook = self.playbook_manager.get_playbook()
        
        # Extract with limits for speed
        extracted_items = self.knowledge_extractor.extract_from_document(
            document=document,
            extraction_types=self.config.extraction_types,
            min_confidence=self.config.min_extraction_confidence,
            granularity_level=self.config.granularity_level,
            batch_size=self.config.batch_size,
            max_clauses=max_clauses
        )
        
        # Classify
        classified_items = self.knowledge_classifier.classify_batch(
            extracted_items=extracted_items,
            existing_playbook=playbook
        )
        
        # Validate
        validated_items = self.knowledge_validator.validate_batch(
            classified_items=classified_items,
            existing_playbook=playbook
        )
        
        # Group by type
        preview_by_type = {}
        for validated in validated_items[:max_items]:
            k_type = validated.classified.final_section
            if k_type not in preview_by_type:
                preview_by_type[k_type] = []
            
            preview_by_type[k_type].append({
                "content": validated.classified.extracted.content,
                "confidence": validated.classified.final_confidence,
                "importance": validated.classified.importance_score,
                "is_valid": validated.is_valid,
                "is_duplicate": validated.classified.is_duplicate,
                "validation_score": validated.validation_score
            })
        
        return {
            "document_title": document.title,
            "total_clauses": len(document.get_all_clauses_flat()),
            "total_extracted": len(extracted_items),
            "total_valid": sum(1 for v in validated_items if v.is_valid),
            "preview_by_type": preview_by_type
        }

