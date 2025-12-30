"""
Enrichment Pipeline - Integrates ACE framework for intelligent playbook enrichment.

Uses Generator, Reflector, and Curator agents to process extracted knowledge.
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import LLMConfig
from llm_client import create_client, LLMClient
from playbook import PlaybookManager, Playbook
from retriever import PlaybookRetriever
from playbook_enricher.document_parser import DocumentParser, ParsedDocument
from playbook_enricher.knowledge_extractor import KnowledgeExtractor, ExtractedKnowledge
from playbook_enricher.enricher_agents import EnricherGenerator, EnricherReflector, EnricherCurator
from playbook_enricher.granularity import GranularityLevel
from playbook_enricher.redundancy import decide_add_vs_skip_or_modify
from utils import logger


@dataclass
class EnrichmentConfig:
    """Configuration for enrichment pipeline."""
    granularity_level: GranularityLevel = GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE
    llm_config: Optional[LLMConfig] = None
    extraction_sections: List[str] = field(default_factory=lambda: ["strategies", "pitfalls", "templates", "definitions"])
    # Redundancy control: prevent duplicates, and upgrade existing bullets when better content arrives
    dedupe_enabled: bool = True
    dedupe_similarity_threshold: float = 0.86
    upgrade_similarity_threshold: float = 0.78
    upgrade_margin: float = 0.08


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
    Main enrichment pipeline integrating ACE framework.
    
    Flow:
    1. Extract raw knowledge from document
    2. For each extracted item:
       a. Generator: Evaluate if knowledge is specific and valuable
       b. Reflector: Assess quality and determine section
       c. Curator: Decide whether to add to playbook
    3. Apply curator decisions
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
        self.knowledge_extractor = KnowledgeExtractor(self.llm_client)
        
        # Enrichment-specific ACE agents
        self.generator = EnricherGenerator(self.llm_client, self.retriever)
        self.reflector = EnricherReflector(self.llm_client)
        self.curator = EnricherCurator(self.llm_client)
    
    def run_from_file(self, file_path: str) -> EnrichmentResult:
        """Run enrichment from a document file."""
        document = self.document_parser.parse_json_file(file_path)
        return self.run_from_document(document)
    
    def run_from_json_string(self, json_string: str) -> EnrichmentResult:
        """Run enrichment from a JSON string."""
        document = self.document_parser.parse_json_string(json_string)
        return self.run_from_document(document)
    
    def run_from_document(self, document: ParsedDocument) -> EnrichmentResult:
        """Run enrichment from a parsed document."""
        logger.info(f"Starting enrichment for: {document.title}")
        
        playbook = self.playbook_manager.get_playbook()
        
        # Step 1: Extract raw knowledge
        logger.info(f"Step 1: Extracting knowledge (granularity: {self.config.granularity_level.value})")
        logger.info(f"Target sections: {', '.join(self.config.extraction_sections)}")
        extracted_items = self.knowledge_extractor.extract_from_document(
            document=document,
            granularity_level=self.config.granularity_level,
            allowed_sections=self.config.extraction_sections
        )
        logger.info(f"Extracted {len(extracted_items)} knowledge items")
        
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
        
        # Step 2: Process each item through ACE pipeline
        logger.info("Step 2: Processing through ACE framework")
        added_bullet_ids = []
        skipped_count = 0
        
        for idx, item in enumerate(extracted_items, 1):
            logger.debug(f"Processing item {idx}/{len(extracted_items)}")
            
            # Use ACE to evaluate and apply operations
            operations = self._evaluate_knowledge(item, playbook)
            
            # Post-process operations to avoid redundant ADDs (and upgrade existing bullets when better)
            effective_ops = self._postprocess_operations(item, operations, playbook)

            if effective_ops:
                # Apply all curator operations
                for op in effective_ops:
                    op_type = op.get("type")
                    
                    if op_type == "ADD":
                        section = op.get("section", item.section)
                        content = op.get("content", item.content)
                        bullet = playbook.add_bullet(section, content)
                        added_bullet_ids.append(bullet.id)
                        logger.debug(f"Added bullet {bullet.id} to {section}")
                        
                        # Update retriever
                        if self.retriever:
                            self.retriever.add_bullet(bullet.id, bullet.content, section)
                    
                    elif op_type == "REMOVE":
                        bullet_id = op.get("bullet_id")
                        reason = op.get("reason", "")
                        if bullet_id:
                            playbook.remove_bullet(bullet_id, reason)
                            logger.debug(f"Removed bullet {bullet_id}: {reason}")
                            
                            # Update retriever
                            if self.retriever:
                                self.retriever.remove_bullet(bullet_id)
                    
                    elif op_type == "MODIFY":
                        bullet_id = op.get("bullet_id")
                        new_content = op.get("new_content")
                        reason = op.get("reason", "")
                        reset_harmful = op.get("reset_harmful", False)
                        if bullet_id and new_content:
                            playbook.modify_bullet(bullet_id, new_content, reason, reset_harmful)
                            logger.debug(f"Modified bullet {bullet_id}: {reason}")
                            
                            # Update retriever - find section by searching all sections
                            if self.retriever:
                                section = self._find_bullet_section(playbook, bullet_id)
                                if section:
                                    # retriever.update_bullet signature is (bullet_id, new_content, helpful_count?, harmful_count?)
                                    self.retriever.update_bullet(bullet_id, new_content)
                    
                    elif op_type == "MERGE":
                        source_ids = op.get("source_bullet_ids", [])
                        target_section = op.get("target_section")
                        merged_content = op.get("merged_content")
                        reason = op.get("reason", "")
                        if len(source_ids) >= 2 and target_section and merged_content:
                            new_bullet = playbook.merge_bullets(
                                source_ids, target_section, merged_content, reason
                            )
                            if new_bullet:
                                added_bullet_ids.append(new_bullet.id)
                                logger.debug(f"Merged {len(source_ids)} bullets into {new_bullet.id}")
                                
                                # Update retriever
                                if self.retriever:
                                    for src_id in source_ids:
                                        self.retriever.remove_bullet(src_id)
                                    self.retriever.add_bullet(new_bullet.id, merged_content, target_section)
            else:
                skipped_count += 1
                logger.debug(f"Skipped item: not specific enough or duplicate")
        
        # Save playbook
        self.playbook_manager.save()
        
        result = EnrichmentResult(
            document_title=document.title,
            document_type=document.document_type,
            total_extracted=len(extracted_items),
            total_processed=len(extracted_items),
            total_added=len(added_bullet_ids),
            total_skipped=skipped_count,
            added_bullet_ids=added_bullet_ids
        )
        
        logger.info("=" * 80)
        logger.info(result.get_summary())
        logger.info("=" * 80)
        
        return result

    def _postprocess_operations(
        self,
        item: ExtractedKnowledge,
        operations: List[Dict[str, Any]],
        playbook: Playbook
    ) -> List[Dict[str, Any]]:
        """
        Enforce deterministic redundancy control on top of LLM curator outputs.

        Key behavior:
        - For curator ADD ops: if content already exists (or is a near-duplicate) in the same section,
          skip the op.
        - If the incoming content is clearly better (e.g. a substantive definition replacing a pointer),
          convert ADD -> MODIFY on the existing bullet instead of adding a duplicate.
        """
        if not operations:
            return []

        if not self.config.dedupe_enabled:
            return operations

        effective: List[Dict[str, Any]] = []

        for op in operations:
            op_type = (op.get("type") or "").upper()
            if op_type != "ADD":
                effective.append(op)
                continue

            section = op.get("section", item.section)
            content = op.get("content", item.content)

            decision = decide_add_vs_skip_or_modify(
                playbook=playbook,
                section=section,
                new_content=content,
                retriever=self.retriever,
                duplicate_similarity_threshold=self.config.dedupe_similarity_threshold,
                upgrade_similarity_threshold=self.config.upgrade_similarity_threshold,
                upgrade_margin=self.config.upgrade_margin,
            )

            if decision.action == "ADD":
                effective.append(op)
                continue

            if decision.action == "SKIP":
                logger.info(
                    f"Redundancy: SKIP ADD into {section} (matched {decision.target_bullet_id}, {decision.reason})"
                )
                continue

            if decision.action == "MODIFY" and decision.target_bullet_id:
                logger.info(
                    f"Redundancy: MODIFY {decision.target_bullet_id} instead of ADD into {section} ({decision.reason})"
                )
                effective.append({
                    "type": "MODIFY",
                    "bullet_id": decision.target_bullet_id,
                    "new_content": content,
                    "reason": f"Auto-updated by ACE enrichment: {decision.reason}",
                    "reset_harmful": False
                })
                continue

            # Safe fallback: if decision was malformed, keep the original op.
            effective.append(op)

        return effective
    
    def _find_bullet_section(self, playbook: Playbook, bullet_id: str) -> Optional[str]:
        """Find which section a bullet belongs to."""
        for section_name in ["strategies", "pitfalls", "templates", "definitions"]:
            section_bullets = playbook.get_section(section_name)
            for bullet in section_bullets:
                if bullet.id == bullet_id:
                    return section_name
        return None
    
    def _evaluate_knowledge(
        self,
        item: ExtractedKnowledge,
        playbook: Playbook
    ) -> List[Dict[str, Any]]:
        """
        Evaluate knowledge using enrichment-specific ACE framework.
        
        Returns:
            List of operations (ADD, REMOVE, MODIFY, MERGE)
        """
        source_info = f"{item.document_title} - {item.source_clause_title}"
        
        try:
            # Step 1: Generator validates the extracted knowledge
            logger.info(f"Evaluating: {item.content[:80]}...")
            
            generator_output = self.generator.validate(
                content=item.content,
                section=item.section,
                playbook=playbook,
                source_info=source_info
            )
            
            logger.info(f"Generator: {generator_output.recommendation} | Valid={generator_output.is_valid} | Duplicate={generator_output.is_duplicate}")
            logger.info(f"Generator reasoning: {generator_output.reasoning}")
            
            # Step 2: Reflector assesses quality
            reflector_output = self.reflector.assess(
                content=item.content,
                section=item.section,
                generator_output=generator_output,
                source_info=source_info
            )
            
            logger.info(f"Reflector: Quality={reflector_output.quality_score:.2f}, Specificity={reflector_output.specificity_score:.2f}, Reusability={reflector_output.reusability_score:.2f}")
            logger.info(f"Reflector recommendation: {reflector_output.recommendation}")
            
            # Step 3: Curator decides final operations
            curator_output = self.curator.decide(
                content=item.content,
                section=item.section,
                generator_output=generator_output,
                reflector_output=reflector_output,
                playbook=playbook
            )
            
            logger.info(f"Curator: {len(curator_output.operations)} operations")
            logger.info(f"Curator reasoning: {curator_output.reasoning}")
            
            return curator_output.operations
        
        except Exception as e:
            logger.warning(f"Error evaluating knowledge: {e}")
            return []
    
    def get_extraction_preview(
        self,
        document: ParsedDocument,
        max_items: int = 10
    ) -> Dict[str, Any]:
        """Get a preview of what would be extracted."""
        logger.info("Preview mode: extracting sample items")
        logger.info(f"Using extraction sections: {', '.join(self.config.extraction_sections)}")
        
        extracted_items = self.knowledge_extractor.extract_from_document(
            document=document,
            granularity_level=self.config.granularity_level,
            allowed_sections=self.config.extraction_sections
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

