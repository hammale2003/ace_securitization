"""
Playbook Enricher for Trainer Mode.

Applies validated knowledge to the playbook using bulk operations:
- Bulk ADD operations
- MERGE with existing knowledge
- Update definitions
- Cross-reference bullets
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from playbook import PlaybookManager, Bullet, OperationResult
from trainer.knowledge_validator import ValidationResult


@dataclass
class EnrichmentResult:
    """Result of playbook enrichment operation."""
    total_validated: int
    total_added: int
    total_skipped: int
    total_merged: int
    added_bullets: List[Bullet] = field(default_factory=list)
    skipped_items: List[Dict[str, Any]] = field(default_factory=list)
    merged_items: List[Dict[str, Any]] = field(default_factory=list)
    operation_results: List[OperationResult] = field(default_factory=list)
    enrichment_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_validated": self.total_validated,
            "total_added": self.total_added,
            "total_skipped": self.total_skipped,
            "total_merged": self.total_merged,
            "added_bullets": [b.to_dict() for b in self.added_bullets],
            "skipped_items": self.skipped_items,
            "merged_items": self.merged_items,
            "operation_results": [
                {
                    "success": r.success,
                    "operation_type": r.operation_type,
                    "bullet_id": r.bullet_id,
                    "message": r.message
                }
                for r in self.operation_results
            ],
            "enrichment_summary": self.enrichment_summary
        }


class PlaybookEnricher:
    """
    Enriches the playbook with validated knowledge.
    
    Handles:
    - Bulk ADD operations for new knowledge
    - MERGE operations for similar knowledge
    - Cross-referencing between bullets
    - Metadata enrichment
    """
    
    def __init__(self, playbook_manager: PlaybookManager):
        self.playbook_manager = playbook_manager
    
    def enrich(
        self,
        validated_items: List[ValidationResult],
        auto_merge_threshold: float = 0.75,
        skip_invalid: bool = True
    ) -> EnrichmentResult:
        """
        Enrich playbook with validated knowledge items.
        
        Args:
            validated_items: List of validated knowledge items
            auto_merge_threshold: Similarity threshold for auto-merge
            skip_invalid: Whether to skip invalid items
        
        Returns:
            EnrichmentResult with operation details
        """
        added_bullets = []
        skipped_items = []
        merged_items = []
        operation_results = []
        
        for validated in validated_items:
            # Skip invalid items if configured
            if skip_invalid and not validated.is_valid:
                skipped_items.append({
                    "content": validated.classified.extracted.content[:100],
                    "reason": "Invalid",
                    "issues": validated.issues,
                    "validation_score": validated.validation_score
                })
                continue
            
            # Skip duplicates (high similarity above threshold)
            if validated.classified.is_duplicate:
                skipped_items.append({
                    "content": validated.classified.extracted.content[:100],
                    "reason": "Duplicate (high similarity)",
                    "duplicate_of": validated.classified.duplicate_of,
                    "similarity": validated.classified.similarity_score
                })
                continue
            
            # Check if should merge (similar but not duplicate)
            # Merge when similarity is high but below duplicate threshold
            should_merge = (
                validated.classified.similarity_score >= auto_merge_threshold and
                not validated.classified.is_duplicate and
                validated.classified.duplicate_of is not None
            )
            
            if should_merge and validated.classified.duplicate_of:
                # Perform MERGE operation
                merge_result = self._merge_knowledge(validated)
                operation_results.append(merge_result)
                
                if merge_result.success:
                    merged_items.append({
                        "new_content": validated.classified.extracted.content[:100],
                        "merged_into": validated.classified.duplicate_of,
                        "similarity": validated.classified.similarity_score
                    })
            else:
                # Perform ADD operation
                add_result = self._add_knowledge(validated)
                operation_results.append(add_result)
                
                if add_result.success:
                    # Invalidate cache and reload to get the newly added bullet
                    self.playbook_manager._playbook = None
                    playbook = self.playbook_manager.get_playbook()
                    section = validated.classified.final_section
                    bullets = playbook.get_section(section)
                    
                    # Find the newly added bullet by ID
                    for bullet in bullets:
                        if bullet.id == add_result.bullet_id:
                            added_bullets.append(bullet)
                            break
                else:
                    # ADD failed, add to skipped
                    skipped_items.append({
                        "content": validated.classified.extracted.content[:100],
                        "reason": f"ADD operation failed: {add_result.message}",
                        "section": validated.classified.final_section
                    })
        
        summary = self._generate_summary(
            len(validated_items),
            len(added_bullets),
            len(skipped_items),
            len(merged_items)
        )
        
        return EnrichmentResult(
            total_validated=len(validated_items),
            total_added=len(added_bullets),
            total_skipped=len(skipped_items),
            total_merged=len(merged_items),
            added_bullets=added_bullets,
            skipped_items=skipped_items,
            merged_items=merged_items,
            operation_results=operation_results,
            enrichment_summary=summary
        )
    
    def _add_knowledge(self, validated: ValidationResult) -> OperationResult:
        """
        Add new knowledge to playbook.
        
        Args:
            validated: Validated knowledge item
        
        Returns:
            OperationResult
        """
        section = validated.classified.final_section
        content = validated.classified.extracted.content
        
        # Enrich content with source metadata
        enriched_content = self._enrich_content(validated)
        
        # Create ADD operation
        operation = {
            "type": "ADD",  # Changed from "op" to "type" to match playbook.py
            "section": section,
            "content": enriched_content,
            "metadata": {
                "source": "trainer_mode",
                "source_document": validated.classified.extracted.metadata.get("document_type", "Unknown"),
                "source_clause": validated.classified.extracted.source_clause_uid,
                "confidence": validated.classified.final_confidence,
                "importance": validated.classified.importance_score,
                "extracted_at": datetime.utcnow().isoformat()
            }
        }
        
        # Apply operation
        try:
            results = self.playbook_manager.apply_operations([operation])
            result = results[0] if results else OperationResult(
                success=False,
                operation_type="ADD",
                message="No result returned from apply_operations"
            )
            return result
        except Exception as e:
            return OperationResult(
                success=False,
                operation_type="ADD",
                message=f"Exception: {str(e)}"
            )
    
    def _merge_knowledge(self, validated: ValidationResult) -> OperationResult:
        """
        Merge knowledge with existing bullet.
        
        Args:
            validated: Validated knowledge item
        
        Returns:
            OperationResult
        """
        target_bullet_id = validated.classified.duplicate_of
        new_content = validated.classified.extracted.content
        
        # Create MERGE operation
        operation = {
            "type": "MERGE",  # Changed from "op" to "type" to match playbook.py
            "target_id": target_bullet_id,
            "source_content": new_content,
            "merge_strategy": "append"  # or "replace", "combine"
        }
        
        # Apply operation
        results = self.playbook_manager.apply_operations([operation])
        return results[0] if results else OperationResult(
            success=False,
            operation_type="MERGE",
            message="Failed to merge knowledge"
        )
    
    def _enrich_content(self, validated: ValidationResult) -> str:
        """
        Enrich content with additional context if needed.
        
        Args:
            validated: Validated knowledge item
        
        Returns:
            Enriched content string
        """
        content = validated.classified.extracted.content
        
        # For definitions, ensure proper format
        if validated.classified.final_section == "definitions":
            if ":" not in content and "means" not in content.lower():
                # Try to extract term from source
                source_title = validated.classified.extracted.source_clause_title
                if source_title:
                    content = f"{source_title}: {content}"
        
        # For strategies, ensure actionable format
        if validated.classified.final_section == "strategies":
            # Content should be actionable - already in good format from extraction
            pass
        
        return content
    
    def _generate_summary(
        self,
        total: int,
        added: int,
        skipped: int,
        merged: int
    ) -> str:
        """Generate human-readable enrichment summary."""
        lines = []
        lines.append(f"Enrichment Summary")
        lines.append(f"  Total validated: {total}")
        lines.append(f"  Added: {added}")
        lines.append(f"  Merged: {merged}")
        lines.append(f"  Skipped: {skipped}")
        
        if added > 0:
            lines.append(f"  Playbook enriched with {added} new knowledge items")
        
        return "\n".join(lines)

