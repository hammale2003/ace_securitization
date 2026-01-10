"""
Curator Agent - Deterministic application of Validator recommendations.

Applies delta updates to playbook.json based on Validator recommendations.
Operations: ADD (new bullet), MODIFY (update existing), SKIP (reject)
Deduplication at write-time using redundancy.py logic.
Tracks all changes for audit logging.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from playbook import Playbook, Bullet
from playbook_enricher.validator_agent import ValidatorOutput
from playbook_enricher.redundancy import decide_add_vs_skip_or_modify
from utils import logger


@dataclass
class CuratorOperation:
    """A single curator operation."""
    type: str  # ADD, MODIFY, SKIP
    section: str
    content: Optional[str] = None
    bullet_id: Optional[str] = None
    new_content: Optional[str] = None
    reason: str = ""
    reset_harmful: bool = False


@dataclass
class CuratorResult:
    """Result from curator processing."""
    operations: List[CuratorOperation]
    added_bullet_ids: List[str] = field(default_factory=list)
    modified_bullet_ids: List[str] = field(default_factory=list)
    skipped_count: int = 0
    audit_log: List[Dict[str, Any]] = field(default_factory=list)


class CuratorAgent:
    """
    Deterministic Curator Agent.
    
    Applies Validator recommendations to playbook using redundancy.py logic.
    No LLM calls - purely deterministic based on Validator outputs.
    """
    
    def __init__(
        self,
        dedupe_enabled: bool = True,
        dedupe_similarity_threshold: float = 0.86,
        upgrade_similarity_threshold: float = 0.78,
        upgrade_margin: float = 0.08,
        retriever=None
    ):
        """
        Initialize Curator Agent.
        
        Args:
            dedupe_enabled: Enable deduplication
            dedupe_similarity_threshold: Threshold for duplicate detection
            upgrade_similarity_threshold: Threshold for upgrade detection
            upgrade_margin: Quality margin for upgrades
            retriever: Optional retriever for similarity search
        """
        self.dedupe_enabled = dedupe_enabled
        self.dedupe_similarity_threshold = dedupe_similarity_threshold
        self.upgrade_similarity_threshold = upgrade_similarity_threshold
        self.upgrade_margin = upgrade_margin
        self.retriever = retriever
    
    def apply_recommendations(
        self,
        items: List[Dict[str, Any]],
        validator_outputs: List[ValidatorOutput],
        playbook: Playbook
    ) -> CuratorResult:
        """
        Apply Validator recommendations to playbook.
        
        Args:
            items: Original extracted items (with content, section, etc.)
            validator_outputs: Validator outputs (one per item)
            playbook: Playbook to update
        
        Returns:
            CuratorResult with operations and audit log
        """
        if len(items) != len(validator_outputs):
            logger.error(f"Mismatch: {len(items)} items but {len(validator_outputs)} validator outputs")
            return CuratorResult(operations=[])
        
        operations = []
        added_bullet_ids = []
        modified_bullet_ids = []
        skipped_count = 0
        audit_log = []
        
        # Track items added in this batch for intra-batch deduplication
        batch_added_items = {}  # section -> list of (content, item_index) tuples
        
        for idx, (item, validator_output) in enumerate(zip(items, validator_outputs)):
            # Determine content to use (enriched if available, otherwise original)
            content = validator_output.enriched_content or item["content"]
            section = item["section"]
            
            # Create audit log entry
            audit_entry = {
                "item_index": idx,
                "section": section,
                "original_content": item["content"],
                "enriched_content": validator_output.enriched_content,
                "validator_recommendation": validator_output.recommendation,
                "validator_reasoning": validator_output.reasoning,
                "is_reusable": validator_output.is_reusable,
                "is_correct": validator_output.is_correct,
                "is_duplicate": validator_output.is_duplicate,
                "quality_score": validator_output.quality_score,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Apply recommendation
            if validator_output.recommendation == "SKIP":
                operation = CuratorOperation(
                    type="SKIP",
                    section=section,
                    reason=f"Validator recommendation: {validator_output.reasoning}"
                )
                operations.append(operation)
                skipped_count += 1
                audit_entry["curator_action"] = "SKIP"
                audit_entry["curator_reason"] = operation.reason
                audit_log.append(audit_entry)
                continue
            
            # Check if content should be skipped due to low quality
            if not validator_output.is_reusable or not validator_output.is_correct:
                operation = CuratorOperation(
                    type="SKIP",
                    section=section,
                    reason=f"Not reusable or incorrect: {validator_output.reasoning}"
                )
                operations.append(operation)
                skipped_count += 1
                audit_entry["curator_action"] = "SKIP"
                audit_entry["curator_reason"] = operation.reason
                audit_log.append(audit_entry)
                continue
            
            # Check for duplicates within current batch
            if self._is_duplicate_in_batch(content, section, batch_added_items, playbook):
                operation = CuratorOperation(
                    type="SKIP",
                    section=section,
                    reason="Duplicate within current batch"
                )
                operations.append(operation)
                skipped_count += 1
                audit_entry["curator_action"] = "SKIP"
                audit_entry["curator_reason"] = operation.reason
                audit_log.append(audit_entry)
                continue
            
            # Apply ADD or MODIFY recommendation
            if validator_output.recommendation == "ADD":
                # Use redundancy logic to decide final action
                if self.dedupe_enabled:
                    redundancy_decision = decide_add_vs_skip_or_modify(
                        playbook=playbook,
                        section=section,
                        new_content=content,
                        retriever=self.retriever,
                        duplicate_similarity_threshold=self.dedupe_similarity_threshold,
                        upgrade_similarity_threshold=self.upgrade_similarity_threshold,
                        upgrade_margin=self.upgrade_margin
                    )
                    
                    if redundancy_decision.action == "SKIP":
                        operation = CuratorOperation(
                            type="SKIP",
                            section=section,
                            reason=f"Redundancy check: {redundancy_decision.reason}"
                        )
                        operations.append(operation)
                        skipped_count += 1
                        audit_entry["curator_action"] = "SKIP"
                        audit_entry["curator_reason"] = redundancy_decision.reason
                        audit_entry["redundancy_similarity"] = redundancy_decision.similarity
                        audit_log.append(audit_entry)
                        continue
                    
                    elif redundancy_decision.action == "MODIFY" and redundancy_decision.target_bullet_id:
                        # Build update_reason for auto-upgrade
                        update_reason = f"Auto-upgrade: {redundancy_decision.reason}"
                        operation = CuratorOperation(
                            type="MODIFY",
                            section=section,
                            bullet_id=redundancy_decision.target_bullet_id,
                            new_content=content,
                            reason=update_reason,
                            reset_harmful=False
                        )
                        operations.append(operation)
                        modified_bullet_ids.append(redundancy_decision.target_bullet_id)
                        audit_entry["curator_action"] = "MODIFY"
                        audit_entry["curator_reason"] = redundancy_decision.reason
                        audit_entry["update_reason"] = update_reason
                        audit_entry["target_bullet_id"] = redundancy_decision.target_bullet_id
                        audit_entry["redundancy_similarity"] = redundancy_decision.similarity
                        audit_log.append(audit_entry)
                        continue
                
                # ADD operation
                operation = CuratorOperation(
                    type="ADD",
                    section=section,
                    content=content,
                    reason=f"Validator recommendation: {validator_output.reasoning}"
                )
                operations.append(operation)
                
                # Track in batch context (will be updated after actual add)
                if section not in batch_added_items:
                    batch_added_items[section] = []
                batch_added_items[section].append((content, idx))
                
                audit_entry["curator_action"] = "ADD"
                audit_entry["curator_reason"] = operation.reason
                audit_log.append(audit_entry)
            
            elif validator_output.recommendation == "MODIFY":
                # Get bullet_id from similar_bullets
                bullet_id = validator_output.similar_bullets[0] if validator_output.similar_bullets else None
                
                if not bullet_id:
                    operation = CuratorOperation(
                        type="SKIP",
                        section=section,
                        reason="MODIFY requested but no bullet_id provided"
                    )
                    operations.append(operation)
                    skipped_count += 1
                    audit_entry["curator_action"] = "SKIP"
                    audit_entry["curator_reason"] = operation.reason
                    audit_log.append(audit_entry)
                    continue
                
                # Check if quality is sufficient for upgrade
                if validator_output.quality_score < 0.60:
                    operation = CuratorOperation(
                        type="SKIP",
                        section=section,
                        reason=f"Quality score {validator_output.quality_score:.2f} too low for MODIFY"
                    )
                    operations.append(operation)
                    skipped_count += 1
                    audit_entry["curator_action"] = "SKIP"
                    audit_entry["curator_reason"] = operation.reason
                    audit_log.append(audit_entry)
                    continue
                
                # CRITICAL: For MODIFY, enriched_content should be provided. If not, use content but log warning
                if not validator_output.enriched_content:
                    logger.warning(f"MODIFY recommendation for bullet {bullet_id} but no enriched_content provided. Using original content.")
                    audit_entry["warning"] = "MODIFY without enriched_content - using original content"
                
                # Verify that the new content is actually different from existing
                existing_bullet = playbook.get_bullet_by_id(bullet_id)
                if existing_bullet:
                    from playbook_enricher.redundancy import normalize_text
                    existing_normalized = normalize_text(existing_bullet.content)
                    new_normalized = normalize_text(content)
                    
                    if existing_normalized == new_normalized:
                        logger.warning(f"MODIFY requested for bullet {bullet_id} but content is identical. Skipping.")
                        operation = CuratorOperation(
                            type="SKIP",
                            section=section,
                            reason="MODIFY requested but new content is identical to existing"
                        )
                        operations.append(operation)
                        skipped_count += 1
                        audit_entry["curator_action"] = "SKIP"
                        audit_entry["curator_reason"] = operation.reason
                        audit_log.append(audit_entry)
                        continue
                
                operation = CuratorOperation(
                    type="MODIFY",
                    section=section,
                    bullet_id=bullet_id,
                    new_content=content,
                    reason=validator_output.update_reason or f"Validator recommendation: {validator_output.reasoning}",
                    reset_harmful=False
                )
                operations.append(operation)
                modified_bullet_ids.append(bullet_id)
                audit_entry["curator_action"] = "MODIFY"
                audit_entry["curator_reason"] = operation.reason
                audit_entry["update_reason"] = validator_output.update_reason
                audit_entry["target_bullet_id"] = bullet_id
                audit_entry["existing_content_preview"] = existing_bullet.content[:100] if existing_bullet else None
                audit_entry["new_content_preview"] = content[:100]
                audit_log.append(audit_entry)
        
        return CuratorResult(
            operations=operations,
            added_bullet_ids=added_bullet_ids,
            modified_bullet_ids=modified_bullet_ids,
            skipped_count=skipped_count,
            audit_log=audit_log
        )
    
    def execute_operations(
        self,
        operations: List[CuratorOperation],
        playbook: Playbook,
        retriever=None
    ) -> Dict[str, Any]:
        """
        Execute curator operations on playbook.
        
        Returns:
            Dict with added_bullet_ids, modified_bullet_ids, skipped_count
        """
        added_bullet_ids = []
        modified_bullet_ids = []
        skipped_count = 0
        
        for op in operations:
            if op.type == "ADD":
                bullet = playbook.add_bullet(op.section, op.content)
                added_bullet_ids.append(bullet.id)
                logger.debug(f"Added bullet {bullet.id} to {op.section}")
                
                # Update retriever if available
                if retriever:
                    retriever.add_bullet(bullet.id, bullet.content, op.section)
            
            elif op.type == "MODIFY":
                if op.bullet_id and op.new_content:
                    # Get existing bullet to verify it exists
                    existing_bullet = playbook.get_bullet_by_id(op.bullet_id)
                    if not existing_bullet:
                        logger.error(f"MODIFY operation failed: bullet {op.bullet_id} not found in playbook")
                        skipped_count += 1
                        continue
                    
                    # Log before modification
                    logger.info(f"Modifying bullet {op.bullet_id} in section {op.section}")
                    logger.debug(f"  Existing content: {existing_bullet.content[:150]}...")
                    logger.debug(f"  New content: {op.new_content[:150]}...")
                    logger.debug(f"  Reason: {op.reason}")
                    
                    # Apply modification
                    modified_bullet = playbook.modify_bullet(
                        op.bullet_id,
                        op.new_content,
                        op.reason,
                        op.reset_harmful
                    )
                    
                    if modified_bullet:
                        modified_bullet_ids.append(op.bullet_id)
                        logger.info(f"Successfully modified bullet {op.bullet_id} (revision {modified_bullet.revision_count})")
                        
                        # Verify the modification was applied
                        verify_bullet = playbook.get_bullet_by_id(op.bullet_id)
                        if verify_bullet and verify_bullet.content == op.new_content.strip():
                            logger.debug(f"Verification: bullet {op.bullet_id} content updated correctly")
                        else:
                            logger.error(f"VERIFICATION FAILED: bullet {op.bullet_id} content was not updated correctly!")
                            logger.error(f"  Expected: {op.new_content[:150]}...")
                            logger.error(f"  Actual: {verify_bullet.content[:150] if verify_bullet else 'BULLET NOT FOUND'}...")
                        
                        # Update retriever if available
                        if retriever:
                            section = self._find_bullet_section(playbook, op.bullet_id)
                            if section:
                                retriever.update_bullet(op.bullet_id, op.new_content)
                                logger.debug(f"Updated retriever index for bullet {op.bullet_id}")
                    else:
                        logger.error(f"MODIFY operation failed: modify_bullet returned None for {op.bullet_id}")
                        skipped_count += 1
                else:
                    logger.error(f"MODIFY operation skipped: missing bullet_id or new_content")
                    skipped_count += 1
            
            elif op.type == "SKIP":
                skipped_count += 1
                logger.debug(f"Skipped: {op.reason}")
        
        return {
            "added_bullet_ids": added_bullet_ids,
            "modified_bullet_ids": modified_bullet_ids,
            "skipped_count": skipped_count
        }
    
    def _is_duplicate_in_batch(
        self,
        content: str,
        section: str,
        batch_added_items: Dict[str, List[tuple]],
        playbook: Playbook
    ) -> bool:
        """Check if content is a duplicate of items already added in the current batch."""
        if section not in batch_added_items:
            return False
        
        from playbook_enricher.redundancy import (
            normalize_text,
            extract_definition_term,
            content_quality_score
        )
        from playbook import compute_semantic_similarity
        
        # Check against each item in the batch
        for batch_content, _ in batch_added_items[section]:
            # 1. Check for exact duplicates (normalized)
            if normalize_text(batch_content) == normalize_text(content):
                logger.debug(f"Found exact duplicate in batch")
                return True
            
            # 2. For definitions, check if same term
            if section == "definitions":
                term1 = extract_definition_term(batch_content)
                term2 = extract_definition_term(content)
                if term1 and term2 and term1.lower() == term2.lower():
                    # Same term - check if one is clearly better
                    batch_q = content_quality_score(section, batch_content)
                    new_q = content_quality_score(section, content)
                    
                    # If new content is not significantly better, skip as duplicate
                    if new_q <= batch_q + self.upgrade_margin:
                        logger.debug(f"Found duplicate definition term in batch: {term1}")
                        return True
            
            # 3. Check semantic similarity
            similarity = compute_semantic_similarity(batch_content, content)
            if similarity >= self.dedupe_similarity_threshold:
                # Check if new content is significantly better
                batch_q = content_quality_score(section, batch_content)
                new_q = content_quality_score(section, content)
                
                # If new content is not significantly better, skip as duplicate
                if new_q <= batch_q + self.upgrade_margin:
                    logger.debug(f"Found duplicate in batch: similarity={similarity:.2f}")
                    return True
        
        return False
    
    def _find_bullet_section(self, playbook: Playbook, bullet_id: str) -> Optional[str]:
        """Find which section a bullet belongs to."""
        for section_name in ["strategies", "pitfalls", "definitions"]:
            section_bullets = playbook.get_section(section_name)
            for bullet in section_bullets:
                if bullet.id == bullet_id:
                    return section_name
        return None

