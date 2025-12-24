"""
Playbook data model and management for the ACE system.

The playbook is the evolving context that accumulates domain knowledge.
Each bullet has metadata (id, helpful/harmful counts) and content.

Supports operations: ADD, REMOVE, MODIFY, MERGE
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import threading

from config import PLAYBOOK_SECTIONS, SECTION_PREFIXES, PlaybookConfig
from utils import logger


@dataclass
class Bullet:
    """A single knowledge bullet in the playbook."""
    id: str
    content: str
    helpful_count: int = 0
    harmful_count: int = 0
    neutral_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    revision_count: int = 0  # Track number of modifications
    archived: bool = False
    archived_at: Optional[str] = None
    archive_reason: Optional[str] = None
    merged_from: Optional[List[str]] = None  # IDs of bullets merged into this one
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bullet":
        # Handle older data without new fields
        defaults = {
            "revision_count": 0,
            "archived": False,
            "archived_at": None,
            "archive_reason": None,
            "merged_from": None
        }
        for key, default in defaults.items():
            if key not in data:
                data[key] = default
        return cls(**data)
    
    def mark_helpful(self):
        self.helpful_count += 1
        self.updated_at = datetime.utcnow().isoformat()
    
    def mark_harmful(self):
        self.harmful_count += 1
        self.updated_at = datetime.utcnow().isoformat()
    
    def mark_neutral(self):
        self.neutral_count += 1
        self.updated_at = datetime.utcnow().isoformat()
    
    @property
    def effectiveness_score(self) -> float:
        """Calculate effectiveness score for ranking/pruning."""
        total = self.helpful_count + self.harmful_count + self.neutral_count
        if total == 0:
            return 0.0
        return (self.helpful_count - self.harmful_count) / total
    
    @property
    def total_usage(self) -> int:
        """Total number of times this bullet was used."""
        return self.helpful_count + self.harmful_count + self.neutral_count
    
    def format_for_prompt(self) -> str:
        """Format bullet for inclusion in LLM prompts."""
        return f"[{self.id}] helpful={self.helpful_count} harmful={self.harmful_count} :: {self.content}"
    
    def archive(self, reason: str) -> None:
        """Mark bullet as archived."""
        self.archived = True
        self.archived_at = datetime.utcnow().isoformat()
        self.archive_reason = reason


@dataclass
class OperationResult:
    """Result of applying a playbook operation."""
    success: bool
    operation_type: str
    bullet_id: Optional[str] = None
    message: str = ""
    affected_bullets: List[str] = field(default_factory=list)


@dataclass
class Playbook:
    """
    The evolving playbook containing accumulated domain knowledge.
    
    Structured into sections: strategies, pitfalls, templates, definitions, code_snippets
    Also maintains an archive of removed/merged bullets.
    """
    strategies: List[Bullet] = field(default_factory=list)
    pitfalls: List[Bullet] = field(default_factory=list)
    templates: List[Bullet] = field(default_factory=list)
    definitions: List[Bullet] = field(default_factory=list)
    code_snippets: List[Bullet] = field(default_factory=list)
    archived_bullets: List[Bullet] = field(default_factory=list)  # Archive for removed bullets
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "version": "1.1",  # Updated version for new features
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "total_updates": 0,
                "total_removes": 0,
                "total_modifies": 0,
                "total_merges": 0
            }
    
    def get_section(self, section_name: str) -> List[Bullet]:
        """Get bullets from a specific section."""
        return getattr(self, section_name, [])
    
    def _get_section_list(self, section: str) -> Optional[List[Bullet]]:
        """Get the actual list object for a section."""
        if section in PLAYBOOK_SECTIONS:
            return getattr(self, section)
        return None
    
    def add_bullet(self, section: str, content: str) -> Bullet:
        """Add a new bullet to the specified section."""
        prefix = SECTION_PREFIXES.get(section, "unk")
        section_list = self.get_section(section)
        
        # Generate unique ID
        counter = len(section_list) + 1
        bullet_id = f"{prefix}-{counter:05d}"
        
        # Check for ID collision and increment if needed
        existing_ids = {b.id for b in section_list}
        while bullet_id in existing_ids:
            counter += 1
            bullet_id = f"{prefix}-{counter:05d}"
        
        bullet = Bullet(id=bullet_id, content=content)
        section_list.append(bullet)
        
        self.metadata["updated_at"] = datetime.utcnow().isoformat()
        self.metadata["total_updates"] = self.metadata.get("total_updates", 0) + 1
        
        return bullet
    
    def remove_bullet(self, bullet_id: str, reason: str = "Removed by Curator", 
                      archive: bool = True) -> Optional[Bullet]:
        """
        Remove a bullet from the playbook.
        
        Args:
            bullet_id: ID of the bullet to remove
            reason: Reason for removal
            archive: If True, move to archive instead of deleting
        
        Returns:
            The removed bullet, or None if not found
        """
        for section in PLAYBOOK_SECTIONS:
            section_list = self._get_section_list(section)
            if section_list is None:
                continue
            
            for i, bullet in enumerate(section_list):
                if bullet.id == bullet_id:
                    removed = section_list.pop(i)
                    
                    if archive:
                        removed.archive(reason)
                        self.archived_bullets.append(removed)
                    
                    self.metadata["updated_at"] = datetime.utcnow().isoformat()
                    self.metadata["total_removes"] = self.metadata.get("total_removes", 0) + 1
                    
                    return removed
        
        return None
    
    def modify_bullet(self, bullet_id: str, new_content: str, 
                      reason: str = "Modified by Curator",
                      reset_harmful: bool = False) -> Optional[Bullet]:
        """
        Modify an existing bullet's content.
        
        Args:
            bullet_id: ID of the bullet to modify
            new_content: New content for the bullet
            reason: Reason for modification (logged in metadata)
            reset_harmful: If True, reset harmful count to 0
        
        Returns:
            The modified bullet, or None if not found
        """
        bullet = self.get_bullet_by_id(bullet_id)
        if bullet is None:
            return None
        
        # Update content
        bullet.content = new_content
        bullet.updated_at = datetime.utcnow().isoformat()
        bullet.revision_count += 1
        
        if reset_harmful:
            bullet.harmful_count = 0
        
        self.metadata["updated_at"] = datetime.utcnow().isoformat()
        self.metadata["total_modifies"] = self.metadata.get("total_modifies", 0) + 1
        
        return bullet
    
    def merge_bullets(self, source_ids: List[str], target_section: str, 
                      merged_content: str, reason: str = "Merged by Curator",
                      archive_sources: bool = True) -> Optional[Bullet]:
        """
        Merge multiple bullets into a single new bullet.
        
        Args:
            source_ids: List of bullet IDs to merge
            target_section: Section for the new merged bullet
            merged_content: Content of the merged bullet
            reason: Reason for merging
            archive_sources: If True, archive source bullets instead of deleting
        
        Returns:
            The new merged bullet, or None if operation failed
        """
        if len(source_ids) < 2:
            return None
        
        # Collect source bullets
        source_bullets = []
        for bullet_id in source_ids:
            bullet = self.get_bullet_by_id(bullet_id)
            if bullet:
                source_bullets.append(bullet)
        
        if len(source_bullets) < 2:
            return None
        
        # Sum up counts from sources
        total_helpful = sum(b.helpful_count for b in source_bullets)
        total_harmful = sum(b.harmful_count for b in source_bullets)
        total_neutral = sum(b.neutral_count for b in source_bullets)
        
        # Create new merged bullet
        prefix = SECTION_PREFIXES.get(target_section, "unk")
        section_list = self.get_section(target_section)
        counter = len(section_list) + 1
        bullet_id = f"{prefix}-{counter:05d}"
        
        existing_ids = {b.id for b in section_list}
        while bullet_id in existing_ids:
            counter += 1
            bullet_id = f"{prefix}-{counter:05d}"
        
        merged_bullet = Bullet(
            id=bullet_id,
            content=merged_content,
            helpful_count=total_helpful,
            harmful_count=total_harmful,
            neutral_count=total_neutral,
            merged_from=source_ids
        )
        
        section_list.append(merged_bullet)
        
        # Remove source bullets
        for bullet_id in source_ids:
            self.remove_bullet(bullet_id, reason=f"Merged into {merged_bullet.id}", 
                             archive=archive_sources)
        
        self.metadata["updated_at"] = datetime.utcnow().isoformat()
        self.metadata["total_merges"] = self.metadata.get("total_merges", 0) + 1
        
        return merged_bullet
    
    def get_bullet_by_id(self, bullet_id: str) -> Optional[Bullet]:
        """Find a bullet by its ID across all sections."""
        for section in PLAYBOOK_SECTIONS:
            for bullet in self.get_section(section):
                if bullet.id == bullet_id:
                    return bullet
        return None
    
    def get_archived_bullet_by_id(self, bullet_id: str) -> Optional[Bullet]:
        """Find an archived bullet by ID."""
        for bullet in self.archived_bullets:
            if bullet.id == bullet_id:
                return bullet
        return None
    
    def restore_bullet(self, bullet_id: str, target_section: str = None) -> Optional[Bullet]:
        """
        Restore an archived bullet.
        
        Args:
            bullet_id: ID of the archived bullet
            target_section: Section to restore to (uses original section by default)
        
        Returns:
            The restored bullet, or None if not found
        """
        for i, bullet in enumerate(self.archived_bullets):
            if bullet.id == bullet_id:
                restored = self.archived_bullets.pop(i)
                restored.archived = False
                restored.archived_at = None
                restored.archive_reason = None
                
                # Determine target section from ID prefix
                if target_section is None:
                    prefix = bullet_id.split("-")[0]
                    for section, p in SECTION_PREFIXES.items():
                        if p == prefix:
                            target_section = section
                            break
                    target_section = target_section or "strategies"
                
                section_list = self._get_section_list(target_section)
                if section_list is not None:
                    section_list.append(restored)
                
                return restored
        
        return None
    
    def update_bullet_tags(self, bullet_tags: List[Dict[str, str]]):
        """Update bullet helpful/harmful counts based on tags."""
        for tag_info in bullet_tags:
            bullet_id = tag_info.get("id")
            tag = tag_info.get("tag", "neutral")
            
            bullet = self.get_bullet_by_id(bullet_id)
            if bullet:
                if tag == "helpful":
                    bullet.mark_helpful()
                elif tag == "harmful":
                    bullet.mark_harmful()
                else:
                    bullet.mark_neutral()
    
    def get_bullets_for_auto_removal(self, harmful_threshold: int = 5, 
                                     effectiveness_threshold: float = -0.3) -> List[Bullet]:
        """
        Get bullets that should be considered for automatic removal.
        
        Args:
            harmful_threshold: Minimum harmful count to trigger
            effectiveness_threshold: Maximum effectiveness score to trigger
        
        Returns:
            List of bullets that meet removal criteria
        """
        candidates = []
        for section in PLAYBOOK_SECTIONS:
            for bullet in self.get_section(section):
                if bullet.harmful_count >= harmful_threshold:
                    candidates.append(bullet)
                elif bullet.effectiveness_score <= effectiveness_threshold and bullet.total_usage >= 3:
                    candidates.append(bullet)
        return candidates
    
    def get_all_bullets(self) -> List[Bullet]:
        """Get all bullets across all sections."""
        all_bullets = []
        for section in PLAYBOOK_SECTIONS:
            all_bullets.extend(self.get_section(section))
        return all_bullets
    
    def format_for_prompt(self) -> str:
        """Format the entire playbook for inclusion in LLM prompts."""
        sections = []
        
        for section_name in PLAYBOOK_SECTIONS:
            section_bullets = self.get_section(section_name)
            if section_bullets:
                header = section_name.upper().replace("_", " ")
                bullets_text = "\n".join(b.format_for_prompt() for b in section_bullets)
                sections.append(f"### {header}\n{bullets_text}")
        
        if not sections:
            return "(Playbook is empty - no accumulated knowledge yet)"
        
        return "\n\n".join(sections)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the playbook."""
        stats = {
            "total_bullets": 0,
            "sections": {},
            "archived_count": len(self.archived_bullets)
        }
        for section in PLAYBOOK_SECTIONS:
            bullets = self.get_section(section)
            stats["sections"][section] = len(bullets)
            stats["total_bullets"] += len(bullets)
        
        stats["metadata"] = self.metadata
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert playbook to dictionary for JSON serialization."""
        return {
            "strategies": [b.to_dict() for b in self.strategies],
            "pitfalls": [b.to_dict() for b in self.pitfalls],
            "templates": [b.to_dict() for b in self.templates],
            "definitions": [b.to_dict() for b in self.definitions],
            "code_snippets": [b.to_dict() for b in self.code_snippets],
            "archived_bullets": [b.to_dict() for b in self.archived_bullets],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Playbook":
        """Create playbook from dictionary."""
        playbook = cls()
        
        for section in PLAYBOOK_SECTIONS:
            section_data = data.get(section, [])
            section_list = []
            for bullet_data in section_data:
                section_list.append(Bullet.from_dict(bullet_data))
            setattr(playbook, section, section_list)
        
        # Load archived bullets
        archived_data = data.get("archived_bullets", [])
        playbook.archived_bullets = [Bullet.from_dict(b) for b in archived_data]
        
        playbook.metadata = data.get("metadata", {})
        return playbook


class PlaybookManager:
    """
    Manages playbook persistence and thread-safe operations.
    
    Supports all CRUD operations: ADD, REMOVE, MODIFY, MERGE
    """
    
    def __init__(self, config: PlaybookConfig = None):
        self.config = config or PlaybookConfig()
        self._playbook: Optional[Playbook] = None
        self._lock = threading.RLock()
        
        # Optional retriever integration
        self._retriever = None
    
    def set_retriever(self, retriever) -> None:
        """Set the retriever for embedding updates."""
        self._retriever = retriever
    
    def load(self) -> Playbook:
        """Load playbook from disk or create empty one."""
        with self._lock:
            path = Path(self.config.path)
            
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._playbook = Playbook.from_dict(data)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Could not load playbook, creating new one: {e}")
                    self._playbook = Playbook()
            else:
                self._playbook = Playbook()
            
            return self._playbook
    
    def save(self):
        """Save playbook to disk."""
        with self._lock:
            if self._playbook is None:
                return
            
            path = Path(self.config.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._playbook.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_playbook(self) -> Playbook:
        """Get the current playbook, loading if necessary."""
        with self._lock:
            if self._playbook is None:
                return self.load()
            return self._playbook
    
    def apply_operations(self, operations: List[Dict[str, Any]]) -> List[OperationResult]:
        """
        Apply curator operations to the playbook.
        
        Supports: ADD, REMOVE, MODIFY, MERGE
        
        Returns:
            List of OperationResult objects
        """
        results = []
        
        with self._lock:
            playbook = self.get_playbook()
            
            for op in operations:
                op_type = op.get("type", "").upper()
                
                if op_type == "ADD":
                    result = self._apply_add(playbook, op)
                elif op_type == "REMOVE":
                    result = self._apply_remove(playbook, op)
                elif op_type == "MODIFY":
                    result = self._apply_modify(playbook, op)
                elif op_type == "MERGE":
                    result = self._apply_merge(playbook, op)
                else:
                    result = OperationResult(
                        success=False,
                        operation_type=op_type,
                        message=f"Unknown operation type: {op_type}"
                    )
                
                results.append(result)
            
            self.save()
        
        return results
    
    def _apply_add(self, playbook: Playbook, op: Dict[str, Any]) -> OperationResult:
        """Apply ADD operation."""
        section = op.get("section", "strategies")
        content = op.get("content", "")
        
        if not content.strip():
            return OperationResult(
                success=False,
                operation_type="ADD",
                message="Empty content"
            )
        
        if section not in PLAYBOOK_SECTIONS:
            section = "strategies"
        
        bullet = playbook.add_bullet(section, content.strip())
        
        # Update retriever index
        if self._retriever:
            self._retriever.add_bullet(
                bullet_id=bullet.id,
                content=bullet.content,
                section=section
            )
        
        return OperationResult(
            success=True,
            operation_type="ADD",
            bullet_id=bullet.id,
            message=f"Added bullet {bullet.id} to {section}",
            affected_bullets=[bullet.id]
        )
    
    def _apply_remove(self, playbook: Playbook, op: Dict[str, Any]) -> OperationResult:
        """Apply REMOVE operation."""
        bullet_id = op.get("bullet_id")
        reason = op.get("reason", "Removed by Curator")
        
        if not bullet_id:
            return OperationResult(
                success=False,
                operation_type="REMOVE",
                message="No bullet_id provided"
            )
        
        removed = playbook.remove_bullet(bullet_id, reason=reason, archive=True)
        
        if removed is None:
            return OperationResult(
                success=False,
                operation_type="REMOVE",
                bullet_id=bullet_id,
                message=f"Bullet {bullet_id} not found"
            )
        
        # Update retriever index
        if self._retriever:
            self._retriever.remove_bullet(bullet_id)
        
        return OperationResult(
            success=True,
            operation_type="REMOVE",
            bullet_id=bullet_id,
            message=f"Removed and archived bullet {bullet_id}: {reason}",
            affected_bullets=[bullet_id]
        )
    
    def _apply_modify(self, playbook: Playbook, op: Dict[str, Any]) -> OperationResult:
        """Apply MODIFY operation."""
        bullet_id = op.get("bullet_id")
        new_content = op.get("new_content", "")
        reason = op.get("reason", "Modified by Curator")
        reset_harmful = op.get("reset_harmful", False)
        
        if not bullet_id or not new_content.strip():
            return OperationResult(
                success=False,
                operation_type="MODIFY",
                message="Missing bullet_id or new_content"
            )
        
        modified = playbook.modify_bullet(
            bullet_id, 
            new_content.strip(), 
            reason=reason,
            reset_harmful=reset_harmful
        )
        
        if modified is None:
            return OperationResult(
                success=False,
                operation_type="MODIFY",
                bullet_id=bullet_id,
                message=f"Bullet {bullet_id} not found"
            )
        
        # Update retriever index
        if self._retriever:
            self._retriever.update_bullet(bullet_id, new_content.strip())
        
        return OperationResult(
            success=True,
            operation_type="MODIFY",
            bullet_id=bullet_id,
            message=f"Modified bullet {bullet_id} (revision {modified.revision_count}): {reason}",
            affected_bullets=[bullet_id]
        )
    
    def _apply_merge(self, playbook: Playbook, op: Dict[str, Any]) -> OperationResult:
        """Apply MERGE operation."""
        source_ids = op.get("source_bullet_ids", [])
        target_section = op.get("target_section", "strategies")
        merged_content = op.get("merged_content", "")
        reason = op.get("reason", "Merged by Curator")
        
        if len(source_ids) < 2:
            return OperationResult(
                success=False,
                operation_type="MERGE",
                message="Need at least 2 source bullets to merge"
            )
        
        if not merged_content.strip():
            return OperationResult(
                success=False,
                operation_type="MERGE",
                message="Missing merged_content"
            )
        
        if target_section not in PLAYBOOK_SECTIONS:
            target_section = "strategies"
        
        merged = playbook.merge_bullets(
            source_ids,
            target_section,
            merged_content.strip(),
            reason=reason
        )
        
        if merged is None:
            return OperationResult(
                success=False,
                operation_type="MERGE",
                message=f"Could not merge bullets: {source_ids}"
            )
        
        # Update retriever index
        if self._retriever:
            for source_id in source_ids:
                self._retriever.remove_bullet(source_id)
            self._retriever.add_bullet(
                bullet_id=merged.id,
                content=merged.content,
                section=target_section,
                helpful_count=merged.helpful_count,
                harmful_count=merged.harmful_count
            )
        
        return OperationResult(
            success=True,
            operation_type="MERGE",
            bullet_id=merged.id,
            message=f"Merged {source_ids} into {merged.id}",
            affected_bullets=source_ids + [merged.id]
        )
    
    def update_tags(self, bullet_tags: List[Dict[str, str]]):
        """Update bullet tags based on reflector feedback."""
        with self._lock:
            playbook = self.get_playbook()
            playbook.update_bullet_tags(bullet_tags)
            self.save()
    
    def auto_cleanup(self, harmful_threshold: int = 5,
                     effectiveness_threshold: float = -0.3) -> List[str]:
        """
        Automatically remove bullets that are consistently harmful.
        
        Returns list of removed bullet IDs.
        """
        with self._lock:
            playbook = self.get_playbook()
            candidates = playbook.get_bullets_for_auto_removal(
                harmful_threshold, effectiveness_threshold
            )
            
            removed_ids = []
            for bullet in candidates:
                reason = f"Auto-removed: harmful={bullet.harmful_count}, effectiveness={bullet.effectiveness_score:.2f}"
                if playbook.remove_bullet(bullet.id, reason=reason, archive=True):
                    removed_ids.append(bullet.id)
                    if self._retriever:
                        self._retriever.remove_bullet(bullet.id)
            
            if removed_ids:
                self.save()
            
            return removed_ids


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.
    
    This is a simple implementation using word overlap.
    For production, use sentence-transformers or similar.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def deduplicate_playbook(playbook: Playbook, threshold: float = 0.85) -> List[str]:
    """
    Remove semantically similar bullets from the playbook.
    
    Returns list of removed bullet IDs.
    """
    removed_ids = []
    
    for section in PLAYBOOK_SECTIONS:
        section_list = playbook.get_section(section)
        if len(section_list) < 2:
            continue
        
        to_remove = set()
        
        for i, bullet1 in enumerate(section_list):
            if bullet1.id in to_remove:
                continue
            
            for bullet2 in section_list[i+1:]:
                if bullet2.id in to_remove:
                    continue
                
                similarity = compute_semantic_similarity(bullet1.content, bullet2.content)
                
                if similarity >= threshold:
                    # Keep the one with better effectiveness score
                    if bullet1.effectiveness_score >= bullet2.effectiveness_score:
                        to_remove.add(bullet2.id)
                    else:
                        to_remove.add(bullet1.id)
                        break
        
        # Remove duplicates (archive them)
        for bullet_id in to_remove:
            playbook.remove_bullet(bullet_id, reason="Deduplicated", archive=True)
        
        removed_ids.extend(to_remove)
    
    return removed_ids