"""
Playbook data model and management for the ACE system.

The playbook is the evolving context that accumulates domain knowledge.
Each bullet has metadata (id, helpful/harmful counts) and content.
"""
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import threading

from config import PLAYBOOK_SECTIONS, SECTION_PREFIXES, PlaybookConfig


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
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bullet":
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
    
    def format_for_prompt(self) -> str:
        """Format bullet for inclusion in LLM prompts."""
        return f"[{self.id}] helpful={self.helpful_count} harmful={self.harmful_count} :: {self.content}"


@dataclass
class Playbook:
    """
    The evolving playbook containing accumulated domain knowledge.
    
    Structured into sections: strategies, pitfalls, templates, definitions, code_snippets
    """
    strategies: List[Bullet] = field(default_factory=list)
    pitfalls: List[Bullet] = field(default_factory=list)
    templates: List[Bullet] = field(default_factory=list)
    definitions: List[Bullet] = field(default_factory=list)
    code_snippets: List[Bullet] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "total_updates": 0
            }
    
    def get_section(self, section_name: str) -> List[Bullet]:
        """Get bullets from a specific section."""
        return getattr(self, section_name, [])
    
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
    
    def get_bullet_by_id(self, bullet_id: str) -> Optional[Bullet]:
        """Find a bullet by its ID across all sections."""
        for section in PLAYBOOK_SECTIONS:
            for bullet in self.get_section(section):
                if bullet.id == bullet_id:
                    return bullet
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
            "sections": {}
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
        
        playbook.metadata = data.get("metadata", {})
        return playbook


class PlaybookManager:
    """
    Manages playbook persistence and thread-safe operations.
    """
    
    def __init__(self, config: PlaybookConfig = None):
        self.config = config or PlaybookConfig()
        self._playbook: Optional[Playbook] = None
        self._lock = threading.RLock()
    
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
                    print(f"Warning: Could not load playbook, creating new one: {e}")
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
    
    def apply_operations(self, operations: List[Dict[str, Any]]) -> List[Bullet]:
        """
        Apply curator operations to the playbook.
        
        Operations format:
        [
            {"type": "ADD", "section": "strategies", "content": "..."},
            ...
        ]
        """
        added_bullets = []
        
        with self._lock:
            playbook = self.get_playbook()
            
            for op in operations:
                op_type = op.get("type", "").upper()
                section = op.get("section", "strategies")
                content = op.get("content", "")
                
                if op_type == "ADD" and content.strip():
                    # Validate section name
                    if section not in PLAYBOOK_SECTIONS:
                        section = "strategies"  # Default to strategies
                    
                    bullet = playbook.add_bullet(section, content.strip())
                    added_bullets.append(bullet)
            
            self.save()
        
        return added_bullets
    
    def update_tags(self, bullet_tags: List[Dict[str, str]]):
        """Update bullet tags based on reflector feedback."""
        with self._lock:
            playbook = self.get_playbook()
            playbook.update_bullet_tags(bullet_tags)
            self.save()


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
        
        # Remove duplicates
        section_list[:] = [b for b in section_list if b.id not in to_remove]
        removed_ids.extend(to_remove)
    
    return removed_ids
