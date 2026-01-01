"""
Playbook Module for ACORD Clause Extraction.

The playbook stores and manages evolving knowledge about clause extraction:
- Strategies: Approaches for finding and extracting clauses
- Definitions: Clause type definitions and characteristics
- Pitfalls: Common mistakes to avoid
- Templates: Patterns and examples for clauses
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


@dataclass
class PlaybookBullet:
    """A single bullet point in the playbook."""
    id: str
    content: str
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "initial"  # 'initial', 'learned', 'user'
    confidence: float = 1.0
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlaybookBullet':
        return cls(**data)
    
    def matches_tags(self, query_tags: List[str]) -> bool:
        """Check if bullet matches any of the query tags."""
        if not query_tags:
            return True
        return bool(set(self.tags) & set(query_tags))


@dataclass
class Playbook:
    """The evolving playbook containing extraction knowledge."""
    strategies: List[PlaybookBullet] = field(default_factory=list)
    definitions: List[PlaybookBullet] = field(default_factory=list)
    pitfalls: List[PlaybookBullet] = field(default_factory=list)
    templates: List[PlaybookBullet] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                'version': '1.0.0',
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
    
    def get_section(self, section: str) -> List[PlaybookBullet]:
        """Get bullets from a specific section."""
        sections = {
            'strategies': self.strategies,
            'definitions': self.definitions,
            'pitfalls': self.pitfalls,
            'templates': self.templates
        }
        return sections.get(section, [])
    
    def add_bullet(self, section: str, content: str, tags: List[str] = None,
                   source: str = "learned", confidence: float = 1.0) -> PlaybookBullet:
        """Add a new bullet to a section."""
        section_list = self.get_section(section)
        
        # Generate unique ID
        bullet_id = self._generate_id(section, content)
        
        bullet = PlaybookBullet(
            id=bullet_id,
            content=content,
            tags=tags or [],
            source=source,
            confidence=confidence
        )
        
        section_list.append(bullet)
        self.metadata['updated_at'] = datetime.utcnow().isoformat()
        
        return bullet
    
    def _generate_id(self, section: str, content: str) -> str:
        """Generate unique ID for bullet."""
        prefix = section[:3]
        hash_input = f"{section}:{content}:{datetime.utcnow().isoformat()}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        # Count existing bullets in section to get sequence number
        section_bullets = self.get_section(section)
        seq_num = len(section_bullets) + 1
        
        return f"{prefix}-{seq_num:05d}-{hash_suffix}"
    
    def find_by_id(self, bullet_id: str) -> Optional[PlaybookBullet]:
        """Find a bullet by its ID."""
        for section in [self.strategies, self.definitions, self.pitfalls, self.templates]:
            for bullet in section:
                if bullet.id == bullet_id:
                    return bullet
        return None
    
    def remove_bullet(self, bullet_id: str) -> bool:
        """Remove a bullet by ID."""
        for section in [self.strategies, self.definitions, self.pitfalls, self.templates]:
            for i, bullet in enumerate(section):
                if bullet.id == bullet_id:
                    section.pop(i)
                    self.metadata['updated_at'] = datetime.utcnow().isoformat()
                    return True
        return False
    
    def update_bullet(self, bullet_id: str, content: str = None, 
                      tags: List[str] = None) -> bool:
        """Update a bullet's content or tags."""
        bullet = self.find_by_id(bullet_id)
        if bullet:
            if content is not None:
                bullet.content = content
            if tags is not None:
                bullet.tags = tags
            bullet.updated_at = datetime.utcnow().isoformat()
            self.metadata['updated_at'] = datetime.utcnow().isoformat()
            return True
        return False
    
    def get_all_bullets(self) -> List[PlaybookBullet]:
        """Get all bullets from all sections."""
        return self.strategies + self.definitions + self.pitfalls + self.templates
    
    def search(self, query: str, section: str = None, 
               tags: List[str] = None) -> List[PlaybookBullet]:
        """Search bullets by content and/or tags."""
        if section:
            bullets = self.get_section(section)
        else:
            bullets = self.get_all_bullets()
        
        results = []
        query_lower = query.lower()
        
        for bullet in bullets:
            # Check content match
            if query_lower in bullet.content.lower():
                if tags is None or bullet.matches_tags(tags):
                    results.append(bullet)
        
        return results
    
    def get_relevant_bullets(self, clause_type: str, top_k: int = 10) -> List[PlaybookBullet]:
        """Get bullets relevant to a specific clause type."""
        all_bullets = self.get_all_bullets()
        
        # Score bullets by relevance
        scored = []
        clause_type_lower = clause_type.lower()
        clause_words = set(clause_type_lower.split())
        
        for bullet in all_bullets:
            score = 0.0
            content_lower = bullet.content.lower()
            
            # Check if clause type appears in content
            if clause_type_lower in content_lower:
                score += 2.0
            
            # Check for word overlap
            content_words = set(re.findall(r'\w+', content_lower))
            overlap = len(clause_words & content_words)
            score += overlap * 0.5
            
            # Check tags
            for tag in bullet.tags:
                if any(word in tag.lower() for word in clause_words):
                    score += 1.0
            
            # Boost based on confidence and usage
            score *= bullet.confidence
            score += bullet.usage_count * 0.1
            
            if score > 0:
                scored.append((bullet, score))
        
        # Sort by score and return top_k
        scored.sort(key=lambda x: -x[1])
        return [bullet for bullet, score in scored[:top_k]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get playbook statistics."""
        return {
            'total_bullets': len(self.get_all_bullets()),
            'sections': {
                'strategies': len(self.strategies),
                'definitions': len(self.definitions),
                'pitfalls': len(self.pitfalls),
                'templates': len(self.templates)
            },
            'by_source': self._count_by_source(),
            'metadata': self.metadata
        }
    
    def _count_by_source(self) -> Dict[str, int]:
        """Count bullets by source."""
        counts = {}
        for bullet in self.get_all_bullets():
            counts[bullet.source] = counts.get(bullet.source, 0) + 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert playbook to dictionary."""
        return {
            'strategies': [b.to_dict() for b in self.strategies],
            'definitions': [b.to_dict() for b in self.definitions],
            'pitfalls': [b.to_dict() for b in self.pitfalls],
            'templates': [b.to_dict() for b in self.templates],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Playbook':
        """Create playbook from dictionary."""
        return cls(
            strategies=[PlaybookBullet.from_dict(b) for b in data.get('strategies', [])],
            definitions=[PlaybookBullet.from_dict(b) for b in data.get('definitions', [])],
            pitfalls=[PlaybookBullet.from_dict(b) for b in data.get('pitfalls', [])],
            templates=[PlaybookBullet.from_dict(b) for b in data.get('templates', [])],
            metadata=data.get('metadata', {})
        )
    
    def to_prompt_context(self, clause_type: str = None, max_bullets: int = 15) -> str:
        """Format playbook as context for LLM prompts."""
        if clause_type:
            bullets = self.get_relevant_bullets(clause_type, top_k=max_bullets)
        else:
            bullets = self.get_all_bullets()[:max_bullets]
        
        if not bullets:
            return "No relevant playbook entries available."
        
        sections = {'strategies': [], 'definitions': [], 'pitfalls': [], 'templates': []}
        
        for bullet in bullets:
            # Determine section from ID prefix
            if bullet.id.startswith('str'):
                sections['strategies'].append(bullet)
            elif bullet.id.startswith('def'):
                sections['definitions'].append(bullet)
            elif bullet.id.startswith('pit'):
                sections['pitfalls'].append(bullet)
            elif bullet.id.startswith('tem'):
                sections['templates'].append(bullet)
        
        context_parts = []
        
        if sections['definitions']:
            context_parts.append("## Definitions")
            for b in sections['definitions']:
                context_parts.append(f"- [{b.id}] {b.content}")
        
        if sections['strategies']:
            context_parts.append("\n## Strategies")
            for b in sections['strategies']:
                context_parts.append(f"- [{b.id}] {b.content}")
        
        if sections['pitfalls']:
            context_parts.append("\n## Pitfalls to Avoid")
            for b in sections['pitfalls']:
                context_parts.append(f"- [{b.id}] {b.content}")
        
        if sections['templates']:
            context_parts.append("\n## Templates/Patterns")
            for b in sections['templates']:
                context_parts.append(f"- [{b.id}] {b.content}")
        
        return "\n".join(context_parts)


class PlaybookManager:
    """Manager for loading, saving, and manipulating playbooks."""
    
    def __init__(self, path: str = "playbook.json"):
        self.path = Path(path)
        self.playbook = Playbook()
        
        if self.path.exists():
            self.load()
    
    def load(self) -> Playbook:
        """Load playbook from file."""
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.playbook = Playbook.from_dict(data)
        return self.playbook
    
    def save(self) -> None:
        """Save playbook to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.playbook.to_dict(), f, indent=2, ensure_ascii=False)
    
    def reset(self) -> None:
        """Reset playbook to empty state."""
        self.playbook = Playbook()
        self.save()
    
    def get_playbook(self) -> Playbook:
        """Get current playbook."""
        return self.playbook
    
    def get_stats(self) -> Dict[str, Any]:
        """Get playbook statistics."""
        return self.playbook.get_stats()


def create_initial_acord_playbook() -> Playbook:
    """Create initial playbook with domain knowledge for legal clause extraction."""
    playbook = Playbook()
    
    # Add strategies
    strategies = [
        ("When extracting clauses, first scan for section headers that match or relate to the clause type (e.g., 'TERMINATION', 'GOVERNING LAW').", 
         ["clause_identification", "document_structure"]),
        ("Pay attention to enumerated lists (a), (b), (c) or numbered sections as they often contain detailed clause provisions.",
         ["clause_structure", "enumeration"]),
        ("Cross-references like 'Section 5.2' or 'pursuant to Article III' indicate related content that may be part of the same clause.",
         ["cross_references", "document_navigation"]),
        ("Look for trigger phrases: 'may terminate' for termination, 'shall be governed by' for governing law, 'shall not exceed' for liability caps.",
         ["trigger_phrases", "clause_identification"]),
        ("When a clause is not explicitly labeled, search for its functional elements based on what the clause type is meant to accomplish.",
         ["implicit_clauses", "functional_analysis"]),
    ]
    
    for content, tags in strategies:
        playbook.add_bullet("strategies", content, tags, source="initial")
    
    # Add definitions
    definitions = [
        ("Termination For Convenience: Allows ending the agreement without cause, typically requiring advance notice (e.g., 30 or 60 days).",
         ["termination", "clause_types"]),
        ("Governing Law: Specifies which jurisdiction's laws apply. Usually states 'shall be governed by and construed in accordance with the laws of [State/Country]'.",
         ["jurisdiction", "clause_types"]),
        ("Change of Control: Triggered when ownership/control changes, often allowing termination or requiring consent.",
         ["corporate_events", "clause_types"]),
        ("Cap On Liability: Limits maximum damages, often as a fixed amount or multiple of fees paid.",
         ["liability", "clause_types"]),
        ("Non-Transferable License: Prohibits assigning or sublicensing rights to third parties.",
         ["licensing", "clause_types"]),
        ("IP Ownership Assignment: Transfers intellectual property rights, typically for work product created under the agreement.",
         ["intellectual_property", "clause_types"]),
    ]
    
    for content, tags in definitions:
        playbook.add_bullet("definitions", content, tags, source="initial")
    
    # Add pitfalls
    pitfalls = [
        ("Do not confuse 'Governing Law' (jurisdiction choice) with 'Dispute Resolution' (arbitration/litigation process).",
         ["common_mistakes", "clause_distinction"]),
        ("Do not confuse 'Cap on Liability' (maximum amount) with 'Limitation of Liability' (excluded damage types).",
         ["common_mistakes", "liability"]),
        ("Termination 'for convenience' differs from 'for cause' - convenience allows ending without breach.",
         ["common_mistakes", "termination"]),
        ("Include complete provisions with exceptions and qualifications, not just the main rule.",
         ["extraction_quality", "completeness"]),
    ]
    
    for content, tags in pitfalls:
        playbook.add_bullet("pitfalls", content, tags, source="initial")
    
    # Add templates
    templates = [
        ("Termination For Convenience pattern: '[Party] may terminate this Agreement [upon/with] [X] days [prior] written notice [without cause].'",
         ["termination", "pattern"]),
        ("Governing Law pattern: 'This Agreement shall be governed by and construed in accordance with the laws of [State/Country], without regard to conflict of laws principles.'",
         ["governing_law", "pattern"]),
        ("Cap On Liability pattern: 'In no event shall [Party's] liability exceed [the fees paid/amount] [in the preceding 12 months].'",
         ["liability", "pattern"]),
    ]
    
    for content, tags in templates:
        playbook.add_bullet("templates", content, tags, source="initial")
    
    return playbook
