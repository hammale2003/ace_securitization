"""
Knowledge Classifier Agent for Trainer Mode.

Categorizes extracted knowledge, assigns confidence scores,
detects duplicates, and prioritizes by importance.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from llm_client import LLMClient
from embeddings import EmbeddingModel, cosine_similarity_matrix
from playbook import Playbook
from trainer.knowledge_extractor import ExtractedKnowledge


@dataclass
class ClassifiedKnowledge:
    """Represents classified and scored knowledge."""
    extracted: ExtractedKnowledge
    final_section: str  # Final playbook section assignment
    final_confidence: float  # Adjusted confidence after classification
    importance_score: float  # Priority score for enrichment
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None  # Bullet ID if duplicate
    similarity_score: float = 0.0  # Similarity to existing bullets
    classification_reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = self.extracted.to_dict()
        result.update({
            "final_section": self.final_section,
            "final_confidence": self.final_confidence,
            "importance_score": self.importance_score,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "similarity_score": self.similarity_score,
            "classification_reasoning": self.classification_reasoning
        })
        return result


class KnowledgeClassifier:
    """
    Classifies extracted knowledge and detects duplicates.
    
    - Validates section assignment
    - Adjusts confidence scores
    - Detects duplicates using semantic similarity
    - Prioritizes knowledge by importance
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        embedding_model: EmbeddingModel,
        duplicate_threshold: float = 0.85
    ):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.duplicate_threshold = duplicate_threshold
    
    def classify_batch(
        self,
        extracted_items: List[ExtractedKnowledge],
        existing_playbook: Playbook
    ) -> List[ClassifiedKnowledge]:
        """
        Classify a batch of extracted knowledge items.
        
        Args:
            extracted_items: List of extracted knowledge
            existing_playbook: Current playbook for duplicate detection
        
        Returns:
            List of classified knowledge items
        """
        classified = []
        
        for item in extracted_items:
            classified_item = self.classify_single(item, existing_playbook)
            classified.append(classified_item)
        
        return classified
    
    def classify_single(
        self,
        extracted: ExtractedKnowledge,
        existing_playbook: Playbook
    ) -> ClassifiedKnowledge:
        """
        Classify a single extracted knowledge item.
        
        Args:
            extracted: Extracted knowledge item
            existing_playbook: Current playbook
        
        Returns:
            Classified knowledge item
        """
        # 1. Validate section assignment
        final_section = self._validate_section(extracted)
        
        # 2. Adjust confidence based on quality checks
        final_confidence = self._adjust_confidence(extracted)
        
        # 3. Calculate importance score
        importance_score = self._calculate_importance(extracted)
        
        # 4. Check for duplicates
        is_duplicate, duplicate_of, similarity = self._check_duplicate(
            extracted,
            existing_playbook,
            final_section
        )
        
        # 5. Generate reasoning
        reasoning = self._generate_classification_reasoning(
            extracted,
            final_section,
            final_confidence,
            importance_score,
            is_duplicate
        )
        
        return ClassifiedKnowledge(
            extracted=extracted,
            final_section=final_section,
            final_confidence=final_confidence,
            importance_score=importance_score,
            is_duplicate=is_duplicate,
            duplicate_of=duplicate_of,
            similarity_score=similarity,
            classification_reasoning=reasoning
        )
    
    def _validate_section(self, extracted: ExtractedKnowledge) -> str:
        """
        Validate and potentially correct the section assignment.
        
        Args:
            extracted: Extracted knowledge
        
        Returns:
            Validated section name
        """
        # Map extraction types to playbook sections
        type_to_section = {
            "strategies": "strategies",
            "definitions": "definitions",
            "templates": "templates",
            "pitfalls": "pitfalls",
            "code_snippets": "code_snippets"
        }
        
        return type_to_section.get(extracted.knowledge_type, "strategies")
    
    def _adjust_confidence(self, extracted: ExtractedKnowledge) -> float:
        """
        Adjust confidence based on quality indicators.
        
        Args:
            extracted: Extracted knowledge
        
        Returns:
            Adjusted confidence score
        """
        confidence = extracted.confidence
        
        # Boost confidence if:
        # - Has specific examples
        if extracted.examples:
            confidence = min(1.0, confidence + 0.05)
        
        # - Has related terms (shows comprehensiveness)
        if len(extracted.related_terms) >= 3:
            confidence = min(1.0, confidence + 0.05)
        
        # - Content is substantial (not too short)
        if len(extracted.content) >= 100:
            confidence = min(1.0, confidence + 0.05)
        
        # Reduce confidence if:
        # - Content is too short
        if len(extracted.content) < 30:
            confidence = max(0.0, confidence - 0.2)
        
        # - No examples
        if not extracted.examples:
            confidence = max(0.0, confidence - 0.05)
        
        # - Contains vague terms
        vague_terms = ["may", "could", "possibly", "perhaps", "unclear"]
        if any(term in extracted.content.lower() for term in vague_terms):
            confidence = max(0.0, confidence - 0.1)
        
        return confidence
    
    def _calculate_importance(self, extracted: ExtractedKnowledge) -> float:
        """
        Calculate importance score for prioritization.
        
        Args:
            extracted: Extracted knowledge
        
        Returns:
            Importance score (0.0 to 1.0)
        """
        score = 0.5  # Base score
        
        # Higher importance for:
        # - Strategies and pitfalls (actionable knowledge)
        if extracted.knowledge_type in ["strategies", "pitfalls"]:
            score += 0.2
        
        # - Definitions (foundational knowledge)
        if extracted.knowledge_type == "definitions":
            score += 0.15
        
        # - Templates (reusable patterns)
        if extracted.knowledge_type == "templates":
            score += 0.15
        
        # - High confidence
        score += extracted.confidence * 0.2
        
        # - Multiple examples
        score += min(0.1, len(extracted.examples) * 0.03)
        
        # - Rich context
        if len(extracted.related_terms) >= 3:
            score += 0.1
        
        # - From key sections (Security Trustee, Waterfall, etc.)
        key_terms = ["security trustee", "waterfall", "enforcement", "default", "indemnification"]
        if any(term in extracted.source_clause_title.lower() for term in key_terms):
            score += 0.1
        
        return min(1.0, score)
    
    def _check_duplicate(
        self,
        extracted: ExtractedKnowledge,
        existing_playbook: Playbook,
        section: str
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if extracted knowledge is duplicate of existing bullet.
        
        Args:
            extracted: Extracted knowledge
            existing_playbook: Current playbook
            section: Target section
        
        Returns:
            Tuple of (is_duplicate, duplicate_bullet_id, similarity_score)
        """
        # Get existing bullets in the same section
        existing_bullets = existing_playbook.get_section(section)
        
        if not existing_bullets:
            return False, None, 0.0
        
        # Compute embeddings
        extracted_embedding = self.embedding_model.embed(extracted.content)
        
        # Get embeddings for all existing bullets
        bullet_texts = [bullet.content for bullet in existing_bullets]
        bullet_embeddings = self.embedding_model.embed_batch(bullet_texts)
        
        # Ensure bullet_embeddings is 2D (handle single bullet case)
        if bullet_embeddings.ndim == 1:
            bullet_embeddings = bullet_embeddings.reshape(1, -1)
        
        # Compute similarities in batch (much faster!)
        if len(bullet_embeddings) > 0:
            similarities = cosine_similarity_matrix(extracted_embedding, bullet_embeddings)
            max_idx = int(np.argmax(similarities))
            max_similarity = float(similarities[max_idx])
            most_similar_id = existing_bullets[max_idx].id
        else:
            max_similarity = 0.0
            most_similar_id = None
        
        # Check if duplicate
        is_duplicate = max_similarity >= self.duplicate_threshold
        
        return is_duplicate, most_similar_id if is_duplicate else None, max_similarity
    
    def _generate_classification_reasoning(
        self,
        extracted: ExtractedKnowledge,
        final_section: str,
        final_confidence: float,
        importance_score: float,
        is_duplicate: bool
    ) -> str:
        """Generate human-readable classification reasoning."""
        parts = []
        
        parts.append(f"Classified as '{final_section}'")
        parts.append(f"Confidence: {final_confidence:.2f}")
        parts.append(f"Importance: {importance_score:.2f}")
        
        if is_duplicate:
            parts.append("[DUPLICATE] Duplicate detected")
        else:
            parts.append("[UNIQUE] Unique knowledge")
        
        return " | ".join(parts)

