"""
Knowledge Validator Agent for Trainer Mode.

Validates extracted and classified knowledge for:
- Consistency with existing playbook
- Legal/domain accuracy
- Conflict resolution
- Quality assurance
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from llm_client import LLMClient
from playbook import Playbook
from trainer.knowledge_classifier import ClassifiedKnowledge


@dataclass
class ValidationResult:
    """Result of knowledge validation."""
    classified: ClassifiedKnowledge
    is_valid: bool
    validation_score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = self.classified.to_dict()
        result.update({
            "is_valid": self.is_valid,
            "validation_score": self.validation_score,
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "conflicts": self.conflicts
        })
        return result


class KnowledgeValidator:
    """
    Validates extracted knowledge before adding to playbook.
    
    Performs:
    - Consistency checks with existing knowledge
    - Legal accuracy validation
    - Conflict detection and resolution
    - Quality gates
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        min_validation_score: float = 0.6
    ):
        self.llm_client = llm_client
        self.min_validation_score = min_validation_score
    
    def validate_batch(
        self,
        classified_items: List[ClassifiedKnowledge],
        existing_playbook: Playbook
    ) -> List[ValidationResult]:
        """
        Validate a batch of classified knowledge items.
        
        Args:
            classified_items: List of classified knowledge
            existing_playbook: Current playbook
        
        Returns:
            List of validation results
        """
        validated = []
        
        for item in classified_items:
            result = self.validate_single(item, existing_playbook)
            validated.append(result)
        
        return validated
    
    def validate_single(
        self,
        classified: ClassifiedKnowledge,
        existing_playbook: Playbook
    ) -> ValidationResult:
        """
        Validate a single classified knowledge item.
        
        Args:
            classified: Classified knowledge item
            existing_playbook: Current playbook
        
        Returns:
            Validation result
        """
        issues = []
        warnings = []
        recommendations = []
        conflicts = []
        
        # 1. Basic quality checks
        quality_score = self._check_quality(classified, issues, warnings)
        
        # 2. Consistency checks
        consistency_score = self._check_consistency(
            classified,
            existing_playbook,
            conflicts,
            warnings
        )
        
        # 3. Domain accuracy checks
        accuracy_score = self._check_accuracy(classified, issues, warnings)
        
        # 4. Generate recommendations
        self._generate_recommendations(classified, recommendations)
        
        # Calculate overall validation score
        validation_score = (quality_score + consistency_score + accuracy_score) / 3.0
        
        # Determine if valid
        is_valid = (
            validation_score >= self.min_validation_score and
            len(issues) == 0 and
            not classified.is_duplicate
        )
        
        return ValidationResult(
            classified=classified,
            is_valid=is_valid,
            validation_score=validation_score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            conflicts=conflicts
        )
    
    def _check_quality(
        self,
        classified: ClassifiedKnowledge,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """
        Check basic quality of extracted knowledge.
        
        Returns quality score (0.0 to 1.0)
        """
        score = 1.0
        content = classified.extracted.content
        
        # Check minimum length
        if len(content) < 20:
            issues.append("Content too short (< 20 characters)")
            score -= 0.5
        
        # Check maximum length
        if len(content) > 1000:
            warnings.append("Content very long (> 1000 characters) - consider splitting")
            score -= 0.1
        
        # Check for placeholder text
        placeholders = ["TODO", "TBD", "FIXME", "[...]", "XXX"]
        if any(p in content for p in placeholders):
            issues.append("Contains placeholder text")
            score -= 0.3
        
        # Check confidence threshold
        if classified.final_confidence < 0.5:
            warnings.append(f"Low confidence: {classified.final_confidence:.2f}")
            score -= 0.2
        
        # Check for complete sentences
        if not content.strip().endswith(('.', '!', '?', '"', "'")):
            warnings.append("Content doesn't end with proper punctuation")
            score -= 0.1
        
        # Check for meaningful content (not just definitions without context)
        if classified.final_section == "definitions" and ":" not in content and "means" not in content.lower():
            warnings.append("Definition format unclear - consider adding 'means' or ':' structure")
            score -= 0.1
        
        return max(0.0, score)
    
    def _check_consistency(
        self,
        classified: ClassifiedKnowledge,
        existing_playbook: Playbook,
        conflicts: List[Dict[str, str]],
        warnings: List[str]
    ) -> float:
        """
        Check consistency with existing playbook knowledge.
        
        Returns consistency score (0.0 to 1.0)
        """
        score = 1.0
        
        # If duplicate, flag it
        if classified.is_duplicate:
            conflicts.append({
                "type": "duplicate",
                "existing_bullet_id": classified.duplicate_of,
                "similarity": f"{classified.similarity_score:.2f}",
                "recommendation": "Skip or merge with existing bullet"
            })
            score -= 0.5
        
        # Check for contradictions with existing knowledge
        # (This would require LLM analysis for semantic contradictions)
        # For now, we use simple keyword-based checks
        
        existing_bullets = existing_playbook.get_section(classified.final_section)
        content_lower = classified.extracted.content.lower()
        
        # Check for potential contradictions
        contradiction_keywords = [
            ("must", "must not"),
            ("always", "never"),
            ("required", "prohibited"),
            ("shall", "shall not")
        ]
        
        for bullet in existing_bullets:
            bullet_lower = bullet.content.lower()
            for pos_kw, neg_kw in contradiction_keywords:
                if pos_kw in content_lower and neg_kw in bullet_lower:
                    warnings.append(f"Potential contradiction with bullet {bullet.id}")
                    score -= 0.1
                    break
        
        return max(0.0, score)
    
    def _check_accuracy(
        self,
        classified: ClassifiedKnowledge,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """
        Check domain/legal accuracy of knowledge.
        
        Returns accuracy score (0.0 to 1.0)
        """
        score = 1.0
        content = classified.extracted.content
        
        # Check for overly generic statements
        generic_terms = ["generally", "usually", "typically", "often", "sometimes"]
        generic_count = sum(1 for term in generic_terms if term in content.lower())
        if generic_count >= 2:
            warnings.append("Contains multiple generic qualifiers - may lack specificity")
            score -= 0.1
        
        # Check for unsupported claims
        claim_indicators = ["always", "never", "all", "none", "every"]
        if any(term in content.lower() for term in claim_indicators):
            if not classified.extracted.examples:
                warnings.append("Strong claim without supporting examples")
                score -= 0.15
        
        # Check for proper terminology (securitization-specific)
        if classified.final_section in ["strategies", "definitions"]:
            # Should contain domain-specific terms
            domain_terms = [
                "securitization", "tranche", "waterfall", "spe", "spv", "issuer",
                "originator", "servicer", "trustee", "noteholder", "underlying",
                "collateral", "credit enhancement", "subordination"
            ]
            has_domain_term = any(term in content.lower() for term in domain_terms)
            if not has_domain_term and len(content) > 50:
                warnings.append("May lack securitization-specific terminology")
                score -= 0.1
        
        return max(0.0, score)
    
    def _generate_recommendations(
        self,
        classified: ClassifiedKnowledge,
        recommendations: List[str]
    ) -> None:
        """Generate recommendations for improvement."""
        
        # Recommend adding examples
        if not classified.extracted.examples:
            recommendations.append("Consider adding specific examples from source document")
        
        # Recommend adding related terms
        if len(classified.extracted.related_terms) < 2:
            recommendations.append("Consider identifying related terms/concepts")
        
        # Recommend for high-importance items
        if classified.importance_score >= 0.8:
            recommendations.append("High importance - prioritize for review and enrichment")
        
        # Recommend review for low confidence
        if classified.final_confidence < 0.7:
            recommendations.append("Low confidence - recommend human review before adding")
        
        # Recommend merge for near-duplicates
        if classified.similarity_score >= 0.7 and not classified.is_duplicate:
            recommendations.append(f"High similarity ({classified.similarity_score:.2f}) - consider merging with existing bullet")

