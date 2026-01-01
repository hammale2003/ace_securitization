"""
Redundancy and "upgrade" logic for playbook enrichment.

Goal:
- Prevent ACE enrichment from adding duplicate bullets into the playbook.
- If the incoming bullet is a clearly better version of an existing one,
  update (MODIFY) the existing bullet instead of adding a new duplicate.

Uses embedding-based semantic similarity for robust duplicate detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Tuple, List, Literal

from playbook import Playbook, Bullet, compute_semantic_similarity
from embeddings import cosine_similarity
from utils import logger


RedundancyAction = Literal["ADD", "SKIP", "MODIFY"]


@dataclass(frozen=True)
class RedundancyDecision:
    action: RedundancyAction
    target_bullet_id: Optional[str] = None
    reason: str = ""
    similarity: float = 0.0


_POINTER_DEF_PATTERNS = [
    r"\bhas the meaning given to it in\b",
    r"\bhas the meaning given in\b",
    r"\bmeans the meaning set out in\b",
    r"\bhas the meaning set out in\b",
    r"\brefers to clause\b",
    r"\bset out in clause\b",
]


def normalize_text(text: str) -> str:
    """Normalize text for exact-match duplicate detection."""
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    # Light punctuation normalization for common near-exact duplicates
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    t = t.strip('."\' ')
    return t


def is_pointer_definition(text: str) -> bool:
    t = normalize_text(text)
    return any(re.search(pat, t) for pat in _POINTER_DEF_PATTERNS)


def extract_definition_term(text: str) -> Optional[str]:
    """
    Attempt to extract the defined term key.

    Supports common formats:
    - Term: means ...
    - "Term" means ...
    - Term means ...
    """
    if not text:
        return None

    raw = text.strip()
    # Term: ...
    m = re.match(r'^\s*"?([A-Za-z0-9][A-Za-z0-9\s\/\-\(\)&]+?)"?\s*:\s+', raw)
    if m:
        term = normalize_text(m.group(1))
        return term or None

    # "Term" means ...
    m = re.match(r'^\s*"([^"]+)"\s+(?:means|has the meaning)\b', raw, flags=re.IGNORECASE)
    if m:
        term = normalize_text(m.group(1))
        return term or None

    # Term means ...
    m = re.match(r'^\s*([A-Za-z0-9][A-Za-z0-9\s\/\-\(\)&]+?)\s+(?:means|has the meaning)\b', raw, flags=re.IGNORECASE)
    if m:
        term = normalize_text(m.group(1))
        # Avoid obviously-too-long captures (e.g. full sentence starts)
        if term and len(term.split()) <= 8:
            return term

    return None


def _sequence_similarity(a: str, b: str) -> float:
    """Character-level similarity (robust for near-duplicates)."""
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _embedding_similarity(a: str, b: str, embedding_model=None) -> Optional[float]:
    """
    Compute embedding-based semantic similarity if embedding model is available.
    Returns None if embeddings are not available.
    """
    if embedding_model is None:
        return None
    
    try:
        emb_a = embedding_model.embed(a)
        emb_b = embedding_model.embed(b)
        sim = float(cosine_similarity(emb_a, emb_b))
        logger.debug(f"Embedding similarity computed: {sim:.3f}")
        return sim
    except Exception as e:
        logger.warning(f"Failed to compute embedding similarity: {e}")
        return None


def _hybrid_similarity(a: str, b: str, embedding_model=None) -> float:
    """
    Hybrid similarity: combines embedding-based semantic similarity with 
    character-level and word-overlap methods for robust duplicate detection.
    
    Priority:
    1. Embedding-based semantic similarity (if available) - most robust
    2. SequenceMatcher (character-level)
    3. Word-overlap Jaccard
    4. Token containment
    """
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    # Token containment: if one bullet is basically an expanded version of the other,
    # Jaccard can be misleadingly low; containment catches that.
    def containment(x: str, y: str) -> float:
        x_words = set(x.split())
        y_words = set(y.split())
        if not x_words or not y_words:
            return 0.0
        inter = len(x_words & y_words)
        denom = min(len(x_words), len(y_words))
        return inter / denom if denom else 0.0

    similarities = [
        _sequence_similarity(a_n, b_n),
        compute_semantic_similarity(a_n, b_n),
        containment(a_n, b_n),
    ]
    
    # Add embedding similarity if available (highest priority)
    emb_sim = _embedding_similarity(a, b, embedding_model)
    if emb_sim is not None:
        similarities.insert(0, emb_sim)
        logger.debug(f"Using embedding similarity: {emb_sim:.3f} (fallback methods: {similarities[1:]})")
    else:
        logger.debug(f"No embedding similarity available, using fallback methods: {similarities}")
    
    # Return max similarity (embedding-based is most robust)
    final_sim = max(similarities)
    logger.debug(f"Final hybrid similarity: {final_sim:.3f}")
    return final_sim


def content_quality_score(section: str, text: str) -> float:
    """
    Heuristic quality score in [0, 1].
    Higher means: more likely to be the "better" version to keep.
    """
    if not text or not text.strip():
        return 0.0

    t = text.strip()
    t_norm = normalize_text(t)

    # Length signal
    length = len(t_norm)
    length_score = min(length / 450.0, 0.45)  # cap

    # Penalize very short bullets
    short_penalty = 0.15 if length < 60 else 0.0

    if section == "definitions":
        term = extract_definition_term(t) or ""
        has_termish_prefix = 0.15 if term else 0.0
        has_means = 0.15 if re.search(r"\bmeans\b", t, flags=re.IGNORECASE) else 0.0
        pointer_penalty = 0.35 if is_pointer_definition(t) else 0.0
        return max(
            0.0,
            min(1.0, 0.25 + length_score + has_termish_prefix + has_means - pointer_penalty - short_penalty),
        )
    
    if section == "strategies":
        # Strategies quality: penalize transaction-specific content, reward reusable patterns
        has_transaction_specific = 0.0
        # Penalize if contains specific dates, amounts, or clause references
        if re.search(r'\b(20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december)\b', t, flags=re.IGNORECASE):
            has_transaction_specific = -0.15
        if re.search(r'\bclause\s+\d+\.\d+', t, flags=re.IGNORECASE):
            has_transaction_specific = -0.10
        if re.search(r'\$\d+|\d+%', t):
            has_transaction_specific = -0.10
        
        # Reward reusable patterns
        has_actionable = 0.15 if re.search(r'\b(use|establish|implement|apply|consider|ensure|avoid|prevent)\b', t, flags=re.IGNORECASE) else 0.0
        has_structure = 0.1 if any(x in t for x in [":", ";", "—", "- "]) else 0.0
        has_example = 0.1 if re.search(r'\b(e\.g\.|for example|such as)\b', t, flags=re.IGNORECASE) else 0.0
        has_general_terms = 0.1 if re.search(r'\b(any|each|all|every|general|common|typical)\b', t, flags=re.IGNORECASE) else 0.0
        
        return max(0.0, min(1.0, 0.25 + length_score + has_actionable + has_structure + has_example + has_general_terms + has_transaction_specific - short_penalty))

    # Other sections (pitfalls, templates): modest signals
    has_structure = 0.1 if any(x in t for x in [":", ";", "—", "- "]) else 0.0
    has_example = 0.05 if re.search(r"\b(e\.g\.|for example)\b", t, flags=re.IGNORECASE) else 0.0
    return max(0.0, min(1.0, 0.20 + length_score + has_structure + has_example - short_penalty))

def _best_existing_match_fallback(
    existing_bullets: List[Bullet],
    new_content: str,
    embedding_model=None
) -> Tuple[Optional[Bullet], float]:
    best: Optional[Bullet] = None
    best_sim = 0.0
    for b in existing_bullets:
        sim = _hybrid_similarity(new_content, b.content, embedding_model)
        if sim > best_sim:
            best_sim = sim
            best = b
    return best, best_sim


def decide_add_vs_skip_or_modify(
    *,
    playbook: Playbook,
    section: str,
    new_content: str,
    retriever=None,
    duplicate_similarity_threshold: float = 0.86,
    upgrade_similarity_threshold: float = 0.78,
    upgrade_margin: float = 0.08,
) -> RedundancyDecision:
    """
    Decide what to do with an incoming ADD operation.
    
    Uses embedding-based semantic similarity when available for robust duplicate detection.
    """
    # Extract embedding model from retriever if available
    embedding_model = None
    if retriever is not None and hasattr(retriever, 'embedding_model'):
        embedding_model = retriever.embedding_model
        logger.info(f"Embedding model available for {section} section: {type(embedding_model).__name__}")
    else:
        logger.warning(f"No embedding model available for {section} section - using fallback similarity methods")
    
    section_bullets = playbook.get_section(section) or []

    new_norm = normalize_text(new_content)
    if not new_norm:
        return RedundancyDecision(action="SKIP", reason="Empty content")

    # 1) Exact duplicates (normalized)
    for b in section_bullets:
        if normalize_text(b.content) == new_norm:
            return RedundancyDecision(
                action="SKIP",
                target_bullet_id=b.id,
                reason="Exact duplicate in same section",
                similarity=1.0,
            )

    # 2) Definitions: term-keyed updates
    if section == "definitions":
        term = extract_definition_term(new_content)

        # If we can extract a term, use term-based deduplication
        if term:
            same_term = [b for b in section_bullets if extract_definition_term(b.content) == term]
            if same_term:
                # Prefer upgrading pointer -> substantive definition
                new_is_pointer = is_pointer_definition(new_content)
                best_existing = max(same_term, key=lambda b: content_quality_score(section, b.content))

                if is_pointer_definition(best_existing.content) and not new_is_pointer:
                    sim = _hybrid_similarity(new_content, best_existing.content, embedding_model)
                    logger.info(f"MODIFY definition bullet {best_existing.id} for term '{term}': pointer → substantive (sim={sim:.3f})")
                    logger.info(f"  Existing: {best_existing.content[:100]}...")
                    logger.info(f"  New: {new_content[:100]}...")
                    return RedundancyDecision(
                        action="MODIFY",
                        target_bullet_id=best_existing.id,
                        reason=f'Upgrade definition for term "{term}" (pointer → substantive)',
                        similarity=sim,
                    )

                new_q = content_quality_score(section, new_content)
                old_q = content_quality_score(section, best_existing.content)
                if new_q >= old_q + upgrade_margin:
                    sim = _hybrid_similarity(new_content, best_existing.content, embedding_model)
                    logger.info(f"MODIFY definition bullet {best_existing.id} for term '{term}': quality {old_q:.2f} → {new_q:.2f} (sim={sim:.3f})")
                    logger.info(f"  Existing: {best_existing.content[:100]}...")
                    logger.info(f"  New: {new_content[:100]}...")
                    return RedundancyDecision(
                        action="MODIFY",
                        target_bullet_id=best_existing.id,
                        reason=f'Upgrade definition for term "{term}" (quality {old_q:.2f} → {new_q:.2f})',
                        similarity=sim,
                    )
                return RedundancyDecision(
                    action="SKIP",
                    target_bullet_id=best_existing.id,
                    reason=f'Definition for term "{term}" already exists and is not worse',
                    similarity=_hybrid_similarity(new_content, best_existing.content, embedding_model),
                )
    
    if embedding_model is not None:
        strategy_similarity_threshold = 0.60  # Embeddings are very robust, lower threshold to catch more semantic duplicates
    else:
        strategy_similarity_threshold = 0.82  # Word-overlap needs higher threshold
    
    if section == "strategies":
        best_strategy: Optional[Bullet] = None
        best_strategy_sim: float = 0.0
        
        if retriever is not None:
            try:
                results = retriever.search(new_content, sections=[section], top_k=5)
                if results:
                    # Check top results for high similarity using hybrid similarity
                    for result in results:
                        bullet = playbook.get_bullet_by_id(result.bullet_id)
                        if bullet:
                            sim = _hybrid_similarity(new_content, bullet.content, embedding_model)
                            if sim > best_strategy_sim:
                                best_strategy_sim = sim
                                best_strategy = bullet
            except Exception:
                pass
        
        # Fallback to deterministic search
        if best_strategy is None:
            best_strategy, best_strategy_sim = _best_existing_match_fallback(section_bullets, new_content, embedding_model)
        
        if best_strategy:
            logger.info(f"Found similar strategy: {best_strategy.id} with similarity={best_strategy_sim:.3f} (threshold={strategy_similarity_threshold:.3f})")
            logger.debug(f"  Strategy content: {best_strategy.content[:150]}...")
            logger.debug(f"  New content: {new_content[:150]}...")
        
        if best_strategy and best_strategy_sim >= strategy_similarity_threshold:
            new_q = content_quality_score(section, new_content)
            old_q = content_quality_score(section, best_strategy.content)
            
            logger.info(f"Strategy similarity check: sim={best_strategy_sim:.3f} >= threshold={strategy_similarity_threshold:.3f}, quality: old={old_q:.2f}, new={new_q:.2f}")
            
            # For strategies, prefer the more reusable/general one
            if new_q >= old_q + upgrade_margin:
                logger.info(f"MODIFY strategy bullet {best_strategy.id}: similarity={best_strategy_sim:.3f}, quality {old_q:.2f} → {new_q:.2f}")
                logger.info(f"  Existing: {best_strategy.content[:100]}...")
                logger.info(f"  New: {new_content[:100]}...")
                return RedundancyDecision(
                    action="MODIFY",
                    target_bullet_id=best_strategy.id,
                    reason=f'Upgrade strategy (quality {old_q:.2f} → {new_q:.2f}, sim={best_strategy_sim:.2f})',
                    similarity=best_strategy_sim,
                )
            logger.info(f"SKIP strategy bullet {best_strategy.id}: similarity={best_strategy_sim:.3f} (existing quality {old_q:.2f} >= new {new_q:.2f})")
            logger.info(f"  Existing: {best_strategy.content[:100]}...")
            logger.info(f"  New: {new_content[:100]}...")
            return RedundancyDecision(
                action="SKIP",
                target_bullet_id=best_strategy.id,
                reason=f'Similar strategy already exists (sim={best_strategy_sim:.2f})',
                similarity=best_strategy_sim,
            )
        # If strategy check didn't find a match, continue to general similarity check below

    # 3) Similarity-based duplicate/upgrade detection (for all sections, including strategies if no match found above)
    best_bullet: Optional[Bullet] = None
    best_sim: float = 0.0

    if retriever is not None:
        try:
            results = retriever.search(new_content, sections=[section], top_k=3)
            if results:
                # Use top result; it is already section-filtered
                top = results[0]
                best_bullet = playbook.get_bullet_by_id(top.bullet_id)
                # Use hybrid similarity for consistency
                if best_bullet:
                    best_sim = _hybrid_similarity(new_content, best_bullet.content, embedding_model)
        except Exception:
            # Fallback to deterministic similarity
            best_bullet, best_sim = _best_existing_match_fallback(section_bullets, new_content, embedding_model)
    else:
        best_bullet, best_sim = _best_existing_match_fallback(section_bullets, new_content, embedding_model)

    if best_bullet is None:
        logger.debug(f"No similar {section} bullets found, will ADD")
        return RedundancyDecision(action="ADD", reason="No similar bullets found")

    # If similarity is high enough, treat as redundant unless this is a clear upgrade.
    # For strategies with embeddings, use a slightly lower threshold in the general check too
    effective_threshold = duplicate_similarity_threshold
    if section == "strategies" and embedding_model is not None:
        effective_threshold = min(duplicate_similarity_threshold, 0.75)  # More aggressive for strategies with embeddings
    
    logger.info(f"General similarity check for {section}: sim={best_sim:.3f}, threshold={effective_threshold:.3f}, bullet_id={best_bullet.id}")
    logger.debug(f"  Existing: {best_bullet.content[:150]}...")
    logger.debug(f"  New: {new_content[:150]}...")
    
    if best_sim >= effective_threshold:
        new_q = content_quality_score(section, new_content)
        old_q = content_quality_score(section, best_bullet.content)
        if new_q >= old_q + upgrade_margin and best_sim >= upgrade_similarity_threshold:
            logger.info(f"MODIFY {section} bullet {best_bullet.id}: similarity={best_sim:.3f}, quality {old_q:.2f} → {new_q:.2f}")
            logger.info(f"  Existing: {best_bullet.content[:100]}...")
            logger.info(f"  New: {new_content[:100]}...")
            return RedundancyDecision(
                action="MODIFY",
                target_bullet_id=best_bullet.id,
                reason=f"Upgrade near-duplicate (quality {old_q:.2f} → {new_q:.2f}, sim={best_sim:.2f})",
                similarity=best_sim,
            )
        logger.info(f"SKIP {section} bullet {best_bullet.id}: similarity={best_sim:.3f} (existing quality {old_q:.2f} >= new {new_q:.2f})")
        logger.info(f"  Existing: {best_bullet.content[:100]}...")
        logger.info(f"  New: {new_content[:100]}...")
        return RedundancyDecision(
            action="SKIP",
            target_bullet_id=best_bullet.id,
            reason=f"Near-duplicate in same section (sim={best_sim:.2f})",
            similarity=best_sim,
        )

    # Otherwise: allow ADD
    return RedundancyDecision(action="ADD", reason=f"Below duplicate threshold (sim={best_sim:.2f})", similarity=best_sim)


