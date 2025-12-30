"""
Redundancy and "upgrade" logic for playbook enrichment.

Goal:
- Prevent ACE enrichment from adding duplicate bullets into the playbook.
- If the incoming bullet is a clearly better version of an existing one,
  update (MODIFY) the existing bullet instead of adding a new duplicate.

This module is intentionally deterministic (no LLM calls).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Tuple, List, Literal

from playbook import Playbook, Bullet, compute_semantic_similarity


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


def _hybrid_similarity(a: str, b: str) -> float:
    """
    Hybrid similarity: max of (SequenceMatcher, simple word-overlap).
    `compute_semantic_similarity` is a word-overlap Jaccard-like score.
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

    return max(
        _sequence_similarity(a_n, b_n),
        compute_semantic_similarity(a_n, b_n),
        containment(a_n, b_n),
    )


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

    # Other sections: modest signals
    has_structure = 0.1 if any(x in t for x in [":", ";", "—", "- "]) else 0.0
    has_example = 0.05 if re.search(r"\b(e\.g\.|for example)\b", t, flags=re.IGNORECASE) else 0.0
    return max(0.0, min(1.0, 0.20 + length_score + has_structure + has_example - short_penalty))

def _best_existing_match_fallback(
    existing_bullets: List[Bullet],
    new_content: str,
) -> Tuple[Optional[Bullet], float]:
    best: Optional[Bullet] = None
    best_sim = 0.0
    for b in existing_bullets:
        sim = _hybrid_similarity(new_content, b.content)
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

    - SKIP if exact duplicate (normalized) exists in the same section.
    - If a strong match exists, SKIP unless the new content is clearly better.
    - For definitions: if same term exists, prefer the better definition (often
      upgrading pointer definitions to substantive ones).
    """
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
                    return RedundancyDecision(
                        action="MODIFY",
                        target_bullet_id=best_existing.id,
                        reason=f'Upgrade definition for term "{term}" (pointer → substantive)',
                        similarity=_hybrid_similarity(new_content, best_existing.content),
                    )

                new_q = content_quality_score(section, new_content)
                old_q = content_quality_score(section, best_existing.content)
                if new_q >= old_q + upgrade_margin:
                    return RedundancyDecision(
                        action="MODIFY",
                        target_bullet_id=best_existing.id,
                        reason=f'Upgrade definition for term "{term}" (quality {old_q:.2f} → {new_q:.2f})',
                        similarity=_hybrid_similarity(new_content, best_existing.content),
                    )
                return RedundancyDecision(
                    action="SKIP",
                    target_bullet_id=best_existing.id,
                    reason=f'Definition for term "{term}" already exists and is not worse',
                    similarity=_hybrid_similarity(new_content, best_existing.content),
                )

    # 3) Similarity-based duplicate/upgrade detection
    best_bullet: Optional[Bullet] = None
    best_sim: float = 0.0

    if retriever is not None:
        try:
            results = retriever.search(new_content, sections=[section], top_k=3)
            if results:
                # Use top result; it is already section-filtered
                top = results[0]
                best_bullet = playbook.get_bullet_by_id(top.bullet_id)
                best_sim = float(top.score)
        except Exception:
            # Fallback to deterministic similarity
            best_bullet, best_sim = _best_existing_match_fallback(section_bullets, new_content)
    else:
        best_bullet, best_sim = _best_existing_match_fallback(section_bullets, new_content)

    if best_bullet is None:
        return RedundancyDecision(action="ADD", reason="No similar bullets found")

    # If similarity is high enough, treat as redundant unless this is a clear upgrade.
    if best_sim >= duplicate_similarity_threshold:
        new_q = content_quality_score(section, new_content)
        old_q = content_quality_score(section, best_bullet.content)
        if new_q >= old_q + upgrade_margin and best_sim >= upgrade_similarity_threshold:
            return RedundancyDecision(
                action="MODIFY",
                target_bullet_id=best_bullet.id,
                reason=f"Upgrade near-duplicate (quality {old_q:.2f} → {new_q:.2f}, sim={best_sim:.2f})",
                similarity=best_sim,
            )
        return RedundancyDecision(
            action="SKIP",
            target_bullet_id=best_bullet.id,
            reason=f"Near-duplicate in same section (sim={best_sim:.2f})",
            similarity=best_sim,
        )

    # Otherwise: allow ADD
    return RedundancyDecision(action="ADD", reason=f"Below duplicate threshold (sim={best_sim:.2f})", similarity=best_sim)


