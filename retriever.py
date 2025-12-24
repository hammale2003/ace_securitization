"""
Playbook Retriever for semantic search over playbook bullets.

Implements RAG-style retrieval to reduce token consumption by only
sending relevant bullets to the Generator.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from embeddings import (
    EmbeddingModel, 
    EmbeddingConfig, 
    create_embedding_model,
    cosine_similarity_matrix
)
from utils import logger


@dataclass
class RetrieverConfig:
    """Configuration for the playbook retriever."""
    # Retrieval parameters
    top_k: int = 10
    similarity_threshold: float = 0.3
    max_tokens_budget: int = 2000  # Approximate token limit for context
    
    # Embedding configuration
    embedding_provider: str = "sentence-transformers"  
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 256
    
    # Index persistence
    index_path: Optional[str] = None  # Path to save/load embeddings
    
    # Scoring weights for hybrid retrieval
    semantic_weight: float = 0.6
    keyword_weight: float = 0.2
    effectiveness_weight: float = 0.1
    recency_weight: float = 0.1
    
    # Feature flags
    enable_hybrid_scoring: bool = True
    boost_high_effectiveness: bool = True
    min_playbook_size_for_retrieval: int = 15  # Skip retrieval if playbook is smaller


@dataclass
class RetrievedBullet:
    """A bullet retrieved from the playbook with its relevance score."""
    bullet_id: str
    content: str
    section: str
    score: float
    helpful_count: int = 0
    harmful_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bullet_id": self.bullet_id,
            "content": self.content,
            "section": self.section,
            "score": self.score,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count
        }


class PlaybookRetriever:
    """
    Retrieves relevant playbook bullets using semantic search.
    
    Maintains an embedding index of all bullets and performs
    similarity search to find the most relevant ones for a given query.
    """
    
    def __init__(self, config: RetrieverConfig = None):
        self.config = config or RetrieverConfig()
        
        # Initialize embedding model
        embed_config = EmbeddingConfig(
            provider=self.config.embedding_provider,
            model_name=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim
        )
        self.embedding_model = create_embedding_model(embed_config)
        
        # Index storage
        self._bullet_ids: List[str] = []  # Maps index position to bullet ID
        self._bullet_sections: List[str] = []  # Section for each bullet
        self._bullet_contents: List[str] = []  # Content for keyword matching
        self._embeddings: Optional[np.ndarray] = None  # Shape: (n_bullets, dim)
        self._bullet_metadata: Dict[str, Dict[str, Any]] = {}  # Additional metadata
        
        # Load existing index if path provided
        if self.config.index_path:
            self._load_index()
    
    def index_playbook(self, playbook) -> int:
        """
        Index all bullets in a playbook.
        
        Args:
            playbook: Playbook object with bullets
        
        Returns:
            Number of bullets indexed
        """
        from config import PLAYBOOK_SECTIONS
        
        # Clear existing index
        self._bullet_ids = []
        self._bullet_sections = []
        self._bullet_contents = []
        self._bullet_metadata = {}
        
        all_contents = []
        
        for section in PLAYBOOK_SECTIONS:
            bullets = playbook.get_section(section)
            for bullet in bullets:
                self._bullet_ids.append(bullet.id)
                self._bullet_sections.append(section)
                self._bullet_contents.append(bullet.content)
                all_contents.append(bullet.content)
                
                self._bullet_metadata[bullet.id] = {
                    "helpful_count": bullet.helpful_count,
                    "harmful_count": bullet.harmful_count,
                    "effectiveness_score": bullet.effectiveness_score,
                    "updated_at": bullet.updated_at
                }
        
        # Compute embeddings
        if all_contents:
            self._embeddings = self.embedding_model.embed_batch(all_contents)
        else:
            self._embeddings = np.array([])
        
        # Save index if path configured
        if self.config.index_path:
            self._save_index()
        
        return len(self._bullet_ids)
    
    def add_bullet(self, bullet_id: str, content: str, section: str, 
                   helpful_count: int = 0, harmful_count: int = 0,
                   updated_at: str = None) -> None:
        """
        Add a single bullet to the index.
        
        Called when a new bullet is added via ADD operation.
        """
        # Compute embedding
        embedding = self.embedding_model.embed(content)
        
        # Add to index
        self._bullet_ids.append(bullet_id)
        self._bullet_sections.append(section)
        self._bullet_contents.append(content)
        
        # Add embedding
        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])
        
        # Store metadata
        effectiveness = (helpful_count - harmful_count) / max(helpful_count + harmful_count, 1)
        self._bullet_metadata[bullet_id] = {
            "helpful_count": helpful_count,
            "harmful_count": harmful_count,
            "effectiveness_score": effectiveness,
            "updated_at": updated_at or datetime.utcnow().isoformat()
        }
        
        # Save index
        if self.config.index_path:
            self._save_index()
    
    def update_bullet(self, bullet_id: str, new_content: str, 
                      helpful_count: int = None, harmful_count: int = None) -> bool:
        """
        Update a bullet's embedding (called after MODIFY operation).
        
        Returns True if bullet was found and updated.
        """
        try:
            idx = self._bullet_ids.index(bullet_id)
        except ValueError:
            return False
        
        # Update content and embedding
        self._bullet_contents[idx] = new_content
        new_embedding = self.embedding_model.embed(new_content)
        self._embeddings[idx] = new_embedding
        
        # Update metadata
        if bullet_id in self._bullet_metadata:
            if helpful_count is not None:
                self._bullet_metadata[bullet_id]["helpful_count"] = helpful_count
            if harmful_count is not None:
                self._bullet_metadata[bullet_id]["harmful_count"] = harmful_count
            
            h = self._bullet_metadata[bullet_id]["helpful_count"]
            harm = self._bullet_metadata[bullet_id]["harmful_count"]
            self._bullet_metadata[bullet_id]["effectiveness_score"] = (h - harm) / max(h + harm, 1)
            self._bullet_metadata[bullet_id]["updated_at"] = datetime.utcnow().isoformat()
        
        if self.config.index_path:
            self._save_index()
        
        return True
    
    def remove_bullet(self, bullet_id: str) -> bool:
        """
        Remove a bullet from the index (called after REMOVE operation).
        
        Returns True if bullet was found and removed.
        """
        try:
            idx = self._bullet_ids.index(bullet_id)
        except ValueError:
            return False
        
        # Remove from all lists
        self._bullet_ids.pop(idx)
        self._bullet_sections.pop(idx)
        self._bullet_contents.pop(idx)
        
        # Remove embedding
        if self._embeddings is not None and len(self._embeddings) > 0:
            self._embeddings = np.delete(self._embeddings, idx, axis=0)
        
        # Remove metadata
        self._bullet_metadata.pop(bullet_id, None)
        
        if self.config.index_path:
            self._save_index()
        
        return True
    
    def search(self, query: str, top_k: int = None, 
               sections: List[str] = None) -> List[RetrievedBullet]:
        """
        Search for relevant bullets given a query.
        
        Args:
            query: The search query (typically the user's question)
            top_k: Number of results to return (default from config)
            sections: Optional list of sections to search (None = all)
        
        Returns:
            List of RetrievedBullet objects sorted by relevance
        """
        if len(self._bullet_ids) == 0:
            return []
        
        top_k = top_k or self.config.top_k
        
        # Compute query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Compute semantic similarities
        semantic_scores = cosine_similarity_matrix(query_embedding, self._embeddings)
        
        # Compute final scores
        if self.config.enable_hybrid_scoring:
            final_scores = self._compute_hybrid_scores(query, semantic_scores)
        else:
            final_scores = semantic_scores
        
        # Filter by section if specified
        if sections:
            for i, section in enumerate(self._bullet_sections):
                if section not in sections:
                    final_scores[i] = -1.0
        
        # Filter by similarity threshold
        valid_mask = final_scores >= self.config.similarity_threshold
        
        # Get top-k indices
        if valid_mask.sum() == 0:
            # No results above threshold, return top results anyway
            top_indices = np.argsort(final_scores)[-top_k:][::-1]
        else:
            # Mask out low scores and get top-k
            masked_scores = np.where(valid_mask, final_scores, -np.inf)
            top_indices = np.argsort(masked_scores)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            if final_scores[idx] < 0:
                continue
                
            bullet_id = self._bullet_ids[idx]
            metadata = self._bullet_metadata.get(bullet_id, {})
            
            results.append(RetrievedBullet(
                bullet_id=bullet_id,
                content=self._bullet_contents[idx],
                section=self._bullet_sections[idx],
                score=float(final_scores[idx]),
                helpful_count=metadata.get("helpful_count", 0),
                harmful_count=metadata.get("harmful_count", 0)
            ))
        
        return results
    
    def _compute_hybrid_scores(self, query: str, semantic_scores: np.ndarray) -> np.ndarray:
        """
        Compute hybrid scores combining semantic, keyword, effectiveness, and recency.
        """
        n = len(self._bullet_ids)
        
        # Keyword scores (simple term overlap)
        keyword_scores = self._compute_keyword_scores(query)
        
        # Effectiveness scores
        effectiveness_scores = np.zeros(n)
        for i, bullet_id in enumerate(self._bullet_ids):
            meta = self._bullet_metadata.get(bullet_id, {})
            eff = meta.get("effectiveness_score", 0.0)
            # Normalize to [0, 1] range (original is [-1, 1])
            effectiveness_scores[i] = (eff + 1) / 2
        
        # Recency scores
        recency_scores = self._compute_recency_scores()
        
        # Combine with weights
        final_scores = (
            self.config.semantic_weight * semantic_scores +
            self.config.keyword_weight * keyword_scores +
            self.config.effectiveness_weight * effectiveness_scores +
            self.config.recency_weight * recency_scores
        )
        
        return final_scores
    
    def _compute_keyword_scores(self, query: str) -> np.ndarray:
        """Compute simple keyword overlap scores."""
        import re
        
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        scores = np.zeros(len(self._bullet_contents))
        
        for i, content in enumerate(self._bullet_contents):
            content_tokens = set(re.findall(r'\b\w+\b', content.lower()))
            if len(query_tokens) == 0:
                scores[i] = 0
            else:
                overlap = len(query_tokens & content_tokens)
                scores[i] = overlap / len(query_tokens)
        
        return scores
    
    def _compute_recency_scores(self) -> np.ndarray:
        """Compute recency scores based on update timestamps."""
        scores = np.zeros(len(self._bullet_ids))
        
        timestamps = []
        for bullet_id in self._bullet_ids:
            meta = self._bullet_metadata.get(bullet_id, {})
            ts = meta.get("updated_at", "2000-01-01T00:00:00")
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                timestamps.append(dt.timestamp())
            except:
                timestamps.append(0)
        
        if len(timestamps) > 0:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            if max_ts > min_ts:
                scores = np.array([(ts - min_ts) / (max_ts - min_ts) for ts in timestamps])
            else:
                scores = np.ones(len(timestamps)) * 0.5
        
        return scores
    
    def _save_index(self) -> None:
        """Save the index to disk."""
        if not self.config.index_path:
            return
        
        path = Path(self.config.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self._embeddings is not None:
            np.save(str(path) + ".npy", self._embeddings)
        
        # Save metadata
        index_data = {
            "bullet_ids": self._bullet_ids,
            "bullet_sections": self._bullet_sections,
            "bullet_contents": self._bullet_contents,
            "bullet_metadata": self._bullet_metadata
        }
        
        with open(str(path) + ".json", "w") as f:
            json.dump(index_data, f)
    
    def _load_index(self) -> bool:
        """Load the index from disk."""
        if not self.config.index_path:
            return False
        
        path = Path(self.config.index_path)
        embeddings_path = str(path) + ".npy"
        metadata_path = str(path) + ".json"
        
        try:
            if Path(embeddings_path).exists() and Path(metadata_path).exists():
                self._embeddings = np.load(embeddings_path)
                
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                
                self._bullet_ids = data["bullet_ids"]
                self._bullet_sections = data["bullet_sections"]
                self._bullet_contents = data["bullet_contents"]
                self._bullet_metadata = data["bullet_metadata"]
                
                return True
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
        
        return False
    
    def should_use_retrieval(self, playbook_size: int) -> bool:
        """
        Determine if retrieval should be used based on playbook size.
        
        For small playbooks, it's better to just send everything.
        """
        return playbook_size >= self.config.min_playbook_size_for_retrieval
    
    def format_retrieved_for_prompt(self, bullets: List[RetrievedBullet]) -> str:
        """
        Format retrieved bullets for inclusion in the Generator prompt.
        """
        if not bullets:
            return "(No relevant playbook bullets found for this query)"
        
        sections_dict: Dict[str, List[str]] = {}
        
        for bullet in bullets:
            section = bullet.section.upper().replace("_", " ")
            if section not in sections_dict:
                sections_dict[section] = []
            
            formatted = f"[{bullet.bullet_id}] helpful={bullet.helpful_count} harmful={bullet.harmful_count} :: {bullet.content}"
            sections_dict[section].append(formatted)
        
        parts = []
        for section, items in sections_dict.items():
            parts.append(f"### {section}")
            parts.extend(items)
        
        return "\n\n".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            "total_bullets": len(self._bullet_ids),
            "embedding_dim": self.embedding_model.dimension,
            "sections": dict(zip(*np.unique(self._bullet_sections, return_counts=True))) if self._bullet_sections else {},
            "index_path": self.config.index_path
        }