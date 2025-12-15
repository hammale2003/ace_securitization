"""
Embedding model abstraction for the ACE system.

Supports multiple embedding providers: sentence-transformers, OpenAI, and a simple fallback.
Used for semantic retrieval of playbook bullets.
"""
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    provider: str = "simple"  # simple, sentence-transformers, openai
    model_name: str = "all-MiniLM-L6-v2"  # for sentence-transformers
    openai_model: str = "text-embedding-3-small"  # for openai
    api_key: Optional[str] = None
    cache_embeddings: bool = True
    embedding_dim: int = 384  # dimension for sentence-transformers


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts at once."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SimpleEmbedding(EmbeddingModel):
    """
    Simple TF-IDF-like embedding using word hashing.
    
    This is a fallback when no ML libraries are available.
    It's not as good as neural embeddings but works without dependencies.
    """
    
    def __init__(self, dim: int = 256):
        self._dim = dim
        self._vocab: Dict[str, int] = {}
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def _hash_token(self, token: str) -> int:
        """Hash a token to a dimension index."""
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return h % self._dim
    
    def embed(self, text: str) -> np.ndarray:
        """Create a simple bag-of-words embedding with hashing."""
        tokens = self._tokenize(text)
        embedding = np.zeros(self._dim, dtype=np.float32)
        
        for token in tokens:
            idx = self._hash_token(token)
            embedding[idx] += 1.0
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        embeddings = [self.embed(text) for text in texts]
        return np.array(embeddings)


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Embedding using sentence-transformers library.
    
    Provides high-quality semantic embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        return self.model.encode(texts, convert_to_numpy=True)


class OpenAIEmbedding(EmbeddingModel):
    """
    Embedding using OpenAI's embedding API.
    """
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
            self.model = model
            # Dimensions for OpenAI models
            self._dim_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            self._dim = self._dim_map.get(model, 1536)
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return np.array(embeddings)


def create_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    """Factory function to create the appropriate embedding model."""
    provider = config.provider.lower()
    
    if provider == "simple":
        return SimpleEmbedding(dim=config.embedding_dim)
    elif provider == "sentence-transformers" or provider == "st":
        return SentenceTransformerEmbedding(model_name=config.model_name)
    elif provider == "openai":
        return OpenAIEmbedding(model=config.openai_model, api_key=config.api_key)
    else:
        # Default to simple
        print(f"Unknown embedding provider '{provider}', using simple embeddings")
        return SimpleEmbedding(dim=config.embedding_dim)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and all corpus vectors.
    
    Args:
        query: Shape (dim,) - single query vector
        corpus: Shape (n, dim) - matrix of corpus vectors
    
    Returns:
        Shape (n,) - similarity scores
    """
    if len(corpus) == 0:
        return np.array([])
    
    # Normalize query
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    
    # Normalize corpus
    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8
    corpus_normalized = corpus / corpus_norms
    
    # Compute similarities
    similarities = np.dot(corpus_normalized, query_norm)
    
    return similarities