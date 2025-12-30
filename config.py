"""
Configuration settings for the ACE Securitization System.

Updated to include retrieval and extended operation settings.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

# Environment variables for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Default paths
DEFAULT_PLAYBOOK_PATH = "playbook.json"
DEFAULT_LOG_PATH = "logs"
DEFAULT_INDEX_PATH = "playbook_index"


# LLM Configuration
@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "openai"  # openai, google, anthropic, mock
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 4096
    stream: bool = True
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = OPENAI_API_KEY
            elif self.provider == "google":
                self.api_key = GOOGLE_API_KEY
            elif self.provider == "anthropic":
                self.api_key = ANTHROPIC_API_KEY


@dataclass
class PlaybookConfig:
    """Configuration for playbook management."""
    path: str = DEFAULT_PLAYBOOK_PATH
    max_bullets_per_section: int = 100
    dedup_similarity_threshold: float = 0.85
    enable_semantic_dedup: bool = True
    
    # Auto-cleanup thresholds
    auto_remove_harmful_threshold: int = 5  # harmful_count to trigger auto-remove
    auto_remove_effectiveness_threshold: float = -0.3  # effectiveness to trigger auto-remove
    archive_removed_bullets: bool = True  # Keep removed bullets in archive


@dataclass
class RetrieverConfig:
    """Configuration for semantic retrieval."""
    enabled: bool = True
    top_k: int = 10
    similarity_threshold: float = 0.3
    max_tokens_budget: int = 2000
    
    # Embedding settings
    embedding_provider: str = "simple"  # simple, sentence-transformers, openai
    embedding_model: str = "all-MiniLM-L6-v2"  # for sentence-transformers
    embedding_dim: int = 256  # for simple embeddings
    
    # Index persistence
    index_path: Optional[str] = DEFAULT_INDEX_PATH
    
    # Hybrid scoring weights
    semantic_weight: float = 0.6
    keyword_weight: float = 0.2
    effectiveness_weight: float = 0.1
    recency_weight: float = 0.1
    
    # When to use retrieval
    min_playbook_size_for_retrieval: int = 15


@dataclass
class ACEConfig:
    """Main configuration for the ACE system."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    playbook: PlaybookConfig = field(default_factory=PlaybookConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    max_reflector_iterations: int = 5
    max_epochs: int = 5
    enable_streaming: bool = True
    log_path: str = DEFAULT_LOG_PATH
    verbose: bool = True
    
    # Feature flags
    enable_retrieval: bool = True
    enable_auto_cleanup: bool = False  # Auto-remove harmful bullets


# Playbook section names
PLAYBOOK_SECTIONS = [
    "strategies",
    "pitfalls", 
    "templates",
    "definitions"
]

# Section prefixes for bullet IDs
SECTION_PREFIXES = {
    "strategies": "str",
    "pitfalls": "pit",
    "templates": "tmp",
    "definitions": "def"
}

# Supported operations
SUPPORTED_OPERATIONS = [
    "ADD",
    "REMOVE", 
    "MODIFY",
    "MERGE"
]