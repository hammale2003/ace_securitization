"""
Configuration for ACORD Clause Extraction Evaluation.

This module contains all configuration classes for the ACET-ACORD evaluation system.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000
    api_key: Optional[str] = None  # If None, uses environment variable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'provider': self.provider,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


@dataclass
class PlaybookConfig:
    """Configuration for the evolving playbook."""
    path: str = "playbook.json"
    max_bullets_per_section: int = 100
    dedup_similarity_threshold: float = 0.85
    retriever_top_k: int = 10


@dataclass
class ReflectorConfig:
    """Configuration for the Reflector agent."""
    max_iterations: int = 3
    enabled: bool = True


@dataclass
class CuratorConfig:
    """Configuration for the Curator agent."""
    enabled: bool = True
    min_confidence_to_add: float = 0.7


@dataclass 
class ACORDConfig:
    """Main configuration for ACORD evaluation."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    playbook: PlaybookConfig = field(default_factory=PlaybookConfig)
    reflector: ReflectorConfig = field(default_factory=ReflectorConfig)
    curator: CuratorConfig = field(default_factory=CuratorConfig)
    
    # ACORD-specific settings
    max_contract_length: int = 8000  # Max characters to include from contract
    include_negative_samples: bool = True  # Include samples where clause is absent
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'llm': self.llm.to_dict(),
            'playbook': {
                'path': self.playbook.path,
                'max_bullets_per_section': self.playbook.max_bullets_per_section,
                'dedup_similarity_threshold': self.playbook.dedup_similarity_threshold,
                'retriever_top_k': self.playbook.retriever_top_k
            },
            'reflector': {
                'max_iterations': self.reflector.max_iterations,
                'enabled': self.reflector.enabled
            },
            'curator': {
                'enabled': self.curator.enabled,
                'min_confidence_to_add': self.curator.min_confidence_to_add
            },
            'max_contract_length': self.max_contract_length,
            'include_negative_samples': self.include_negative_samples
        }


# Default clause types from CUAD/ACORD
ACORD_CLAUSE_TYPES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Shop",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All Remedies",
    "Cap On Liability",
    "Liquidated Damages",
    "Termination For Convenience",
    "Competitive Restriction Exception",
    "Non-Solicitation Of Customers",
    "Non-Solicitation Of Employees",
    "Non-Disparagement",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Audit Rights",
    "Uncapped Liability",
    "Warranty Duration",
    "Post-Termination Services"
]
