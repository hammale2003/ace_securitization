"""
ACE Securitization System

Agentic Context Engineering for Securitization and Structured Finance.

This package implements the ACE framework with three specialized agents:
- Generator: Produces answers using the evolving playbook
- Reflector: Analyzes responses and extracts insights
- Curator: Updates the playbook with new knowledge

Usage:
    from ACE_Framework import ACEPipeline, ACEConfig
    
    pipeline = ACEPipeline()
    result = pipeline.run("What are the key elements of a true sale?")
    print(result.generator_output.final_answer)
"""

__version__ = "1.0.0"
__author__ = "Mourad_hammale"

from config import ACEConfig, LLMConfig, PlaybookConfig
from playbook import Playbook, PlaybookManager, Bullet
from agents import (
    ACEPipeline,
    Generator,
    Reflector, 
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
    ACEPipelineResult
)
from llm_client import LLMClient, create_client

__all__ = [
    # Configuration
    "ACEConfig",
    "LLMConfig", 
    "PlaybookConfig",
    
    # Playbook
    "Playbook",
    "PlaybookManager",
    "Bullet",
    
    # Agents
    "ACEPipeline",
    "Generator",
    "Reflector",
    "Curator",
    
    # Outputs
    "GeneratorOutput",
    "ReflectorOutput", 
    "CuratorOutput",
    "ACEPipelineResult",
    
    # LLM Client
    "LLMClient",
    "create_client",
]
