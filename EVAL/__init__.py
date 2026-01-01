"""
ACORD: ACET Evaluation on Atticus Clause Retrieval Dataset

A self-contained implementation of the ACET (Agentic Context Engineering)
framework specifically designed for legal clause extraction tasks.

Components:
- config: Configuration classes
- llm_client: LLM provider interfaces (OpenAI, Anthropic, Google, Mock)
- playbook: Evolving knowledge storage
- agents: Generator, Reflector, Curator agents
- pipeline: Main orchestration pipeline
- dataset: ACORD data loading and processing
- evaluator: Metrics and evaluation framework

Usage:
    from ACORD import ACORDPipeline, ACORDEvaluator, create_synthetic_dataset
    
    # Quick demo
    pipeline = ACORDPipeline()
    result = pipeline.generate(contract_text, "Termination For Convenience")
    
    # Full evaluation
    dataset = create_synthetic_dataset(100)
    train, _, test = dataset.split()
    
    evaluator = ACORDEvaluator()
    baseline = evaluator.evaluate(test)
    evaluator.train_epoch(train, epoch=0)
    final = evaluator.evaluate(test)
"""

from .config import ACORDConfig, LLMConfig, PlaybookConfig, ACORD_CLAUSE_TYPES
from .llm_client import create_llm_client, BaseLLMClient, LLMResponse
from .playbook import Playbook, PlaybookManager, PlaybookBullet, create_initial_acord_playbook
from .agents import (
    GeneratorAgent, GeneratorOutput,
    ReflectorAgent, ReflectorOutput, ReflectorInsight,
    CuratorAgent, CuratorOutput, CuratorUpdate
)
from .pipeline import ACORDPipeline, PipelineResult, create_pipeline
from .dataset import (
    ACORDSample, ACORDDataset, ACORDLoader,
    create_synthetic_sample, create_synthetic_dataset
)
from .evaluator import (
    ACORDEvaluator, EvaluationReport, SampleResult, EpochMetrics,
    MetricsCalculator, run_full_evaluation
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "ACORDConfig", "LLMConfig", "PlaybookConfig", "ACORD_CLAUSE_TYPES",
    # LLM
    "create_llm_client", "BaseLLMClient", "LLMResponse",
    # Playbook
    "Playbook", "PlaybookManager", "PlaybookBullet", "create_initial_acord_playbook",
    # Agents
    "GeneratorAgent", "GeneratorOutput",
    "ReflectorAgent", "ReflectorOutput", "ReflectorInsight",
    "CuratorAgent", "CuratorOutput", "CuratorUpdate",
    # Pipeline
    "ACORDPipeline", "PipelineResult", "create_pipeline",
    # Dataset
    "ACORDSample", "ACORDDataset", "ACORDLoader",
    "create_synthetic_sample", "create_synthetic_dataset",
    # Evaluator
    "ACORDEvaluator", "EvaluationReport", "SampleResult", "EpochMetrics",
    "MetricsCalculator", "run_full_evaluation"
]
