"""
ACORD Pipeline: Orchestrates ACET agents for clause extraction.

This is the main entry point for running the ACET framework on ACORD tasks.
"""
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List

from config import ACORDConfig, LLMConfig, PlaybookConfig
from llm_client import create_llm_client, BaseLLMClient
from playbook import PlaybookManager, Playbook, create_initial_acord_playbook
from agents import (
    GeneratorAgent, GeneratorOutput,
    ReflectorAgent, ReflectorOutput,
    CuratorAgent, CuratorOutput
)


@dataclass
class PipelineResult:
    """Complete result from a pipeline run."""
    # Generator output
    generator_output: GeneratorOutput
    
    # Reflector output (if reflection was performed)
    reflector_output: Optional[ReflectorOutput] = None
    
    # Curator output (if curation was performed)  
    curator_output: Optional[CuratorOutput] = None
    
    # Metadata
    clause_type: str = ""
    had_ground_truth: bool = False
    playbook_updated: bool = False
    
    @property
    def final_answer(self) -> str:
        """Get the final extracted clause."""
        return self.generator_output.final_answer
    
    @property
    def reasoning(self) -> str:
        """Get the extraction reasoning."""
        return self.generator_output.reasoning
    
    @property
    def confidence(self) -> float:
        """Get the extraction confidence."""
        return self.generator_output.confidence


class ACORDPipeline:
    """
    Main pipeline for ACORD clause extraction using ACET architecture.
    
    The pipeline orchestrates three agents:
    1. Generator: Extracts clauses using playbook knowledge
    2. Reflector: Analyzes results when ground truth is available
    3. Curator: Updates playbook with learned insights
    
    Usage:
        pipeline = ACORDPipeline(config)
        
        # Generation only (no learning)
        result = pipeline.generate(contract_text, clause_type)
        
        # Full pipeline with learning
        result = pipeline.run(contract_text, clause_type, ground_truth="...")
    """
    
    def __init__(self, config: ACORDConfig = None, playbook_path: str = None):
        """
        Initialize the ACORD pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
            playbook_path: Override playbook path from config
        """
        self.config = config or ACORDConfig()
        
        # Override playbook path if provided
        if playbook_path:
            self.config.playbook.path = playbook_path
        
        # Initialize LLM client
        self.llm_client = create_llm_client(self.config.llm)
        
        # Initialize playbook manager
        self.playbook_manager = PlaybookManager(self.config.playbook.path)
        
        # Initialize agents
        self.generator = GeneratorAgent(self.llm_client, self.playbook_manager)
        self.reflector = ReflectorAgent(self.llm_client, self.playbook_manager)
        self.curator = CuratorAgent(
            self.llm_client, 
            self.playbook_manager,
            min_confidence=self.config.curator.min_confidence_to_add,
            max_bullets_per_section=self.config.playbook.max_bullets_per_section
        )
    
    def generate(self, contract_text: str, clause_type: str) -> GeneratorOutput:
        """
        Generate clause extraction without learning.
        
        This is a lightweight call that only uses the Generator agent.
        Use this for inference/evaluation without updating the playbook.
        
        Args:
            contract_text: The contract to extract from
            clause_type: The type of clause to extract
        
        Returns:
            GeneratorOutput with extracted clause
        """
        return self.generator.run(
            contract_text=contract_text,
            clause_type=clause_type,
            max_context_length=self.config.max_contract_length
        )
    
    def run(self, contract_text: str, clause_type: str,
            ground_truth: str = None,
            feedback: str = None,
            stream_callback: Callable[[str, str], None] = None) -> PipelineResult:
        """
        Run the full ACET pipeline with optional learning.
        
        When ground_truth is provided, the pipeline will:
        1. Generate extraction
        2. Reflect on the result vs ground truth
        3. Update playbook with learned insights
        
        Args:
            contract_text: The contract to extract from
            clause_type: The type of clause to extract
            ground_truth: The correct extraction (enables learning)
            feedback: Additional human feedback (optional)
            stream_callback: Callback(stage, message) for progress updates
        
        Returns:
            PipelineResult with all agent outputs
        """
        if stream_callback:
            stream_callback("generator", "Starting extraction...")
        
        # Step 1: Generate extraction
        generator_output = self.generator.run(
            contract_text=contract_text,
            clause_type=clause_type,
            max_context_length=self.config.max_contract_length
        )
        
        if stream_callback:
            stream_callback("generator", f"Extracted with confidence {generator_output.confidence:.2f}")
        
        result = PipelineResult(
            generator_output=generator_output,
            clause_type=clause_type,
            had_ground_truth=ground_truth is not None
        )
        
        # Step 2: Reflect (if ground truth provided and reflector enabled)
        if ground_truth and self.config.reflector.enabled:
            if stream_callback:
                stream_callback("reflector", "Analyzing extraction...")
            
            reflector_output = self.reflector.run(
                clause_type=clause_type,
                generated_output=generator_output,
                ground_truth=ground_truth
            )
            result.reflector_output = reflector_output
            
            if stream_callback:
                stream_callback("reflector", f"Found {len(reflector_output.insights)} insights")
            
            # Step 3: Curate (if curator enabled and reflector suggests updates)
            if self.config.curator.enabled and reflector_output.should_update_playbook:
                if stream_callback:
                    stream_callback("curator", "Updating playbook...")
                
                curator_output = self.curator.run(reflector_output)
                result.curator_output = curator_output
                result.playbook_updated = len(curator_output.updates) > 0
                
                if stream_callback:
                    stream_callback("curator", f"Applied {len(curator_output.updates)} updates")
        
        return result
    
    def get_playbook(self) -> Playbook:
        """Get the current playbook."""
        return self.playbook_manager.get_playbook()
    
    def get_playbook_stats(self) -> Dict[str, Any]:
        """Get playbook statistics."""
        return self.playbook_manager.get_stats()
    
    def reset_playbook(self, use_initial: bool = True) -> None:
        """
        Reset the playbook.
        
        Args:
            use_initial: If True, reset to initial ACORD playbook; else empty
        """
        if use_initial:
            self.playbook_manager.playbook = create_initial_acord_playbook()
        else:
            self.playbook_manager.playbook = Playbook()
        self.playbook_manager.save()
    
    def save_playbook(self, path: str = None) -> None:
        """Save playbook to file."""
        if path:
            original_path = self.playbook_manager.path
            self.playbook_manager.path = path
            self.playbook_manager.save()
            self.playbook_manager.path = original_path
        else:
            self.playbook_manager.save()
    
    def load_playbook(self, path: str) -> None:
        """Load playbook from file."""
        self.playbook_manager.path = path
        self.playbook_manager.load()


def create_pipeline(
    provider: str = "openai",
    model: str = "gpt-4",
    playbook_path: str = "playbook.json",
    temperature: float = 0.0,
    **kwargs
) -> ACORDPipeline:
    """
    Factory function to create an ACORD pipeline with common settings.
    
    Args:
        provider: LLM provider ("openai", "anthropic", "google", "mock")
        model: Model name
        playbook_path: Path to playbook file
        temperature: Generation temperature
        **kwargs: Additional config overrides
    
    Returns:
        Configured ACORDPipeline
    """
    config = ACORDConfig(
        llm=LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature
        ),
        playbook=PlaybookConfig(
            path=playbook_path
        )
    )
    
    # Apply any additional overrides
    if 'max_contract_length' in kwargs:
        config.max_contract_length = kwargs['max_contract_length']
    if 'reflector_enabled' in kwargs:
        config.reflector.enabled = kwargs['reflector_enabled']
    if 'curator_enabled' in kwargs:
        config.curator.enabled = kwargs['curator_enabled']
    
    return ACORDPipeline(config)
