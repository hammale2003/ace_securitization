"""
ACE Agents: Generator, Reflector, and Curator.

These three agents work together to process questions and evolve the playbook.
"""
import json
from typing import Dict, List, Any, Optional, Generator as GenType, Callable
from dataclasses import dataclass, field
from datetime import datetime

from config import ACEConfig, LLMConfig
from llm_client import LLMClient, create_client, LLMResponse
from playbook import Playbook, PlaybookManager, Bullet
from prompts import (
    GENERATOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    CURATOR_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    format_generator_user_message,
    format_reflector_user_message,
    format_curator_user_message,
    format_refinement_user_message,
    get_empty_generator_response,
    get_empty_reflector_response,
    get_empty_curator_response
)


@dataclass
class GeneratorOutput:
    """Output from the Generator agent."""
    reasoning: str
    bullet_ids: List[str]
    final_answer: str
    raw_response: Optional[LLMResponse] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "bullet_ids": self.bullet_ids,
            "final_answer": self.final_answer
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratorOutput":
        return cls(
            reasoning=data.get("reasoning", ""),
            bullet_ids=data.get("bullet_ids", []),
            final_answer=data.get("final_answer", "")
        )
    
    @classmethod
    def from_llm_response(cls, response: LLMResponse) -> "GeneratorOutput":
        parsed = response.parse_json()
        if parsed:
            return cls(
                reasoning=parsed.get("reasoning", ""),
                bullet_ids=parsed.get("bullet_ids", []),
                final_answer=parsed.get("final_answer", ""),
                raw_response=response
            )
        # Fallback: treat entire response as final answer
        return cls(
            reasoning="",
            bullet_ids=[],
            final_answer=response.content,
            raw_response=response
        )


@dataclass
class ReflectorOutput:
    """Output from the Reflector agent."""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[Dict[str, str]]
    raw_response: Optional[LLMResponse] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "error_identification": self.error_identification,
            "root_cause_analysis": self.root_cause_analysis,
            "correct_approach": self.correct_approach,
            "key_insight": self.key_insight,
            "bullet_tags": self.bullet_tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectorOutput":
        return cls(
            reasoning=data.get("reasoning", ""),
            error_identification=data.get("error_identification", ""),
            root_cause_analysis=data.get("root_cause_analysis", ""),
            correct_approach=data.get("correct_approach", ""),
            key_insight=data.get("key_insight", ""),
            bullet_tags=data.get("bullet_tags", [])
        )
    
    @classmethod
    def from_llm_response(cls, response: LLMResponse) -> "ReflectorOutput":
        parsed = response.parse_json()
        if parsed:
            return cls(
                reasoning=parsed.get("reasoning", ""),
                error_identification=parsed.get("error_identification", ""),
                root_cause_analysis=parsed.get("root_cause_analysis", ""),
                correct_approach=parsed.get("correct_approach", ""),
                key_insight=parsed.get("key_insight", ""),
                bullet_tags=parsed.get("bullet_tags", []),
                raw_response=response
            )
        return cls(
            reasoning=response.content,
            error_identification="",
            root_cause_analysis="",
            correct_approach="",
            key_insight="",
            bullet_tags=[],
            raw_response=response
        )


@dataclass
class CuratorOutput:
    """Output from the Curator agent."""
    reasoning: str
    operations: List[Dict[str, Any]]
    raw_response: Optional[LLMResponse] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "operations": self.operations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CuratorOutput":
        return cls(
            reasoning=data.get("reasoning", ""),
            operations=data.get("operations", [])
        )
    
    @classmethod
    def from_llm_response(cls, response: LLMResponse) -> "CuratorOutput":
        parsed = response.parse_json()
        if parsed:
            return cls(
                reasoning=parsed.get("reasoning", ""),
                operations=parsed.get("operations", []),
                raw_response=response
            )
        return cls(
            reasoning=response.content,
            operations=[],
            raw_response=response
        )


class Generator:
    """
    The Generator agent produces answers using the playbook.
    
    It reads the playbook, applies relevant knowledge, and generates
    structured responses with reasoning and bullet references.
    """
    
    def __init__(self, client: LLMClient):
        self.client = client
    
    def generate(
        self,
        question: str,
        playbook: Playbook,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> GeneratorOutput:
        """
        Generate an answer for the given question using the playbook.
        
        Args:
            question: The user's question
            playbook: The current playbook with accumulated knowledge
            stream_callback: Optional callback for streaming tokens
        
        Returns:
            GeneratorOutput with reasoning, bullet_ids, and final_answer
        """
        playbook_text = playbook.format_for_prompt()
        user_message = format_generator_user_message(playbook_text, question)
        
        if stream_callback and self.client.config.stream:
            # Streaming mode
            full_response = ""
            for chunk in self.client.stream_chat(GENERATOR_SYSTEM_PROMPT, user_message):
                full_response += chunk
                stream_callback(chunk)
            
            response = LLMResponse(content=full_response)
        else:
            # Non-streaming mode
            response = self.client.chat(GENERATOR_SYSTEM_PROMPT, user_message)
        
        return GeneratorOutput.from_llm_response(response)
    
    def generate_stream(
        self,
        question: str,
        playbook: Playbook
    ) -> GenType[str, None, GeneratorOutput]:
        """
        Generator that yields tokens and returns the final output.
        
        Usage:
            gen = generator.generate_stream(question, playbook)
            for token in gen:
                print(token, end="")
            output = gen.value  # Not available in Python, use generate() with callback instead
        """
        playbook_text = playbook.format_for_prompt()
        user_message = format_generator_user_message(playbook_text, question)
        
        full_response = ""
        for chunk in self.client.stream_chat(GENERATOR_SYSTEM_PROMPT, user_message):
            full_response += chunk
            yield chunk
        
        response = LLMResponse(content=full_response)
        return GeneratorOutput.from_llm_response(response)


class Reflector:
    """
    The Reflector agent analyzes Generator output and extracts insights.
    
    It identifies errors, root causes, and key learnings that should
    be added to the playbook.
    """
    
    def __init__(self, client: LLMClient, max_iterations: int = 5):
        self.client = client
        self.max_iterations = max_iterations
    
    def reflect(
        self,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> ReflectorOutput:
        """
        Analyze the Generator's output and produce reflection.
        
        Args:
            question: The original question
            generator_output: The Generator's response
            playbook: The current playbook
            ground_truth: Optional correct answer for comparison
            feedback: Optional human feedback
            stream_callback: Optional callback for streaming tokens
        
        Returns:
            ReflectorOutput with analysis and insights
        """
        playbook_text = playbook.format_for_prompt()
        user_message = format_reflector_user_message(
            question=question,
            generator_output=generator_output.to_dict(),
            playbook_text=playbook_text,
            ground_truth=ground_truth,
            feedback=feedback
        )
        
        if stream_callback and self.client.config.stream:
            full_response = ""
            for chunk in self.client.stream_chat(REFLECTOR_SYSTEM_PROMPT, user_message):
                full_response += chunk
                stream_callback(chunk)
            response = LLMResponse(content=full_response)
        else:
            response = self.client.chat(REFLECTOR_SYSTEM_PROMPT, user_message)
        
        return ReflectorOutput.from_llm_response(response)
    
    def reflect_with_refinement(
        self,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        iterations: Optional[int] = None,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> ReflectorOutput:
        """
        Perform iterative reflection refinement.
        
        Multiple passes to improve the quality of insights.
        """
        num_iterations = iterations or self.max_iterations
        
        # Initial reflection
        current_reflection = self.reflect(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            stream_callback=stream_callback
        )
        
        # Iterative refinement
        playbook_text = playbook.format_for_prompt()
        
        for i in range(1, num_iterations):
            user_message = format_refinement_user_message(
                question=question,
                generator_output=generator_output.to_dict(),
                previous_reflection=current_reflection.to_dict(),
                playbook_text=playbook_text,
                iteration=i + 1
            )
            
            if stream_callback and self.client.config.stream:
                full_response = ""
                for chunk in self.client.stream_chat(REFINEMENT_SYSTEM_PROMPT, user_message):
                    full_response += chunk
                    stream_callback(chunk)
                response = LLMResponse(content=full_response)
            else:
                response = self.client.chat(REFINEMENT_SYSTEM_PROMPT, user_message)
            
            refined = ReflectorOutput.from_llm_response(response)
            
            # Only update if refinement produced valid output
            if refined.key_insight or refined.reasoning:
                current_reflection = refined
        
        return current_reflection


class Curator:
    """
    The Curator agent updates the playbook based on Reflector insights.
    
    It determines what new knowledge should be added, avoiding
    redundancy and maintaining organization.
    """
    
    def __init__(self, client: LLMClient, playbook_manager: PlaybookManager):
        self.client = client
        self.playbook_manager = playbook_manager
    
    def curate(
        self,
        question: str,
        generator_output: GeneratorOutput,
        reflector_output: ReflectorOutput,
        playbook: Playbook,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> CuratorOutput:
        """
        Determine what updates to make to the playbook.
        
        Args:
            question: The original question
            generator_output: The Generator's response
            reflector_output: The Reflector's analysis
            playbook: The current playbook
            stream_callback: Optional callback for streaming tokens
        
        Returns:
            CuratorOutput with reasoning and operations to apply
        """
        playbook_text = playbook.format_for_prompt()
        user_message = format_curator_user_message(
            question=question,
            generator_output=generator_output.to_dict(),
            reflector_output=reflector_output.to_dict(),
            playbook_text=playbook_text
        )
        
        if stream_callback and self.client.config.stream:
            full_response = ""
            for chunk in self.client.stream_chat(CURATOR_SYSTEM_PROMPT, user_message):
                full_response += chunk
                stream_callback(chunk)
            response = LLMResponse(content=full_response)
        else:
            response = self.client.chat(CURATOR_SYSTEM_PROMPT, user_message)
        
        return CuratorOutput.from_llm_response(response)
    
    def apply_updates(
        self,
        curator_output: CuratorOutput,
        reflector_output: ReflectorOutput
    ) -> List[Bullet]:
        """
        Apply the curator's operations to the playbook.
        
        Also updates bullet tags based on reflector feedback.
        
        Returns:
            List of newly added bullets
        """
        # Apply bullet tags from reflector
        if reflector_output.bullet_tags:
            self.playbook_manager.update_tags(reflector_output.bullet_tags)
        
        # Apply curator operations (additions)
        added_bullets = self.playbook_manager.apply_operations(curator_output.operations)
        
        return added_bullets


@dataclass
class ACEPipelineResult:
    """Result from running the full ACE pipeline."""
    question: str
    generator_output: GeneratorOutput
    reflector_output: ReflectorOutput
    curator_output: CuratorOutput
    added_bullets: List[Bullet]
    playbook_stats: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "generator_output": self.generator_output.to_dict(),
            "reflector_output": self.reflector_output.to_dict(),
            "curator_output": self.curator_output.to_dict(),
            "added_bullets": [b.to_dict() for b in self.added_bullets],
            "playbook_stats": self.playbook_stats,
            "timestamp": self.timestamp
        }


class ACEPipeline:
    """
    The main ACE pipeline that orchestrates Generator, Reflector, and Curator.
    """
    
    def __init__(self, config: ACEConfig = None):
        self.config = config or ACEConfig()
        
        # Initialize LLM client
        self.client = create_client(self.config.llm)
        
        # Initialize playbook manager
        self.playbook_manager = PlaybookManager(self.config.playbook)
        
        # Initialize agents
        self.generator = Generator(self.client)
        self.reflector = Reflector(self.client, self.config.max_reflector_iterations)
        self.curator = Curator(self.client, self.playbook_manager)
    
    def run(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        stream_callbacks: Optional[Dict[str, Callable[[str], None]]] = None
    ) -> ACEPipelineResult:
        """
        Run the full ACE pipeline for a question.
        
        Args:
            question: The user's question
            ground_truth: Optional correct answer for training
            feedback: Optional human feedback
            stream_callbacks: Dict with keys 'generator', 'reflector', 'curator'
                             mapping to streaming callback functions
        
        Returns:
            ACEPipelineResult with all outputs and updated playbook stats
        """
        callbacks = stream_callbacks or {}
        playbook = self.playbook_manager.get_playbook()
        
        # Step 1: Generate answer
        generator_output = self.generator.generate(
            question=question,
            playbook=playbook,
            stream_callback=callbacks.get("generator")
        )
        
        # Step 2: Reflect on the answer
        reflector_output = self.reflector.reflect(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            stream_callback=callbacks.get("reflector")
        )
        
        # Step 3: Curate new knowledge
        curator_output = self.curator.curate(
            question=question,
            generator_output=generator_output,
            reflector_output=reflector_output,
            playbook=playbook,
            stream_callback=callbacks.get("curator")
        )
        
        # Step 4: Apply updates
        added_bullets = self.curator.apply_updates(curator_output, reflector_output)
        
        # Get updated stats
        playbook_stats = self.playbook_manager.get_playbook().get_stats()
        
        return ACEPipelineResult(
            question=question,
            generator_output=generator_output,
            reflector_output=reflector_output,
            curator_output=curator_output,
            added_bullets=added_bullets,
            playbook_stats=playbook_stats
        )
    
    def run_with_refinement(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        reflector_iterations: int = 3,
        stream_callbacks: Optional[Dict[str, Callable[[str], None]]] = None
    ) -> ACEPipelineResult:
        """
        Run the ACE pipeline with iterative reflector refinement.
        """
        callbacks = stream_callbacks or {}
        playbook = self.playbook_manager.get_playbook()
        
        # Step 1: Generate answer
        generator_output = self.generator.generate(
            question=question,
            playbook=playbook,
            stream_callback=callbacks.get("generator")
        )
        
        # Step 2: Reflect with refinement
        reflector_output = self.reflector.reflect_with_refinement(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            iterations=reflector_iterations,
            stream_callback=callbacks.get("reflector")
        )
        
        # Step 3: Curate new knowledge
        curator_output = self.curator.curate(
            question=question,
            generator_output=generator_output,
            reflector_output=reflector_output,
            playbook=playbook,
            stream_callback=callbacks.get("curator")
        )
        
        # Step 4: Apply updates
        added_bullets = self.curator.apply_updates(curator_output, reflector_output)
        
        playbook_stats = self.playbook_manager.get_playbook().get_stats()
        
        return ACEPipelineResult(
            question=question,
            generator_output=generator_output,
            reflector_output=reflector_output,
            curator_output=curator_output,
            added_bullets=added_bullets,
            playbook_stats=playbook_stats
        )
    
    def generate_only(
        self,
        question: str,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> GeneratorOutput:
        """
        Only run the Generator without reflection or curation.
        
        Useful for inference after the playbook has been trained.
        """
        playbook = self.playbook_manager.get_playbook()
        return self.generator.generate(
            question=question,
            playbook=playbook,
            stream_callback=stream_callback
        )
    
    def get_playbook(self) -> Playbook:
        """Get the current playbook."""
        return self.playbook_manager.get_playbook()
    
    def get_playbook_stats(self) -> Dict[str, Any]:
        """Get playbook statistics."""
        return self.playbook_manager.get_playbook().get_stats()
