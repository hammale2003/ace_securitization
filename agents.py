"""
ACE Agents: Generator, Reflector, and Curator.

These three agents work together to process questions and evolve the playbook.
Now with semantic retrieval support and extended operations (ADD, REMOVE, MODIFY, MERGE).
"""
import json
from typing import Dict, List, Any, Optional, Generator as GenType, Callable
from dataclasses import dataclass, field
from datetime import datetime

from config import ACEConfig, LLMConfig
from llm_client import LLMClient, create_client, LLMResponse
from playbook import Playbook, PlaybookManager, Bullet, OperationResult
from retriever import PlaybookRetriever, RetrieverConfig, RetrievedBullet
from prompts import (
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_PROSEMIRROR_SYSTEM_PROMPT,
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
from prompts_reformulation_agent import (
    get_reformulation_prompt,
    format_reformulation_user_message,
    parse_reformulation_response
)


@dataclass
class GeneratorOutput:
    """Output from the Generator agent."""
    reasoning: str
    bullet_ids: List[str]
    final_answer: str
    final_answer_prosemirror: Optional[Dict[str, Any]] = None  # ProseMirror JSON format
    reformulation_result: Optional[Dict[str, Any]] = None  # For reformulation modes: {success, alternatives, failure_reason}
    raw_response: Optional[LLMResponse] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "reasoning": self.reasoning,
            "bullet_ids": self.bullet_ids,
            "final_answer": self.final_answer
        }
        if self.final_answer_prosemirror:
            result["final_answer_prosemirror"] = self.final_answer_prosemirror
        if self.reformulation_result:
            result["reformulation_result"] = self.reformulation_result
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratorOutput":
        return cls(
            reasoning=data.get("reasoning", ""),
            bullet_ids=data.get("bullet_ids", []),
            final_answer=data.get("final_answer", ""),
            final_answer_prosemirror=data.get("final_answer_prosemirror"),
            reformulation_result=data.get("reformulation_result")
        )
    
    @classmethod
    def from_llm_response(cls, response: LLMResponse, prosemirror_mode: bool = False) -> "GeneratorOutput":
        parsed = response.parse_json()
        if parsed:
            if prosemirror_mode:
                # ProseMirror mode - final_answer_prosemirror is the main output
                prosemirror_doc = parsed.get("final_answer_prosemirror")
                # Extract plain text from prosemirror for final_answer fallback
                plain_text = cls._extract_text_from_prosemirror(prosemirror_doc) if prosemirror_doc else ""
                return cls(
                    reasoning=parsed.get("reasoning", ""),
                    bullet_ids=parsed.get("bullet_ids", []),
                    final_answer=plain_text,
                    final_answer_prosemirror=prosemirror_doc,
                    raw_response=response
                )
            else:
                return cls(
                    reasoning=parsed.get("reasoning", ""),
                    bullet_ids=parsed.get("bullet_ids", []),
                    final_answer=parsed.get("final_answer", ""),
                    final_answer_prosemirror=parsed.get("final_answer_prosemirror"),
                    raw_response=response
                )
        return cls(
            reasoning="",
            bullet_ids=[],
            final_answer=response.content,
            raw_response=response
        )
    
    @staticmethod
    def _extract_text_from_prosemirror(doc: Dict[str, Any]) -> str:
        """Extract plain text from a ProseMirror document."""
        if not doc or not isinstance(doc, dict):
            return ""
        
        texts = []
        
        def extract_from_node(node):
            if isinstance(node, dict):
                if node.get("type") == "text":
                    texts.append(node.get("text", ""))
                elif "content" in node:
                    for child in node["content"]:
                        extract_from_node(child)
            elif isinstance(node, list):
                for item in node:
                    extract_from_node(item)
        
        extract_from_node(doc)
        return " ".join(texts)


@dataclass
class ReflectorOutput:
    """Output from the Reflector agent."""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[Dict[str, str]]
    removal_candidates: List[str] = field(default_factory=list)
    modification_suggestions: List[Dict[str, str]] = field(default_factory=list)
    extracted_strategies: List[str] = field(default_factory=list)  # Strategies from ground truth comparison
    extracted_pitfalls: List[str] = field(default_factory=list)     # Pitfalls from ground truth comparison
    ground_truth_definition: Optional[Dict[str, Any]] = None
    raw_response: Optional[LLMResponse] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "reasoning": self.reasoning,
            "error_identification": self.error_identification,
            "root_cause_analysis": self.root_cause_analysis,
            "correct_approach": self.correct_approach,
            "key_insight": self.key_insight,
            "bullet_tags": self.bullet_tags,
            "removal_candidates": self.removal_candidates,
            "modification_suggestions": self.modification_suggestions,
            "extracted_strategies": self.extracted_strategies,
            "extracted_pitfalls": self.extracted_pitfalls
        }
        if self.ground_truth_definition:
            result["ground_truth_definition"] = self.ground_truth_definition
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectorOutput":
        return cls(
            reasoning=data.get("reasoning", ""),
            error_identification=data.get("error_identification", ""),
            root_cause_analysis=data.get("root_cause_analysis", ""),
            correct_approach=data.get("correct_approach", ""),
            key_insight=data.get("key_insight", ""),
            bullet_tags=data.get("bullet_tags", []),
            removal_candidates=data.get("removal_candidates", []),
            modification_suggestions=data.get("modification_suggestions", []),
            extracted_strategies=data.get("extracted_strategies", []),
            extracted_pitfalls=data.get("extracted_pitfalls", []),
            ground_truth_definition=data.get("ground_truth_definition")
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
                removal_candidates=parsed.get("removal_candidates", []),
                modification_suggestions=parsed.get("modification_suggestions", []),
                extracted_strategies=parsed.get("extracted_strategies", []),
                extracted_pitfalls=parsed.get("extracted_pitfalls", []),
                ground_truth_definition=parsed.get("ground_truth_definition"),
                raw_response=response
            )
        return cls(
            reasoning=response.content,
            error_identification="",
            root_cause_analysis="",
            correct_approach="",
            key_insight="",
            bullet_tags=[],
            removal_candidates=[],
            modification_suggestions=[],
            extracted_strategies=[],
            extracted_pitfalls=[],
            ground_truth_definition=None,
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
    
    Now supports semantic retrieval to only include relevant bullets.
    Supports both plain text and ProseMirror JSON output formats.
    """
    
    def __init__(self, client: LLMClient, retriever: Optional[PlaybookRetriever] = None):
        self.client = client
        self.retriever = retriever
    
    def generate(
        self,
        question: str,
        playbook: Playbook,
        stream_callback: Optional[Callable[[str], None]] = None,
        use_retrieval: bool = True,
        output_format: str = "text",  # "text" or "prosemirror"
        mode: str = "answer",  # "answer", "enrich", "derive", "remediate", "explore"
        # Reformulation-specific parameters
        reference_clause: str = "",
        constraints: str = "",
        issues: str = "",
        user_prompt: str = "",
        additional_instructions: str = ""
    ) -> GeneratorOutput:
        """
        Generate an answer or reformulation using the playbook.
        
        Args:
            question: The user's question or clause to reformulate
            playbook: The current playbook with accumulated knowledge
            stream_callback: Optional callback for streaming tokens
            use_retrieval: If True and retriever is set, use semantic retrieval
            output_format: "text" for plain text, "prosemirror" for ProseMirror JSON
            mode: "answer" for Q&A, or reformulation modes: "enrich", "derive", "remediate", "explore"
            reference_clause: Canon clause for reformulation modes
            constraints: Constraints for "derive" mode
            issues: Issues for "remediate" mode
            user_prompt: User request for "explore" mode
            additional_instructions: Extra guidance for reformulation
        
        Returns:
            GeneratorOutput with reasoning, bullet_ids, and final_answer
        """
        # Check if this is a reformulation request
        is_reformulation = mode in ["enrich", "derive", "remediate", "explore"]
        
        if is_reformulation:
            # Reformulation mode
            return self._generate_reformulation(
                clause=question,
                playbook=playbook,
                mode=mode,
                output_format=output_format,
                reference_clause=reference_clause,
                constraints=constraints,
                issues=issues,
                user_prompt=user_prompt,
                additional_instructions=additional_instructions,
                stream_callback=stream_callback
            )
        else:
            # Standard Q&A mode
            return self._generate_answer(
                question=question,
                playbook=playbook,
                use_retrieval=use_retrieval,
                output_format=output_format,
                stream_callback=stream_callback
            )
    
    def _generate_answer(
        self,
        question: str,
        playbook: Playbook,
        use_retrieval: bool,
        output_format: str,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> GeneratorOutput:
        """Standard Q&A generation (existing functionality)."""
        # Determine whether to use retrieval
        playbook_size = len(playbook.get_all_bullets())
        
        if (use_retrieval and 
            self.retriever is not None and 
            self.retriever.should_use_retrieval(playbook_size)):
            # Use semantic retrieval
            retrieved_bullets = self.retriever.search(question)
            playbook_text = self.retriever.format_retrieved_for_prompt(retrieved_bullets)
        else:
            # Use full playbook
            playbook_text = playbook.format_for_prompt()
        
        user_message = format_generator_user_message(playbook_text, question)
        
        # Select system prompt based on output format
        if output_format == "prosemirror":
            system_prompt = GENERATOR_PROSEMIRROR_SYSTEM_PROMPT
        else:
            system_prompt = GENERATOR_SYSTEM_PROMPT
        
        if stream_callback and self.client.config.stream:
            full_response = ""
            for chunk in self.client.stream_chat(system_prompt, user_message):
                full_response += chunk
                stream_callback(chunk)
            response = LLMResponse(content=full_response)
        else:
            response = self.client.chat(system_prompt, user_message)
        
        return GeneratorOutput.from_llm_response(
            response, 
            prosemirror_mode=(output_format == "prosemirror")
        )
    
    def _generate_reformulation(
        self,
        clause: str,
        playbook: Playbook,
        mode: str,
        output_format: str,
        reference_clause: str,
        constraints: str,
        issues: str,
        user_prompt: str,
        additional_instructions: str,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> GeneratorOutput:
        """Reformulation generation (new functionality)."""
        # Get playbook context
        playbook_context = playbook.format_for_prompt()
        
        # Get appropriate reformulation prompt
        system_prompt = get_reformulation_prompt(mode, output_format)
        
        # Format user message for reformulation
        user_message = format_reformulation_user_message(
            mode=mode,
            clause=clause,
            reference_clause=reference_clause,
            playbook_context=playbook_context,
            additional_instructions=additional_instructions,
            constraints=constraints,
            issues=issues,
            user_prompt=user_prompt
        )
        
        if stream_callback and self.client.config.stream:
            full_response = ""
            for chunk in self.client.stream_chat(system_prompt, user_message):
                full_response += chunk
                stream_callback(chunk)
            response = LLMResponse(content=full_response)
        else:
            response = self.client.chat(system_prompt, user_message)
        
        # Parse reformulation response
        result = parse_reformulation_response(response.content)
        
        # Convert to GeneratorOutput format - keep the full reformulation structure
        if result["success"] and result["alternatives"]:
            # Use the top-ranked alternative for backward compatibility (final_answer field)
            best_alt = result["alternatives"][0]
            content = best_alt.get("content", "")
            
            # Handle both text and ProseMirror formats
            if isinstance(content, dict):
                # ProseMirror format
                final_answer_text = GeneratorOutput._extract_text_from_prosemirror(content)
            else:
                # Plain text
                final_answer_text = content
            
            return GeneratorOutput(
                reasoning=f"Reformulation mode: {mode}. Generated {len(result['alternatives'])} alternative(s).",
                bullet_ids=[],  # Reformulation doesn't use specific bullets
                final_answer=final_answer_text,  # Best alternative for backward compatibility
                reformulation_result=result,  # Full structured result with all alternatives
                raw_response=response
            )
        else:
            # Reformulation failed
            return GeneratorOutput(
                reasoning=f"Reformulation failed: {result.get('failure_reason', 'Unknown error')}",
                bullet_ids=[],
                final_answer=f"ERROR: Could not reformulate clause. {result.get('failure_reason', '')}",
                reformulation_result=result,  # Include failure info
                raw_response=response
            )
    
    def generate_stream(
        self,
        question: str,
        playbook: Playbook
    ) -> GenType[str, None, GeneratorOutput]:
        """Generator that yields tokens and returns the final output."""
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
    
    Now also identifies candidates for REMOVE and MODIFY operations.
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
        """Perform iterative reflection refinement."""
        num_iterations = iterations or self.max_iterations
        
        current_reflection = self.reflect(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            stream_callback=stream_callback
        )
        
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
            
            if refined.key_insight or refined.reasoning:
                current_reflection = refined
        
        return current_reflection


class Curator:
    """
    The Curator agent updates the playbook based on Reflector insights.
    
    Now supports all operations: ADD, REMOVE, MODIFY, MERGE
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
    ) -> List[OperationResult]:
        """
        Apply the curator's operations to the playbook.
        
        Also updates bullet tags based on reflector feedback.
        
        Returns:
            List of OperationResult objects
        """
        # Apply bullet tags from reflector
        if reflector_output.bullet_tags:
            self.playbook_manager.update_tags(reflector_output.bullet_tags)
        
        # Apply curator operations (ADD, REMOVE, MODIFY, MERGE)
        results = self.playbook_manager.apply_operations(curator_output.operations)
        
        return results


@dataclass
class ACEPipelineResult:
    """Result from running the full ACE pipeline."""
    question: str
    generator_output: GeneratorOutput
    reflector_output: ReflectorOutput
    curator_output: CuratorOutput
    operation_results: List[OperationResult]
    playbook_stats: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    retrieval_used: bool = False
    retrieved_bullet_count: int = 0
    
    # For backwards compatibility
    @property
    def added_bullets(self) -> List[Bullet]:
        """Get bullets that were added (for backwards compatibility)."""
        added = []
        for result in self.operation_results:
            if result.success and result.operation_type == "ADD" and result.bullet_id:
                added.append(Bullet(id=result.bullet_id, content=""))
        return added
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "generator_output": self.generator_output.to_dict(),
            "reflector_output": self.reflector_output.to_dict(),
            "curator_output": self.curator_output.to_dict(),
            "operation_results": [
                {
                    "success": r.success,
                    "operation_type": r.operation_type,
                    "bullet_id": r.bullet_id,
                    "message": r.message,
                    "affected_bullets": r.affected_bullets
                }
                for r in self.operation_results
            ],
            "playbook_stats": self.playbook_stats,
            "timestamp": self.timestamp,
            "retrieval_used": self.retrieval_used,
            "retrieved_bullet_count": self.retrieved_bullet_count
        }


class ACEPipeline:
    """
    The main ACE pipeline that orchestrates Generator, Reflector, and Curator.
    
    Now with semantic retrieval support and extended operations.
    """
    
    def __init__(self, config: ACEConfig = None, enable_retrieval: bool = True):
        self.config = config or ACEConfig()
        
        # Initialize LLM client
        self.client = create_client(self.config.llm)
        
        # Initialize playbook manager
        self.playbook_manager = PlaybookManager(self.config.playbook)
        
        # Initialize retriever if enabled
        self.retriever = None
        if enable_retrieval:
            retriever_config = RetrieverConfig(
                top_k=10,
                similarity_threshold=0.3,
                embedding_provider="simple",
                min_playbook_size_for_retrieval=15
            )
            self.retriever = PlaybookRetriever(retriever_config)
            
            # Index the playbook
            playbook = self.playbook_manager.get_playbook()
            self.retriever.index_playbook(playbook)
            
            # Link retriever to playbook manager for automatic index updates
            self.playbook_manager.set_retriever(self.retriever)
        
        # Initialize agents
        self.generator = Generator(self.client, self.retriever)
        self.reflector = Reflector(self.client, self.config.max_reflector_iterations)
        self.curator = Curator(self.client, self.playbook_manager)
    
    def run(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        stream_callbacks: Optional[Dict[str, Callable[[str], None]]] = None,
        use_retrieval: bool = True,
        output_format: str = "text",  # "text" or "prosemirror"
        mode: str = "answer",  # "answer", "enrich", "derive", "remediate", "explore"
        # Reformulation-specific parameters
        reference_clause: str = "",
        constraints: str = "",
        issues: str = "",
        user_prompt: str = "",
        additional_instructions: str = ""
    ) -> ACEPipelineResult:
        """
        Run the full ACE pipeline for a question or clause reformulation.
        
        Args:
            question: The user's question or clause to reformulate
            ground_truth: Optional correct answer for training
            feedback: Optional human feedback
            stream_callbacks: Dict with callbacks for each agent
            use_retrieval: Whether to use semantic retrieval
            output_format: "text" for plain text, "prosemirror" for ProseMirror JSON
            mode: "answer" for Q&A, or reformulation: "enrich", "derive", "remediate", "explore"
            reference_clause: Canon clause for reformulation modes
            constraints: Constraints for "derive" mode
            issues: Issues for "remediate" mode
            user_prompt: User request for "explore" mode
            additional_instructions: Extra guidance for reformulation
        """
        callbacks = stream_callbacks or {}
        playbook = self.playbook_manager.get_playbook()
        
        # Track retrieval usage
        retrieval_used = False
        retrieved_count = 0
        
        if use_retrieval and self.retriever:
            playbook_size = len(playbook.get_all_bullets())
            retrieval_used = self.retriever.should_use_retrieval(playbook_size)
            if retrieval_used:
                retrieved = self.retriever.search(question)
                retrieved_count = len(retrieved)
        
        # Step 1: Generate answer or reformulation
        generator_output = self.generator.generate(
            question=question,
            playbook=playbook,
            stream_callback=callbacks.get("generator"),
            use_retrieval=use_retrieval,
            output_format=output_format,
            mode=mode,
            reference_clause=reference_clause,
            constraints=constraints,
            issues=issues,
            user_prompt=user_prompt,
            additional_instructions=additional_instructions
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
        operation_results = self.curator.apply_updates(curator_output, reflector_output)
        
        # Get updated stats
        playbook_stats = self.playbook_manager.get_playbook().get_stats()
        
        return ACEPipelineResult(
            question=question,
            generator_output=generator_output,
            reflector_output=reflector_output,
            curator_output=curator_output,
            operation_results=operation_results,
            playbook_stats=playbook_stats,
            retrieval_used=retrieval_used,
            retrieved_bullet_count=retrieved_count
        )
    
    def run_with_refinement(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        reflector_iterations: int = 3,
        stream_callbacks: Optional[Dict[str, Callable[[str], None]]] = None,
        use_retrieval: bool = True
    ) -> ACEPipelineResult:
        """Run the ACE pipeline with iterative reflector refinement."""
        callbacks = stream_callbacks or {}
        playbook = self.playbook_manager.get_playbook()
        
        retrieval_used = False
        retrieved_count = 0
        
        if use_retrieval and self.retriever:
            playbook_size = len(playbook.get_all_bullets())
            retrieval_used = self.retriever.should_use_retrieval(playbook_size)
            if retrieval_used:
                retrieved = self.retriever.search(question)
                retrieved_count = len(retrieved)
        
        generator_output = self.generator.generate(
            question=question,
            playbook=playbook,
            stream_callback=callbacks.get("generator"),
            use_retrieval=use_retrieval
        )
        
        reflector_output = self.reflector.reflect_with_refinement(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            iterations=reflector_iterations,
            stream_callback=callbacks.get("reflector")
        )
        
        curator_output = self.curator.curate(
            question=question,
            generator_output=generator_output,
            reflector_output=reflector_output,
            playbook=playbook,
            stream_callback=callbacks.get("curator")
        )
        
        operation_results = self.curator.apply_updates(curator_output, reflector_output)
        
        playbook_stats = self.playbook_manager.get_playbook().get_stats()
        
        return ACEPipelineResult(
            question=question,
            generator_output=generator_output,
            reflector_output=reflector_output,
            curator_output=curator_output,
            operation_results=operation_results,
            playbook_stats=playbook_stats,
            retrieval_used=retrieval_used,
            retrieved_bullet_count=retrieved_count
        )
    
    def generate_only(
        self,
        question: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        use_retrieval: bool = True,
        output_format: str = "text",  # "text" or "prosemirror"
        mode: str = "answer",  # "answer", "enrich", "derive", "remediate", "explore"
        # Reformulation-specific parameters
        reference_clause: str = "",
        constraints: str = "",
        issues: str = "",
        user_prompt: str = "",
        additional_instructions: str = ""
    ) -> GeneratorOutput:
        """
        Only run the Generator without reflection or curation.
        
        Useful for inference after the playbook has been trained.
        
        Args:
            question: The user's question or clause to reformulate
            stream_callback: Optional callback for streaming
            use_retrieval: Whether to use semantic retrieval
            output_format: "text" for plain text, "prosemirror" for ProseMirror JSON
            mode: "answer" for Q&A, or reformulation: "enrich", "derive", "remediate", "explore"
            reference_clause: Canon clause for reformulation modes
            constraints: Constraints for "derive" mode
            issues: Issues for "remediate" mode
            user_prompt: User request for "explore" mode
            additional_instructions: Extra guidance for reformulation
        """
        playbook = self.playbook_manager.get_playbook()
        return self.generator.generate(
            question=question,
            playbook=playbook,
            stream_callback=stream_callback,
            use_retrieval=use_retrieval,
            output_format=output_format,
            mode=mode,
            reference_clause=reference_clause,
            constraints=constraints,
            issues=issues,
            user_prompt=user_prompt,
            additional_instructions=additional_instructions
        )
    
    def auto_cleanup(self, harmful_threshold: int = 5,
                     effectiveness_threshold: float = -0.3) -> List[str]:
        """
        Automatically remove harmful bullets without LLM calls.
        
        Returns list of removed bullet IDs.
        """
        return self.playbook_manager.auto_cleanup(
            harmful_threshold=harmful_threshold,
            effectiveness_threshold=effectiveness_threshold
        )
    
    def reindex_playbook(self) -> int:
        """
        Re-index the playbook for retrieval.
        
        Call this after manual playbook modifications.
        Returns number of bullets indexed.
        """
        if self.retriever:
            playbook = self.playbook_manager.get_playbook()
            return self.retriever.index_playbook(playbook)
        return 0
    
    def get_playbook(self) -> Playbook:
        """Get the current playbook."""
        return self.playbook_manager.get_playbook()
    
    def get_playbook_stats(self) -> Dict[str, Any]:
        """Get playbook statistics."""
        return self.playbook_manager.get_playbook().get_stats()
    
    def get_retriever_stats(self) -> Optional[Dict[str, Any]]:
        """Get retriever statistics."""
        if self.retriever:
            return self.retriever.get_stats()
        return None