"""
ACET Agents for ACORD Clause Extraction.

Implements the three-agent architecture:
- Generator: Extracts clauses using playbook knowledge
- Reflector: Analyzes outputs and identifies learning opportunities  
- Curator: Updates the playbook with new knowledge
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from llm_client import BaseLLMClient, LLMResponse
from playbook import Playbook, PlaybookManager, PlaybookBullet


# ============================================================================
# Data Classes for Agent Outputs
# ============================================================================

@dataclass
class GeneratorOutput:
    """Output from the Generator agent."""
    final_answer: str
    reasoning: str
    bullet_ids: List[str]
    confidence: float
    raw_response: str = ""


@dataclass
class ReflectorInsight:
    """A single insight from the Reflector."""
    insight_type: str  # 'strategy', 'definition', 'pitfall', 'template'
    content: str
    confidence: float
    tags: List[str] = field(default_factory=list)


@dataclass
class ReflectorOutput:
    """Output from the Reflector agent."""
    analysis: str
    insights: List[ReflectorInsight]
    overall_confidence: float
    should_update_playbook: bool
    raw_response: str = ""


@dataclass
class CuratorUpdate:
    """A single update action from the Curator."""
    action: str  # 'add', 'modify', 'remove'
    section: str  # 'strategies', 'definitions', 'pitfalls', 'templates'
    content: str
    bullet_id: Optional[str] = None  # For modify/remove
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class CuratorOutput:
    """Output from the Curator agent."""
    updates: List[CuratorUpdate]
    reasoning: str
    raw_response: str = ""


# ============================================================================
# Base Agent Class
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for ACET agents."""
    
    def __init__(self, llm_client: BaseLLMClient, playbook_manager: PlaybookManager):
        self.llm = llm_client
        self.playbook_manager = playbook_manager
    
    @property
    def playbook(self) -> Playbook:
        return self.playbook_manager.get_playbook()
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the agent's task."""
        pass


# ============================================================================
# Generator Agent
# ============================================================================

class GeneratorAgent(BaseAgent):
    """
    Generator Agent: Extracts clauses from contracts using playbook knowledge.
    
    The Generator:
    1. Retrieves relevant playbook bullets for the clause type
    2. Constructs a prompt with contract context and playbook guidance
    3. Generates the clause extraction with reasoning
    """
    
    SYSTEM_PROMPT = """You are an expert legal document analyst specializing in contract clause extraction.
Your task is to extract specific clause types from contracts accurately and completely.

When extracting clauses:
1. Look for the exact clause type requested
2. Extract the COMPLETE clause text verbatim from the contract
3. Include any exceptions, qualifications, or related sub-provisions
4. If the clause is not present, clearly state that

Always explain your reasoning before providing the extracted clause."""

    EXTRACTION_PROMPT_TEMPLATE = """## Playbook Knowledge
{playbook_context}

## Contract Text
{contract_text}

## Task
Extract the **{clause_type}** clause from the contract above.

Instructions:
1. First, explain your reasoning for identifying (or not identifying) this clause
2. Reference any playbook bullets (by ID) that helped guide your extraction
3. Provide the extracted clause text verbatim, or state if not found

Format your response as:
REASONING: [Your step-by-step reasoning]
BULLETS_USED: [Comma-separated bullet IDs, or "none"]
CONFIDENCE: [0.0-1.0 score]
EXTRACTED_CLAUSE: [The exact clause text, or "NOT_FOUND: [explanation]"]"""

    def run(self, contract_text: str, clause_type: str, 
            max_context_length: int = 8000) -> GeneratorOutput:
        """
        Extract a clause from a contract.
        
        Args:
            contract_text: The full contract text
            clause_type: The type of clause to extract
            max_context_length: Maximum characters of contract to include
        
        Returns:
            GeneratorOutput with extracted clause and metadata
        """
        # Truncate contract if needed
        if len(contract_text) > max_context_length:
            contract_text = contract_text[:max_context_length] + "\n... [truncated]"
        
        # Get relevant playbook context
        playbook_context = self.playbook.to_prompt_context(clause_type, max_bullets=15)
        
        # Build prompt
        prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(
            playbook_context=playbook_context,
            contract_text=contract_text,
            clause_type=clause_type
        )
        
        # Call LLM
        response = self.llm.complete_single(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.0
        )
        
        # Parse response
        return self._parse_response(response.content)
    
    def _parse_response(self, response: str) -> GeneratorOutput:
        """Parse Generator response into structured output."""
        reasoning = ""
        bullet_ids = []
        confidence = 0.5
        final_answer = ""
        
        # Extract REASONING
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=BULLETS_USED:|CONFIDENCE:|EXTRACTED_CLAUSE:|$)', 
                                    response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # Extract BULLETS_USED
        bullets_match = re.search(r'BULLETS_USED:\s*(.+?)(?=CONFIDENCE:|EXTRACTED_CLAUSE:|$)', 
                                  response, re.DOTALL | re.IGNORECASE)
        if bullets_match:
            bullets_str = bullets_match.group(1).strip()
            if bullets_str.lower() != 'none':
                bullet_ids = [b.strip() for b in bullets_str.split(',') if b.strip()]
        
        # Extract CONFIDENCE
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5
        
        # Extract EXTRACTED_CLAUSE
        clause_match = re.search(r'EXTRACTED_CLAUSE:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
        if clause_match:
            final_answer = clause_match.group(1).strip()
        else:
            # Fallback: use the last part of the response
            final_answer = response.split('\n')[-1].strip()
        
        return GeneratorOutput(
            final_answer=final_answer,
            reasoning=reasoning,
            bullet_ids=bullet_ids,
            confidence=confidence,
            raw_response=response
        )


# ============================================================================
# Reflector Agent
# ============================================================================

class ReflectorAgent(BaseAgent):
    """
    Reflector Agent: Analyzes extraction results and identifies learning opportunities.
    
    The Reflector:
    1. Compares generated output to ground truth (when available)
    2. Identifies what worked and what didn't
    3. Generates insights that could improve future extractions
    """
    
    SYSTEM_PROMPT = """You are an expert at analyzing legal document extraction results and identifying patterns for improvement.
Your task is to compare extraction outputs to ground truth and generate actionable insights.

Focus on:
1. What strategies led to successful extractions
2. What patterns or definitions would have helped
3. What pitfalls or mistakes should be avoided
4. What templates or examples would be useful

Generate insights that are specific, actionable, and would help improve future extractions."""

    REFLECTION_PROMPT_TEMPLATE = """## Current Playbook Knowledge
{playbook_context}

## Extraction Task
Clause Type: {clause_type}

## Generated Output
{generated_output}

## Ground Truth
{ground_truth}

## Analysis Task
Compare the generated output to the ground truth and identify:
1. What aspects of the extraction were correct or incorrect
2. What strategies, definitions, pitfalls, or templates would improve future extractions
3. Whether the playbook should be updated with new knowledge

Format your response as JSON:
{{
    "analysis": "Your detailed analysis of the extraction quality",
    "insights": [
        {{
            "type": "strategy|definition|pitfall|template",
            "content": "The specific insight",
            "confidence": 0.0-1.0,
            "tags": ["relevant", "tags"]
        }}
    ],
    "overall_confidence": 0.0-1.0,
    "should_update_playbook": true|false
}}"""

    def run(self, clause_type: str, generated_output: GeneratorOutput,
            ground_truth: str) -> ReflectorOutput:
        """
        Analyze extraction results and generate insights.
        
        Args:
            clause_type: The type of clause being extracted
            generated_output: The Generator's output
            ground_truth: The correct extraction (if known)
        
        Returns:
            ReflectorOutput with analysis and insights
        """
        playbook_context = self.playbook.to_prompt_context(clause_type, max_bullets=10)
        
        prompt = self.REFLECTION_PROMPT_TEMPLATE.format(
            playbook_context=playbook_context,
            clause_type=clause_type,
            generated_output=f"Answer: {generated_output.final_answer}\nReasoning: {generated_output.reasoning}",
            ground_truth=ground_truth
        )
        
        response = self.llm.complete_single(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.2  # Slight temperature for diversity
        )
        
        return self._parse_response(response.content)
    
    def _parse_response(self, response: str) -> ReflectorOutput:
        """Parse Reflector response into structured output."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            insights = []
            for insight_data in data.get('insights', []):
                insights.append(ReflectorInsight(
                    insight_type=insight_data.get('type', 'strategy'),
                    content=insight_data.get('content', ''),
                    confidence=float(insight_data.get('confidence', 0.5)),
                    tags=insight_data.get('tags', [])
                ))
            
            return ReflectorOutput(
                analysis=data.get('analysis', ''),
                insights=insights,
                overall_confidence=float(data.get('overall_confidence', 0.5)),
                should_update_playbook=data.get('should_update_playbook', False),
                raw_response=response
            )
        
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback parsing
            return ReflectorOutput(
                analysis=response,
                insights=[],
                overall_confidence=0.5,
                should_update_playbook=False,
                raw_response=response
            )


# ============================================================================
# Curator Agent
# ============================================================================

class CuratorAgent(BaseAgent):
    """
    Curator Agent: Updates the playbook with new knowledge from Reflector insights.
    
    The Curator:
    1. Reviews Reflector insights for quality and relevance
    2. Checks for duplicates and conflicts with existing playbook
    3. Decides which updates to apply
    4. Applies incremental delta updates to the playbook
    """
    
    SYSTEM_PROMPT = """You are an expert at curating and organizing knowledge bases for legal document analysis.
Your task is to review proposed insights and decide how to update the playbook.

When curating:
1. Only add high-quality, specific, actionable insights
2. Avoid duplicating existing playbook content
3. Ensure new content is well-written and clear
4. Consider whether content should modify existing entries vs. create new ones

Be selective - only recommend updates that will genuinely improve future extractions."""

    CURATION_PROMPT_TEMPLATE = """## Current Playbook
{playbook_context}

## Proposed Insights from Reflector
{insights}

## Curation Task
Review the proposed insights and decide which updates to apply to the playbook.
For each insight, decide whether to:
- ADD: Create a new playbook entry
- MODIFY: Update an existing entry (specify which one)
- SKIP: Don't add (explain why - duplicate, low quality, etc.)

Format your response as JSON:
{{
    "updates": [
        {{
            "action": "add|modify|remove",
            "section": "strategies|definitions|pitfalls|templates",
            "content": "The content to add or the updated content",
            "bullet_id": "existing-id (for modify/remove only)",
            "tags": ["relevant", "tags"],
            "confidence": 0.0-1.0
        }}
    ],
    "reasoning": "Overall explanation of your curation decisions"
}}"""

    def __init__(self, llm_client: BaseLLMClient, playbook_manager: PlaybookManager,
                 min_confidence: float = 0.7, max_bullets_per_section: int = 100):
        super().__init__(llm_client, playbook_manager)
        self.min_confidence = min_confidence
        self.max_bullets_per_section = max_bullets_per_section
    
    def run(self, reflector_output: ReflectorOutput) -> CuratorOutput:
        """
        Process Reflector insights and update the playbook.
        
        Args:
            reflector_output: Output from the Reflector agent
        
        Returns:
            CuratorOutput with applied updates
        """
        if not reflector_output.should_update_playbook or not reflector_output.insights:
            return CuratorOutput(updates=[], reasoning="No updates needed.", raw_response="")
        
        # Filter insights by confidence
        high_confidence_insights = [
            i for i in reflector_output.insights 
            if i.confidence >= self.min_confidence
        ]
        
        if not high_confidence_insights:
            return CuratorOutput(
                updates=[], 
                reasoning="No insights met the confidence threshold.",
                raw_response=""
            )
        
        # Format insights for prompt
        insights_str = "\n".join([
            f"- [{i.insight_type}] (confidence: {i.confidence:.2f}) {i.content}"
            for i in high_confidence_insights
        ])
        
        playbook_context = self.playbook.to_prompt_context(max_bullets=20)
        
        prompt = self.CURATION_PROMPT_TEMPLATE.format(
            playbook_context=playbook_context,
            insights=insights_str
        )
        
        response = self.llm.complete_single(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.0
        )
        
        curator_output = self._parse_response(response.content)
        
        # Apply updates to playbook
        self._apply_updates(curator_output.updates)
        
        return curator_output
    
    def _parse_response(self, response: str) -> CuratorOutput:
        """Parse Curator response into structured output."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            updates = []
            for update_data in data.get('updates', []):
                if update_data.get('action') in ['add', 'modify', 'remove']:
                    updates.append(CuratorUpdate(
                        action=update_data['action'],
                        section=update_data.get('section', 'strategies'),
                        content=update_data.get('content', ''),
                        bullet_id=update_data.get('bullet_id'),
                        tags=update_data.get('tags', []),
                        confidence=float(update_data.get('confidence', 0.8))
                    ))
            
            return CuratorOutput(
                updates=updates,
                reasoning=data.get('reasoning', ''),
                raw_response=response
            )
        
        except (json.JSONDecodeError, ValueError):
            return CuratorOutput(updates=[], reasoning="Failed to parse response", raw_response=response)
    
    def _apply_updates(self, updates: List[CuratorUpdate]) -> None:
        """Apply updates to the playbook."""
        for update in updates:
            section = update.section
            
            # Check section limits
            current_count = len(self.playbook.get_section(section))
            
            if update.action == 'add':
                if current_count < self.max_bullets_per_section:
                    # Check for duplicates (simple similarity check)
                    if not self._is_duplicate(update.content, section):
                        self.playbook.add_bullet(
                            section=section,
                            content=update.content,
                            tags=update.tags,
                            source="learned",
                            confidence=update.confidence
                        )
            
            elif update.action == 'modify' and update.bullet_id:
                self.playbook.update_bullet(
                    bullet_id=update.bullet_id,
                    content=update.content,
                    tags=update.tags
                )
            
            elif update.action == 'remove' and update.bullet_id:
                self.playbook.remove_bullet(update.bullet_id)
        
        # Save playbook after updates
        self.playbook_manager.save()
    
    def _is_duplicate(self, content: str, section: str, threshold: float = 0.8) -> bool:
        """Check if content is too similar to existing bullets."""
        content_words = set(content.lower().split())
        
        for bullet in self.playbook.get_section(section):
            bullet_words = set(bullet.content.lower().split())
            
            if not content_words or not bullet_words:
                continue
            
            # Jaccard similarity
            intersection = len(content_words & bullet_words)
            union = len(content_words | bullet_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                return True
        
        return False
