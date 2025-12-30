"""
ACE Framework adapted for Playbook Enrichment.

Specialized agents for validating and curating extracted knowledge from documents.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from llm_client import LLMClient
from playbook import Playbook
from retriever import PlaybookRetriever
from utils import logger


@dataclass
class EnricherGeneratorOutput:
    """Output from Enricher Generator."""
    is_valid: bool
    is_duplicate: bool
    similar_bullets: List[str]
    reasoning: str
    recommendation: str  # "ADD", "MODIFY", "MERGE", "SKIP"


@dataclass
class EnricherReflectorOutput:
    """Output from Enricher Reflector."""
    quality_score: float
    specificity_score: float
    reusability_score: float
    enriched_content: Optional[str]  # Complete definition for pointer definitions
    issues: List[str]
    strengths: List[str]
    recommendation: str


@dataclass
class EnricherCuratorOutput:
    """Output from Enricher Curator."""
    operations: List[Dict[str, Any]]
    reasoning: str


class EnricherGenerator:
    """
    Generator for enrichment validation.
    
    Validates if extracted knowledge is:
    - Specific (not generic)
    - Not duplicate
    - Properly classified
    """
    
    SYSTEM_PROMPT = """You are the Enrichment Validator for a securitization playbook.

Your job is to validate extracted knowledge from legal documents and determine if it should be added to the playbook AS-IS.

CRITICAL RULES:
1. The knowledge was EXTRACTED from a real document - validate it, don't transform it
2. You must validate if it's SPECIFIC enough (not generic legal principles)
3. You must check for DUPLICATES using semantic search
4. You must verify SECTION classification is correct
5. DO NOT create meta-knowledge about the validation process itself
6. DO NOT extract strategies/pitfalls about how to validate - just validate the content

Return JSON:
{
  "is_valid": true/false,
  "is_duplicate": true/false,
  "similar_bullets": ["bullet-id-1", "bullet-id-2"],
  "reasoning": "Brief explanation of validation decision",
  "recommendation": "ADD|SKIP"
}

Recommendations:
- ADD: Valid and should be added (be permissive - trust the extraction)
- SKIP: Only if clearly generic, exact duplicate IN THE SAME SECTION, or completely wrong section

CRITICAL DUPLICATE LOGIC:
- Check for duplicates ONLY within the target section (definitions vs definitions, strategies vs strategies)
- A definition of "X" is NOT a duplicate of a strategy about "X" - they serve different purposes
- Example: If playbook has strategy "Use ABS Transaction for exits" and you're validating definition "ABS Transaction: means...", this is NOT a duplicate (different sections)

IMPORTANT: 
- If recommending ADD, the extracted content will be added EXACTLY as provided
- Be PERMISSIVE - if it's from a real document and somewhat specific, recommend ADD
- Better to add and let users remove later than to miss valuable content
- Only SKIP if you're very confident it's not useful"""
    
    def __init__(self, llm_client: LLMClient, retriever: Optional[PlaybookRetriever] = None):
        self.llm_client = llm_client
        self.retriever = retriever
    
    def validate(
        self,
        content: str,
        section: str,
        playbook: Playbook,
        source_info: str
    ) -> EnricherGeneratorOutput:
        """Validate extracted knowledge."""
        
        # Get similar bullets from playbook
        similar_bullets = []
        if self.retriever:
            results = self.retriever.search(content, sections=[section], top_k=5)
            similar_bullets = [r.bullet_id for r in results if r.score > 0.75]
        
        # Build playbook context
        playbook_context = self._build_playbook_context(playbook, section, similar_bullets)
        
        user_message = f"""EXTRACTED KNOWLEDGE TO VALIDATE:

Source: {source_info}
Section: {section}

Content:
{content}

VALIDATION CHECKS:

1. POINTER DEFINITION HANDLING:
   If this is a pointer definition (e.g., "X has the meaning given in clause Y"), you MUST:
   - Mark is_valid = true (it's a valid term to track)
   - Mark is_duplicate = false (unless exact term exists)
   - Recommendation = ADD
   - In your reasoning, note: "Pointer definition - will be enriched by Reflector"

2. SPECIFICITY: Is this specific to this transaction structure?
   - NOT generic principles like "SPVs are bankruptcy remote"
   - YES specific patterns like "Bifurcated EMI vs Credit Institution eligibility"

3. DUPLICATION: Search results from playbook:
{playbook_context}

4. SECTION: Is "{section}" the correct classification?
   - strategies: Best practices, methodologies
   - pitfalls: Mistakes to avoid
   - templates: Reusable clause patterns
   - definitions: Legal term definitions

Validate and provide recommendation."""
        
        try:
            response = self.llm_client.chat(self.SYSTEM_PROMPT, user_message)
            parsed = response.parse_json()
            
            return EnricherGeneratorOutput(
                is_valid=parsed.get("is_valid", False),
                is_duplicate=parsed.get("is_duplicate", False),
                similar_bullets=parsed.get("similar_bullets", []),
                reasoning=parsed.get("reasoning", ""),
                recommendation=parsed.get("recommendation", "SKIP")
            )
        
        except Exception as e:
            logger.error(f"Generator validation error: {e}")
            return EnricherGeneratorOutput(
                is_valid=False,
                is_duplicate=False,
                similar_bullets=[],
                reasoning=f"Error: {e}",
                recommendation="SKIP"
            )
    
    def _build_playbook_context(
        self,
        playbook: Playbook,
        section: str,
        similar_bullet_ids: List[str]
    ) -> str:
        """Build context of similar bullets."""
        if not similar_bullet_ids:
            return "No similar bullets found in playbook."
        
        lines = ["Similar bullets found:"]
        for bullet_id in similar_bullet_ids[:3]:
            bullet = playbook.get_bullet_by_id(bullet_id)
            if bullet:
                lines.append(f"- [{bullet_id}] {bullet.content[:150]}...")
        
        return "\n".join(lines)


class EnricherReflector:
    """
    Reflector for enrichment quality assessment.
    
    Analyzes the quality of extracted knowledge and Generator's validation.
    """
    
    SYSTEM_PROMPT = """You are the Quality Assessor for playbook enrichment.

CRITICAL TASK: Detect and enrich pointer definitions.

STEP 1 - DETECT POINTER DEFINITIONS:
A pointer definition contains phrases like:
- "has the meaning given to it in clause X"
- "has the meaning given in clause X"
- "means the meaning set out in clause X"
- "refers to clause X"

STEP 2 - IF POINTER DEFINITION DETECTED:
You MUST provide a complete substantive definition based on securitization domain knowledge.

Example Input: "ABS Transaction Fee" has the meaning given to it in clause 8.2(b) (Voluntary Cancellation)
Example Output enriched_content: "ABS Transaction Fee: means the fee payable to the lender upon voluntary cancellation or prepayment in connection with an ABS transaction, typically calculated as a percentage of the prepaid amount to compensate for lost interest income."

Set scores: quality=0.90, specificity=0.85, reusability=0.85, recommendation=ADD

STEP 3 - IF SUBSTANTIVE DEFINITION (not a pointer):
Score normally, set enriched_content=null, recommend ADD if scores > 0.60

Return JSON:
{
  "quality_score": 0.85,
  "specificity_score": 0.90,
  "reusability_score": 0.80,
  "enriched_content": "Complete definition (ONLY for pointers, otherwise null)",
  "issues": [],
  "strengths": [],
  "recommendation": "ADD|SKIP"
}

CRITICAL: For ANY pointer definition, enriched_content MUST be filled with a complete definition."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def assess(
        self,
        content: str,
        section: str,
        generator_output: EnricherGeneratorOutput,
        source_info: str
    ) -> EnricherReflectorOutput:
        """Assess quality of extracted knowledge."""
        
        user_message = f"""KNOWLEDGE TO ASSESS:

Source: {source_info}
Section: {section}

Content:
{content}

GENERATOR VALIDATION:
- Valid: {generator_output.is_valid}
- Duplicate: {generator_output.is_duplicate}
- Similar bullets: {', '.join(generator_output.similar_bullets) if generator_output.similar_bullets else 'None'}
- Recommendation: {generator_output.recommendation}
- Reasoning: {generator_output.reasoning}

Assess quality and provide scores."""
        
        try:
            response = self.llm_client.chat(self.SYSTEM_PROMPT, user_message)
            parsed = response.parse_json()
            
            return EnricherReflectorOutput(
                quality_score=parsed.get("quality_score", 0.0),
                specificity_score=parsed.get("specificity_score", 0.0),
                reusability_score=parsed.get("reusability_score", 0.0),
                enriched_content=parsed.get("enriched_content"),
                issues=parsed.get("issues", []),
                strengths=parsed.get("strengths", []),
                recommendation=parsed.get("recommendation", "SKIP")
            )
        
        except Exception as e:
            logger.error(f"Reflector assessment error: {e}")
            return EnricherReflectorOutput(
                quality_score=0.0,
                specificity_score=0.0,
                reusability_score=0.0,
                enriched_content=None,
                issues=[f"Error: {e}"],
                strengths=[],
                recommendation="SKIP"
            )


class EnricherCurator:
    """
    Curator for enrichment operations.
    
    Decides final operations (ADD, REMOVE, MODIFY, MERGE) based on Generator and Reflector.
    """
    
    SYSTEM_PROMPT = """You are the Playbook Curator for enrichment operations.

Your job is to decide if extracted knowledge should be added to the playbook AS-IS.

CRITICAL RULES:
1. If Reflector provides "enriched_content", use that instead of original content
2. For pointer definitions: ALWAYS use enriched_content from Reflector
3. For substantive content: use original content as-is
4. DO NOT create meta-knowledge about the validation process
5. BE PERMISSIVE - if Generator says valid and Reflector scores are decent, ADD IT

Available Operations:

1. ADD: Create new bullet
   {
     "type": "ADD",
     "section": "strategies|pitfalls|templates|definitions",
     "content": "Use enriched_content if provided by Reflector, otherwise use original content"
   }

2. SKIP: No operation (return empty operations list)

Decision Rules:
- If Reflector scores < 0.70: SKIP
- If Generator says SKIP: SKIP  
- If Generator says ADD and Reflector agrees: ADD with EXACT extracted content
- DO NOT create new strategies/pitfalls about the validation process
- DO NOT transform the content - add it AS-IS or skip it
- If Generator says ADD and Reflector scores >= 0.60: ADD (lowered threshold)
-Default to ADD when in doubt - better to add and let users remove later

Return JSON:
{
  "operations": [{"type": "ADD", "section": "...", "content": "EXACT extracted content"}],
  "reasoning": "Brief explanation"
}

If SKIP, return empty operations list.

EXAMPLES:

CORRECT - Add definition:
Content: "ABS Transaction: means any securitisation..."
Generator: is_valid=true, is_duplicate=false, recommendation=ADD
Reflector: scores all > 0.60
Operations: [{"type": "ADD", "section": "definitions", "content": "ABS Transaction: means any securitisation..."}]

WRONG - Creating meta-knowledge:
Content: "ABS Transaction: means any securitisation..."
Operations: [{"type": "ADD", "section": "strategies", "content": "Always check for duplicates..."}]
^ DO NOT DO THIS"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def decide(
        self,
        content: str,
        section: str,
        generator_output: EnricherGeneratorOutput,
        reflector_output: EnricherReflectorOutput,
        playbook: Playbook
    ) -> EnricherCuratorOutput:
        """Decide final operations."""
        
        user_message = f"""DECISION REQUIRED:

Content: {content}
Section: {section}

GENERATOR VALIDATION:
- Valid: {generator_output.is_valid}
- Duplicate: {generator_output.is_duplicate}
- Similar bullets: {generator_output.similar_bullets}
- Recommendation: {generator_output.recommendation}
- Reasoning: {generator_output.reasoning}

REFLECTOR ASSESSMENT:
- Quality Score: {reflector_output.quality_score}
- Specificity Score: {reflector_output.specificity_score}
- Reusability Score: {reflector_output.reusability_score}
- Enriched Content: {reflector_output.enriched_content or "None (use original)"}
- Issues: {reflector_output.issues}
- Strengths: {reflector_output.strengths}
- Recommendation: {reflector_output.recommendation}

IMPORTANT: If enriched_content is provided, use it in the ADD operation instead of the original content.

Decide operations."""
        
        try:
            response = self.llm_client.chat(self.SYSTEM_PROMPT, user_message)
            parsed = response.parse_json()
            
            return EnricherCuratorOutput(
                operations=parsed.get("operations", []),
                reasoning=parsed.get("reasoning", "")
            )
        
        except Exception as e:
            logger.error(f"Curator decision error: {e}")
            return EnricherCuratorOutput(
                operations=[],
                reasoning=f"Error: {e}"
            )

