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

DEFINITION VALIDATION:
For definitions section, check if the definition is:
- Properly formatted: "TERM: means [definition]" or "TERM has the meaning..."
- If malformed (sentence fragment, missing term, negative statement), still recommend ADD but note in reasoning that it needs fixing
- The Reflector will fix malformed definitions, so don't SKIP them - just note the issue

Return JSON:
{
  "is_valid": true/false,
  "is_duplicate": true/false,
  "similar_bullets": ["bullet-id-1", "bullet-id-2"],
  "reasoning": "Brief explanation of validation decision. If definition is malformed, note: 'Malformed definition - needs formatting fix'",
  "recommendation": "ADD|SKIP"
}

Recommendations:
- ADD: Valid and should be added (be permissive - trust the extraction)
  - Even if definition is malformed, recommend ADD (Reflector will fix it)
  - Only SKIP if it's clearly not a definition at all (e.g., a warranty clause)
- MODIFY: Duplicate found but new item is MORE PRECISE/COMPLETE than existing
  - Include the bullet_id to modify in similar_bullets
  - New content has more detail, better formatting, or clearer language
- SKIP: Only if clearly generic, exact duplicate with same/better quality, completely wrong section, or not a definition at all

CRITICAL DUPLICATE LOGIC:
- Check for duplicates ONLY within the target section (definitions vs definitions, strategies vs strategies)
- A definition of "X" is NOT a duplicate of a strategy about "X" - they serve different purposes
- Example: If playbook has strategy "Use ABS Transaction for exits" and you're validating definition "ABS Transaction: means...", this is NOT a duplicate (different sections)

IMPORTANT: 
- If recommending ADD, the extracted content will be processed by Reflector who will fix formatting issues
- Be PERMISSIVE - if it's from a real document and somewhat specific, recommend ADD
- Better to add and let users remove later than to miss valuable content
- Only SKIP if you're very confident it's not useful or not a definition at all"""
    
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

1. DEFINITION FORMAT CHECK (if section is "definitions"):
   Check if the definition is properly formatted:
   - Proper format: "TERM: means [definition]" or "TERM has the meaning..."
   - Malformed examples to watch for:
     * Sentence fragments: "The temporary waiver... (the "Effective Date")"
     * Missing term: "all authorisations... (each, an "Authorisation")"
     * Just a reference: "Section 7... (the "Blocking Law")"
     * Negative statements: "no Receivable... is "stock"..."
   - If malformed: Still recommend ADD, but note in reasoning: "Malformed definition - needs formatting fix by Reflector"
   - If not a definition at all (e.g., warranty clause): Recommend SKIP

2. POINTER DEFINITION HANDLING:
   If this is a pointer definition (e.g., "X has the meaning given in clause Y"), you MUST:
   - Mark is_valid = true (it's a valid term to track)
   - Mark is_duplicate = false (unless exact term exists)
   - Recommendation = ADD
   - In your reasoning, note: "Pointer definition - will be enriched by Reflector"

3. SPECIFICITY: Is this specific to this transaction structure?
   - NOT generic principles like "SPVs are bankruptcy remote"
   - YES specific patterns like "Bifurcated EMI vs Credit Institution eligibility"

4. DUPLICATION: Search results from playbook:
{playbook_context}

5. SECTION: Is "{section}" the correct classification?
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
    
    def validate_batch(
        self,
        items: List[Dict[str, Any]],
        playbook: Playbook
    ) -> List[EnricherGeneratorOutput]:
        """Validate multiple extracted knowledge items in a single batch."""
        if not items:
            return []
        
        # Build batch context
        batch_items = []
        for item in items:
            content = item["content"]
            section = item["section"]
            source_info = item["source_info"]
            
            # Get similar bullets for this item
            similar_bullets = []
            if self.retriever:
                results = self.retriever.search(content, sections=[section], top_k=5)
                similar_bullets = [r.bullet_id for r in results if r.score > 0.75]
            
            playbook_context = self._build_playbook_context(playbook, section, similar_bullets)
            
            batch_items.append({
                "content": content,
                "section": section,
                "source_info": source_info,
                "playbook_context": playbook_context
            })
        
        # Build batch prompt
        items_text = []
        for idx, item in enumerate(batch_items, 1):
            items_text.append(f"""
ITEM {idx}:
Source: {item['source_info']}
Section: {item['section']}
Content: {item['content']}
Similar bullets: {item['playbook_context']}
""")
        
        user_message = f"""BATCH VALIDATION: Validate {len(batch_items)} extracted knowledge items.

{''.join(items_text)}

CRITICAL RULES - BE PERMISSIVE:
1. The knowledge was EXTRACTED from a real document - validate it, don't transform it
2. Be PERMISSIVE - if it's from a real document and somewhat specific, recommend ADD
3. Better to add and let users remove later than to miss valuable content
4. Only SKIP if you're very confident it's not useful or not a definition at all

For EACH item, validate:
1. DEFINITION FORMAT CHECK (if section is "definitions"):
   - Properly formatted: "TERM: means [definition]" or "TERM has the meaning..."
   - If malformed, still recommend ADD (Reflector will fix it)
   - Only SKIP if clearly not a definition at all (e.g., warranty clause)

2. POINTER DEFINITION HANDLING:
   - If pointer definition (e.g., "X has the meaning given in clause Y"), mark is_valid=true, recommendation=ADD
   - Note in reasoning: "Pointer definition - will be enriched by Reflector"

3. SPECIFICITY: Is this specific to this transaction structure?
   - NOT generic principles like "SPVs are bankruptcy remote"
   - YES specific patterns like "Bifurcated EMI vs Credit Institution eligibility"
   - If somewhat specific, recommend ADD

4. DUPLICATION: 
   - Check similar bullets provided from playbook
   - Check if SAME TERM exists in other items in this batch (for definitions)
   - Check if very similar content exists in other items in this batch
   - If duplicate found in playbook: Compare quality
     * If new item is MORE PRECISE/COMPLETE: mark is_duplicate=true, recommendation=MODIFY, include bullet_id in similar_bullets
     * If existing is better or equal: mark is_duplicate=true, recommendation=SKIP
   - If duplicate found WITHIN THIS BATCH: mark is_duplicate=true and recommendation=SKIP
   - For definitions: if same term exists in batch, prefer substantive over pointer

5. SECTION: Is the classification correct?
   - strategies: Best practices, methodologies
   - pitfalls: Mistakes to avoid
   - templates: Reusable clause patterns
   - definitions: Legal term definitions

IMPORTANT FOR DEFINITIONS:
- If multiple items in this batch define the same term (e.g., "ABS Transaction Fee"), mark duplicates as is_duplicate=true
- Prefer substantive definitions over pointer definitions
- If one item is a pointer and another is substantive for the same term, mark the pointer as duplicate

CRITICAL: You MUST return a valid JSON array. Return ONLY the JSON array, no other text.

Return JSON array with one object per item:
[
  {{
    "item_index": 1,
    "is_valid": true,
    "is_duplicate": false,
    "similar_bullets": ["bullet-id-1", "bullet-id-2"],
    "reasoning": "Brief explanation (mention if duplicate found in batch or playbook)",
    "recommendation": "ADD"
  }},
  {{
    "item_index": 2,
    "is_valid": true,
    "is_duplicate": true,
    "similar_bullets": ["str-00123"],
    "reasoning": "Duplicate found but new item is more precise/complete than str-00123",
    "recommendation": "MODIFY"
  }}
]

REMEMBER: 
- Be PERMISSIVE - only SKIP if clearly generic, exact duplicate, wrong section, or not a definition at all
- Return a JSON ARRAY (starts with [ and ends with ])
- Array must have EXACTLY {len(batch_items)} items
- Each item must have item_index, is_valid, is_duplicate, similar_bullets, reasoning, and recommendation
- Return ONLY the JSON array, no markdown code blocks, no explanations before/after"""
        
        try:
            response = self.llm_client.chat(self.SYSTEM_PROMPT, user_message)
            parsed = response.parse_json()
            
            # Log the raw response content for debugging
            logger.debug(f"Batch validation raw response content (first 500 chars): {response.content[:500]}")
            
            # Check if parsing failed
            if parsed is None:
                logger.error(f"Batch validation failed to parse JSON. Raw response: {response.content[:1000]}")
                # Fallback: process items individually
                logger.warning(f"Falling back to individual processing for batch of {len(batch_items)} items")
                results = []
                for item in batch_items:
                    try:
                        individual_result = self.validate(
                            content=item["content"],
                            section=item["section"],
                            playbook=playbook,
                            source_info=item["source_info"]
                        )
                        results.append(individual_result)
                    except Exception as e:
                        logger.error(f"Individual validation failed for item: {e}")
                        results.append(EnricherGeneratorOutput(
                            is_valid=False,
                            is_duplicate=False,
                            similar_bullets=[],
                            reasoning=f"Error: {e}",
                            recommendation="SKIP"
                        ))
                return results
            
            # Log the parsed response for debugging
            logger.debug(f"Batch validation parsed response type: {type(parsed)}")
            
            # Ensure we got a list
            if not isinstance(parsed, list):
                logger.warning(f"Batch validation returned non-list: {type(parsed)}. Value: {parsed}")
                # Try to extract array from the response
                if isinstance(parsed, dict) and "extracted_items" in parsed:
                    parsed = parsed["extracted_items"]
                elif isinstance(parsed, dict) and "items" in parsed:
                    parsed = parsed["items"]
                else:
                    parsed = [parsed] if parsed else []
            
            if len(parsed) != len(batch_items):
                logger.warning(f"Batch validation returned {len(parsed)} items, expected {len(batch_items)}")
            
            # Map results back to items
            results = []
            for idx, item in enumerate(batch_items):
                if idx < len(parsed):
                    result = parsed[idx]
                    # Log individual result for debugging
                    logger.debug(f"Item {idx+1} validation result: {result}")
                    
                    # Handle case where result might be a dict with nested structure
                    if isinstance(result, dict):
                        results.append(EnricherGeneratorOutput(
                            is_valid=result.get("is_valid", False),
                            is_duplicate=result.get("is_duplicate", False),
                            similar_bullets=result.get("similar_bullets", []),
                            reasoning=result.get("reasoning", ""),
                            recommendation=result.get("recommendation", "SKIP")
                        ))
                    else:
                        logger.warning(f"Item {idx+1} result is not a dict: {type(result)}")
                        results.append(EnricherGeneratorOutput(
                            is_valid=False,
                            is_duplicate=False,
                            similar_bullets=[],
                            reasoning="Batch processing error: invalid result format",
                            recommendation="SKIP"
                        ))
                else:
                    # Fallback if LLM didn't return enough items
                    logger.warning(f"Missing validation result for item {idx+1}")
                    results.append(EnricherGeneratorOutput(
                        is_valid=False,
                        is_duplicate=False,
                        similar_bullets=[],
                        reasoning="Batch processing error: missing result",
                        recommendation="SKIP"
                    ))
            
            return results
        
        except Exception as e:
            logger.error(f"Generator batch validation error: {e}", exc_info=True)
            # Fallback: process items individually
            logger.warning(f"Exception in batch validation, falling back to individual processing for {len(items)} items")
            results = []
            for item in batch_items:
                try:
                    individual_result = self.validate(
                        content=item["content"],
                        section=item["section"],
                        playbook=playbook,
                        source_info=item["source_info"]
                    )
                    results.append(individual_result)
                except Exception as e2:
                    logger.error(f"Individual validation failed for item: {e2}")
                    results.append(EnricherGeneratorOutput(
                        is_valid=False,
                        is_duplicate=False,
                        similar_bullets=[],
                        reasoning=f"Error: {e2}",
                        recommendation="SKIP"
                    ))
            return results


class EnricherReflector:
    """
    Reflector for enrichment quality assessment.
    
    Analyzes the quality of extracted knowledge and Generator's validation.
    """
    
    SYSTEM_PROMPT = """You are the Quality Assessor for playbook enrichment.

CRITICAL TASK: Detect and fix malformed definitions.

STEP 1 - DETECT POINTER DEFINITIONS:
A pointer definition contains phrases like:
- "has the meaning given to it in clause X"
- "has the meaning given in clause X"
- "means the meaning set out in clause X"
- "refers to clause X"

STEP 2 - DETECT MALFORMED DEFINITIONS:
A definition is malformed if it:
- Is a sentence fragment without the term being defined (e.g., "The temporary waiver granted in this email will take effect on [PARTY_NAME] (the "Effective Date")")
- Contains a term in quotes but doesn't start with "TERM: means..." format (e.g., "all authorisations... (each, an "Authorisation")")
- Is a negative statement or warranty, not a definition (e.g., "no Receivable or its Related Rights is "stock"...")
- References a term but doesn't define it (e.g., "Section 7 of the German Foreign Trade Regulation (the "Blocking Law")")
- Is a list item without the term (e.g., "(i) the Data Protection Laws... ("Privacy Commitments")")

STEP 3 - FIX MALFORMED DEFINITIONS:
For ANY malformed definition, you MUST:
1. Extract the term being defined (usually in quotes like "Term" or (the "Term"))
2. Convert to proper format: "TERM: means [definition text]"
3. Remove placeholder text like [PARTY_NAME] and replace with appropriate language
4. For negative statements/warranties that aren't definitions: mark as SKIP with low scores

Examples of fixes:
Input: "The temporary waiver granted in this email will take effect on [PARTY_NAME] (the "Effective Date")"
Output: "Effective Date: means the date on which the temporary waiver granted in this email takes effect."

Input: "all authorisations, consents, licences, filings, approvals and permissions, including, where relevant, as required by the Financial Conduct Authority or under Data Protection Laws (each, an "Authorisation")"
Output: "Authorisation: means all authorisations, consents, licences, filings, approvals and permissions, including, where relevant, as required by the Financial Conduct Authority or under Data Protection Laws."

Input: "Section 7 of the German Foreign Trade Regulation (Außenwirtschaftsverordnung) (the "Blocking Law")"
Output: "Blocking Law: means Section 7 of the German Foreign Trade Regulation (Außenwirtschaftsverordnung)."

Input: "(i) the Data Protection Laws, (ii) contractual obligations and other commitments, and (iii) its own policies and procedures, in each case as relating to privacy, data protection, and the processing of Personal Data ("Privacy Commitments")"
Output: "Privacy Commitments: means (i) the Data Protection Laws, (ii) contractual obligations and other commitments, and (iii) its own policies and procedures, in each case as relating to privacy, data protection, and the processing of Personal Data."

STEP 4 - IF POINTER DEFINITION DETECTED:
You MUST provide a complete substantive definition based on securitization domain knowledge.

Example Input: "ABS Transaction Fee" has the meaning given to it in clause 8.2(b) (Voluntary Cancellation)
Example Output enriched_content: "ABS Transaction Fee: means the fee payable to the lender upon voluntary cancellation or prepayment in connection with an ABS transaction, typically calculated as a percentage of the prepaid amount to compensate for lost interest income."

STEP 5 - IF PROPERLY FORMATTED DEFINITION:
Score normally, set enriched_content=null, recommend ADD if scores > 0.60

Return JSON:
{
  "quality_score": 0.85,
  "specificity_score": 0.90,
  "reusability_score": 0.80,
  "enriched_content": "Fixed definition in proper format (for malformed/pointer definitions, otherwise null)",
  "issues": ["List any issues found"],
  "strengths": ["List strengths"],
  "recommendation": "ADD|SKIP"
}

CRITICAL RULES:
- For ANY malformed or pointer definition, enriched_content MUST be filled with a complete, properly formatted definition
- Format: "TERM: means [definition text]"
- For negative statements/warranties that aren't definitions: set low scores (< 0.50) and recommend SKIP
- If you cannot extract a meaningful term or the content is not a definition at all, recommend SKIP with low scores"""
    
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

CRITICAL: If Generator noted "Malformed definition" in reasoning, you MUST:
1. Extract the term being defined (usually in quotes)
2. Convert to proper format: "TERM: means [definition text]"
3. Provide the fixed definition in enriched_content
4. Set appropriate scores (0.70+ if fixable, <0.50 if not a definition at all)

Assess quality, fix any malformed definitions, and provide scores."""
        
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
    
    def assess_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[EnricherReflectorOutput]:
        """Assess quality of multiple extracted knowledge items in a single batch."""
        if not items:
            return []
        
        # Build batch prompt
        items_text = []
        for idx, item in enumerate(items, 1):
            items_text.append(f"""
ITEM {idx}:
Source: {item['source_info']}
Section: {item['section']}
Content: {item['content']}
Generator Validation:
- Valid: {item['generator_output'].is_valid}
- Duplicate: {item['generator_output'].is_duplicate}
- Similar bullets: {', '.join(item['generator_output'].similar_bullets) if item['generator_output'].similar_bullets else 'None'}
- Recommendation: {item['generator_output'].recommendation}
- Reasoning: {item['generator_output'].reasoning}
""")
        
        user_message = f"""BATCH QUALITY ASSESSMENT: Assess {len(items)} extracted knowledge items.

{''.join(items_text)}

For EACH item, assess quality, fix malformed definitions, and enrich pointer definitions.

CRITICAL FOR POINTER DEFINITIONS:
- If content is like "TERM has the meaning in clause X" or "TERM is defined in clause Y"
- You MUST provide enriched_content with the actual substantive definition
- Example: "After Tax Basis" has the meaning in Clause 10.8 → enriched_content: "After Tax Basis: means the amount calculated after deducting all applicable taxes..."

For malformed definitions: Convert to "TERM: means [definition]" format in enriched_content.
For properly formatted definitions: Set enriched_content to null.

Return JSON array with one object per item:
[
  {{
    "item_index": 1,
    "quality_score": 0.85,
    "specificity_score": 0.90,
    "reusability_score": 0.80,
    "enriched_content": "REQUIRED for pointer/malformed definitions, null otherwise",
    "issues": ["List any issues"],
    "strengths": ["List strengths"],
    "recommendation": "ADD|SKIP"
  }},
  ...
]

Ensure array length matches number of items ({len(items)})."""
        
        try:
            response = self.llm_client.chat(self.SYSTEM_PROMPT, user_message)
            parsed = response.parse_json()
            
            # Check if parsing failed
            if parsed is None:
                logger.error(f"Reflector batch assessment failed to parse JSON. Raw response: {response.content[:1000]}")
                # Fallback: process items individually
                logger.warning(f"Falling back to individual processing for Reflector batch of {len(items)} items")
                results = []
                for item in items:
                    try:
                        individual_result = self.assess(
                            content=item["content"],
                            section=item["section"],
                            generator_output=item["generator_output"],
                            source_info=item["source_info"]
                        )
                        results.append(individual_result)
                    except Exception as e:
                        logger.error(f"Individual Reflector assessment failed for item: {e}")
                        results.append(EnricherReflectorOutput(
                            quality_score=0.0,
                            specificity_score=0.0,
                            reusability_score=0.0,
                            enriched_content=None,
                            issues=[f"Error: {e}"],
                            strengths=[],
                            recommendation="SKIP"
                        ))
                return results
            
            # Ensure we got a list
            if not isinstance(parsed, list):
                logger.warning(f"Reflector batch returned non-list: {type(parsed)}. Converting to list.")
                if isinstance(parsed, dict) and "extracted_items" in parsed:
                    parsed = parsed["extracted_items"]
                elif isinstance(parsed, dict) and "items" in parsed:
                    parsed = parsed["items"]
                else:
                    parsed = [parsed] if parsed else []
            
            if len(parsed) != len(items):
                logger.warning(f"Reflector batch returned {len(parsed)} items, expected {len(items)}")
            
            # Map results back to items
            results = []
            for idx, item in enumerate(items):
                if idx < len(parsed):
                    result = parsed[idx]
                    if isinstance(result, dict):
                        results.append(EnricherReflectorOutput(
                            quality_score=result.get("quality_score", 0.0),
                            specificity_score=result.get("specificity_score", 0.0),
                            reusability_score=result.get("reusability_score", 0.0),
                            enriched_content=result.get("enriched_content"),
                            issues=result.get("issues", []),
                            strengths=result.get("strengths", []),
                            recommendation=result.get("recommendation", "SKIP")
                        ))
                    else:
                        logger.warning(f"Reflector item {idx+1} result is not a dict: {type(result)}")
                        results.append(EnricherReflectorOutput(
                            quality_score=0.0,
                            specificity_score=0.0,
                            reusability_score=0.0,
                            enriched_content=None,
                            issues=["Batch processing error: invalid result format"],
                            strengths=[],
                            recommendation="SKIP"
                        ))
                else:
                    # Fallback if LLM didn't return enough items
                    logger.warning(f"Missing Reflector result for item {idx+1}")
                    results.append(EnricherReflectorOutput(
                        quality_score=0.0,
                        specificity_score=0.0,
                        reusability_score=0.0,
                        enriched_content=None,
                        issues=["Batch processing error: missing result"],
                        strengths=[],
                        recommendation="SKIP"
                    ))
            
            return results
        
        except Exception as e:
            logger.error(f"Reflector batch assessment error: {e}", exc_info=True)
            # Fallback: process items individually
            logger.warning(f"Exception in Reflector batch, falling back to individual processing for {len(items)} items")
            results = []
            for item in items:
                try:
                    individual_result = self.assess(
                        content=item["content"],
                        section=item["section"],
                        generator_output=item["generator_output"],
                        source_info=item["source_info"]
                    )
                    results.append(individual_result)
                except Exception as e2:
                    logger.error(f"Individual Reflector assessment failed for item: {e2}")
                    results.append(EnricherReflectorOutput(
                        quality_score=0.0,
                        specificity_score=0.0,
                        reusability_score=0.0,
                        enriched_content=None,
                        issues=[f"Error: {e2}"],
                        strengths=[],
                        recommendation="SKIP"
                    ))
            return results


class EnricherCurator:
    """
    Curator for enrichment operations.
    
    Decides final operations (ADD, REMOVE, MODIFY, MERGE) based on Generator and Reflector.
    """
    
    SYSTEM_PROMPT = """You are the Playbook Curator for enrichment operations.

Your job is to decide if extracted knowledge should be added to the playbook AS-IS.

CRITICAL RULES:
1. If Reflector provides "enriched_content", you MUST use it as the "content" field - DO NOT use original content
2. For pointer definitions: Reflector ALWAYS enriches them - ALWAYS use enriched_content
3. For substantive content: If enriched_content is null, use original content
4. DO NOT create meta-knowledge about the validation process
5. BE PERMISSIVE - if Generator says valid and Reflector scores are decent, ADD IT

EXAMPLES:
✓ CORRECT - Pointer definition with enriched content:
Original: "ABS Fee" has meaning in clause 5.2
Reflector enriched_content: "ABS Fee: means the fee payable for asset-backed securitization..."
YOUR OPERATION: {{"type": "ADD", "section": "definitions", "content": "ABS Fee: means the fee payable for asset-backed securitization..."}}

✗ WRONG - Using original pointer:
YOUR OPERATION: {{"type": "ADD", "section": "definitions", "content": "ABS Fee" has meaning in clause 5.2}}
^ NEVER DO THIS - Always use enriched_content for pointers!

Available Operations:

1. ADD: Create new bullet
   {
     "type": "ADD",
     "section": "USE THE EXACT SECTION PROVIDED IN THE INPUT - DO NOT CHANGE IT",
     "content": "Use enriched_content if provided by Reflector, otherwise use original content"
   }

2. MODIFY: Update existing bullet with better content
   {
     "type": "MODIFY",
     "bullet_id": "ID from Generator's similar_bullets",
     "new_content": "Use enriched_content if provided by Reflector, otherwise use original content",
     "reason": "Why this is better (e.g., 'More precise definition')"
   }

3. SKIP: No operation (return empty operations list)

Decision Rules (APPLY STRICTLY):
- If Generator says ADD and Reflector scores >= 0.60: ADD (use enriched_content if provided, otherwise original)
- If Generator says ADD and Reflector scores < 0.60 but > 0.40: STILL ADD (be permissive)
- If Generator says MODIFY and Reflector scores >= 0.60: MODIFY (use first bullet_id from similar_bullets)
- If Generator says MODIFY and Reflector scores < 0.60: SKIP (don't downgrade quality)
- If Generator says SKIP: SKIP
- If Reflector scores are all 0.0 (parsing error): If Generator says ADD/MODIFY, trust Generator
- DO NOT create new strategies/pitfalls about the validation process
- DO NOT transform the content - add it AS-IS or skip it
- DO NOT CHANGE THE SECTION - use the exact section provided in the input
- Default to ADD when in doubt - better to add and let users remove later

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
        playbook: Playbook,
        allowed_sections: Optional[List[str]] = None
    ) -> EnricherCuratorOutput:
        """Decide final operations."""
        
        # If allowed_sections specified, ensure section is in the list
        if allowed_sections and section not in allowed_sections:
            logger.warning(f"Item section '{section}' not in allowed sections {allowed_sections}, using first allowed section")
            section = allowed_sections[0] if allowed_sections else section
        
        allowed_sections_text = f"\nCRITICAL: You can ONLY use these sections: {', '.join(allowed_sections)}. DO NOT change the section to something else." if allowed_sections else ""
        
        user_message = f"""DECISION REQUIRED:

Content: {content}
Section: {section}{allowed_sections_text}

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

CRITICAL INSTRUCTION: 
- If Reflector provided enriched_content (shown above as NOT "None"), you MUST use the enriched_content as the "content" in your ADD/MODIFY operation.
- If enriched_content is "None", use the original content shown at the top.
- For pointer definitions, Reflector ALWAYS provides enriched_content - you MUST use it.
- Use the section "{section}" - DO NOT change it to a different section.{' Only sections allowed: ' + ', '.join(allowed_sections) if allowed_sections else ''}

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
    
    def decide_batch(
        self,
        items: List[Dict[str, Any]],
        playbook: Playbook,
        allowed_sections: Optional[List[str]] = None
    ) -> List[EnricherCuratorOutput]:
        """Decide final operations for multiple items in a single batch."""
        if not items:
            return []
        
        # Build batch prompt
        items_text = []
        for idx, item in enumerate(items, 1):
            enriched_note = ""
            if item['reflector_output'].enriched_content:
                enriched_note = f"\nReflector Enriched Content: {item['reflector_output'].enriched_content[:200]}..."
            
            items_text.append(f"""
ITEM {idx}:
Original Content: {item['content']}
Section: {item['section']}
Generator: Valid={item['generator_output'].is_valid}, Duplicate={item['generator_output'].is_duplicate}, Recommendation={item['generator_output'].recommendation}
Reflector: Quality={item['reflector_output'].quality_score:.2f}, Specificity={item['reflector_output'].specificity_score:.2f}, Reusability={item['reflector_output'].reusability_score:.2f}, Recommendation={item['reflector_output'].recommendation}{enriched_note}
""")
        
        allowed_sections_text = f"\n\nCRITICAL SECTION CONSTRAINT: You can ONLY use these sections: {', '.join(allowed_sections)}. For each item, use the section it was extracted for - DO NOT change it to a different section." if allowed_sections else ""
        
        user_message = f"""BATCH CURATION: Decide operations for {len(items)} items.

{''.join(items_text)}

CRITICAL DECISION RULES:
- If Generator says ADD and Reflector scores >= 0.60: ADD
  * If "Reflector Enriched Content" is shown → USE THAT as content
  * If no enriched content → use Original Content
- If Generator says ADD and Reflector scores are 0.0 (parsing error): STILL ADD (use original)
- If Generator says ADD and Reflector scores < 0.60 but > 0.40: STILL ADD (be permissive)
- If Generator says MODIFY and Reflector scores >= 0.60: MODIFY
  * If "Reflector Enriched Content" is shown → USE THAT as new_content
  * If no enriched content → use Original Content
- If Generator says MODIFY and Reflector scores < 0.60: SKIP (don't downgrade quality)
- If Generator says SKIP: SKIP
- If Reflector scores < 0.40 AND Generator says SKIP: SKIP
- Default to ADD when Generator says ADD - better to add and let users remove later
- DO NOT CHANGE THE SECTION - use the exact section from each item
{allowed_sections_text}

For EACH item, decide if it should be added to the playbook.
CRITICAL CONTENT INSTRUCTION: If an item has "Reflector Enriched Content" shown above, YOU MUST USE THAT ENRICHED VERSION as the content in your operation, NOT the "Original Content".
CRITICAL SECTION RULE: Use the EXACT section the item was extracted for - DO NOT EVER change the section.

Return JSON array with one object per item:
[
  {{
    "item_index": 1,
    "operations": [{{"type": "ADD", "section": "USE ORIGINAL SECTION FROM ITEM", "content": "..."}}],
    "reasoning": "Brief explanation"
  }},
  {{
    "item_index": 2,
    "operations": [{{"type": "MODIFY", "bullet_id": "str-00123", "new_content": "...", "reason": "New version is more precise"}}],
    "reasoning": "Upgrading existing bullet with better content"
  }}
]

CRITICAL: 
- Return ONLY the JSON array, no markdown, no explanations.
- Use the section from the item (shown above) - DO NOT change it.
- If SKIP, return empty operations list.
- Ensure array length matches number of items ({len(items)})."""
        
        try:
            response = self.llm_client.chat(self.SYSTEM_PROMPT, user_message)
            parsed = response.parse_json()
            
            # Check if parsing failed
            if parsed is None:
                logger.error(f"Curator batch decision failed to parse JSON. Raw response: {response.content[:1000]}")
                # Fallback: process items individually OR use simple logic
                logger.warning(f"Falling back to individual processing for Curator batch of {len(items)} items")
                results = []
                for item in items:
                    try:
                        # Use simple logic: if Generator says ADD, add it
                        gen_rec = item['generator_output'].recommendation
                        ref_scores = item['reflector_output']
                        
                        if gen_rec == "ADD":
                            # Use enriched_content if available, otherwise original
                            content = item['reflector_output'].enriched_content or item['content']
                            results.append(EnricherCuratorOutput(
                                operations=[{
                                    "type": "ADD",
                                    "section": item['section'],
                                    "content": content
                                }],
                                reasoning=f"Generator recommends ADD. Reflector scores: Q={ref_scores.quality_score:.2f}, S={ref_scores.specificity_score:.2f}, R={ref_scores.reusability_score:.2f}"
                            ))
                        elif gen_rec == "MODIFY" and ref_scores.quality_score >= 0.60:
                            # Get bullet_id from similar_bullets
                            bullet_id = item['generator_output'].similar_bullets[0] if item['generator_output'].similar_bullets else None
                            if bullet_id:
                                content = item['reflector_output'].enriched_content or item['content']
                                results.append(EnricherCuratorOutput(
                                    operations=[{
                                        "type": "MODIFY",
                                        "bullet_id": bullet_id,
                                        "new_content": content,
                                        "reason": "New version is more precise/complete"
                                    }],
                                    reasoning=f"Generator recommends MODIFY. Reflector scores: Q={ref_scores.quality_score:.2f}, S={ref_scores.specificity_score:.2f}"
                                ))
                            else:
                                results.append(EnricherCuratorOutput(
                                    operations=[],
                                    reasoning="MODIFY requested but no bullet_id provided"
                                ))
                        else:
                            results.append(EnricherCuratorOutput(
                                operations=[],
                                reasoning=f"Generator recommends {gen_rec}"
                            ))
                    except Exception as e2:
                        logger.error(f"Individual Curator decision failed for item: {e2}")
                        results.append(EnricherCuratorOutput(
                            operations=[],
                            reasoning=f"Error: {e2}"
                        ))
                return results
            
            # Ensure we got a list
            if not isinstance(parsed, list):
                logger.warning(f"Curator batch returned non-list: {type(parsed)}. Converting to list.")
                if isinstance(parsed, dict) and "extracted_items" in parsed:
                    parsed = parsed["extracted_items"]
                elif isinstance(parsed, dict) and "items" in parsed:
                    parsed = parsed["items"]
                else:
                    parsed = [parsed] if parsed else []
            
            if len(parsed) != len(items):
                logger.warning(f"Curator batch returned {len(parsed)} items, expected {len(items)}")
            
            # Map results back to items
            results = []
            for idx, item in enumerate(items):
                if idx < len(parsed):
                    result = parsed[idx]
                    if isinstance(result, dict):
                        results.append(EnricherCuratorOutput(
                            operations=result.get("operations", []),
                            reasoning=result.get("reasoning", "")
                        ))
                    else:
                        logger.warning(f"Curator item {idx+1} result is not a dict: {type(result)}")
                        # Fallback logic
                        gen_rec = item['generator_output'].recommendation
                        if gen_rec == "ADD":
                            content = item['reflector_output'].enriched_content or item['content']
                            results.append(EnricherCuratorOutput(
                                operations=[{"type": "ADD", "section": item['section'], "content": content}],
                                reasoning="Fallback: Generator recommends ADD"
                            ))
                        elif gen_rec == "MODIFY" and item['reflector_output'].quality_score >= 0.60:
                            bullet_id = item['generator_output'].similar_bullets[0] if item['generator_output'].similar_bullets else None
                            if bullet_id:
                                content = item['reflector_output'].enriched_content or item['content']
                                results.append(EnricherCuratorOutput(
                                    operations=[{"type": "MODIFY", "bullet_id": bullet_id, "new_content": content, "reason": "More precise"}],
                                    reasoning="Fallback: Generator recommends MODIFY"
                                ))
                            else:
                                results.append(EnricherCuratorOutput(operations=[], reasoning="Fallback: MODIFY without bullet_id"))
                        else:
                            results.append(EnricherCuratorOutput(
                                operations=[],
                                reasoning="Fallback: Generator recommends SKIP"
                            ))
                else:
                    # Fallback if LLM didn't return enough items - use simple logic
                    logger.warning(f"Missing Curator result for item {idx+1}, using fallback logic")
                    gen_rec = item['generator_output'].recommendation
                    if gen_rec == "ADD":
                        content = item['reflector_output'].enriched_content or item['content']
                        results.append(EnricherCuratorOutput(
                            operations=[{"type": "ADD", "section": item['section'], "content": content}],
                            reasoning="Fallback: Generator recommends ADD"
                        ))
                    elif gen_rec == "MODIFY" and item['reflector_output'].quality_score >= 0.60:
                        bullet_id = item['generator_output'].similar_bullets[0] if item['generator_output'].similar_bullets else None
                        if bullet_id:
                            content = item['reflector_output'].enriched_content or item['content']
                            results.append(EnricherCuratorOutput(
                                operations=[{"type": "MODIFY", "bullet_id": bullet_id, "new_content": content, "reason": "More precise"}],
                                reasoning="Fallback: Generator recommends MODIFY"
                            ))
                        else:
                            results.append(EnricherCuratorOutput(operations=[], reasoning="Fallback: MODIFY without bullet_id"))
                    else:
                        results.append(EnricherCuratorOutput(
                            operations=[],
                            reasoning="Fallback: Generator recommends SKIP"
                        ))
            
            return results
        
        except Exception as e:
            logger.error(f"Curator batch decision error: {e}", exc_info=True)
            # Fallback: use simple logic based on Generator recommendation
            logger.warning(f"Exception in Curator batch, using fallback logic for {len(items)} items")
            results = []
            for item in items:
                try:
                    gen_rec = item['generator_output'].recommendation
                    if gen_rec == "ADD":
                        content = item['reflector_output'].enriched_content or item['content']
                        results.append(EnricherCuratorOutput(
                            operations=[{"type": "ADD", "section": item['section'], "content": content}],
                            reasoning=f"Fallback: Generator recommends ADD. Error: {e}"
                        ))
                    elif gen_rec == "MODIFY" and item['reflector_output'].quality_score >= 0.60:
                        bullet_id = item['generator_output'].similar_bullets[0] if item['generator_output'].similar_bullets else None
                        if bullet_id:
                            content = item['reflector_output'].enriched_content or item['content']
                            results.append(EnricherCuratorOutput(
                                operations=[{"type": "MODIFY", "bullet_id": bullet_id, "new_content": content, "reason": "More precise"}],
                                reasoning=f"Fallback: Generator recommends MODIFY. Error: {e}"
                            ))
                        else:
                            results.append(EnricherCuratorOutput(operations=[], reasoning="Fallback: MODIFY without bullet_id"))
                    else:
                        results.append(EnricherCuratorOutput(
                            operations=[],
                            reasoning=f"Fallback: Generator recommends {gen_rec}"
                        ))
                except Exception as e2:
                    logger.error(f"Fallback Curator logic failed for item: {e2}")
                    results.append(EnricherCuratorOutput(
                        operations=[],
                        reasoning=f"Error: {e2}"
                    ))
            return results

