"""
Validator Agent - Batch processes items and validates them.

Combines Generator + Reflector logic:
- Validates: is_reusable, is_correct, is_duplicate
- Enriches pointer definitions with substantive content
- Outputs recommendation: ADD / SKIP / MODIFY
- Returns JSON array matching input batch size
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from llm_client import LLMClient
from playbook import Playbook
from retriever import PlaybookRetriever
from utils import logger


@dataclass
class ValidatorOutput:
    """Output from Validator Agent."""
    is_reusable: bool
    is_correct: bool
    is_duplicate: bool
    similar_bullets: List[str]
    enriched_content: Optional[str]  # For pointer definitions
    quality_score: float
    specificity_score: float
    reusability_score: float
    recommendation: str  # ADD, SKIP, MODIFY
    reasoning: str
    update_reason: Optional[str] = None  # Specific reason for MODIFY (why the bullet is being modified)


class ValidatorAgent:
    """
    Validator Agent - Batch processes items with configurable batch size.
    
    Validates each item:
    - is_reusable: Can this be applied to other transactions?
    - is_correct: Is the content accurate and well-formed?
    - is_duplicate: Does this already exist in the playbook?
    
    Enriches pointer definitions with substantive content.
    """
    
    SYSTEM_PROMPT = """You are the Validator for a securitization playbook enrichment system.

Your job is to validate extracted knowledge from legal documents and determine if it should be added to the playbook.

CRITICAL RULES:
1. The knowledge was EXTRACTED from a real document - validate it, don't transform it
2. You must validate THREE aspects:
   - is_reusable: Can this be applied to other transactions? (not transaction-specific)
   - is_correct: Is the content accurate and well-formed?
   - is_duplicate: Does similar content already exist in the playbook?
3. For definitions: Check format and enrich pointer definitions
4. Be PERMISSIVE - if it's from a real document and somewhat specific, recommend ADD

VALIDATION CRITERIA:

1. is_reusable:
   - TRUE: Content can be applied to different transactions (abstracted principles, patterns)
   - FALSE: Content is too transaction-specific (mentions specific parties, dates, amounts)

2. is_correct:
   - TRUE: Content is accurate, well-formed, and makes sense
   - FALSE: Content is malformed, incorrect, or nonsensical
   - For definitions: Check if properly formatted ("TERM: means..." or "TERM has the meaning...")
   - For malformed definitions: Still mark is_correct=true if fixable, but note in reasoning

3. is_duplicate:
   - TRUE: Similar content already exists in the playbook (check similar_bullets)
   - FALSE: This is new content
   - Check ONLY within the same section (definitions vs definitions, strategies vs strategies)

POINTER DEFINITION ENRICHMENT:
If you detect a definition with clause references (e.g., "X has the meaning given in clause Y" OR "X means... specified in clauses 3.1 (d), (e) or (h)"), you MUST:
1. Mark is_correct=true (it's a valid term to track)
2. Provide enriched_content with a complete substantive definition based on securitization domain knowledge
3. Format: "TERM: means [complete definition]" - NO clause references, NO "clause X", NO "specified in clauses"
4. Remove ALL clause references and replace with substantive content
5. Example: "ABS Transaction Fee" has the meaning in clause 8.2(b) 
   → enriched_content: "ABS Transaction Fee: means the fee payable to the lender upon voluntary cancellation or prepayment in connection with an ABS transaction, typically calculated as a percentage of the prepaid amount to compensate for lost interest income."
6. Example: "Agreed Ad Hoc Funding Amount" means... specified in clauses 3.1 (d), (e) or (h)
   → enriched_content: "Agreed Ad Hoc Funding Amount: means, as at any Cut-Off Date, any Subordinated Advances made by the Subordinated Lender to fund specified ad hoc costs payable by the Issuer in accordance with the Transaction Documents which are pre-approved in writing by the Facility Agent (such approval not to be unreasonably withheld or delayed, including situations where withholding consent would result in not being able to prevent the occurrence of, or allow the remediation of, an Early Amortisation Event or an Event of Default) or any other costs and expenses (but not principal repayment amounts or redemption amounts on the Senior Facility or Mezzanine Notes) related to transaction purposes such as fees, expenses, and other costs as typically defined in subordinated facility agreements."

INCOMPLETE DEFINITION HANDLING (CRITICAL):
If a definition contains "..." (ellipsis) indicating the LLM extracted an incomplete definition:
1. Mark is_correct=true (you will complete it)
2. You MUST provide enriched_content with the COMPLETE definition
3. Use the partial definition as a starting point and complete it based on:
   - The context from the source document (if available in source_info)
   - Your securitization domain knowledge
   - Standard securitization terminology and patterns
4. Remove ALL "..." ellipsis and provide the full, complete definition
5. Format: "TERM: means [complete definition]" - NO ellipsis, NO truncation
6. Example: "Senior Borrowing Base" means... the product of: the Senior Advance Rate... and the Senior Net Eligible Receivables Balance... plus the sum of [Cash Accounts]...
   → enriched_content: "Senior Borrowing Base: means the product of: the Senior Advance Rate and the Senior Net Eligible Receivables Balance, plus the sum of Cash Accounts; or when a Senior Advance Rate Reduction Event is continuing, the Senior Advance Rate Reduction Event Borrowing Base."
7. Example: "Senior Advance Rate Reduction Event" means... the aggregate Outstanding Principal Balance of Purchased Receivables that are outstanding in respect of Borrowers that are paying by way of direct debit or Push Payments... is greater than 19.0 per cent...; or the Default Rate... is greater than 1.25 per cent...
   → enriched_content: "Senior Advance Rate Reduction Event: means the occurrence of any of the following events: (a) the aggregate Outstanding Principal Balance of Purchased Receivables that are outstanding in respect of Borrowers that are paying by way of direct debit or Push Payments is greater than 19.0 per cent of the aggregate Outstanding Principal Balance of all Purchased Receivables; or (b) the Default Rate is greater than 1.25 per cent of the aggregate Outstanding Principal Balance of all Purchased Receivables."
8. Mark recommendation=ADD (with enriched_content) - incomplete definitions should be completed, not skipped

MALFORMED DEFINITION HANDLING:
If a definition is malformed (sentence fragment, missing term, negative statement):
1. Mark is_correct=true if you can fix it
2. Extract the term and convert to proper format in enriched_content
3. Example: "The temporary waiver... (the "Effective Date")"
   → enriched_content: "Effective Date: means the date on which the temporary waiver takes effect."

SCORING:
- quality_score: Overall quality (0.0-1.0)
- specificity_score: How specific vs generic (0.0-1.0)
- reusability_score: How reusable across transactions (0.0-1.0)

RECOMMENDATIONS:
- ADD: Valid, reusable, and should be added (use enriched_content if provided, otherwise original)
- MODIFY: Duplicate found but new item is MORE PRECISE/COMPLETE (include bullet_id in similar_bullets)
  * CRITICAL: When recommending MODIFY, you MUST provide enriched_content with the improved/complete version
  * enriched_content should be the BETTER version that will replace the existing bullet
  * If the new content is not significantly better, recommend SKIP instead
  * Example: If existing bullet is incomplete or has errors, provide enriched_content with the complete/corrected version
  * CRITICAL: You MUST provide "update_reason" explaining WHY this modification improves the bullet
- SKIP: Generic, duplicate with same/better quality, wrong section, or not fixable

Return JSON:
{
  "is_reusable": true,
  "is_correct": true,
  "is_duplicate": false,
  "similar_bullets": ["bullet-id-1"],
  "enriched_content": "For pointer/malformed definitions: complete definition, otherwise null",
  "quality_score": 0.85,
  "specificity_score": 0.90,
  "reusability_score": 0.80,
  "recommendation": "ADD",
  "reasoning": "Brief explanation",
  "update_reason": "For MODIFY only: specific reason why the bullet is being updated (e.g., 'Added missing clause references', 'Completed incomplete definition', 'Fixed formatting errors')"
}"""

    def __init__(self, llm_client: LLMClient, retriever: Optional[PlaybookRetriever] = None):
        self.llm_client = llm_client
        self.retriever = retriever
    
    def validate_batch(
        self,
        items: List[Dict[str, Any]],
        playbook: Playbook,
        batch_size: int = 5
    ) -> List[ValidatorOutput]:
        """
        Batch process items with configurable batch size.
        
        Args:
            items: List of items to validate, each with:
                - content: str
                - section: str
                - source_info: str
            playbook: Current playbook state
            batch_size: Number of items to process per batch
        
        Returns:
            List of ValidatorOutput, one per input item
        """
        if not items:
            return []
        
        all_results = []
        
        # Process in batches
        num_batches = (len(items) + batch_size - 1) // batch_size
        logger.info(f"Validating {len(items)} items in {num_batches} batches (batch_size={batch_size})")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(items))
            batch_items = items[start_idx:end_idx]
            
            logger.info(f"Processing validation batch {batch_idx + 1}/{num_batches} ({len(batch_items)} items)")
            
            batch_results = self._validate_single_batch(batch_items, playbook)
            all_results.extend(batch_results)
        
        return all_results
    
    def _validate_single_batch(
        self,
        batch_items: List[Dict[str, Any]],
        playbook: Playbook
    ) -> List[ValidatorOutput]:
        """Validate a single batch of items."""
        # Build batch context with similar bullets
        batch_data = []
        for item in batch_items:
            content = item["content"]
            section = item["section"]
            
            # Get similar bullets from playbook
            similar_bullets = []
            if self.retriever:
                try:
                    results = self.retriever.search(content, sections=[section], top_k=5)
                    similar_bullets = [r.bullet_id for r in results if r.score > 0.75]
                except Exception as e:
                    logger.warning(f"Retriever search failed: {e}")
            
            # Build playbook context
            playbook_context = self._build_playbook_context(playbook, section, similar_bullets)
            
            batch_data.append({
                "content": content,
                "section": section,
                "source_info": item.get("source_info", ""),
                "similar_bullets": similar_bullets,
                "playbook_context": playbook_context
            })
        
        # Build batch prompt
        items_text = []
        for idx, item in enumerate(batch_data, 1):
            items_text.append(f"""
ITEM {idx}:
Source: {item['source_info']}
Section: {item['section']}
Content: {item['content']}
Similar bullets in playbook: {item['playbook_context']}
""")
        
        user_message = f"""BATCH VALIDATION: Validate {len(batch_data)} extracted knowledge items.

{''.join(items_text)}

For EACH item, validate:
1. is_reusable: Can this be applied to other transactions?
2. is_correct: Is the content accurate and well-formed?
3. is_duplicate: Does similar content exist in the playbook (check similar bullets)?

SPECIAL HANDLING:
- For incomplete definitions (containing "..."): Mark is_correct=true, provide enriched_content with COMPLETE definition, recommendation=ADD
- For pointer definitions: Provide enriched_content with complete substantive definition
- For malformed definitions: Fix format in enriched_content
- For properly formatted definitions: Set enriched_content to null

CRITICAL: You MUST return a valid JSON array. Return ONLY the JSON array, no other text.

Return JSON array with one object per item:
[
  {{
    "item_index": 1,
    "is_reusable": true,
    "is_correct": true,
    "is_duplicate": false,
    "similar_bullets": ["bullet-id-1"],
    "enriched_content": "For pointer/malformed definitions: complete definition, otherwise null",
    "quality_score": 0.85,
    "specificity_score": 0.90,
    "reusability_score": 0.80,
    "recommendation": "ADD",
    "reasoning": "Brief explanation",
    "update_reason": "For MODIFY only: specific reason why the bullet is being updated"
  }},
  ...
]

REMEMBER:
- Array must have EXACTLY {len(batch_data)} items
- Each item must have all required fields
- Return ONLY the JSON array, no markdown code blocks, no explanations"""
        
        try:
            response = self.llm_client.chat(self.SYSTEM_PROMPT, user_message)
            parsed = response.parse_json()
            
            # Handle parsing errors
            if parsed is None:
                logger.error(f"Batch validation failed to parse JSON. Falling back to individual processing.")
                return self._validate_individually(batch_items, playbook)
            
            # Ensure we got a list
            if not isinstance(parsed, list):
                logger.warning(f"Batch validation returned non-list: {type(parsed)}. Converting.")
                if isinstance(parsed, dict):
                    parsed = [parsed]
                else:
                    parsed = []
            
            if len(parsed) != len(batch_data):
                logger.warning(f"Batch validation returned {len(parsed)} items, expected {len(batch_data)}")
            
            # Map results back to items
            results = []
            for idx, item in enumerate(batch_data):
                if idx < len(parsed):
                    result = parsed[idx]
                    if isinstance(result, dict):
                        results.append(ValidatorOutput(
                            is_reusable=result.get("is_reusable", False),
                            is_correct=result.get("is_correct", False),
                            is_duplicate=result.get("is_duplicate", False),
                            similar_bullets=result.get("similar_bullets", []),
                            enriched_content=result.get("enriched_content"),
                            quality_score=result.get("quality_score", 0.0),
                            specificity_score=result.get("specificity_score", 0.0),
                            reusability_score=result.get("reusability_score", 0.0),
                            recommendation=result.get("recommendation", "SKIP"),
                            reasoning=result.get("reasoning", ""),
                            update_reason=result.get("update_reason")
                        ))
                    else:
                        logger.warning(f"Item {idx+1} result is not a dict: {type(result)}")
                        results.append(self._create_skip_output(f"Invalid result format"))
                else:
                    logger.warning(f"Missing validation result for item {idx+1}")
                    results.append(self._create_skip_output("Missing result"))
            
            return results
        
        except Exception as e:
            logger.error(f"Batch validation error: {e}", exc_info=True)
            logger.warning(f"Falling back to individual processing for {len(batch_items)} items")
            return self._validate_individually(batch_items, playbook)
    
    def _validate_individually(
        self,
        items: List[Dict[str, Any]],
        playbook: Playbook
    ) -> List[ValidatorOutput]:
        """Fallback: validate items individually."""
        results = []
        for item in items:
            try:
                # Get similar bullets
                similar_bullets = []
                if self.retriever:
                    try:
                        results_search = self.retriever.search(item["content"], sections=[item["section"]], top_k=5)
                        similar_bullets = [r.bullet_id for r in results_search if r.score > 0.75]
                    except Exception:
                        pass
                
                playbook_context = self._build_playbook_context(playbook, item["section"], similar_bullets)
                
                user_message = f"""VALIDATE THIS ITEM:

Source: {item.get('source_info', '')}
Section: {item['section']}
Content: {item['content']}
Similar bullets: {playbook_context}

Validate and provide recommendation."""
                
                response = self.llm_client.chat(self.SYSTEM_PROMPT, user_message)
                parsed = response.parse_json()
                
                if parsed and isinstance(parsed, dict):
                    results.append(ValidatorOutput(
                        is_reusable=parsed.get("is_reusable", False),
                        is_correct=parsed.get("is_correct", False),
                        is_duplicate=parsed.get("is_duplicate", False),
                        similar_bullets=parsed.get("similar_bullets", []),
                        enriched_content=parsed.get("enriched_content"),
                        quality_score=parsed.get("quality_score", 0.0),
                        specificity_score=parsed.get("specificity_score", 0.0),
                        reusability_score=parsed.get("reusability_score", 0.0),
                        recommendation=parsed.get("recommendation", "SKIP"),
                        reasoning=parsed.get("reasoning", ""),
                        update_reason=parsed.get("update_reason")
                    ))
                else:
                    results.append(self._create_skip_output("Parsing error"))
            
            except Exception as e:
                logger.error(f"Individual validation failed: {e}")
                results.append(self._create_skip_output(f"Error: {e}"))
        
        return results
    
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
    
    def _create_skip_output(self, reason: str) -> ValidatorOutput:
        """Create a SKIP output with default values."""
        return ValidatorOutput(
            is_reusable=False,
            is_correct=False,
            is_duplicate=False,
            similar_bullets=[],
            enriched_content=None,
            quality_score=0.0,
            specificity_score=0.0,
            reusability_score=0.0,
            recommendation="SKIP",
            reasoning=reason
        )

