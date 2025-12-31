"""
Knowledge Extractor for Playbook Enricher.

Extracts raw knowledge from document clauses for ACE processing.
"""
from typing import List, Dict, Any
from dataclasses import dataclass, field

from llm_client import LLMClient
from playbook_enricher.document_parser import ParsedDocument, ParsedClause
from playbook_enricher.granularity import GranularityLevel
from utils import logger


@dataclass
class ExtractedKnowledge:
    """Raw knowledge extracted from a document clause."""
    content: str
    section: str  # strategies, pitfalls, templates, or definitions
    source_clause_uid: str
    source_clause_title: str
    source_clause_text: str
    document_title: str
    document_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "section": self.section,
            "source_clause_uid": self.source_clause_uid,
            "source_clause_title": self.source_clause_title,
            "source_clause_text": self.source_clause_text[:200] + "..." if len(self.source_clause_text) > 200 else self.source_clause_text,
            "document_title": self.document_title,
            "document_type": self.document_type
        }


class KnowledgeExtractor:
    """Extracts raw knowledge from securitization documents."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def _build_extraction_prompt(self, allowed_sections: List[str]) -> str:
        """Build extraction prompt based on allowed sections."""
        
        section_descriptions = {
            "strategies": "STRATEGIES: Best practices, approaches, methodologies for drafting/analysis",
            "pitfalls": "PITFALLS: Common mistakes, things to avoid, red flags",
            "templates": "TEMPLATES: Reusable clause patterns, boilerplate structures",
            "definitions": "DEFINITIONS: Key terms with precise legal meanings"
        }
        
        section_examples = {
            "strategies": 'STRATEGY: "Jurisdictional Completeness Rule: When deriving variants for jurisdictions (UK, EU, US, etc.), ALWAYS generate: (1) Each jurisdiction standalone, AND (2) The combined/cross-border variant. A transaction may have entities in multiple jurisdictions simultaneously."',
            "pitfalls": 'PITFALL: "Deemed Collections Omission: Failing to include \'Deemed Collections\' in the definition of Collections. This is a critical error in securitization drafting, as it prevents the Issuer from capturing value when the Seller reduces a receivable balance through non-cash means (e.g., product returns or set-off)."',
            "templates": 'TEMPLATE: "Revolving vs Term Facility Pattern: [if is_revolving = TRUE] \'Upon the terms and subject to the conditions of this Agreement, each {{holder}} hereby grant to the Issuer, during the Revolving Period, a revolving loan facility in an amount equal to the {{name}} Total Commitments.\' [if is_revolving = FALSE] \'Upon the terms and subject to the conditions of this Agreement, each {{holder}} hereby grant to the Issuer a term loan facility in an amount equal to the {{name}} Total Commitments.\'"',
            "definitions": 'DEFINITION: "ABS Transaction: means any securitisation of any Receivables originated by the Seller" OR "ABS Transaction Fee has the meaning given to it in clause 8.2(b)" OR "Collections: Means, in respect of any Purchased Underlying Exposure: (a) all cash collections..."'
        }
        
        sections_text = "\n".join([f"{i+1}. {section_descriptions[s]}" for i, s in enumerate(allowed_sections)])
        examples_text = "\n- ".join([section_examples[s] for s in allowed_sections])
        allowed_sections_list = "|".join(allowed_sections)
        
        return f"""You are extracting knowledge from a securitization document for playbook enrichment.

CRITICAL RULES - READ CAREFULLY:
1. EXTRACT ONLY - DO NOT INVENT OR HALLUCINATE
2. Extract ONLY from the text provided in the document
3. You MUST extract ONLY for these sections: {', '.join(allowed_sections)}
4. DO NOT extract content for sections not in the allowed list
5. BE COMPREHENSIVE - Extract ALL items that match the allowed sections, not just a few

ALLOWED SECTIONS FOR THIS EXTRACTION:
{sections_text}

SECTION-SPECIFIC EXTRACTION RULES:

{"- DEFINITIONS: Extract ALL term definitions including pointer definitions (e.g., 'X has the meaning in clause Y'). Extract EXACTLY as written. Extract EVERY definition you find." if "definitions" in allowed_sections else ""}
{"- STRATEGIES: REFORMULATE clauses into reusable best practices. DO NOT copy-paste raw contract text. Extract the UNDERLYING PATTERN/PRINCIPLE that can be applied to different transactions. Remove transaction-specific names/details. Format as actionable guidance. Example: Raw clause 'Each party shall enter into Transaction Documents to incorporate definitions by reference' → Strategy 'Use master framework agreements to incorporate common definitions by reference across transaction documents, reducing redundancy and ensuring consistency.'" if "strategies" in allowed_sections else ""}
{"- PITFALLS: REFORMULATE into reusable warnings. Extract the UNDERLYING MISTAKE/RISK that can occur in different contexts. Remove transaction-specific details. Format as clear warnings. Example: Raw clause mentioning a specific error → Pitfall 'Risk: Failing to include deemed collections in the Collections definition prevents capturing value from non-cash receivable reductions.'" if "pitfalls" in allowed_sections else ""}
{"- TEMPLATES: Extract reusable clause PATTERNS with conditional variants [if...] or parameterized patterns using {{placeholders}}. Generalize transaction-specific language. Extract EVERY template pattern you find." if "templates" in allowed_sections else ""}

Extract information that is:
- For DEFINITIONS: EXPLICITLY PRESENT in the document text (extract exactly as written)
- For STRATEGIES/PITFALLS: REFORMULATED from clauses into reusable insights (abstract the principle, remove transaction-specific names)
- For TEMPLATES: Generalized patterns with {{placeholders}} for variable elements
- Specific to this transaction structure OR reusable patterns (not generic legal principles)
- Belongs to ONE of the allowed sections: {', '.join(allowed_sections)}
- Applicable to similar transactions in different contexts

CRITICAL - REFORMULATION FOR STRATEGIES/PITFALLS:
- DO NOT copy-paste raw contract sentences
- ABSTRACT the underlying principle/pattern
- REMOVE transaction-specific entity names, dates, amounts
- FORMAT as actionable guidance that applies to ANY similar transaction
- Think: "What is the reusable insight here that another lawyer could apply?"

IMPORTANT - BE COMPREHENSIVE:
- Do NOT skip items because you think they're "too obvious" or "too simple"
- Do NOT limit yourself to a small number - extract EVERYTHING that matches

Return JSON:
{{
  "extracted_items": [
    {{
      "section": "{allowed_sections_list}",
      "content": "For definitions: exact text. For strategies/pitfalls: reformulated reusable insight. For templates: generalized pattern.",
      "reasoning": "Why this belongs to this section and how it was reformulated (if applicable)"
    }}
  ]
}}

Examples from actual playbook:
- {examples_text}

EXTRACTION RULES:
- Extract EVERYTHING that matches the allowed sections: {', '.join(allowed_sections)}
- If you extract a definition but "definitions" is not allowed, DISCARD it
- If you extract a strategy but "strategies" is not allowed, DISCARD it
- Only verify the SECTION is correct
- DO NOT limit the number of items - extract ALL matching content

Your role: EXTRACT EVERYTHING COMPREHENSIVELY. ACE's role: VALIDATE and DECIDE what to keep.
If you find 0 items, that's fine. But if you find 10 items, extract all 10, not just 2-3."""
    
    def extract_from_document(
        self,
        document: ParsedDocument,
        granularity_level: GranularityLevel = GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE,
        allowed_sections: List[str] = None
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge using specified granularity."""
        if allowed_sections is None:
            allowed_sections = ["strategies", "pitfalls", "templates", "definitions"]
        
        operative_clauses = document.get_operative_clauses()
        
        if granularity_level == GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE:
            return self._extract_clause_by_clause(document, operative_clauses, allowed_sections)
        else:
            return self._extract_full_document(document, operative_clauses, allowed_sections)
    
    def _extract_clause_by_clause(
        self,
        document: ParsedDocument,
        clauses: List[ParsedClause],
        allowed_sections: List[str]
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge clause by clause."""
        logger.info(f"Extracting clause-by-clause from {len(clauses)} operative clauses")
        logger.info(f"Allowed sections: {', '.join(allowed_sections)}")
        
        extraction_prompt = self._build_extraction_prompt(allowed_sections)
        
        all_extracted = []
        for idx, clause in enumerate(clauses, 1):
            logger.debug(f"Processing clause {idx}/{len(clauses)}: {clause.uid}")
            
            clause_text = clause.get_full_text()
            if not clause_text or len(clause_text.strip()) < 50:
                continue
            
            user_message = f"""Document: {document.title}
Type: {document.document_type}
Clause: {clause.title_text or clause.uid}

Text:
{clause_text}

TASK: Extract knowledge that matches the allowed sections: {', '.join(allowed_sections)}

CRITICAL INSTRUCTIONS:
- For DEFINITIONS: Extract term definitions EXACTLY as written
- For STRATEGIES: REFORMULATE clauses into reusable best practices - abstract the principle, remove transaction-specific names/dates/amounts
- For PITFALLS: REFORMULATE into reusable warnings - extract the underlying risk/mistake
- For TEMPLATES: Generalize into patterns with {{placeholders}}

DO NOT copy-paste raw contract text for strategies/pitfalls. Reformulate into actionable guidance."""
            
            try:
                response = self.llm_client.chat(extraction_prompt, user_message)
                parsed = response.parse_json()
                
                if parsed and "extracted_items" in parsed:
                    for item in parsed["extracted_items"]:
                        section = item.get("section")
                        content = item.get("content")
                 
                        if not content or not section:
                            logger.debug(f"Skipping item: missing content or section")
                            continue
                        
                        if section not in allowed_sections:
                            logger.warning(f"Skipping item: section '{section}' not in allowed sections {allowed_sections}")
                            continue
                        
                        all_extracted.append(ExtractedKnowledge(
                            content=content,
                            section=section,
                            source_clause_uid=clause.uid,
                            source_clause_title=clause.title_text,
                            source_clause_text=clause_text,
                            document_title=document.title,
                            document_type=document.document_type
                        ))
            except Exception as e:
                logger.warning(f"Failed to extract from clause {clause.uid}: {e}")
                continue
        
        logger.info(f"Extracted {len(all_extracted)} knowledge items")
        return all_extracted
    
    def _extract_full_document(
        self,
        document: ParsedDocument,
        clauses: List[ParsedClause],
        allowed_sections: List[str]
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge from full document in one pass (or chunked for large documents)."""
        logger.info(f"Extracting from full document with {len(clauses)} operative clauses")
        logger.info(f"Allowed sections: {', '.join(allowed_sections)}")
        
        # Check if document is too large and needs chunking
        MAX_CLAUSES_PER_CHUNK = 500  # Conservative limit to avoid token overflow
        
        if len(clauses) > MAX_CLAUSES_PER_CHUNK:
            logger.warning(f"Document has {len(clauses)} clauses, which exceeds {MAX_CLAUSES_PER_CHUNK}. Using chunked extraction.")
            return self._extract_full_document_chunked(document, clauses, allowed_sections, MAX_CLAUSES_PER_CHUNK)
        
        extraction_prompt = self._build_extraction_prompt(allowed_sections)
        document_text = self._build_document_text(document, clauses)
        
        user_message = f"""Document: {document.title}
Type: {document.document_type}

Full Document Text:
{document_text}

Extract ONLY from the text above. Do not invent or generalize.
Focus on transaction-specific patterns that are explicitly present in the document."""
        
        try:
            response = self.llm_client.chat(extraction_prompt, user_message)
            parsed = response.parse_json()
            
            all_extracted = []
            if parsed and "extracted_items" in parsed:
                for item in parsed["extracted_items"]:
                    section = item.get("section")
                    content = item.get("content")
                    
                
                    if not content or not section:
                        logger.debug(f"Skipping item: missing content or section")
                        continue
                    
                    if section not in allowed_sections:
                        logger.warning(f"Skipping item: section '{section}' not in allowed sections {allowed_sections}")
                        continue
                    
                    all_extracted.append(ExtractedKnowledge(
                        content=content,
                        section=section,
                        source_clause_uid=document.document_uid,
                        source_clause_title=document.title,
                        source_clause_text="Full document",
                        document_title=document.title,
                        document_type=document.document_type
                    ))
            
            logger.info(f"Extracted {len(all_extracted)} knowledge items")
            return all_extracted
        except Exception as e:
            logger.error(f"Failed to extract from full document: {e}")
            return []
    
    def _extract_full_document_chunked(
        self,
        document: ParsedDocument,
        clauses: List[ParsedClause],
        allowed_sections: List[str],
        chunk_size: int
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge from large documents by processing in chunks."""
        all_extracted = []
        num_chunks = (len(clauses) + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing document in {num_chunks} chunks of ~{chunk_size} clauses each")
        
        extraction_prompt = self._build_extraction_prompt(allowed_sections)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(clauses))
            chunk_clauses = clauses[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (clauses {start_idx + 1}-{end_idx})")
            
            chunk_text = self._build_document_text(document, chunk_clauses)
            
            user_message = f"""Document: {document.title} (Part {chunk_idx + 1}/{num_chunks})
Type: {document.document_type}

Document Text (Clauses {start_idx + 1}-{end_idx} of {len(clauses)}):
{chunk_text}

Extract ONLY from the text above. Do not invent or generalize.
Focus on transaction-specific patterns that are explicitly present in the document."""
            
            try:
                response = self.llm_client.chat(extraction_prompt, user_message)
                parsed = response.parse_json()
                
                # Log raw response for debugging
                if parsed and "extracted_items" in parsed:
                    raw_count = len(parsed["extracted_items"])
                    logger.debug(f"Chunk {chunk_idx + 1}/{num_chunks}: LLM returned {raw_count} raw items")
                else:
                    logger.warning(f"Chunk {chunk_idx + 1}/{num_chunks}: LLM returned no 'extracted_items' field. Response: {parsed}")
                
                if parsed and "extracted_items" in parsed:
                    chunk_extracted = 0
                    skipped_missing = 0
                    skipped_wrong_section = 0
                    
                    for item in parsed["extracted_items"]:
                        section = item.get("section")
                        content = item.get("content")
                        
                        if not content or not section:
                            skipped_missing += 1
                            logger.debug(f"Skipping item: missing content or section. Item: {item}")
                            continue
                        
                        if section not in allowed_sections:
                            skipped_wrong_section += 1
                            logger.warning(f"Skipping item: section '{section}' not in allowed sections {allowed_sections}. Content: {content[:100]}...")
                            continue
                        
                        all_extracted.append(ExtractedKnowledge(
                            content=content,
                            section=section,
                            source_clause_uid=f"{document.document_uid}_chunk_{chunk_idx + 1}",
                            source_clause_title=f"{document.title} (Part {chunk_idx + 1}/{num_chunks})",
                            source_clause_text=f"Chunk {chunk_idx + 1}/{num_chunks}",
                            document_title=document.title,
                            document_type=document.document_type
                        ))
                        chunk_extracted += 1
                    
                    logger.info(f"Chunk {chunk_idx + 1}/{num_chunks}: Extracted {chunk_extracted} items (skipped {skipped_missing} missing, {skipped_wrong_section} wrong section)")
                    
                    # Log breakdown by section
                    if chunk_extracted > 0:
                        section_counts = {}
                        for item in all_extracted[-chunk_extracted:]:
                            section_counts[item.section] = section_counts.get(item.section, 0) + 1
                        logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} breakdown: {section_counts}")
                else:
                    logger.warning(f"Chunk {chunk_idx + 1}/{num_chunks}: No items extracted")
                    
            except Exception as e:
                logger.error(f"Failed to extract from chunk {chunk_idx + 1}/{num_chunks}: {e}")
                continue
        
        logger.info(f"Total extracted from all chunks: {len(all_extracted)} knowledge items")
        return all_extracted
    
    def _build_document_text(self, document: ParsedDocument, clauses: List[ParsedClause]) -> str:
        """Build consolidated document text."""
        parts = [f"Document: {document.title}", f"Type: {document.document_type}", ""]
        
        for clause in clauses:
            clause_text = clause.get_full_text()
            if clause_text:
                parts.append(f"[{clause.uid}] {clause.title_text or 'Untitled'}")
                parts.append(clause_text)
                parts.append("")
        
        return "\n".join(parts)

