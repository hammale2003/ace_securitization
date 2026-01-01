"""
Extractor Agent - Specialized extraction with 4 section-specific prompts.

Extracts knowledge from documents using specialized prompts for:
- strategies: Abstract into reusable principles, NEVER copy-paste contract text
- definitions: Extract exactly, include pointer definitions
- pitfalls: Abstract into reusable warnings
- templates: Convert to {{PLACEHOLDERS}} and [IF]...[ENDIF] patterns

Supports two granularity modes:
- OPERATIVE_CLAUSE_BY_CLAUSE: Process each clause individually
- FULL_DOCUMENT: Process entire document (or chunked for large docs)
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass

from llm_client import LLMClient
from playbook_enricher.document_parser import ParsedDocument, ParsedClause
from playbook_enricher.granularity import GranularityLevel
from utils import logger


def clean_extracted_text(text: str) -> str:
    """
    Clean formatting artifacts from extracted text.
    
    Removes:
    - Newline characters (\\n, \\r)
    - Escaped characters (\\\\)
    - Triple quotes (triple double quotes)
    - Extra whitespace
    - Other formatting artifacts
    
    Preserves the actual content and meaning.
    """
    if not text:
        return ""
    
    # Remove newline characters
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    
    # Remove escaped backslashes
    text = text.replace("\\\\", "")
    
    # Remove triple quotes
    text = text.replace('"""', '"').replace("'''", "'")
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


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


# Specialized system prompts for each section
STRATEGIES_EXTRACTION_PROMPT = """You are extracting STRATEGIES (best practices, methodologies) from securitization documents.

CRITICAL RULES:
1. REFORMULATE clauses into reusable best practices - DO NOT copy-paste raw contract text
2. ABSTRACT the underlying principle/pattern that can be applied to different transactions
3. REMOVE transaction-specific entity names, dates, amounts, clause references, document-specific details
4. CLEAN formatting: Remove all \\n, \\\\, triple quotes, and other formatting artifacts from original document
5. FORMAT as clean, actionable guidance that applies to ANY similar transaction
6. Think: "What is the reusable insight here that another lawyer could apply when drafting ANY securitization contract?"

EXTRACTION PROCESS:
- Read the clause text
- Identify the underlying pattern or best practice
- Remove ALL transaction-specific details (party names, dates, amounts, clause references, document titles)
- Clean all formatting artifacts (\\n, \\\\, triple quotes, extra spaces, line breaks)
- Reformulate as a generalizable, reusable strategy
- Ensure the strategy is applicable to ANY securitization transaction, not just this one

QUALITY REQUIREMENTS:
- Strategy must be reusable across different transactions
- Must NOT reference specific parties, dates, amounts, or clause numbers
- Must be formatted as clean text (no \\n, \\\\, triple quotes, or other artifacts)
- Must be actionable guidance that an LLM can use to answer securitization questions

Example transformations:
Raw: "Each party shall enter into Transaction Documents to incorporate definitions by reference"
Strategy: "Use master framework agreements to incorporate common definitions by reference across transaction documents, reducing redundancy and ensuring consistency."

Raw: "The Seller shall provide monthly reports to the Issuer detailing all Collections received"
Strategy: "Establish regular reporting mechanisms (e.g., monthly) for Collections to enable timely monitoring and reconciliation."

Raw with artifacts: "The Seller\\n shall provide\\n monthly reports..."
Strategy: "Establish regular reporting mechanisms (e.g., monthly) for Collections to enable timely monitoring and reconciliation."

Return JSON:
{
  "extracted_items": [
    {
      "section": "strategies",
      "content": "Clean, reformulated strategy as actionable guidance (NO formatting artifacts, NO transaction-specific details)",
      "reasoning": "How this was abstracted from the clause"
    }
  ]
}

CRITICAL: Output clean text only. Remove ALL \\n, \\\\, triple quotes, and formatting artifacts. Make strategies reusable for ANY contract.

BE COMPREHENSIVE - Extract ALL strategies you find, not just a few."""

DEFINITIONS_EXTRACTION_PROMPT = """You are extracting DEFINITIONS (key terms with precise legal meanings) from securitization documents.

CRITICAL SELECTIVITY RULES:
1. Extract ONLY domain-specific securitization terms - NOT generic terms that any LLM can define
2. Focus on terms specific to Tranchify Ltd and securitization transactions
3. SKIP generic terms like: "Agreement", "Party", "Date", "Person", "Company", "Document", etc.
4. EXTRACT terms that are specific to Tranchify Ltd and securitization transactions
5. Include substantive definitions AND pointer definitions (they will be enriched later)
6. CLEAN formatting: Remove formatting artifacts (\\n, \\\\, triple quotes) but preserve the actual definition text

CRITICAL COMPLETENESS REQUIREMENT - ABSOLUTE MANDATORY:
1. You MUST extract COMPLETE definitions - NEVER use "..." (ellipsis) to truncate definitions
2. Extract the ENTIRE definition text from the source document, including ALL sub-clauses, conditions, qualifications, and complete sentences
3. If a definition spans multiple sentences or paragraphs, extract ALL of them
4. If a definition has multiple parts (a), (b), (c), etc., extract ALL parts completely
5. NEVER output incomplete definitions with "..." - this is STRICTLY FORBIDDEN
6. If you cannot find the complete definition in the provided text, DO NOT extract it with "..."
7. Search the entire document/clause text to find the complete definition before extracting

WHAT TO EXTRACT (Domain-Specific):
- Securitization-specific terms (Collections, Receivables, ABS Transaction, etc.)
- Transaction structure terms (Subordinated Advances, Senior Facility, Mezzanine Notes, etc.)
- Financial terms specific to securitization (Cut-Off Date, Early Amortisation Event, etc.)
- Terms with clause references (will be enriched by validator)

WHAT TO SKIP (Generic):
- Generic legal terms (Agreement, Party, Person, Company, Document, etc.)
- Common English words
- Terms that any LLM can define without domain knowledge

Example extractions (EXTRACT - COMPLETE definitions only):
- "ABS Transaction: means any securitisation of any Receivables originated by the Seller"
- "Collections: Means, in respect of any Purchased Underlying Exposure: (a) all cash collections and other cash proceeds of the Receivable, including finance charges (if any), all VAT (if any) and cash proceeds of any Related Rights; (b) any proceeds received by the Seller of the Receivable from insurance policies; (c) any Deemed Collection; (d) any proceeds from the sale of the Purchased Receivable; and (e) any Repurchase Price"
- "Cut-Off Date" means the date specified in the Transaction Documents
- "Agreed Ad Hoc Funding Amount" means, as at any Cut-Off Date, any Subordinated Advances made by the Subordinated Lender to fund specified ad hoc costs payable by the Issuer in accordance with the Transaction Documents which are pre-approved in writing by the Facility Agent (contains clause references - will be enriched)

Example SKIP (DO NOT EXTRACT):
- "Agreement" means this document
- "Party" means each party to this Agreement
- "Date" means any calendar date

FORBIDDEN - NEVER EXTRACT LIKE THIS (incomplete definitions):
- "Collections: Means, in respect of any Purchased Underlying Exposure: (a) all cash collections..." ❌
- "Senior Borrowing Base" means... the product of: the Senior Advance Rate... ❌
- "Senior Commitment" means: in relation to the Original Senior Lender, the amount set opposite its name...; ❌

Return JSON:
{
  "extracted_items": [
    {
      "section": "definitions",
      "content": "COMPLETE definition text - NO ellipsis (...), NO truncation, full definition from source",
      "reasoning": "Why this term is domain-specific and important"
    }
  ]
}

CRITICAL REMINDERS:
- Only extract domain-specific securitization terms. Skip generic terms.
- Extract COMPLETE definitions - NEVER use "..." ellipsis.
- If you cannot find the complete definition, DO NOT extract it."""

PITFALLS_EXTRACTION_PROMPT = """You are extracting PITFALLS (common mistakes, things to avoid, red flags) from securitization documents.

CRITICAL RULES:
1. REFORMULATE into reusable warnings - DO NOT copy-paste raw contract text
2. Extract the UNDERLYING MISTAKE/RISK that can occur in different contexts
3. REMOVE transaction-specific details (party names, dates, amounts, clause references)
4. CLEAN formatting: Remove all \\n, \\\\, triple quotes, and other formatting artifacts
5. FORMAT as clean, clear warnings about what to avoid

EXTRACTION PROCESS:
- Read the clause text
- Identify the mistake, risk, or thing to avoid
- Abstract it into a generalizable warning
- Remove all transaction-specific details
- Clean all formatting artifacts (\\n, \\\\, triple quotes, extra spaces, line breaks)

Example transformations:
Raw clause mentioning: "Failing to include Deemed Collections in the Collections definition"
Pitfall: "Deemed Collections Omission: Failing to include 'Deemed Collections' in the definition of Collections. This is a critical error in securitization drafting, as it prevents the Issuer from capturing value when the Seller reduces a receivable balance through non-cash means (e.g., product returns or set-off)."

Raw: "Not specifying the calculation method for fees can lead to disputes"
Pitfall: "Fee Calculation Ambiguity: Failing to specify the exact calculation method for transaction fees can lead to disputes and inconsistent application across different transactions."

Return JSON:
{
  "extracted_items": [
    {
      "section": "pitfalls",
      "content": "Clean, reformulated pitfall as clear warning (NO formatting artifacts)",
      "reasoning": "How this was abstracted from the clause"
    }
  ]
}

CRITICAL: Output clean text only. Remove ALL \\n, \\\\, triple quotes, and formatting artifacts.

BE COMPREHENSIVE - Extract ALL pitfalls you find."""

TEMPLATES_EXTRACTION_PROMPT = """You are extracting TEMPLATES (reusable clause patterns) from securitization documents.

CRITICAL RULES:
1. Convert clause patterns into reusable templates with {{PLACEHOLDERS}}
2. Use [IF condition]...[ENDIF] patterns for conditional variants
3. Generalize transaction-specific language
4. CLEAN formatting: Remove all \\n, \\\\, triple quotes, and other formatting artifacts
5. Preserve the structure and logic of the clause

TEMPLATE PATTERNS:
- {{PLACEHOLDER}} for variable elements (party names, amounts, dates, etc.)
- [IF condition]...[ENDIF] for conditional logic
- [IF condition = TRUE]...[ELSE]...[ENDIF] for alternatives

Example transformations:
Raw: "Upon the terms and subject to the conditions of this Agreement, each Lender hereby grants to the Issuer, during the Revolving Period, a revolving loan facility in an amount equal to the Total Commitments."
Template: "Revolving vs Term Facility Pattern: [IF is_revolving = TRUE] 'Upon the terms and subject to the conditions of this Agreement, each {{holder}} hereby grants to the Issuer, during the Revolving Period, a revolving loan facility in an amount equal to the {{name}} Total Commitments.' [IF is_revolving = FALSE] 'Upon the terms and subject to the conditions of this Agreement, each {{holder}} hereby grants to the Issuer a term loan facility in an amount equal to the {{name}} Total Commitments.' [ENDIF]"

Raw: "The Seller shall pay fees calculated as 0.5% of the outstanding balance"
Template: "Fee Calculation Pattern: The {{party}} shall pay fees calculated as {{fee_percentage}}% of the {{calculation_base}}."

Return JSON:
{
  "extracted_items": [
    {
      "section": "templates",
      "content": "Clean, generalized template with {{placeholders}} and [IF] patterns (NO formatting artifacts)",
      "reasoning": "How this was generalized from the clause"
    }
  ]
}

CRITICAL: Output clean text only. Remove ALL \\n, \\\\, triple quotes, and formatting artifacts.

BE COMPREHENSIVE - Extract ALL template patterns you find."""


class ExtractorAgent:
    """
    Extractor Agent with 4 specialized system prompts.
    
    Supports two granularity modes:
    - clause-by-clause: Process each clause individually
    - full-document: Process entire document (or chunked for large docs)
    """
    
    # Map section to specialized prompt
    SECTION_PROMPTS = {
        "strategies": STRATEGIES_EXTRACTION_PROMPT,
        "definitions": DEFINITIONS_EXTRACTION_PROMPT,
        "pitfalls": PITFALLS_EXTRACTION_PROMPT,
        "templates": TEMPLATES_EXTRACTION_PROMPT
    }
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def extract_from_document(
        self,
        document: ParsedDocument,
        granularity_level: GranularityLevel = GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE,
        allowed_sections: List[str] = None
    ) -> List[ExtractedKnowledge]:
        """
        Extract knowledge using specialized prompts per section.
        
        Args:
            document: Parsed document
            granularity_level: Clause-by-clause or full-document
            allowed_sections: Sections to extract (strategies, definitions, pitfalls, templates)
        
        Returns:
            List of extracted knowledge items
        """
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
        """Extract knowledge clause by clause using specialized prompts."""
        logger.info(f"Extracting clause-by-clause from {len(clauses)} operative clauses")
        logger.info(f"Allowed sections: {', '.join(allowed_sections)}")
        
        all_extracted = []
        
        for idx, clause in enumerate(clauses, 1):
            logger.debug(f"Processing clause {idx}/{len(clauses)}: {clause.uid}")
            
            clause_text = clause.get_full_text()
            if not clause_text or len(clause_text.strip()) < 50:
                continue
            
            # Extract for each allowed section using specialized prompt
            for section in allowed_sections:
                if section not in self.SECTION_PROMPTS:
                    logger.warning(f"Unknown section: {section}, skipping")
                    continue
                
                system_prompt = self.SECTION_PROMPTS[section]
                
                user_message = f"""Document: {document.title}
Type: {document.document_type}
Clause: {clause.title_text or clause.uid}

Text:
{clause_text}

TASK: Extract {section} from this clause using the specialized extraction rules."""

                try:
                    response = self.llm_client.chat(system_prompt, user_message)
                    parsed = response.parse_json()
                    
                    if parsed and "extracted_items" in parsed:
                        for item in parsed["extracted_items"]:
                            content = item.get("content")
                            
                            if not content:
                                continue
                            
                            # CRITICAL: Only accept items that match the requested section
                            item_section = item.get("section", section)
                            if item_section != section:
                                logger.warning(f"Item section '{item_section}' doesn't match requested '{section}', skipping item")
                                continue
                            
                            # Double-check: only add if section is in allowed_sections
                            if section not in allowed_sections:
                                logger.warning(f"Section '{section}' not in allowed sections, skipping")
                                continue
                            
                            # Clean formatting artifacts from content
                            cleaned_content = clean_extracted_text(content)
                            if not cleaned_content:
                                logger.debug(f"Skipping item: content is empty after cleaning")
                                continue
                            
                            all_extracted.append(ExtractedKnowledge(
                                content=cleaned_content,
                                section=section,
                                source_clause_uid=clause.uid,
                                source_clause_title=clause.title_text,
                                source_clause_text=clause_text,
                                document_title=document.title,
                                document_type=document.document_type
                            ))
                
                except Exception as e:
                    logger.warning(f"Failed to extract {section} from clause {clause.uid}: {e}")
                    continue
        
        logger.info(f"Extracted {len(all_extracted)} knowledge items")
        
        # Log breakdown by section
        section_counts = {}
        for item in all_extracted:
            section_counts[item.section] = section_counts.get(item.section, 0) + 1
        if section_counts:
            logger.info(f"Section breakdown: {section_counts}")
        
        return all_extracted
    
    def _extract_full_document(
        self,
        document: ParsedDocument,
        clauses: List[ParsedClause],
        allowed_sections: List[str]
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge from full document using specialized prompts."""
        logger.info(f"Extracting from full document with {len(clauses)} operative clauses")
        logger.info(f"Allowed sections: {', '.join(allowed_sections)}")
        
        # Check if document is too large and needs chunking
        MAX_CLAUSES_PER_CHUNK = 500
        
        if len(clauses) > MAX_CLAUSES_PER_CHUNK:
            logger.warning(f"Document has {len(clauses)} clauses, chunking into {MAX_CLAUSES_PER_CHUNK} clause chunks")
            return self._extract_full_document_chunked(document, clauses, allowed_sections, MAX_CLAUSES_PER_CHUNK)
        
        document_text = self._build_document_text(document, clauses)
        all_extracted = []
        
        # Extract for each allowed section
        for section in allowed_sections:
            if section not in self.SECTION_PROMPTS:
                continue
            
            system_prompt = self.SECTION_PROMPTS[section]
            
            user_message = f"""Document: {document.title}
Type: {document.document_type}

Full Document Text:
{document_text}

TASK: Extract {section} from this document using the specialized extraction rules.
Extract ONLY from the text above. Do not invent or generalize."""

            try:
                response = self.llm_client.chat(system_prompt, user_message)
                parsed = response.parse_json()
                
                if parsed and "extracted_items" in parsed:
                    for item in parsed["extracted_items"]:
                        content = item.get("content")
                        
                        if not content:
                            continue
                        
                        # Validate that LLM returned correct section field (log warning if mismatch)
                        item_section = item.get("section", section)
                        if item_section != section:
                            logger.debug(f"LLM returned section '{item_section}' instead of '{section}' - correcting automatically")
                        
                        # Double-check: only add if section is in allowed_sections
                        if section not in allowed_sections:
                            logger.warning(f"Section '{section}' not in allowed sections, skipping")
                            continue
                        
                        # Clean formatting artifacts from content
                        cleaned_content = clean_extracted_text(content)
                        if not cleaned_content:
                            logger.debug(f"Skipping item: content is empty after cleaning")
                            continue
                        
                        all_extracted.append(ExtractedKnowledge(
                            content=cleaned_content,
                            section=section,
                            source_clause_uid=document.document_uid,
                            source_clause_title=document.title,
                            source_clause_text="Full document",
                            document_title=document.title,
                            document_type=document.document_type
                        ))
            
            except Exception as e:
                logger.error(f"Failed to extract {section} from full document: {e}")
                continue
        
        logger.info(f"Extracted {len(all_extracted)} knowledge items")
        
        # Log breakdown by section
        section_counts = {}
        for item in all_extracted:
            section_counts[item.section] = section_counts.get(item.section, 0) + 1
        if section_counts:
            logger.info(f"Section breakdown: {section_counts}")
        
        return all_extracted
    
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
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(clauses))
            chunk_clauses = clauses[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (clauses {start_idx + 1}-{end_idx})")
            
            chunk_text = self._build_document_text(document, chunk_clauses)
            
            # Extract for each allowed section
            for section in allowed_sections:
                if section not in self.SECTION_PROMPTS:
                    continue
                
                system_prompt = self.SECTION_PROMPTS[section]
                
                user_message = f"""Document: {document.title} (Part {chunk_idx + 1}/{num_chunks})
Type: {document.document_type}

Document Text (Clauses {start_idx + 1}-{end_idx} of {len(clauses)}):
{chunk_text}

TASK: Extract {section} from this chunk using the specialized extraction rules.
Extract ONLY from the text above."""

                try:
                    response = self.llm_client.chat(system_prompt, user_message)
                    parsed = response.parse_json()
                    
                    if parsed and "extracted_items" in parsed:
                        for item in parsed["extracted_items"]:
                            content = item.get("content")
                            
                            if not content:
                                continue
                            
                            # Validate that LLM returned correct section field (log warning if mismatch)
                            item_section = item.get("section", section)
                            if item_section != section:
                                logger.debug(f"LLM returned section '{item_section}' instead of '{section}' - correcting automatically")
                            
                            # Double-check: only add if section is in allowed_sections
                            if section not in allowed_sections:
                                logger.warning(f"Section '{section}' not in allowed sections, skipping")
                                continue
                            
                            # Clean formatting artifacts from content
                            cleaned_content = clean_extracted_text(content)
                            if not cleaned_content:
                                logger.debug(f"Skipping item: content is empty after cleaning")
                                continue
                            
                            all_extracted.append(ExtractedKnowledge(
                                content=cleaned_content,
                                section=section,
                                source_clause_uid=f"{document.document_uid}_chunk_{chunk_idx + 1}",
                                source_clause_title=f"{document.title} (Part {chunk_idx + 1}/{num_chunks})",
                                source_clause_text=f"Chunk {chunk_idx + 1}/{num_chunks}",
                                document_title=document.title,
                                document_type=document.document_type
                            ))
                
                except Exception as e:
                    logger.error(f"Failed to extract {section} from chunk {chunk_idx + 1}/{num_chunks}: {e}")
                    continue
        
        logger.info(f"Total extracted from all chunks: {len(all_extracted)} knowledge items")
        
        # Log breakdown by section
        section_counts = {}
        for item in all_extracted:
            section_counts[item.section] = section_counts.get(item.section, 0) + 1
        if section_counts:
            logger.info(f"Section breakdown: {section_counts}")
        
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

