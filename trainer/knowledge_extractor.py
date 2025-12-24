"""
Knowledge Extractor Agent for Trainer Mode.

Extracts strategies, definitions, templates, pitfalls, and code snippets
from parsed securitization documents using LLM analysis.
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from llm_client import LLMClient, LLMResponse
from trainer.document_parser import ParsedDocument, ParsedClause
from trainer.granularity import GranularityLevel
from utils import logger


@dataclass
class ExtractedKnowledge:
    """Represents extracted knowledge from a document clause."""
    knowledge_type: str  # strategies, definitions, templates, pitfalls, code_snippets
    content: str
    source_clause_uid: str
    source_clause_title: str
    source_clause_text: str
    confidence: float = 0.0
    context: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "knowledge_type": self.knowledge_type,
            "content": self.content,
            "source_clause_uid": self.source_clause_uid,
            "source_clause_title": self.source_clause_title,
            "source_clause_text": self.source_clause_text[:200] + "..." if len(self.source_clause_text) > 200 else self.source_clause_text,
            "confidence": self.confidence,
            "context": self.context,
            "examples": self.examples,
            "related_terms": self.related_terms,
            "metadata": self.metadata
        }


class KnowledgeExtractor:
    """
    Extracts structured knowledge from securitization documents.
    
    Uses LLM to identify and extract:
    - Strategies: Patterns, approaches, best practices
    - Definitions: Terms, concepts, legal definitions
    - Templates: Reusable clause structures
    - Pitfalls: Anti-patterns, risks, common errors
    - Code Snippets: Formulas, calculations, logic
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def _get_operative_clauses(self, document: ParsedDocument, max_clauses: Optional[int] = None) -> List[ParsedClause]:
        """Get operative clauses from document."""
        all_clauses = document.get_all_clauses_flat()
        operative = [c for c in all_clauses if c.metadata.get("type") == "operative_clause"]
        return operative[:max_clauses] if max_clauses else operative
    
    def extract_from_document(
        self,
        document: ParsedDocument,
        extraction_types: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        granularity_level: GranularityLevel = GranularityLevel.BATCH,
        batch_size: int = 15,
        max_clauses: Optional[int] = None
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge using specified granularity level."""
        extraction_types = extraction_types or ["strategies", "definitions", "templates", "pitfalls", "code_snippets"]
        
        # Route based on granularity
        methods = {
            GranularityLevel.FULL_DOCUMENT: self._extract_full_document,
            GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE: self._extract_clause_by_clause,
            GranularityLevel.BATCH: self._extract_batch_mode
        }
        
        method = methods[granularity_level]
        
        if granularity_level == GranularityLevel.BATCH:
            return method(document, extraction_types, min_confidence, batch_size, max_clauses)
        elif granularity_level == GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE:
            return method(document, extraction_types, min_confidence, max_clauses)
        else:  # FULL_DOCUMENT
            return method(document, extraction_types, min_confidence, max_clauses)
    def _extract_batch_mode(
        self,
        document: ParsedDocument,
        extraction_types: List[str],
        min_confidence: float,
        batch_size: int,
        max_clauses: Optional[int]
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge using BATCH mode."""
        clauses = self._get_operative_clauses(document, max_clauses)
        logger.info(f"BATCH MODE: {len(clauses)} operative clauses, batch_size={batch_size}")
        
        all_extracted = []
        for extraction_type in extraction_types:
            for i in range(0, len(clauses), batch_size):
                batch = clauses[i:i + batch_size]
                logger.debug(f"{extraction_type}: batch {i//batch_size + 1}/{(len(clauses) + batch_size - 1)//batch_size}")
                
                extracted = self.extract_from_clause_batch(batch, extraction_type, document)
                all_extracted.extend([e for e in extracted if e.confidence >= min_confidence])
        
        return all_extracted
    
    def _extract_clause_by_clause(
        self,
        document: ParsedDocument,
        extraction_types: List[str],
        min_confidence: float,
        max_clauses: Optional[int]
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge using OPERATIVE_CLAUSE_BY_CLAUSE mode."""
        clauses = self._get_operative_clauses(document, max_clauses)
        total_calls = len(clauses) * len(extraction_types)
        logger.info(f"CLAUSE BY CLAUSE: {len(clauses)} operative clauses × {len(extraction_types)} types = {total_calls} LLM calls")
        
        all_extracted = []
        for idx, clause in enumerate(clauses, 1):
            logger.debug(f"Clause {idx}/{len(clauses)}: {clause.title_text[:40] if clause.title_text else 'Untitled'}...")
            for extraction_type in extraction_types:
                extracted = self.extract_from_clause(clause, extraction_type, document)
                all_extracted.extend([e for e in extracted if e.confidence >= min_confidence])
        
        return all_extracted
    
    def _extract_full_document(
        self,
        document: ParsedDocument,
        extraction_types: List[str],
        min_confidence: float,
        max_clauses: Optional[int] = None
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge using FULL_DOCUMENT mode - processes entire document."""
        # FULL_DOCUMENT mode: always process ALL operative clauses (ignore max_clauses)
        clauses = self._get_operative_clauses(document, max_clauses=None)
        logger.info(f"FULL DOCUMENT: Processing ALL {len(clauses)} operative clauses in single LLM call, {len(extraction_types)} extraction types")
        
        document_text = self._build_full_document_text(document, clauses)
        all_extracted = []
        
        for extraction_type in extraction_types:
            system_prompt = self._get_extraction_system_prompt(extraction_type)
            user_prompt = f"""DOCUMENT: {document.title}
TYPE: {document.document_type}

{document_text}

══════════════════════════════════════════════════════════════════════════════════════
MANDATORY COMPREHENSIVE EXTRACTION - NO SAMPLING ALLOWED
══════════════════════════════════════════════════════════════════════════════════════

You MUST extract EVERY SINGLE {extraction_type.upper()} from this entire document.

ZERO TOLERANCE FOR INCOMPLETE EXTRACTION:
• For DEFINITIONS: Extract ALL 100+ defined terms if present - missing ANY term is FAILURE
• For STRATEGIES: Extract ALL 50+ structural patterns - this is NOT a "give me 5 examples" task
• For TEMPLATES: Extract ALL reusable clause structures - EVERY SINGLE ONE
• For PITFALLS: Extract ALL risks and anti-patterns - do not cherry-pick

[UNACCEPTABLE] Returning 10-20 items when document has 100+ relevant items
[REQUIRED] Exhaustive extraction of EVERY relevant item in the document

SPECIAL INSTRUCTION FOR DEFINITIONS:
• If the document contains a dedicated "Definitions" or "Interpretation" clause, extract EVERY SINGLE definition AS-IS
• PRESERVE the full original definition text - DO NOT make it concise or summarize
• You may ENHANCE definitions by adding context, cross-references, or implications, but NEVER shorten them
• Keep the complete legal language intact - definitions must remain legally precise and comprehensive
• If original definition is 200 words, your extraction should be 200+ words (original + enhancements)

QUALITY REQUIREMENTS:
• Each extraction MUST be 50+ words with detailed explanation
• Include WHY it matters + HOW it works + WHAT the implications are
• NO title-only extractions - each item needs full context and rationale
• Reference specific clause titles and mechanisms from the source text

This is building a COMPLETE knowledge base - treat this as a MANDATORY EXHAUSTIVE SCAN.

══════════════════════════════════════════════════════════════════════════════════════
EXPECTED OUTPUT QUANTITY:
══════════════════════════════════════════════════════════════════════════════════════
• Small document (20-50 clauses): Expect 30-80 {extraction_type} items
• Medium document (50-150 clauses): Expect 80-150 {extraction_type} items  
• Large document (150-500 clauses): Expect 150-300 {extraction_type} items
• This document has {len(clauses)} operative clauses

[UNACCEPTABLE OUTPUTS]
• Returning 10-20 items when document has 200+ clauses (FAILURE)
• Extracting only "interesting" or "important" items (WRONG - extract ALL)
• Stopping after first 50 items because you're tired (UNACCEPTABLE)

[ACCEPTABLE OUTPUT]
• Comprehensive JSON array with 100-200+ detailed extraction objects
• Each item 50-150 words with full context and rationale
• Every relevant {extraction_type} from document represented

Return JSON:
{{"extractions": [{{"content": "DETAILED 50+ word explanation...", "confidence": 0.8, "source_clause_title": "Exact clause title", "examples": [{{"text": "quote", "clause_ref": "CLAUSE-X"}}], "related_terms": ["term1", "term2"]}}]}}

START YOUR EXHAUSTIVE EXTRACTION NOW - RETURN 50-200+ ITEMS:
"""
            
            try:
                response = self.llm_client.chat(system_prompt, user_prompt)
                parsed = response.parse_json()
                
                for item in parsed.get("extractions", []):
                    source_title = item.get("source_clause_title", "")
                    clause = next((c for c in clauses if c.title_text and source_title in c.title_text), clauses[0] if clauses else None)
                    
                    if clause and item.get("confidence", 0) >= min_confidence:
                        all_extracted.append(ExtractedKnowledge(
                            knowledge_type=extraction_type,
                            content=item.get("content", ""),
                            source_clause_uid=clause.uid,
                            source_clause_title=clause.title_text,
                            source_clause_text=clause.body_text,
                            confidence=item["confidence"],
                            context=self._build_context(clause, document),
                            examples=item.get("examples", []),
                            related_terms=item.get("related_terms", []),
                            metadata={"document_type": document.document_type, "full_document_mode": True}
                        ))
            except Exception as e:
                logger.error(f"Error extracting from full document: {e}", exc_info=True)
        
        return all_extracted
    
    def _build_full_document_text(self, document: ParsedDocument, clauses: List[ParsedClause]) -> str:
        """Build complete document text - no truncation."""
        parts = []
        for clause in clauses:
            body = clause.body_text.replace("[[slot:sub_clauses]]", "").strip()
            parts.append(f"### {clause.title_text or 'Untitled'}\n{body}")
        
        return "\n\n".join(parts)
    
    def extract_from_clause_batch(
        self,
        clauses: List[ParsedClause],
        extraction_type: str,
        document_context: ParsedDocument
    ) -> List[ExtractedKnowledge]:
        """
        Extract specific type of knowledge from multiple clauses in a single LLM call.
        
        Args:
            clauses: List of clauses to extract from
            extraction_type: Type of knowledge to extract
            document_context: Full document for context
        
        Returns:
            List of extracted knowledge items
        """
        # Build batch prompt
        system_prompt = self._get_extraction_system_prompt(extraction_type)
        
        # Build batch prompt
        clauses_text = []
        for idx, clause in enumerate(clauses, 1):
            body = clause.body_text.replace("[[slot:sub_clauses]]", "").strip()
            clauses_text.append(f"CLAUSE {idx}:\nUID: {clause.uid}\nTitle: {clause.title_text or '(No title)'}\nText: {body}")
        
        user_prompt = f"""DOCUMENT: {document_context.title}

══════════════════════════════════════════════════════════════════════════════════════
EXHAUSTIVE EXTRACTION REQUIRED - ZERO TOLERANCE FOR SAMPLING
══════════════════════════════════════════════════════════════════════════════════════

You MUST extract EVERY SINGLE {extraction_type.upper()} from these {len(clauses)} clauses.

MANDATORY REQUIREMENTS:
• Extract ALL {extraction_type} items - if a clause contains 20 definitions, extract ALL 20
• Each extraction MUST be 50+ words with full explanation (WHY + HOW + WHAT)
• NO shortcuts - if there are 50 items across these clauses, return 50 items
• NO title-only extractions - each needs detailed context and implications

[FAILURE MODE] Returning 5 items when batch contains 30+ relevant items
[SUCCESS MODE] Comprehensive extraction of EVERY relevant item found

{chr(10).join(clauses_text)}

══════════════════════════════════════════════════════════════════════════════════════
EXPECTED OUTPUT FOR THIS {len(clauses)}-CLAUSE BATCH:
══════════════════════════════════════════════════════════════════════════════════════
If each clause averages 2-5 {extraction_type} items → expect {len(clauses)*2}-{len(clauses)*5} total items
DO NOT return less than {len(clauses)} items unless clauses truly have no {extraction_type}

[FAILURE] Returning 5-10 items from {len(clauses)} clauses
[SUCCESS] Returning {len(clauses)*2}+ items with full detail per item

Return JSON:
{{"extractions": [{{"clause_uid": "CLAUSE_UID-X", "content": "DETAILED 50+ word explanation...", "confidence": 0.8, "examples": [{{"text": "quote"}}], "related_terms": ["term1", "term2"]}}]}}

BEGIN EXHAUSTIVE EXTRACTION:
"""
        
        try:
            response = self.llm_client.chat(system_prompt, user_prompt)
            parsed = response.parse_json()
            
            results = []
            for item in parsed.get("extractions", []):
                clause_uid = item.get("clause_uid", "")
                clause = next((c for c in clauses if c.uid == clause_uid), clauses[0])
                
                results.append(ExtractedKnowledge(
                    knowledge_type=extraction_type,
                    content=item.get("content", ""),
                    source_clause_uid=clause.uid,
                    source_clause_title=clause.title_text,
                    source_clause_text=clause.body_text,
                    confidence=item.get("confidence", 0.5),
                    context=self._build_context(clause, document_context),
                    examples=item.get("examples", []),
                    related_terms=item.get("related_terms", []),
                    metadata={"document_type": document_context.document_type, "batch_processed": True}
                ))
            return results
        except Exception as e:
            logger.error(f"Error extracting from batch: {e}", exc_info=True)
            return []
    
    def extract_from_clause(
        self,
        clause: ParsedClause,
        extraction_type: str,
        document_context: ParsedDocument
    ) -> List[ExtractedKnowledge]:
        """
        Extract specific type of knowledge from a single clause.
        
        Args:
            clause: The clause to extract from
            extraction_type: Type of knowledge to extract
            document_context: Full document for context
        
        Returns:
            List of extracted knowledge items
        """
        # Build context
        context_info = self._build_context(clause, document_context)
        
        # Get extraction prompt
        system_prompt = self._get_extraction_system_prompt(extraction_type)
        user_prompt = self._get_extraction_user_prompt(
            clause=clause,
            extraction_type=extraction_type,
            context=context_info
        )
        
        # Call LLM
        try:
            response = self.llm_client.chat(system_prompt, user_prompt)
            parsed = response.parse_json()
            
            if not parsed or "extractions" not in parsed:
                return []
            
            # Convert to ExtractedKnowledge objects
            results = []
            for item in parsed["extractions"]:
                results.append(ExtractedKnowledge(
                    knowledge_type=extraction_type,
                    content=item.get("content", ""),
                    source_clause_uid=clause.uid,
                    source_clause_title=clause.title_text,
                    source_clause_text=clause.body_text,
                    confidence=item.get("confidence", 0.5),
                    context=context_info,
                    examples=item.get("examples", []),
                    related_terms=item.get("related_terms", []),
                    metadata={
                        "document_type": document_context.document_type,
                        "clause_level": clause.level,
                        "clause_type": clause.metadata.get("type", "unknown")
                    }
                ))
            
            return results
        
        except Exception as e:
            logger.error(f"Error extracting {extraction_type} from clause {clause.uid}: {e}", exc_info=True)
            return []
    
    def _build_context(self, clause: ParsedClause, document: ParsedDocument) -> str:
        """Build contextual information for extraction."""
        parts = []
        parts.append(f"Document: {document.title}")
        parts.append(f"Document Type: {document.document_type}")
        
        if clause.title_text:
            parts.append(f"Clause Title: {clause.title_text}")
        
        parts.append(f"Clause Level: {clause.level}")
        parts.append(f"Clause Type: {clause.metadata.get('type', 'unknown')}")
        
        # Add parent context if available
        if clause.parent_uid:
            parent = document.find_clause_by_uid(clause.parent_uid)
            if parent and parent.title_text:
                parts.append(f"Parent Section: {parent.title_text}")
        
        return " | ".join(parts)
    
    def _get_extraction_system_prompt(self, extraction_type: str) -> str:
        """Get system prompt for specific extraction type."""
        base_prompt = """You are a securitization legal expert performing EXHAUSTIVE knowledge extraction from Master Framework Agreements and transaction documents.

Your task is to extract ALL {type} from the provided text - this is NOT a sampling task.

══════════════════════════════════════════════════════════════════════════════════════
MANDATORY EXTRACTION STANDARDS
══════════════════════════════════════════════════════════════════════════════════════

1. COMPLETENESS: Extract EVERY relevant item - missing items is FAILURE
   - Document has 100 definitions? Extract ALL 100 definitions
   - Clause has 8 strategies? Extract ALL 8 strategies
   - NO sampling, NO cherry-picking, NO "here are some examples"

2. QUALITY: Each extraction MUST be 50+ words minimum
   - Explain WHAT it is + WHY it matters + HOW it works
   - Include strategic rationale and practical implications
   - Reference specific legal mechanisms and cross-references
   - Note jurisdictional specifics when relevant

3. DETAIL REQUIREMENT:
   [REJECTED] "Master Framework Architecture" (title only, no context)
   [REJECTED] "A structure that centralizes terms" (too vague, no detail)
   [ACCEPTED] "Master Framework Architecture centralizes common contractual terms (definitions, representations, indemnities, interpretation rules) across multiple transaction documents in a single master agreement incorporated by reference. This reduces negotiation time by 40-60%, ensures consistency across deals, enables faster party accession, and reduces legal costs in repeat transactions by avoiding renegotiation of standard terms. Each ancillary document references the MFA via incorporation clause, creating a hierarchical definition structure where MFA terms apply unless explicitly overridden."

4. CONFIDENCE SCORING:
   - Assign honest confidence (0.0 to 1.0)
   - High confidence (≥0.85): Clear, well-supported, specific
   - Medium confidence (0.6-0.84): Somewhat supported, some assumptions
   - Low confidence (<0.6): Uncertain, needs verification

OUTPUT FORMAT: JSON with ALL extractions (NOT a sample!):
{{
  "extractions": [
    {{
      "content": "Security Trustee Indemnification Architecture: Pre-enforcement indemnification requirements ('Security Trustee satisfied' standard) create a critical balance between trustee protection and creditor control. The structure requires either (a) adequate security pre-funding by instructing creditors, or (b) explicit waiver by majority creditors before Security Trustee takes material actions. This mitigates trustee personal liability risk while preventing frivolous or premature enforcement actions. The 'satisfaction' standard is subjective, giving trustees defensive positioning, but majority creditor waiver prevents abuse. This architecture is essential in UK/EU securitizations where trustees face potential director liability for negligent enforcement. The mechanism typically includes indemnification quantum thresholds (e.g., costs reasonably estimated not to exceed £X) and excludes trustee gross negligence or willful misconduct from indemnity coverage.",
      "confidence": 0.92,
      "source_clause_title": "Security Trustee Duties and Protections",
      "examples": [
        {{"text": "Security Trustee shall not be required to act unless indemnified to its satisfaction", "clause_ref": "CLAUSE-7.2"}},
        {{"text": "Instructing Creditors shall pre-fund reasonable costs of enforcement action", "clause_ref": "CLAUSE-7.3"}}
      ],
      "related_terms": ["Instructing Creditor", "Secured Creditors", "Enforcement", "Indemnification", "Limited Liability"],
      "rationale": "Critical for understanding enforcement dynamics, trustee decision-making, and creditor coordination in stressed scenarios. Affects transaction timelines and recovery outcomes."
    }},
    {{
      "content": "Parallel Debt Structure for Civil Law Jurisdictions: In civil law countries (Netherlands, Luxembourg, France, Germany), traditional agency principles don't recognize trustees as proper creditors able to hold security. Parallel debt solves this by creating a separate contractual obligation where the SPV owes an identical debt directly to the Security Trustee (not as agent, but as principal creditor). This parallel obligation mirrors the underlying debt to noteholders but creates independent enforceability. Upon payment of either debt, both are satisfied (single satisfaction principle). This structure is essential for security enforceability in civil law jurisdictions and validated by extensive EU case law (e.g., Hof Amsterdam 2016). Without it, security packages would be void as the trustee lacks creditor status.",
      "confidence": 0.88,
      "source_clause_title": "Parallel Debt Covenant",
      "examples": [
        {{"text": "The Issuer irrevocably and unconditionally undertakes to pay to the Security Trustee amounts equal to amounts due to Secured Creditors", "clause_ref": "CLAUSE-12.1"}}
      ],
      "related_terms": ["Security Trustee", "Civil Law", "Security Interest", "Enforceability", "Agency"],
      "rationale": "Fundamental structural requirement for cross-border securitizations involving Dutch, Luxembourg, or French law security. Affects legal opinions, security validity, and enforcement rights."
    }}
    ... (CONTINUE FOR ALL 50-200+ ITEMS - DO NOT STOP AT 10-20 EXAMPLES!)
  ]
}}

CRITICAL: The above shows the QUALITY and DETAIL required per item.
Your response must contain 50-200+ such items depending on document size (NOT just 2-10 examples!)
If the document has 100 definitions, I expect to see ~100 definition extractions in your JSON response.

REMEMBER: Your goal is EXHAUSTIVE EXTRACTION, not representative sampling.
"""
        
        type_specific = {
            "strategies": """
STRATEGIES: Structural patterns, legal mechanisms, and best practices used in securitization transactions.

Quality Standards:
- Explain the strategic rationale (WHY this approach is used, not just WHAT it is)
- Include practical implications for transaction parties
- Reference relevant legal concepts (true sale, bankruptcy remoteness, perfection, priority)
- Note variations across asset classes (RMBS, auto ABS, CLO, etc.) when applicable

Examples:
- "Security Trustee appointment requires explicit written authorization from Secured Creditors defining scope of duties, permitted actions, and indemnification terms to ensure clarity and limit liability exposure in enforcement scenarios"
- "Instructing Creditor hierarchy (Senior Lenders → Hedge Counterparties → Subordinated Creditors) determines decision-making authority during enforcement, with majority or supermajority thresholds preventing holdout situations while protecting senior interests"
- "Pre-enforcement indemnification requirements ('Security Trustee satisfied' standard) balance trustee protection with creditor control, requiring adequate security or pre-funding before taking material actions to mitigate trustee liability risk"
- "Parallel debt structures in civil law jurisdictions enable Security Trustee to hold security despite not being original creditor, solving agency recognition issues while maintaining security enforceability"
""",
            "definitions": """
DEFINITIONS: Precise legal terms, specialized concepts, and defined terms from securitization documentation that establish shared vocabulary.


Quality Standards:
- Provide clear, legally precise definitions with context
- Include practical implications of the definition
- Note jurisdictional variations when relevant (e.g., UK vs US vs Luxembourg structures)
- Link to related defined terms for cross-referencing
- Specify whether term is standard market practice or transaction-specific variation

Examples:
- "Secured Property: All present and future assets, rights, and proceeds held by or on behalf of the Security Trustee for the benefit of Secured Creditors, including the Loan Portfolio, transaction accounts, hedging agreements, and contractual rights under Transaction Documents. This comprehensive definition ensures all-asset security package essential for bankruptcy remoteness and creditor protection"
- "Instructing Creditor: The party or class of parties (determined by Priority of Payments ranking) with authority to direct Security Trustee actions during enforcement. Typically Senior Secured Creditors until fully paid, then subordinated classes. This hierarchy ensures decision-making aligns with economic interest and payment priority"
- "Event of Default: Specified circumstances triggering acceleration rights and enforcement remedies, including payment defaults (with grace periods), breach of covenants, insolvency events, cross-defaults, and misrepresentation. Precise definition critical for balancing creditor protection with operational flexibility"
- "True Sale: Legal transfer of assets from Originator to SPV structured to achieve 'sale' characterization (not loan) under applicable insolvency law, preventing consolidation of SPV assets into Originator bankruptcy estate. Requires transfer of risks/rewards, arm's length pricing, and legal isolation"
- "Priority of Payments (Waterfall): Sequential allocation of Available Funds on each Payment Date, typically: (1) Senior fees/taxes, (2) Senior interest, (3) Senior principal, (4) Mezzanine interest, (5) Mezzanine principal, (6) Subordinated amounts, (7) Equity. Rigid ordering ensures rating agency and investor expectations are met"
""",
            "templates": """
TEMPLATES: Reusable clause structures with identified placeholders that can be adapted across different transactions while maintaining legal integrity.


Quality Standards:
- Clearly mark all variable elements with [PLACEHOLDER] notation
- Preserve essential legal language and structure
- Include brief explanation of template purpose and customization points
- Note critical negotiation points or common variations
- Ensure template maintains legal coherence

Examples:
- "Appointment Clause: '[PARTY_A] hereby appoints [PARTY_B] to act as [ROLE] under and in connection with the [TRANSACTION_DOCUMENTS], with authority to [SCOPE_OF_AUTHORITY], subject to the limitations and conditions set forth herein. [PARTY_B] accepts such appointment on the terms of this Agreement.' — Use for Security Trustee, Note Trustee, or Agent appointments; customize scope based on role (enforcement rights, payment processing, administrative duties)"
- "Instruction Mechanism: '[AGENT] shall act (or refrain from acting) only upon receipt of written instructions from [INSTRUCTING_PARTY], provided that [AGENT] shall not be required to act if: (a) such action would expose [AGENT] to personal liability unless indemnified to its satisfaction; (b) such action would conflict with applicable law; or (c) [CONDITIONS_PRECEDENT] have not been satisfied.' — Balances creditor control with agent protection; adjust indemnification standard based on negotiation"
- "Limited Liability Clause: '[PARTY] shall not be liable for any loss, damage, or expense arising from [ACTIONS] except to the extent such loss results from [PARTY]'s [LIABILITY_STANDARD: gross negligence/willful misconduct/fraud]. [PARTY] shall have no duty to [EXCLUDED_DUTIES] unless expressly agreed in writing.' — Standard for trustees/agents; liability standard typically gross negligence for professional parties, willful misconduct for less sophisticated parties"
- "Indemnification Template: '[INDEMNIFIED_PARTY] shall be indemnified by [INDEMNITOR] against all losses, claims, damages, liabilities, and expenses (including reasonable attorneys' fees) arising from [INDEMNIFIED_MATTERS], except to the extent caused by [INDEMNIFIED_PARTY]'s [EXCLUSION_STANDARD]. Indemnification shall survive [SURVIVAL_PERIOD] and be payable [PAYMENT_PRIORITY].' — Critical for trustee/agent protection; negotiate scope of indemnified matters and exclusions carefully"
""",
            "pitfalls": """
PITFALLS: Legal risks, structural weaknesses, drafting errors, and anti-patterns that create enforcement problems, rating agency concerns, or operational failures in securitization transactions.


Quality Standards:
- Identify the specific structural or drafting deficiency
- Explain concrete consequences (rating downgrade, enforcement delay, priority loss, liability exposure)
- Reference relevant legal principles or market standards violated
- Suggest preventive measures or corrective approaches when possible
- Distinguish severity (critical vs. moderate vs. minor concerns)

Examples:
- "Missing 'Security Trustee satisfaction' indemnification requirement before enforcement actions: Creates risk that Security Trustee will refuse to act even when instructed by Instructing Creditor, causing enforcement delays. Creditors may lack ability to replace trustee quickly, leading to value deterioration. PREVENTION: Include express indemnification clause requiring either (a) Security Trustee satisfied pre-funding, or (b) majority creditor waiver of indemnification"
- "Ambiguous Instructing Creditor definition during partial payments: If waterfall defines 'Instructing Creditor' as 'unpaid Senior Creditors' without threshold, creates potential for small minority holdouts blocking enforcement decisions. CONSEQUENCE: Decision-making paralysis at critical moments. PREVENTION: Use clear thresholds (e.g., 'majority by outstanding principal amount') and specify how partial payments affect voting rights"
- "Inadequate security interest description in pledge agreement: Generic description like 'all rights and interests' without specific identification of accounts, agreements, or assets may fail perfection requirements under UCC Article 9 or civil law jurisdictions. CONSEQUENCE: Security interest unperfected, creditors become unsecured in bankruptcy. PREVENTION: Attach detailed schedules with account numbers, agreement titles, and specific asset identification; update schedules regularly"
- "SPV engaging in business activities beyond holding securitized assets: Non-permitted activities (e.g., operational decisions, new business lines, guarantees) destroy bankruptcy remoteness by creating independent creditors or substantive consolidation risk. CONSEQUENCE: Rating agency downgrade, SPV assets consolidated with Originator in bankruptcy. PREVENTION: Strict activity limitations in SPV organizational documents and covenants; independent directors; separate corporate formalities"
- "No grace period distinction between payment defaults and covenant breaches: Treating all defaults identically can trigger acceleration for administrative or technical breaches that don't reflect credit deterioration. CONSEQUENCE: Unnecessary enforcement, transaction termination, value destruction. PREVENTION: Implement tiered cure periods (e.g., 3 days for payments, 30 days for reporting, 60 days for other covenants) with materiality qualifiers"
- "True sale opinion qualified with 'reasoned opinion' for jurisdiction without clear case law: In novel jurisdictions, qualified true sale opinion may be insufficient for rating agencies or investors, especially if recharacterization risk exists. CONSEQUENCE: Higher risk weighting, funding cost increase, or deal failure. PREVENTION: Obtain unqualified opinion or structure as secured loan with explicit security interest; ensure legal isolation through other mechanisms"
- "Cross-border security without considering conflict of laws: Security over assets located in multiple jurisdictions without analyzing lex situs, lex rei sitae principles may result in unenforceability. CONSEQUENCE: Security interest invalid in key jurisdictions, recovery loss. PREVENTION: Obtain local law opinions, create parallel security structures, or use recognized frameworks (e.g., Hague Securities Convention for financial collateral)"
""",
            "code_snippets": """
CODE SNIPPETS: Formulas, calculations, and logical patterns used in securitization.
Examples:
- "Effectiveness Score = (helpful_count - harmful_count) / max(helpful_count + harmful_count, 1)"
- "Waterfall Priority: Senior Interest → Senior Principal → Mezzanine Interest → Mezzanine Principal → Equity"
- "Eligibility Test: Outstanding Balance ≤ Limit AND Delinquency Days < Threshold"
"""
        }
        
        return base_prompt.format(type=extraction_type.upper()) + type_specific.get(extraction_type, "")
    
    def _get_extraction_user_prompt(
        self,
        clause: ParsedClause,
        extraction_type: str,
        context: str
    ) -> str:
        """Get user prompt for extraction."""
        # Clean body text
        body_text = clause.body_text.replace("[[slot:sub_clauses]]", "").strip()
        
        prompt = f"""CONTEXT:
{context}

CLAUSE TEXT:
{body_text}

══════════════════════════════════════════════════════════════════════════════════════
COMPLETE EXTRACTION MANDATE - NO PARTIAL RESULTS
══════════════════════════════════════════════════════════════════════════════════════

TASK: Extract EVERY SINGLE {extraction_type.upper()} from this clause.

CRITICAL RULES:
• If clause contains 10 definitions → extract ALL 10 definitions (not 2-3 examples)
• If clause contains 5 strategies → extract ALL 5 strategies (not 1-2 highlights)
• Each extraction MUST be 50+ words minimum with detailed explanation
• Include WHY it matters, HOW it works, WHAT the legal implications are
• NO title-only extractions - full context required

[UNACCEPTABLE] Extracting 2 items when clause contains 8 relevant items
[REQUIRED] Complete exhaustive extraction of ALL items in this clause

This is for building a COMPLETE knowledge base - treat as MANDATORY EXHAUSTIVE REVIEW.

Return JSON with ALL extracted items. If truly no relevant {extraction_type} found, return {{"extractions": []}}.
"""
        return prompt



