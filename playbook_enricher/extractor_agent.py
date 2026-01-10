"""
Extractor Agent - Specialized extraction with 4 section-specific prompts.

Extracts knowledge from documents using specialized prompts for:
- strategies: Abstract into reusable principles, NEVER copy-paste contract text
- definitions: Extract exactly, include pointer definitions
- pitfalls: Abstract into reusable warnings

Supports two granularity modes:
- OPERATIVE_CLAUSE_BY_CLAUSE: Process each clause individually
- FULL_DOCUMENT: Process entire document
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
    section: str  # strategies, pitfalls, or definitions
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
STRATEGIES_EXTRACTION_PROMPT = """You are extracting STRATEGIES that enable an LLM Generator to perform complex securitization tasks correctly.

TARGET TASKS THE GENERATOR MUST PERFORM:
1. **Clause Interpretation**: Explain dense legal drafting with appropriate nuance to junior lawyers
2. **Precision Redrafting**: Modify clauses (e.g., add tranches) while preserving structure and style
3. **Jurisdictional Adaptation**: Adapt clauses to new legal regimes (e.g., UK → Ireland)
4. **Document Structuring**: Understand and articulate document logic and section interactions
5. **Definition Merging**: Create new definitions that generalize existing patterns
6. **Taxonomy Classification**: Determine where clauses belong in Canon hierarchy
7. **Applicability Analysis**: Determine if clauses should be included in specific deals
8. **Styling**: Render clauses in client-preferred formats
9. **Evaluation**: Detect if clauses have changed significantly from reference versions
10. **Reformulation**: Realign clauses to references under constraints

EXTRACTION CRITERIA - Extract ONLY if ALL of these are true:
1. ❌ **Necessity Test**: The Generator would respond INCORRECTLY without this knowledge
2. ❌ **Non-Obviousness Test**: This is NOT general legal advice or common knowledge
3. ❌ **Tranchify-Specificity Test**: This captures Tranchify/securitization-specific patterns
4. ❌ **Actionability Test**: This provides PRECISE guidance, not vague advice

WHAT TO EXTRACT (with examples):

✅ **Transaction Structure Patterns**
Example: "In Tranchify SPV structures, Priority of Payments waterfalls always distinguish between pre-enforcement and post-enforcement scenarios, with post-enforcement provisions typically including Security Trustee enforcement costs as the first item."

✅ **Jurisdictional-Specific Rules**
Example: "For UK SPV tax exemptions, quoted Eurobond status under s882 ITA 2007 requires listing on a recognized stock exchange; Irish equivalents use s246(3)(h) TCA 1997 and require different documentation."

✅ **Definitional Dependencies and Hierarchies**
Example: "Borrowing Base definitions must always reference 'Net Eligible Receivables Balance' (after reserves/concentrations) not gross balances; Senior Borrowing Base uses Senior Advance Rate; Mezzanine Borrowing Base uses Mezzanine Advance Rate minus Senior Advance."

✅ **Redrafting Constraints**
Example: "When adding new tranches to Priority of Payments, preserve existing waterfall numbering by inserting sub-items (e.g., '(aa)', '(bb)'); use 'ranking pari passu with' for equal priority items; never change numbering of existing senior items."

✅ **Clause Placement Rules (Taxonomy)**
Example: "Representations and Warranties clauses belong in Canon Section 5 (Conditions Precedent and Representations); tax gross-up provisions belong in Section 10 (Tax); payment mechanics belong in Section 7 (Payment Mechanics)."

✅ **Templating Patterns**
Example: "Standard facility commitment language follows: 'Each {{LENDER_TYPE}} hereby grants to the {{BORROWER}}, {{FACILITY_STRUCTURE}} in an amount equal to {{COMMITMENT_DEFINITION}}.' Use [IF is_revolving=TRUE]...[ELSE]...[ENDIF] for revolving vs term variants."

✅ **Applicability Logic**
Example: "ABS Transaction Fee clauses should only be included when the structure anticipates future securitization exit; exclude from warehouse facilities without ABS optionality."

✅ **Domain-Specific Interpretation Rules**
Example: "Collections definitions must include 'Deemed Collections' to capture non-cash value reduction events (e.g., product returns, set-offs); omitting this causes the SPV to lose economic benefit when receivables are reduced without cash payment."

✅ **Cross-Document Reference Patterns**
Example: "Master Framework Agreements incorporate definitions into Facility Agreements via: 'Capitalized terms used but not defined herein have the meanings given in the Master Framework Agreement dated [DATE]'; this must appear in the interpretation section (typically Clause 1)."

WHAT TO SKIP (do NOT extract):

❌ **Generic Legal Advice**
Example: "Review clauses carefully" → TOO VAGUE
Example: "Parties should negotiate terms" → OBVIOUS

❌ **Information LLM Already Knows**
Example: "Securitization involves SPVs" → COMMON KNOWLEDGE
Example: "Agreements require signatures" → BASIC LAW

❌ **Structural Observations Without Utility**
Example: "This document has 20 clauses" → NOT ACTIONABLE
Example: "Definitions appear in Clause 2" → DESCRIPTIVE ONLY

❌ **Transaction-Specific Details**
Example: "XYZ Bank provided £50M on 15 Jan 2023" → DEAL-SPECIFIC
Example: "The Seller is ABC Ltd" → NOT REUSABLE

QUALITY GATE - Before extracting, answer these questions:
1. "If I remove this from the playbook, which of the 11 target tasks will the Generator fail?"
2. "Can the Generator produce this knowledge from its training data alone?"
3. "Is this specific enough to be actionable, or is it vague advice?"

If answers are "NONE", "YES", "VAGUE" → SKIP

FORMATTING RULES:
- Write in clean, complete sentences
- Use specific terminology (not "the parties" but "the Issuer/Servicer/Lender")
- Include concrete examples where helpful (e.g., "s882 ITA 2007", not "relevant tax law")

OUTPUT FORMAT:
{
  "extracted_items": [
    {
      "section": "strategies",
      "content": "Precise, actionable strategy that prevents Generator errors on target tasks",
      "reasoning": "Which target task(s) this enables (use task numbers 1-11)",
      "criticality": "HIGH (critical for task success) / MEDIUM (improves accuracy) / LOW (minor edge case)"
    }
  ]
}

CRITICAL SELECTION CRITERIA:
- Extract only important strategies per document (be HIGHLY selective)
- Prioritize HIGH criticality strategies
- Focus on strategies that enable tasks 1-5 (interpretation, redrafting, adaptation, structuring, merging)
- Only include tasks strategies if they are HIGH criticality

PLAYBOOK AWARENESS:
A summary of the current playbook is provided at the start of the user message under "EXISTING PLAYBOOK COVERAGE".
Before extracting ANY strategy, check the summary:
- If the summary mentions a similar pattern/theme → DON'T extract (already covered)
- If the summary shows strategies about the same topic → DON'T extract (redundant)
- Only extract if the strategy is genuinely NEW and not mentioned in the summary

REMEMBER: The goal is NOT comprehensive extraction. The goal is SELECTIVE extraction of ONLY the knowledge that the Generator CANNOT produce alone and that DIRECTLY enables the 11 target tasks.
"""

DEFINITIONS_EXTRACTION_PROMPT = """You are extracting DEFINITIONS from securitization documents.

CRITICAL STOPPING RULES - DO NOT EXTRACT IF:

1. ❌ **Gemini 3 Pro Already Knows This**
   Test: Ask yourself "Would Gemini 3 Pro define this term correctly in securitization context WITHOUT seeing this document?"
   - If YES → SKIP
   - If NO → Consider extracting
   
   Examples to SKIP:
   - "Securitization: means a structured finance transaction..." → Gemini 3 Pro knows this
   - "SPV: means special purpose vehicle..." → Gemini 3 Pro knows this
   - "Senior Lender: means the entity providing senior debt..." → Obvious from context
   - "Issuer: means the entity issuing securities..." → Generic role

2. ❌ **Playbook Already Contains Similar Strategy/Pitfall**
   Before extracting a definition, check if existing playbook strategies or pitfalls already explain this concept.
   
   Example: If playbook strategy says "Collections must include Deemed Collections to capture non-cash value reductions (set-offs, returns, credit notes)", then you DON'T need to extract the full Collections definition—the strategy already teaches the Generator what matters.
   
   Example: If playbook pitfall says "Use Net Eligible Receivables Balance, not gross balances, in Borrowing Base calculations", then you DON'T need the full Borrowing Base definition—the pitfall already covers the critical knowledge.

3. ❌ **Definition is Generic/Administrative**
   - "Agreement means this document"
   - "Clause means a numbered section"
   - "Party means each party hereto"
   - "Date means any calendar date"
   → These add no value

4. ❌ **Gemini 3 Pro's Generic Definition is Functionally Equivalent**
   Compare what Gemini 3 Pro would generate vs. what the document says:
   
   Document: "Payment Date: means the 15th day of each month"
   Gemini 3 Pro: "Payment Date typically means the scheduled date for payments in structured finance"
   → SIMILAR ENOUGH → SKIP
   
   Document: "Cut-Off Date: means the date as of which Receivables are selected for purchase, being the last day of each calendar month"
   Gemini 3 Pro: "Cut-Off Date is when assets are measured for securitization purposes"
   → SIMILAR ENOUGH → SKIP
   
   Document: "Collections: means (a) all cash collections and proceeds of the Receivable including finance charges, VAT, and Related Rights proceeds; (b) insurance proceeds; (c) any Deemed Collection; (d) sale proceeds; (e) Repurchase Price"
   Gemini 3 Pro: "Collections means cash received from receivables"
   → NOT SIMILAR → Gemini 3 Pro would miss (c) Deemed Collection and other components → EXTRACT

WHAT TO EXTRACT (rare cases where definition adds critical value):

✅ **Terms with Non-Obvious Components**
   - "Collections: means..." → ONLY if it includes non-obvious items like "Deemed Collection"
   - "Borrowing Base: means..." → ONLY if formula is complex or has specific conditions
   - Reason: Gemini 3 Pro would define these terms generically and miss critical legal components

✅ **Terms with Precise Calculation Logic**
   - "Senior Advance Rate: means (a) 85% prior to a Rate Reduction Event; or (b) 75% following a Rate Reduction Event"
   - "Concentration Limit: means no single Obligor shall represent more than 2.5% of Outstanding Balance"
   - Reason: Gemini 3 Pro cannot invent specific percentages or thresholds

✅ **Tranchify-Specific Constructs**
   - "ABS Transaction: means any securitisation of Receivables originated by the Seller which involves marketing and distribution of securities in public markets"
   - Reason: This is company-specific language Gemini 3 Pro wouldn't know

✅ **Terms Where Playbook Strategies/Pitfalls Don't Already Cover**
   - If no strategy/pitfall explains this concept → Consider extracting
   - If strategy/pitfall already teaches what matters → SKIP the full definition

✅ **Jurisdiction-Specific Technical Terms**
   - "Quoted Eurobond: means a security listed on a recognised stock exchange within meaning of s882 Income Tax Act 2007 carrying right to interest"
   - Reason: Specific statutory reference Gemini 3 Pro wouldn't know

✅ **Pointer Definitions** (minimal extraction, will be enriched later)
   - "ABS Transaction Fee has the meaning in clause 8.2(b)"
   - Reason: Marks term for enrichment without duplicating content

COMPARISON METHODOLOGY:

For each definition candidate, perform this test:

STEP 1: Generate what Gemini 3 Pro would say
"What is [TERM] in securitization context?"

STEP 2: Compare with document definition
- Are the key legal components the same?
- Would Gemini 3 Pro's version enable correct Generator output?

STEP 3: Decision
- If Gemini 3 Pro's version is 80 percent+ functionally equivalent → SKIP
- If Gemini 3 Pro would miss critical components → EXTRACT

Example 1:
Term: "Servicer"
Document: "Servicer means the entity responsible for administering and collecting Receivables"
Gemini 3 Pro would say: "Servicer is the entity that services the receivables pool"
→ FUNCTIONALLY EQUIVALENT → SKIP

Example 2:
Term: "Deemed Collection"
Document: "Deemed Collection means any reduction in Outstanding Principal Balance other than by Collection, including set-off, product return, or non-cash adjustment"
Gemini 3 Pro would say: "Deemed Collection is a non-cash reduction in receivables" 
→ NOT EQUIVALENT (Gemini 3 Pro misses set-off, product return specifics) → EXTRACT

Example 3:
Term: "Collections"
Document: "Collections means (a) cash collections...; (b) insurance proceeds; (c) Deemed Collections; (d)..."
Playbook Strategy already says: "Collections must include Deemed Collections to capture non-cash value reductions"
→ STRATEGY COVERS THE CRITICAL KNOWLEDGE → SKIP FULL DEFINITION

COMPLETENESS REQUIREMENT (for terms you DO extract):

**NEVER use ellipsis "..."**
- Extract COMPLETE definition with ALL sub-parts (a), (b), (c)...
- Extract ALL sentences and conditions
- Extract ALL cross-references

FORMATTING:
- Format: "TERM: means [complete definition]" OR "TERM has the meaning in clause X"
- Preserve: ALL substantive content

OUTPUT FORMAT:
{
  "extracted_items": [
    {
      "section": "definitions",
      "content": "COMPLETE definition",
      "reasoning": "Why Gemini 3 Pro would get this wrong OR why playbook doesn't already cover this",
      "gemini_comparison": "What Gemini 3 Pro would say vs. what document says - explain the gap"
    }
  ]
}

GOLDEN RULE:
**When in doubt, DON'T extract.**
Only extract definitions where:
1. Gemini 3 Pro would define incorrectly/incompletely, AND
2. Playbook strategies/pitfalls don't already cover the concept, AND
3. The definition contains non-obvious legal components

PLAYBOOK AWARENESS:
A summary of the current playbook is provided at the start of the user message under "EXISTING PLAYBOOK COVERAGE".
Before extracting ANY definition, check the summary:
- If the term is listed in "DEFINITIONS COVERAGE" → DON'T extract (already defined)
- If the concept is explained in "STRATEGIES COVERAGE" → DON'T extract (strategy covers it)
- Only extract if the term is genuinely NEW and not mentioned in the summary

Most definitions should be SKIPPED. Extract only the truly necessary ones.
"""

PITFALLS_EXTRACTION_PROMPT = """You are extracting PITFALLS (common mistakes, errors to avoid) from securitization documents.

CRITICAL STOPPING RULES - DO NOT EXTRACT IF:

1. ❌ **Gemini 3 Pro Would Naturally Avoid This**
   Test: Ask yourself "Would Gemini 3 Pro make this mistake WITHOUT seeing this document?"
   - If NO (Gemini 3 Pro would naturally avoid this) → SKIP
   - If YES (Gemini 3 Pro would make this error) → Consider extracting
   
   Examples to SKIP:
   - "Ensure parties sign the agreement" → Gemini 3 Pro knows this
   - "Review clauses carefully" → Too vague, not actionable
   - "Don't delete important provisions" → Obvious
   - "Consider tax implications" → Generic advice without specifics

2. ❌ **Playbook Strategies Already Cover This**
   Before extracting a pitfall, check if existing playbook strategies already teach the correct approach.
   
   Example: If playbook strategy says "Use master framework agreements to incorporate definitions by reference, ensuring consistent interpretation across transaction documents", then you DON'T need a pitfall saying "Don't fail to incorporate definitions by reference"—the strategy already teaches the right way.
   
   Example: If playbook strategy says "In Priority of Payments, distinguish pre-enforcement vs post-enforcement scenarios", then you DON'T need a pitfall about missing this distinction—the strategy covers it.

3. ❌ **The Error is Too Generic/Vague**
   - "Be careful with definitions" → WHAT SPECIFICALLY?
   - "Ensure compliance with regulations" → WHICH REGULATIONS? HOW?
   - "Review cross-references" → WHAT ERROR ARE WE PREVENTING?
   → These add no actionable value

4. ❌ **Gemini 3 Pro Would Not Make This Mistake in Practice**
   Some errors are theoretically possible but unlikely:
   
   Example: "Don't define Servicer as the entity providing loans" 
   → Gemini 3 Pro would not confuse Servicer with Lender → SKIP
   
   Example: "Don't use the wrong currency in payment clauses"
   → Gemini 3 Pro would check context → SKIP
   
   Example: "Don't omit Deemed Collections from Collections definition"
   → Gemini 3 Pro WOULD make this mistake (non-obvious component) → EXTRACT

WHAT TO EXTRACT (rare cases where pitfall prevents real errors):

✅ **Critical Drafting Omissions**
   - Mistakes where Gemini 3 Pro would miss non-obvious components
   
   Example: "Deemed Collections Omission: Failing to include 'Deemed Collections' in the Collections definition prevents the Issuer from capturing value when receivables are reduced through non-cash means (product returns, set-offs, credit notes). Without this, the SPV bears economic losses without corresponding adjustments. Always ensure Collections includes: (a) cash collections and proceeds including finance charges and VAT; (b) insurance proceeds; (c) Deemed Collections; (d) sale proceeds; (e) Repurchase Price."
   
   Why Extract: Gemini 3 Pro would define "Collections" as "cash received from receivables" and miss Deemed Collections—this is a non-obvious legal component.

✅ **Structural Errors That Break Cross-References**
   - Mistakes Gemini 3 Pro would make when modifying document structure
   
   Example: "Waterfall Renumbering: When adding new tranches to Priority of Payments, do NOT renumber existing items. Renumbering (e.g., changing item (b) to (c)) breaks all cross-references throughout transaction documents (e.g., 'amounts payable under item (b) of the waterfall'). Instead, insert new items as sub-letters between existing items (e.g., insert '(aa)' between (a) and (b)). Use 'ranking pari passu with [item]' for equal-ranking tranches."
   
   Why Extract: Gemini 3 Pro might logically renumber for cleanliness, not realizing it breaks cross-references.

✅ **Calculation Errors with Non-Obvious Logic**
   - Formula mistakes Gemini 3 Pro would make without seeing the pattern
   
   Example: "Gross vs Net Receivables Error: When defining Borrowing Base, always use 'Net Eligible Receivables Balance' (after concentration limits, dilution reserves, eligibility exclusions), NOT 'Outstanding Principal Balance' (gross). Using gross balances causes over-advancement and covenant breaches. Correct formula: Borrowing Base = Advance Rate × Net Eligible Receivables Balance. Net balance accounts for: (a) concentration limits (e.g., no single obligor >2.5%); (b) dilution reserves (typically 1-3%); (c) ineligible receivables (defaulted, disputed, >90 days past due)."
   
   Why Extract: Gemini 3 Pro might use gross balance (simpler), not knowing net balance is industry standard.

✅ **Jurisdictional Adaptation Errors**
   - Mistakes when adapting clauses across legal systems
   
   Example: "UK-to-Ireland Tax Exemption Error: When adapting UK withholding tax clauses to Ireland, do NOT assume equivalent exemptions. UK quoted Eurobond exemption (section 882 Income Tax Act 2007) requires HMRC recognition and listing on recognized stock exchange. Irish equivalent (section 246(3)(h) Taxes Consolidation Act 1997) has different requirements: (a) listing on recognized stock exchange in EEA/treaty country; (b) payment to non-Irish residents only; (c) different documentation (no clearing system requirement). Always verify jurisdiction-specific exemption criteria and adjust gross-up language accordingly."
   
   Why Extract: Gemini 3 Pro might copy UK language assuming equivalence, missing jurisdictional differences.

✅ **Cross-Reference Integrity Issues**
   - Vague references that create ambiguity
   
   Example: "Master Framework Agreement Version Ambiguity: When referencing the Master Framework Agreement in Facility Agreements, specify the EXACT execution date. Vague references like 'as defined in the Master Framework Agreement' create ambiguity when the MFA is amended. Use precise language: 'as defined in the Master Framework Agreement dated [DATE] (as amended from time to time in accordance with its terms)' or 'as defined in the Master Framework Agreement dated [DATE] as amended by Amendment No. [X] dated [DATE]'."
   
   Why Extract: Gemini 3 Pro might use simpler reference not realizing version control issues.

✅ **Transaction Structure Misapplications**
   - Including clauses that don't fit the deal structure
   
   Example: "ABS Transaction Fee in Warehouse Facilities: Do NOT include 'ABS Transaction Fee' provisions in pure warehouse facilities without ABS exit optionality. This fee compensates lenders for early termination in anticipation of securitization and is embedded in pricing. It is inapplicable when: (a) facility agreement contains no ABS exit mechanism; (b) warehouse is intended as permanent financing; (c) pricing does not reflect ABS optionality. Only include when transaction documents explicitly contemplate future securitization exit with public securities issuance."
   
   Why Extract: Gemini 3 Pro might include this fee generically, not understanding when it applies.

WHAT TO SKIP:

❌ **Obvious Errors Gemini 3 Pro Won't Make**
   - "Don't confuse Senior Lender with Subordinated Lender" → Obvious
   - "Ensure payment dates are specified" → Basic
   - "Don't omit party signatures" → Common sense

❌ **Vague Warnings Without Specifics**
   - "Be careful when drafting definitions" → WHAT ERROR?
   - "Consider regulatory requirements" → WHICH ONES? HOW?
   - "Review tax implications carefully" → NOT ACTIONABLE

❌ **Errors Already Prevented by Strategies**
   - If strategy says "Incorporate definitions by reference using MFA"
   - Then don't add pitfall "Don't fail to incorporate definitions"
   - The strategy already teaches the correct approach

COMPARISON METHODOLOGY:

For each pitfall candidate, perform this test:

STEP 1: Would Gemini 3 Pro make this mistake?
- Consider Gemini 3 Pro's legal knowledge and reasoning ability
- Would it naturally avoid this error?

STEP 2: Do playbook strategies already prevent this?
- Check if existing strategies teach the correct approach
- If strategy covers it, pitfall is redundant

STEP 3: Is this specific and actionable?
- Does it describe a CONCRETE error with clear consequences?
- Does it provide a SPECIFIC correct approach?

STEP 4: Decision
- If Gemini 3 Pro won't make this error → SKIP
- If strategies already prevent this → SKIP
- If vague/generic → SKIP
- If specific, actionable, and fills a gap → EXTRACT

Example 1:
Pitfall Candidate: "Ensure parties are properly identified"
Gemini 3 Pro: Would naturally do this
→ TOO OBVIOUS → SKIP

Example 2:
Pitfall Candidate: "Don't renumber Priority of Payments items when adding tranches"
Gemini 3 Pro: Might renumber for cleanliness, not realizing cross-reference breaks
Playbook: No strategy covers this
→ GEMINI 3 PRO WOULD MAKE THIS ERROR + NOT COVERED → EXTRACT

Example 3:
Pitfall Candidate: "Don't fail to use master framework agreements"
Playbook Strategy: "Use master framework agreements to incorporate definitions by reference"
→ STRATEGY ALREADY COVERS → SKIP

FORMATTING:
- Structure: **[ERROR NAME]: [What NOT to do] + [Why it's wrong/consequences] + [What TO do instead with specifics]**
- Be SPECIFIC: Include concrete examples, clause numbers, formulas, percentages ...
- Use imperative: "Do not...", "Always...", "Never...", "Ensure..."

OUTPUT FORMAT:
{
  "extracted_items": [
    {
      "section": "pitfalls",
      "content": "[ERROR NAME]: Specific mistake + consequences + correct approach with details",
      "reasoning": "Why Gemini 3 Pro would make this mistake AND why playbook strategies don't already prevent it"
    }
  ]
}

GOLDEN RULE:
**When in doubt, DON'T extract.**
Only extract pitfalls where:
1. Gemini 3 Pro would actually make this mistake, AND
2. Playbook strategies don't already prevent it, AND
3. The pitfall describes a specific, concrete error with actionable solution

PLAYBOOK AWARENESS:
A summary of the current playbook is provided at the start of the user message under "EXISTING PLAYBOOK COVERAGE".
Before extracting ANY pitfall, check the summary:
- If a similar warning is in "PITFALLS COVERAGE" → DON'T extract (already warned)
- If a strategy in "STRATEGIES COVERAGE" teaches the correct approach → DON'T extract (strategy prevents the error)
- Only extract if the pitfall is genuinely NEW and not covered in the summary

Most pitfall candidates should be SKIPPED. Extract only the truly necessary ones that fill genuine gaps.
"""



class ExtractorAgent:
    """
    Extractor Agent with 4 specialized system prompts.
    
    Supports two granularity modes:
    - clause-by-clause: Process each clause individually
    - full-document: Process entire document
    """
    
    # Map section to specialized prompt
    SECTION_PROMPTS = {
        "strategies": STRATEGIES_EXTRACTION_PROMPT,
        "definitions": DEFINITIONS_EXTRACTION_PROMPT,
        "pitfalls": PITFALLS_EXTRACTION_PROMPT
    }
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def extract_from_document(
        self,
        document: ParsedDocument,
        granularity_level: GranularityLevel = GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE,
        allowed_sections: List[str] = None,
        playbook = None
    ) -> List[ExtractedKnowledge]:
        """
        Extract knowledge using specialized prompts per section.
        
        Args:
            document: Parsed document
            granularity_level: Clause-by-clause or full-document
            allowed_sections: Sections to extract (strategies, definitions, pitfalls)
            playbook: Current playbook state (for saturation check)
        
        Returns:
            List of extracted knowledge items
        """
        if allowed_sections is None:
            allowed_sections = ["strategies", "pitfalls", "definitions"]
        
        operative_clauses = document.get_operative_clauses()
        
        if granularity_level == GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE:
            return self._extract_clause_by_clause(document, operative_clauses, allowed_sections, playbook)
        else:
            return self._extract_full_document(document, operative_clauses, allowed_sections, playbook)
    
    def _extract_clause_by_clause(
        self,
        document: ParsedDocument,
        clauses: List[ParsedClause],
        allowed_sections: List[str],
        playbook = None
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge clause by clause with playbook awareness."""
        logger.info(f"Extracting clause-by-clause from {len(clauses)} operative clauses")
        logger.info(f"Allowed sections: {', '.join(allowed_sections)}")
        
        all_extracted = []
        
        # Generate playbook summary once (LLM call)
        playbook_summary = self._generate_playbook_summary(playbook, allowed_sections)
        logger.info(f"Playbook summary generated for saturation context")
        
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
                
                user_message = f"""EXISTING PLAYBOOK COVERAGE:
{playbook_summary}

---

NEW DOCUMENT TO ANALYZE:
Document: {document.title}
Type: {document.document_type}
Clause: {clause.title_text or clause.uid}
Note: Clause UIDs show hierarchy (e.g., 2.1 is child of 2, 2.1.1 is child of 2.1).

Clause Text:
{clause_text}

---

TASK: Extract {section} from this clause.
Only extract knowledge that is NOT already covered in the playbook summary above.
If the playbook already covers this concept, return empty list."""

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
        allowed_sections: List[str],
        playbook = None
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge from full document with playbook awareness."""
        logger.info(f"Extracting from full document with {len(clauses)} operative clauses")
        logger.info(f"Allowed sections: {', '.join(allowed_sections)}")
        
        # Generate playbook summary once (LLM call)
        playbook_summary = self._generate_playbook_summary(playbook, allowed_sections)
        logger.info(f"Playbook summary generated for saturation context")
        
        document_text = self._build_document_text(document, clauses)
        all_extracted = []
        
        # Extract for each allowed section
        for section in allowed_sections:
            if section not in self.SECTION_PROMPTS:
                continue
            
            system_prompt = self.SECTION_PROMPTS[section]
            
            user_message = f"""EXISTING PLAYBOOK COVERAGE:
{playbook_summary}

---

NEW DOCUMENT TO ANALYZE:
Document: {document.title}
Type: {document.document_type}
Note: Clause UIDs show hierarchy (e.g., 2.1 is child of 2, 2.1.1 is child of 2.1).

Full Document Text:
{document_text}

---

TASK: Extract {section} from this document.
Only extract knowledge that is NOT already covered in the playbook summary above.
If the playbook already covers these concepts, return empty list or only truly novel items."""

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
    
    def _build_document_text(self, document: ParsedDocument, clauses: List[ParsedClause]) -> str:
        """
        Build consolidated document text preserving FULL hierarchy.
        
        Includes ALL clauses (structural + operative + definitions) with proper
        indentation to show hierarchy. The LLM sees the complete document structure.
        """
        parts = [
            f"Document: {document.title}",
            f"Type: {document.document_type}",
            "",
            "=" * 60,
            "DOCUMENT STRUCTURE (with hierarchy)",
            "=" * 60,
            ""
        ]
        
        def render_clause(clause: ParsedClause, indent_level: int = 0):
            """Recursively render a clause and its children with indentation."""
            indent = "  " * indent_level
            clause_type = clause.metadata.get("type", "clause")
            
            # Header line: [UID] (type) Title
            title = clause.title_text or "Untitled"
            header = f"{indent}[{clause.uid}] ({clause_type}) {title}"
            parts.append(header)
            
            # Body text (excluding slot markers)
            body = clause.body_text.replace("[[slot:sub_clauses]]", "").strip()
            if body:
                # Indent body text
                for line in body.split("\n"):
                    if line.strip():
                        parts.append(f"{indent}  {line.strip()}")
            
            parts.append("")  # Empty line after clause
            
            # Recursively render sub-clauses
            for sub_clause in clause.sub_clauses:
                render_clause(sub_clause, indent_level + 1)
        
        # Render all root clauses with their full hierarchy
        for root_clause in document.clauses:
            render_clause(root_clause, indent_level=0)
        
        return "\n".join(parts)
    
    def _generate_playbook_summary(self, playbook, allowed_sections: List[str]) -> str:
        """
        Use LLM to generate a concise summary of what the playbook already covers.
        
        This summary is then fed to the extraction prompt so the LLM knows
        what knowledge already exists and can avoid extracting duplicates.
        """
        if playbook is None:
            return "The playbook is currently EMPTY. Extract all relevant knowledge."
        
        # Collect all bullets from allowed sections
        all_bullets = []
        section_counts = {}
        
        for section in allowed_sections:
            bullets = playbook.get_section(section) or []
            section_counts[section] = len(bullets)
            for bullet in bullets:
                content = bullet.content if hasattr(bullet, 'content') else str(bullet)
                all_bullets.append(f"[{section}] {content}")
        
        total_bullets = sum(section_counts.values())
        
        if total_bullets == 0:
            return "The playbook is currently EMPTY. Extract all relevant knowledge."
        
        # Build playbook content for summarization
        playbook_content = "\n".join(all_bullets)
        
        summary_prompt = """You are creating a DETAILED inventory of a securitization playbook.

Your summary will be used to PREVENT DUPLICATE EXTRACTION. The extraction agent will read your summary to decide what NOT to extract.

CRITICAL: Be EXHAUSTIVE and SPECIFIC. Include:

1. **STRATEGIES INVENTORY**:
   - List EACH strategy topic/pattern covered (not just themes)
   - Include specific mechanisms mentioned (e.g., "Priority of Payments waterfall", "incorporation by reference")
   - Mention specific roles/parties covered (e.g., "Servicer duties", "Security Trustee role")
   - Note any specific legal/regulatory references mentioned

2. **DEFINITIONS INVENTORY**:
   - List EVERY defined term explicitly (e.g., "ABS Transaction", "Collections", "Deemed Collections")
   - Group related terms together
   - Note if definitions include sub-components or formulas

3. **PITFALLS INVENTORY**:
   - List EACH specific error/warning documented
   - Include the specific mistake pattern (e.g., "using gross vs net balances", "renumbering waterfalls")
   - Mention consequences if documented

FORMAT (be detailed, use up to 1500 words if needed):

=== STRATEGIES COVERED ===
[List each strategy topic with key details, one per line]

=== DEFINITIONS COVERED ===
[List every defined term, grouped by category]

=== PITFALLS COVERED ===
[List each specific warning/error with key details]

=== KEY CONCEPTS ALREADY IN PLAYBOOK ===
[List specific securitization concepts, mechanisms, and patterns that are documented]

The goal is: if the extraction agent sees a concept in your summary, it should NOT extract it again."""

        user_message = f"""Playbook Statistics:
- Strategies: {section_counts.get('strategies', 0)} bullets
- Definitions: {section_counts.get('definitions', 0)} bullets  
- Pitfalls: {section_counts.get('pitfalls', 0)} bullets

FULL Playbook Content (analyze carefully):
{playbook_content}

Generate a DETAILED inventory that will prevent duplicate extraction:"""

        try:
            response = self.llm_client.chat(summary_prompt, user_message)
            summary = response.content.strip()
            logger.info(f"Generated playbook summary ({len(summary)} chars)")
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate playbook summary: {e}")
            # Fallback: include raw bullets
            fallback = f"The playbook contains {total_bullets} bullets. Here are ALL existing bullets:\n\n"
            fallback += playbook_content
            return fallback

