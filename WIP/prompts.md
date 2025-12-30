# ACE Securitization System - Prompts Documentation

This document provides a comprehensive overview of all prompts used in the ACE (Autonomous Curation Engine) Securitization System. The system uses specialized LLM agents to answer legal questions, learn from feedback, and curate a growing knowledge base.

---

## Table of Contents

1. [Prompt Architecture Overview](#prompt-architecture-overview)
2. [Core ACE Agents](#core-ace-agents)
   - [Generator Agent](#1-generator-agent)
   - [Reflector Agent](#2-reflector-agent)
   - [Curator Agent](#3-curator-agent)
3. [Reformulation Agent](#reformulation-agent)
   - [Enrich Mode](#mode-1-enrich)
   - [Derive Mode](#mode-2-derive)
   - [Remediate Mode](#mode-3-remediate)
   - [Explore Mode](#mode-4-explore)
4. [Trainer Pipeline](#trainer-pipeline)
   - [Knowledge Extractor](#knowledge-extractor)
   - [Knowledge Classifier](#knowledge-classifier)
   - [Knowledge Validator](#knowledge-validator)

---

## Prompt Architecture Overview

### System Prompts vs User Prompts

The ACE system uses a **two-part prompting architecture**:

#### **System Prompts**
- Define the agent's **role, capabilities, and constraints**
- Establish **output format requirements** (JSON structure)
- Provide **general instructions** that remain constant across all invocations
- Set the **tone and expertise level** (e.g., "legal expert in securitization")
- Are loaded once per agent initialization and persist across all user queries

#### **User Prompts**
- Contain **specific, query-dependent information**:
  - The user's actual question or task
  - Current playbook content (knowledge base)
  - Ground truth or feedback (for learning)
  - Context from previous agent outputs
- Are dynamically constructed for each interaction
- Include **data markers** like `PLAYBOOK_BEGIN/END`, `QUESTION:`, `GROUND TRUTH:`
- Change with every new query or learning cycle

**Why this separation?**
- **Efficiency**: System prompts define behavior once; user prompts provide variable data
- **Consistency**: Ensures agents behave predictably across all queries
- **Flexibility**: User prompts can be dynamically composed based on available data (playbook, ground truth, etc.)
- **Cost optimization**: System prompts can be cached by the LLM, reducing token usage

---

## Core ACE Agents

The ACE system consists of three primary agents that work in a continuous learning loop:

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  GENERATOR  │  →    │  REFLECTOR  │  →    │   CURATOR   │
│             │       │             │       │             │
│ Answers     │       │ Analyzes    │       │ Updates     │
│ Questions   │       │ Mistakes    │       │ Playbook    │
└─────────────┘       └─────────────┘       └─────────────┘
```

---

## 1. Generator Agent

**Role**: The Generator is a legal expert that answers securitization questions by combining playbook knowledge with general legal expertise.

### Generator System Prompt

**Location**: `prompts.py` → `GENERATOR_SYSTEM_PROMPT`

#### Purpose
Instructs the Generator to provide direct, authoritative answers while citing relevant playbook bullets.

#### Key Features
- **Dual Knowledge Sources**: Uses both playbook bullets AND general legal expertise
- **Critical Rule**: MUST answer even when playbook is empty (uses general knowledge)
- **Direct Answer Format**: No preambles like "In the context of...", just the answer
- **Structured Output**: Returns JSON with `reasoning`, `bullet_ids`, `final_answer`
- **Citation Tracking**: Records which playbook bullets were useful via `bullet_ids`

#### Output Structure
```json
{
  "reasoning": "Internal chain-of-thought analysis",
  "bullet_ids": ["str-00001", "def-00003"],
  "final_answer": "Direct answer text without preamble"
}
```

#### Critical Rules for `final_answer`
- **Definitions**: Give definition directly (e.g., "**Term** means X")
- **Clauses**: Provide clause text only, no explanation
- **Yes/No questions**: Start with "Yes" or "No" then brief explanation
- **No meta-commentary**: Don't say "not defined in playbook" or "according to documents"

#### Why This Prompt Design?
The Generator must be a **reliable legal advisor**, not a passive retriever. If the playbook lacks information, the system should still provide value using the LLM's inherent legal knowledge. This makes the system useful from day one, even with an empty playbook.

---

### Generator ProseMirror Variant

**Location**: `prompts.py` → `GENERATOR_PROSEMIRROR_SYSTEM_PROMPT`

#### Purpose
Same as standard Generator but outputs answers in ProseMirror JSON format for rich text rendering in web applications.

#### ProseMirror Format Example
```json
{
  "type": "doc",
  "content": [
    {
      "type": "paragraph",
      "content": [
        {"type": "text", "text": "\"Sanctioned Country\"", "marks": [{"type": "strong"}]},
        {"type": "text", "text": " means any country subject to comprehensive Sanctions..."}
      ]
    }
  ]
}
```

#### Formatting Rules
- Split content into **logical paragraphs**
- For definitions: Format as **"Term"** means [definition]
- For multi-paragraph clauses: Use separate paragraph objects

---

### Generator User Prompt

**Location**: `prompts.py` → `format_generator_user_message()`

#### Purpose
Delivers the user's question and current playbook content to the Generator.

#### Structure
```
PLAYBOOK_BEGIN
[Current playbook bullets with IDs and content]
PLAYBOOK_END

QUESTION:
[User's actual question]

Please analyze the question and provide your response as a JSON object...
```

#### Why This Format?
- Clear **data boundaries** prevent the LLM from confusing playbook content with instructions
- **Bullet IDs** enable citation tracking
- Simple **question marker** makes the task clear

---

## 2. Reflector Agent

**Role**: The Reflector analyzes the Generator's output, identifies errors by comparing against ground truth, and extracts reusable insights (strategies and pitfalls).

### Reflector System Prompt

**Location**: `prompts.py` → `REFLECTOR_SYSTEM_PROMPT`

#### Purpose
Performs **post-generation quality control** and **knowledge extraction** to improve future responses.

#### Key Capabilities

**1. Ground Truth Comparison (CRITICAL)**
When ground truth is provided, the Reflector performs detailed comparison:
- Identifies **every difference** between Generator output and ground truth
- Notes **missing key points**
- Flags **incorrect statements** that contradict ground truth
- Detects **phrasing differences** that change legal meaning
- Checks for **incomplete coverage**

**2. Strategy Extraction**
Extracts what the ground truth did RIGHT:
- What **approach, structure, or content** made the ground truth correct
- What **best practices** should be replicated in future responses
- Example: *"Always mention the five elements of true sale: legal isolation, substantive consolidation risk..."*

**3. Pitfall Extraction**
Extracts what the Generator did WRONG:
- What **mistakes, omissions, or misconceptions** led to errors
- What **warnings** should guide future responses
- Example: *"Avoid defining bankruptcy remoteness without mentioning SPV structural requirements"*

**4. Root Cause Analysis**
Identifies WHY errors occurred:
- Wrong source of truth?
- Misinterpreted term?
- Missing playbook knowledge?
- Failed to apply relevant bullet?

**5. Bullet Effectiveness Tagging**
Tags each playbook bullet used as:
- `helpful`: Contributed to correct answer
- `harmful`: Led to incorrect answer
- `neutral`: Used but neither helped nor harmed

**6. Definition Detection**
If user asked to define a term and ground truth provides the definition, flags it for addition to playbook.

**7. Conditional Pattern Analysis (CRITICAL)**
Detects and analyzes conditional variants in ground truth:

**Condition Types**:
- **Binary**: [if A] vs [if NOT A]
- **Ordinal**: [if first] vs [if middle] vs [if last]
- **Jurisdictional**: [if UK] vs [if EU] vs [if UK AND EU]
- **Combinatorial**: Multiple conditions creating a matrix

**Analysis Requirements**:
- Extract **all conditions** verbatim
- Classify **condition types**
- Check **completeness** (e.g., jurisdictional must have combined variant)
- Identify **template variables** like `{{deal.tranche[#i].holder}}`
- Flag if Generator **mutated variable syntax**

**8. Securitization Domain Checks**
Validates understanding of Securitization Regulation

#### Output Structure
```json
{
  "reasoning": "Detailed comparison with ground truth",
  "error_identification": "Specific errors with ground truth references",
  "root_cause_analysis": "Why errors occurred",
  "correct_approach": "What should be done instead (based on ground truth)",
  "key_insight": "Principle to remember for future",
  "extracted_strategies": [
    "Strategy 1 from ground truth...",
    "Strategy 2 from ground truth..."
  ],
  "extracted_pitfalls": [
    "Pitfall 1 (what Generator did wrong)...",
    "Pitfall 2 (another mistake)..."
  ],
  "bullet_tags": [
    {"id": "str-00001", "tag": "helpful"}
  ],
  "removal_candidates": ["str-00003"],
  "modification_suggestions": [
    {"id": "str-00002", "suggestion": "Should clarify..."}
  ],
  "ground_truth_definition": {
    "term": "Term Name",
    "definition": "Ground truth definition",
    "should_add_to_playbook": true
  },
  "conditional_analysis": {
    "has_conditions": true,
    "condition_types_found": ["jurisdictional", "positional"],
    "all_conditions_extracted": ["[if UK]...", "[if EU]..."],
    "completeness_issues": ["Missing UK+EU combined variant"],
    "template_variables_found": ["{{deal.tranche[#i].holder}}"],
    "variable_mutation_detected": false,
    "suggested_template": "Full template with all conditions"
  }
}
```

#### Why This Complexity?
The Reflector is the **learning engine** of ACE. It transforms ground truth into reusable knowledge:
- **Strategies** become best practices
- **Pitfalls** become warnings
- **Conditional patterns** become templates
- **Definitions** fill knowledge gaps

This enables the system to **learn once, apply everywhere**.

---

### Reflector User Prompt

**Location**: `prompts.py` → `format_reflector_user_message()`

#### Purpose
Provides the Reflector with all information needed for analysis.

#### Structure
```
ORIGINAL QUESTION:
[User's question]

GENERATOR OUTPUT:
[Generator's JSON response]

CURRENT PLAYBOOK:
[Playbook bullets]

═══════════════════════════════════════
GROUND TRUTH / EXPECTED ANSWER (PRIMARY REFERENCE):
═══════════════════════════════════════
[Correct answer for comparison]
═══════════════════════════════════════

*** CRITICAL: Compare Generator output with this ground truth. Extract strategies and pitfalls. ***

HUMAN FEEDBACK (if provided):
[Additional feedback]

Please analyze...
```

#### Why This Format?
- **Separation markers** (`═══`) make ground truth visually prominent
- **Explicit instruction** reinforces the importance of ground truth comparison
- **Complete context** gives Reflector everything needed for thorough analysis

---

## 3. Curator Agent

**Role**: The Curator maintains the playbook by integrating insights from the Reflector, managing playbook evolution through ADD, MODIFY, REMOVE, and MERGE operations.

### Curator System Prompt

**Location**: `prompts.py` → `CURATOR_SYSTEM_PROMPT`

#### Purpose
Translates Reflector insights into concrete playbook modifications while avoiding redundancy.

#### Key Capabilities

**1. Insight Integration**
- **Extracted Strategies**: Creates ADD operations to `strategies` section
- **Extracted Pitfalls**: Creates ADD operations to `pitfalls` section
- **Ground Truth Definitions**: Creates ADD operations to `definitions` section with format `"[TERM]: [DEFINITION]"`

**2. Conditional Pattern Integration (CRITICAL)**
When Reflector provides `conditional_analysis`:
- **Condition Completeness**: Adds strategies addressing missing variants (e.g., "Jurisdictional Completeness Rule")
- **Templates**: Adds full conditional templates to `templates` section with ALL variants
- **Variable Protection**: Adds pitfalls about preserving template variable syntax
- **Condition-Specific Strategies**: Ensures playbook has strategies for each condition type

**3. Template Variable Sanctity**
- Ensures playbook warns about preserving `{{...}}` syntax exactly
- If Generator mutated variables, creates pitfall entry

**4. Redundancy Avoidance**
- Checks for **similar existing bullets** before adding
- Uses **MODIFY** instead of ADD+REMOVE when refining bullets
- Proposes **MERGE** for redundant bullets covering same concept

**5. Harmful Bullet Management**
- Tracks bullets tagged as `harmful` multiple times
- Proposes **REMOVE** operations for consistently harmful bullets
- Can **MODIFY** with `reset_harmful: true` after fixing

**6. Condition Pattern Recognition**
When adding templates with conditions:
- **ALL variants included** (no partial templates)
- **Jurisdictional**: MUST include combined cross-border variant (UK, EU, UK+EU)
- **Positional**: MUST include all positions (first/senior, middle/mezzanine, last/junior)
- **Binary**: MUST include both states (TRUE/FALSE, revolving/term)

#### Available Operations

**ADD**: Create new bullets
```json
{
  "type": "ADD",
  "section": "strategies",
  "content": "Actionable best practice text..."
}
```

**REMOVE**: Delete harmful bullets
```json
{
  "type": "REMOVE",
  "bullet_id": "pit-00003",
  "reason": "Incorrectly stated, caused errors"
}
```

**MODIFY**: Update existing bullets
```json
{
  "type": "MODIFY",
  "bullet_id": "str-00002",
  "new_content": "Improved content...",
  "reason": "Original was too vague",
  "reset_harmful": true
}
```

**MERGE**: Combine redundant bullets
```json
{
  "type": "MERGE",
  "source_bullet_ids": ["def-00001", "def-00005"],
  "target_section": "definitions",
  "merged_content": "Combined definition...",
  "reason": "Both covered same concept"
}
```

#### Playbook Sections
- **strategies**: General approaches, best practices, methodologies, condition handling rules
- **pitfalls**: Common mistakes, things to avoid, red flags, template variable mutations
- **templates**: Reusable clause structures with ALL conditional variants
- **definitions**: Key terms and precise meanings, condition trigger keywords


#### Output Structure
```json
{
  "reasoning": "Why these operations are needed",
  "operations": [
    {"type": "ADD", "section": "strategies", "content": "..."},
    {"type": "MODIFY", "bullet_id": "str-00001", "new_content": "...", "reason": "..."}
  ]
}
```

#### Why This Design?
The Curator is the **knowledge base manager**. It must:
- **Prioritize high-value additions** (from ground truth)
- **Maintain organization** (correct sections)
- **Prevent bloat** (avoid duplicates)
- **Ensure completeness** (all conditional variants)
- **Track quality** (harmful bullet removal)

---

### Curator User Prompt

**Location**: `prompts.py` → `format_curator_user_message()`

#### Purpose
Provides the Curator with question, Generator output, Reflector analysis, and current playbook.

#### Structure
```
ORIGINAL QUESTION:
[User's question]

GENERATOR'S ATTEMPTED ANSWER:
[Generator JSON]

REFLECTOR'S ANALYSIS:
[Reflector JSON with strategies/pitfalls/conditional_analysis]

CURRENT PLAYBOOK:
[All playbook bullets with IDs]

Based on the Reflector's analysis, determine what operations should be applied to the playbook...
```

---

## Reformulation Agent

**Role**: The Reformulation Agent proposes alternative formulations of legal clauses under explicit instructions, preserving legal meaning unless instructed otherwise. Operates in 4 modes.

### Base Reformulation Principles

**Location**: `prompts_reformulation_agent.py` → `REFORMULATION_BASE_PROMPT`

#### Core Principles
1. **NEVER alter legal meaning** unless explicitly instructed
2. **Preserve defined terms**, party references, and legal obligations
3. **Maintain consistency** with Canon (reference) clause structure
4. **Output ranked alternatives** with confidence scores
5. **Explain clearly** if reformulation is not possible

#### Output Format
```json
{
  "success": true,
  "alternatives": [
    {
      "rank": 1,
      "content": "Reformulated clause text",
      "confidence": 0.95,
      "changes_summary": "Brief description of changes"
    }
  ],
  "failure_reason": null
}
```

#### Universal Rules
- Provide **1-3 ranked alternatives** (rank 1 = best)
- Confidence scores: **0.0 to 1.0**
- If reformulation impossible without changing meaning: **success=false**
- Always explain changes in **changes_summary**
- Preserve all placeholders like **[[var:deal.xyz]]** exactly

---

## Mode 1: ENRICH

**Location**: `prompts_reformulation_agent.py` → `REFORMULATION_ENRICH_SYSTEM_PROMPT`

### Purpose
**EXPAND or COMPLETE** the clause without altering its legal effect.

### What "Enrich" Means
- Add **clarifying language** where ambiguity exists
- Expand **abbreviated expressions**
- Include **standard boilerplate** that is implied but not stated
- Add **cross-references** to related provisions
- Flesh out **incomplete provisions** with standard market language

### Restrictions
- **DON'T** change legal obligations or rights
- **DON'T** add new substantive terms not implied by original
- **DON'T** remove existing content
- **DON'T** alter defined terms or party names

### User Prompt Template
```
CLAUSE TO ENRICH:
[clause text]

PLAYBOOK CONTEXT:
[relevant playbook bullets]

ADDITIONAL INSTRUCTIONS:
[any extra guidance]

Please provide enriched alternatives...
```

### Why Enrich Mode?
Clients often draft **shorthand clauses** that need expansion for legal certainty. Enrich mode adds missing detail while preserving intent.

---

## Mode 2: DERIVE

**Location**: `prompts_reformulation_agent.py` → `REFORMULATION_DERIVE_SYSTEM_PROMPT`

### Purpose
**PROPOSE CLAUSE VARIANTS** under given constraints (rules).

### What "Derive" Means
- Generate alternatives satisfying **specific requirements**
- Adapt clause to **different scenarios** or deal structures
- Create variants for **different jurisdictions** or regulatory contexts
- Produce options with **varying levels** of protection or flexibility

### Constraints May Include
- **Jurisdiction** requirements (e.g., "must comply with English law")
- **Party preferences** (e.g., "more favorable to the Issuer")
- **Structural requirements** (e.g., "must work with revolving structure")
- **Regulatory requirements** (e.g., "must satisfy EU Securitization Regulation")
- **CONDITIONS IN BRACKETS**: [if condition A] requires multiple alternatives per condition

### CRITICAL Pre-Derivation Analysis

**Before generating ANY alternatives, MUST:**

**1. Scan for Condition Triggers**
Identify keywords signaling condition needs:
- `position`, `is_first`, `is_last` → **Positional variants needed**
- `is_revolving` → **Revolving vs Term variants**
- `country`, `jurisdiction`, `UK`, `EU`, `US` → **Jurisdictional variants** (ALWAYS include combined variant)
- `has_X`, `X_enabled` → **Feature toggle variants** (with/without)
- Multiple triggers → **Combinatorial matrix needed**

**2. Identify Template Variables**
Find all `{{...}}` patterns - these are **SACRED**:
- **Never modify syntax** (keep `{{deal.tranche[#i].holder}}` exactly)
- **Never change accompanying verbs** (keep 'hereby grant' even if grammatically odd)
- **Never interpret meaning**
- **Copy exactly** as they appear

### Derivation Rules

**1. Template Variable Handling**
- Variables like `{{deal.tranche[#i].holder}}` must appear **EXACTLY** as provided
- **DO NOT** change `[#i]` to `[i]` or `[0]`
- **DO NOT** change property names or paths
- **DO NOT** "fix" perceived grammatical issues caused by variables
- The verb must match the template, **not your interpretation of plurality**

**2. Positional Awareness**
- **position=0 or "first"**: Senior/most protected tranche
- **position="last"**: Junior/residual tranche
- **position>0 AND position<>"last"**: Mezzanine tranches
- Each may have different: recipients, timing, rights, documentation

**3. Legally Significant Conditions ONLY**
- Create conditions **ONLY** for legally significant differences (obligations, rights, recipients, timing, regulations)
- **DO NOT** create conditions for stylistic or grammatical variations

### CRITICAL Rules for Conditions

When constraints contain bracketed conditions like `[if all accounts are in EU]`:

1. Generate **AT LEAST 2 alternatives** (preferably 2-3) for **EVERY condition**
2. Each alternative = **complete, standalone reformulated clause** tailored to its condition
3. **Group by condition**: alternatives 1-2 address condition 1, alternatives 3-4 address condition 2
4. **Rank sequentially**: ranks 1-2 for first condition, ranks 3-4 for second condition
5. In **changes_summary**, explicitly state: `"Condition [X] - Approach [A/B/C]: [description]"`
6. **Vary drafting approaches** per condition: conservative/detailed, minimal/comprehensive, prescriptive/flexible
7. **TOTAL alternatives** = (number of conditions) × (alternatives per condition, minimum 2)

### Example with Conditions

**Original**: "The Servicer shall maintain all Collection Accounts."

**Constraints**: "[if all issuer accounts and collection accounts are in the EU] [if accounts span multiple jurisdictions including non-EU]"

**Expected Output** (AT LEAST 2 per condition = 4 alternatives):

```json
{
  "success": true,
  "alternatives": [
    {
      "rank": 1,
      "content": "The Servicer shall maintain all Collection Accounts with financial institutions located within the European Union and subject to supervision by competent authorities under EU banking regulations.",
      "confidence": 0.90,
      "changes_summary": "Condition 1 (EU only) - Approach A: Concise version with EU location and regulatory supervision requirements"
    },
    {
      "rank": 2,
      "content": "The Servicer shall maintain all Collection Accounts with financial institutions that are: (a) incorporated and operating within a Member State of the European Union; (b) authorized and supervised by the competent authority in such Member State; and (c) subject to capital adequacy requirements under the EU Capital Requirements Regulation (EU) 575/2013.",
      "confidence": 0.88,
      "changes_summary": "Condition 1 (EU only) - Approach B: Detailed version with specific enumerated requirements and CRR reference"
    },
    {
      "rank": 3,
      "content": "The Servicer shall maintain all Collection Accounts with financial institutions of international standing, provided that any such account located outside the European Union must (i) be with an institution rated at least A- by a recognized rating agency, (ii) be subject to equivalent regulatory supervision, and (iii) comply with applicable anti-money laundering and know-your-customer requirements.",
      "confidence": 0.85,
      "changes_summary": "Condition 2 (multi-jurisdiction) - Approach A: Rating-based approach with regulatory equivalence and AML/KYC requirements"
    },
    {
      "rank": 4,
      "content": "The Servicer shall maintain all Collection Accounts with financial institutions, provided that: (a) any account within the European Union must comply with EU banking regulations; and (b) any account outside the European Union must be with an institution that has entered into a regulatory acknowledgment acceptable to the Security Trustee and maintains deposit insurance or equivalent protection.",
      "confidence": 0.82,
      "changes_summary": "Condition 2 (multi-jurisdiction) - Approach B: Bifurcated structure with EU compliance and non-EU protective measures including regulatory acknowledgment"
    }
  ],
  "failure_reason": null
}
```

### User Prompt Template
```
CLAUSE TO DERIVE FROM:
[clause text]

CONSTRAINTS/RULES:
[constraints and conditions]

PLAYBOOK CONTEXT:
[relevant playbook]

ADDITIONAL INSTRUCTIONS:
[extra guidance]

Please provide derived variants...
```

### Why Derive Mode?
Securitization deals vary by **jurisdiction, asset class, and party preferences**. Derive mode generates compliant variants for different scenarios, saving lawyers hours of drafting time.

---

## Mode 3: REMEDIATE

**Location**: `prompts_reformulation_agent.py` → `REFORMULATION_REMEDIATE_SYSTEM_PROMPT`

### Purpose
**RESTORE COMPLIANCE** of a clause when incompatibility arises.

### What "Remediate" Means
- Fix clauses that have **drifted from Canon** (reference)
- Correct **semantic errors** or inconsistencies
- Realign language that has become **non-compliant**
- Restore **standard market terms** incorrectly modified
- Fix **structural issues** while preserving intended meaning

### Incompatibilities May Include
- **Semantic drift** from Canon clause
- **Inconsistent defined terms**
- **Missing essential provisions**
- **Incorrect legal formulations**
- **Non-standard language** creating ambiguity

### Restrictions
- **DON'T** simply copy Canon clause verbatim (preserve user's style where possible)
- **DON'T** ignore user's intended meaning if it can be preserved compliantly
- **DON'T** make unnecessary changes beyond what's needed for compliance

### Example
**Non-compliant**: "The Seller transfers all risks to the Buyer maybe."

**Issue**: Ambiguous language ("maybe") undermines true sale characterization

**Remediated**: "The Seller hereby transfers, assigns, and conveys to the Buyer all right, title, and interest in and to the Receivables, together with all associated risks and benefits, without recourse to the Seller."

### User Prompt Template
```
CLAUSE TO REMEDIATE:
[clause text]

IDENTIFIED ISSUES:
[specific compliance problems]

PLAYBOOK CONTEXT:
[relevant standards]

ADDITIONAL INSTRUCTIONS:
[extra guidance]

Please provide remediated alternatives...
```

### Why Remediate Mode?
Clauses **drift from standards** during negotiation. Remediate mode restores compliance while respecting the parties' intended meaning.

---

## Mode 4: EXPLORE

**Location**: `prompts_reformulation_agent.py` → `REFORMULATION_EXPLORE_SYSTEM_PROMPT`

### Purpose
**REFORMULATE** in response to open-ended user prompts.

### What "Explore" Means
- Respond to **natural language requests** for modifications
- **Interpret user intent** and propose appropriate reformulations
- Offer **creative alternatives** while maintaining legal soundness
- Suggest **improvements** the user may not have explicitly requested

### User Prompts May Include
- "Make this simpler"
- "Can we add protection for the Servicer?"
- "What if we need to cover multiple jurisdictions?"
- "This feels too aggressive, soften it"
- "How would a US law firm draft this?"

### Restrictions
- **DON'T** ignore the user's explicit request
- **DON'T** make changes unrelated to the prompt
- **DON'T** sacrifice legal precision for readability without warning
- **DON'T** assume intent that contradicts the prompt

### Example
**Original**: "The Calculation Agent shall determine the Interest Rate."

**User prompt**: "Make this more detailed and add fallback provisions"

**Explored**: "The Calculation Agent shall determine the Interest Rate for each Interest Period by reference to the Screen Rate at approximately 11:00 a.m. (London time) on the Interest Determination Date. If the Screen Rate is unavailable, the Calculation Agent shall request the principal London office of each of the Reference Banks to provide a quotation of its rate. If at least two such quotations are provided, the Interest Rate shall be the arithmetic mean of the quotations. If fewer than two quotations are provided, the Interest Rate shall be the Interest Rate in effect for the immediately preceding Interest Period."

### User Prompt Template
```
CLAUSE TO REFORMULATE:
[clause text]

USER REQUEST:
[natural language request]

PLAYBOOK CONTEXT:
[relevant playbook]

ADDITIONAL INSTRUCTIONS:
[extra guidance]

Please provide reformulated alternatives...
```

### Why Explore Mode?
Users don't always know the precise legal term for what they want. Explore mode **interprets intent** and proposes solutions, making the system accessible to non-experts.

---

### ProseMirror Output Variants

All four reformulation modes have **ProseMirror variants** for rich text output:
- `REFORMULATION_ENRICH_PROSEMIRROR_SYSTEM_PROMPT`
- `REFORMULATION_DERIVE_PROSEMIRROR_SYSTEM_PROMPT`
- `REFORMULATION_REMEDIATE_PROSEMIRROR_SYSTEM_PROMPT`
- `REFORMULATION_EXPLORE_PROSEMIRROR_SYSTEM_PROMPT`

#### Purpose
Output reformulated clauses in **ProseMirror JSON format** for web applications requiring structured rich text with defined terms in bold.

#### Two Formats

**Simple Format** (for single-paragraph clauses):
```json
{
  "type": "doc",
  "content": [
    {
      "type": "paragraph",
      "content": [
        {"type": "text", "text": "normal text"},
        {"type": "text", "text": "defined term", "marks": [{"type": "strong"}]},
        {"type": "text", "text": " more text"}
      ]
    }
  ]
}
```

**Structured Format** (for clauses with enumerated sub-clauses):
```json
{
  "main_clause": {
    "type": "doc",
    "content": [
      {
        "type": "paragraph",
        "content": [
          {"type": "text", "text": "\"", "marks": [{"type": "strong"}]},
          {"type": "text", "text": "Event of Default", "marks": [{"type": "strong"}]},
          {"type": "text", "text": "\" means any of:"}
        ]
      },
      {"type": "slot", "attrs": {"name": "sub_clauses"}}
    ]
  },
  "sub_clauses": [
    {
      "type": "doc",
      "content": [
        {
          "type": "paragraph",
          "content": [
            {"type": "text", "text": "(a) failure to pay..."}
          ]
        }
      ]
    }
  ]
}
```

#### When to Use Each Format
- **Simple**: Single-paragraph clauses without enumeration
- **Structured**: Clauses with (a), (b), (c) or (i), (ii), (iii) sub-items

---

## Trainer Pipeline

The Trainer Pipeline extracts knowledge from minified json(for ex. MFA.json) and transaction documents to bootstrap and enrich the playbook.

### Architecture
```
Document
   ↓
[Parser] → ParsedClause objects
   ↓
[Extractor] → Extract strategies, definitions, templates, pitfalls
   ↓
[Classifier] → Validate and classify knowledge type
   ↓
[Validator] → Quality checks, deduplication
   ↓
Playbook Operations
```

---

## Knowledge Extractor

**Location**: `trainer/knowledge_extractor.py` → `_get_extraction_system_prompt()`

### Purpose
Performs **EXHAUSTIVE knowledge extraction** from securitization documents. This is NOT sampling - it extracts EVERY relevant item.

### System Prompt Components

#### Mandatory Extraction Standards

**1. COMPLETENESS**
- Extract **EVERY relevant item** - missing items = FAILURE
- Document has 100 definitions? Extract **ALL 100 definitions**
- Clause has 8 strategies? Extract **ALL 8 strategies**
- **NO sampling**, NO cherry-picking, NO "here are some examples"

**Why this rule?**
Incomplete extraction creates knowledge gaps. The playbook must be comprehensive to be reliable.

**2. QUALITY**
Each extraction MUST be **50+ words minimum**:
- Explain **WHAT** it is
- Explain **WHY** it matters
- Explain **HOW** it works
- Include strategic rationale and practical implications
- Reference specific legal mechanisms and cross-references
- Note jurisdictional specifics when relevant

**Quality Examples**:

**[REJECTED]** "Master Framework Architecture" (title only, no context)

**[REJECTED]** "A structure that centralizes terms" (too vague)

**[ACCEPTED]** "Master Framework Architecture centralizes common contractual terms (definitions, representations, indemnities, interpretation rules) across multiple transaction documents in a single master agreement incorporated by reference. This reduces negotiation time by 40-60%, ensures consistency across deals, enables faster party accession, and reduces legal costs in repeat transactions by avoiding renegotiation of standard terms. Each ancillary document references the MFA via incorporation clause, creating a hierarchical definition structure where MFA terms apply unless explicitly overridden."

**3. DETAIL REQUIREMENT**
Each extraction must provide:
- **Context**: Where it fits in securitization structure
- **Mechanism**: How it works legally/operationally
- **Rationale**: Why it's structured this way
- **Implications**: What happens if done incorrectly
- **Variations**: Different approaches across jurisdictions/asset classes

**4. CONFIDENCE SCORING**
- **High (≥0.85)**: Clear, well-supported, specific
- **Medium (0.6-0.84)**: Somewhat supported, some assumptions
- **Low (<0.6)**: Uncertain, needs verification

#### Output Format
```json
{
  "extractions": [
    {
      "content": "50+ word detailed extraction with WHAT/WHY/HOW...",
      "confidence": 0.92,
      "source_clause_title": "Clause title from document",
      "examples": [
        {
          "text": "Example quote from document",
          "clause_ref": "CLAUSE-7.2"
        }
      ],
      "related_terms": ["Term1", "Term2", "Term3"],
      "rationale": "Why this knowledge matters for practitioners"
    }
  ]
}
```

**CRITICAL**: The above shows the QUALITY required per item. Response must contain **50-200+ such items** depending on document size (NOT just 2-10 examples!).

---

### Extraction Type-Specific Prompts

#### STRATEGIES

**Focus**: Structural patterns, legal mechanisms, and best practices used in securitization transactions.

**Quality Standards**:
- Explain the **strategic rationale** (WHY this approach is used, not just WHAT it is)
- Include **practical implications** for transaction parties
- Reference relevant **legal concepts** (true sale, bankruptcy remoteness, perfection, priority)
- Note **variations** across asset classes (RMBS, auto ABS, CLO, etc.) when applicable


#### DEFINITIONS

**Focus**: Precise legal terms, specialized concepts, and defined terms that establish shared vocabulary.

**Quality Standards**:
- Provide **clear, legally precise definitions** with context
- Include **practical implications** of the definition
- Note **jurisdictional variations** when relevant (UK vs US vs Luxembourg)
- **Link to related defined terms** for cross-referencing
- Specify whether term is **standard market practice** or transaction-specific variation


#### TEMPLATES

**Focus**: Reusable clause structures with identified placeholders adaptable across transactions.

**Quality Standards**:
- Clearly mark all **variable elements** with [PLACEHOLDER] notation
- Preserve **essential legal language** and structure
- Include **brief explanation** of template purpose and customization points
- Note **critical negotiation points** or common variations
- Ensure template **maintains legal coherence**

**Example Extractions**:
- "Appointment Clause: '[PARTY_A] hereby appoints [PARTY_B] to act as [ROLE] under and in connection with the [TRANSACTION_DOCUMENTS], with authority to [SCOPE_OF_AUTHORITY], subject to the limitations and conditions set forth herein. [PARTY_B] accepts such appointment on the terms of this Agreement.' — Use for Security Trustee, Note Trustee, or Agent appointments; customize scope based on role (enforcement rights, payment processing, administrative duties)"
- "Limited Liability Clause: '[PARTY] shall not be liable for any loss, damage, or expense arising from [ACTIONS] except to the extent such loss results from [PARTY]'s [LIABILITY_STANDARD: gross negligence/willful misconduct/fraud]. [PARTY] shall have no duty to [EXCLUDED_DUTIES] unless expressly agreed in writing.' — Standard for trustees/agents; liability standard typically gross negligence for professional parties, willful misconduct for less sophisticated parties"

#### PITFALLS

**Focus**: Legal risks, structural weaknesses, drafting errors, and anti-patterns that create problems.

**Quality Standards**:
- Identify the **specific structural or drafting deficiency**
- Explain **concrete consequences** (rating downgrade, enforcement delay, priority loss, liability exposure)
- Reference **relevant legal principles** or market standards violated
- Suggest **preventive measures** or corrective approaches when possible
- Distinguish **severity** (critical vs. moderate vs. minor concerns)

---

---

### Extraction Granularity Modes

The Knowledge Extractor operates in **three granularity modes** to balance extraction completeness with LLM token limits and processing time:

#### Mode 1: Operative Clause-by-Clause

**Location**: `trainer/knowledge_extractor.py` → `_extract_clause_by_clause()`

**How It Works**:
1. Identifies all **operative clauses** and **definition clauses** in the document
2. Processes **each clause individually** in separate LLM calls
3. For each clause × each extraction type (strategies, definitions, templates, pitfalls) = one LLM call
4. Total LLM calls = number_of_clauses × number_of_extraction_types

**Example**: Document with 50 operative clauses and 4 extraction types = 200 LLM calls

**User Prompt Structure**:
```
CONTEXT:
[Document title, clause location, surrounding context]

CLAUSE TEXT:
[Actual clause body text for this specific clause]

══════════════════════════════════════════════════════════════════
COMPLETE EXTRACTION MANDATE - NO PARTIAL RESULTS
══════════════════════════════════════════════════════════════════

TASK: Extract EVERY SINGLE [TYPE] from this clause.

CRITICAL RULES:
• If clause contains 10 definitions → extract ALL 10 definitions (not 2-3 examples)
• If clause contains 5 strategies → extract ALL 5 strategies (not 1-2 highlights)
• Each extraction MUST be 50+ words minimum with detailed explanation
• Include WHY it matters, HOW it works, WHAT the legal implications are
• NO title-only extractions - full context required

[UNACCEPTABLE] Extracting 2 items when clause contains 8 relevant items
[REQUIRED] Complete exhaustive extraction of ALL items in this clause

Return JSON with 'extractions' array...
```

**Advantages**:
- **Most thorough**: Each clause gets focused attention
- **Precise attribution**: Knowledge directly linked to source clause
- **Context preservation**: Each extraction has clear provenance

**Disadvantages**:
- **Expensive**: Hundreds of LLM calls for large documents
- **Slow**: Sequential processing takes time
- **Potential redundancy**: Same concept may be extracted multiple times from different clauses

**When to Use**:
- Small to medium documents (20-50 clauses)
- When precise clause-level attribution is critical
- When budget and time are not constraints

---

#### Mode 2: Full Document

**Location**: `trainer/knowledge_extractor.py` → `_extract_full_document()`

**How It Works**:
1. Concatenates **ALL operative clauses** into single comprehensive document text
2. Sends **entire document** to LLM in one prompt
3. Requests **exhaustive extraction** of ALL items of each type
4. Total LLM calls = number_of_extraction_types (typically 4)

**Example**: Document with 200 operative clauses and 4 extraction types = only 4 LLM calls

**User Prompt Structure**:
```
DOCUMENT: [Full Document Title]
TYPE: [Document Type]

### Clause 1: Security Trustee Appointment
[Full clause text]

### Clause 2: Definitions
[Full clause text]

### Clause 3: Representations and Warranties
[Full clause text]

... [ALL operative clauses included]

══════════════════════════════════════════════════════════════════
MANDATORY COMPREHENSIVE EXTRACTION - NO SAMPLING ALLOWED
══════════════════════════════════════════════════════════════════

You MUST extract EVERY SINGLE [TYPE] from this entire document.

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

══════════════════════════════════════════════════════════════════
EXPECTED OUTPUT QUANTITY:
══════════════════════════════════════════════════════════════════
• Small document (20-50 clauses): Expect 30-80 items
• Medium document (50-150 clauses): Expect 80-150 items  
• Large document (150-500 clauses): Expect 150-300 items
• This document has [N] operative clauses

[UNACCEPTABLE OUTPUTS]
• Returning 10-20 items when document has 200+ clauses (FAILURE)
• Extracting only "interesting" or "important" items (WRONG - extract ALL)
• Stopping after first 50 items (UNACCEPTABLE)

[ACCEPTABLE OUTPUT]
• Comprehensive JSON array with 100-200+ detailed extraction objects
• Each item 50-150 words with full context and rationale
• Every relevant item from document represented

Return JSON with 'extractions' array containing 50-200+ items...
```

**Advantages**:
- **Extremely efficient**: Only 4 LLM calls regardless of document size
- **Fast**: Parallel extraction types possible
- **Cost-effective**: 50-100x cheaper than clause-by-clause for large documents
- **Cross-clause synthesis**: LLM can identify patterns across entire document

**Disadvantages**:
- **Token limits**: Very large documents (500+ clauses) may exceed context windows
- **Less precise attribution**: Source clause identification is approximate
- **Potential missed items**: LLM may overlook clauses in middle of long document

**When to Use**:
- Large documents (100-500 clauses)
- Master Framework Agreements with comprehensive definitions sections
- When speed and cost efficiency are priorities
- When cross-clause pattern recognition is valuable

**Special Handling for Definitions**:
In full document mode, when extracting definitions, the prompt explicitly instructs to:
- **Preserve original length**: Do not summarize or shorten definitions
- **Enhance, don't reduce**: Add context and implications, but keep all original legal language
- **Extract ALL definitions**: From dedicated "Definitions" or "Interpretation" clauses, every single term must be extracted

---

#### Mode 3: Batch (Hybrid Approach)

**Location**: `trainer/knowledge_extractor.py` → `_extract_batch_mode()`

**How It Works**:
1. Groups operative clauses into **batches** (e.g., 15 clauses per batch)
2. Sends each batch to LLM as a mini-document
3. Total LLM calls = (number_of_clauses / batch_size) × number_of_extraction_types

**Example**: Document with 150 clauses, batch size 15, 4 extraction types = 40 LLM calls

**Advantages**:
- **Balanced**: More thorough than full document, more efficient than clause-by-clause
- **Scalable**: Works well for medium to large documents
- **Context awareness**: LLM sees multiple related clauses together

**Disadvantages**:
- **Moderate cost**: More expensive than full document, cheaper than clause-by-clause
- **Batch boundary issues**: Related clauses may be split across batches

**When to Use**:
- Medium to large documents (50-200 clauses)
- Default mode for most extraction tasks
- When you want balance between thoroughness and efficiency

---

### Operative Clauses vs Definition Clauses

**Operative Clauses**:
- Clauses that establish **rights, obligations, procedures, or mechanisms**
- Examples: "Security Trustee Duties", "Payment Waterfall", "Events of Default"
- Rich source for **strategies, templates, and pitfalls**

**Definition Clauses**:
- Clauses that define **terms and concepts**
- Examples: "Interpretation", "Definitions", "Meanings of Terms"
- Primary source for **definitions** extraction type
- Often contain 50-200+ defined terms in a single clause

**Extraction Strategy**:
- Both clause types are treated as "operative" for extraction purposes
- Full document mode is particularly effective for definition clauses (can extract all 100+ definitions in one pass)
- Clause-by-clause mode gives each definition focused attention but is slower

---

### Why This Emphasis on Exhaustive Extraction?

LLMs have a natural tendency to **sample** (provide representative examples) rather than **enumerate** (list everything). The repeated emphasis on "EVERY", "ALL", "EXHAUSTIVE", "ZERO TOLERANCE" counteracts this tendency.

**Without emphasis**: LLM returns 10-15 "interesting" items
**With emphasis**: LLM returns 100-200+ comprehensive items

This is critical because the playbook must be **complete** to be reliable. Missing knowledge creates gaps in answer quality.

---

## Knowledge Classifier

**Location**: `trainer/prompts_trainer.py` → `CLASSIFICATION_GUIDELINES`

### Purpose
Validates that extracted knowledge is correctly categorized as strategies, definitions, templates, pitfalls, or code snippets.

### Classification Criteria

**Strategies**: How-to knowledge, patterns, approaches
- Example: "Security Trustee appointment requires explicit authorization..."

**Definitions**: What things mean, terminology
- Example: "Secured Property: All assets held by Security Trustee..."

**Templates**: Reusable structures with placeholders
- Example: "[Party A] appoints [Party B] to act as [Role]..."

**Pitfalls**: What to avoid, risks, anti-patterns
- Example: "Missing indemnification creates enforcement risk..."

### Why Classification Matters?
Correct categorization ensures knowledge is **stored in the right playbook section**, making it discoverable when needed.

---

## Knowledge Validator

**Location**: `trainer/prompts_trainer.py` → `VALIDATION_GUIDELINES`

### Purpose
Ensures extracted knowledge meets quality standards before adding to playbook.

### Validation Checks

**1. Quality**
- Minimum length: **≥20 characters**
- Proper formatting
- No placeholders (TODO, TBD, XXX)

**2. Consistency**
- No **contradictions** with existing knowledge
- Uses **domain-appropriate terminology**

**3. Accuracy**
- Proper **legal concepts**
- Correct **securitization mechanics**

**4. Completeness**
- Sufficient **context**
- Adequate **examples** when applicable

**5. Uniqueness**
- Not a **duplicate** of existing bullets

### Quality Gates
- Confidence **≥0.6**
- Length **≥20 characters**
- No placeholder text
- Proper sentence structure

### Why Validation Matters?
The playbook is the system's **source of truth**. Low-quality entries degrade answer quality. Validation ensures only high-value knowledge enters the playbook.

---

## Extraction Quality Guidelines

**Location**: `trainer/prompts_trainer.py` → `EXTRACTION_GUIDELINES`

### Principles

**1. Reusability**
Extract knowledge that applies across **multiple transactions**, not just one deal.

**2. Actionability**
Focus on what practitioners should **DO**, not just what exists.

**3. Specificity**
Include **concrete examples** and references.

**4. Context**
Preserve **jurisdictional and structural context**.

**5. Clarity**
Write in **clear, professional language**.

### What to Avoid
- **Transaction-specific details** (party names, amounts, dates)
- **Overly generic statements** without supporting evidence
- **Verbatim copying** without synthesis
- **Ambiguous or vague language**

### Why These Guidelines?
They ensure extracted knowledge is **practical, precise, and perpetually useful** rather than one-off facts.

---

## Summary: The Complete Prompt Ecosystem

### Core ACE Loop
1. **Generator** answers questions using playbook + general knowledge
2. **Reflector** compares answer to ground truth, extracts strategies/pitfalls
3. **Curator** integrates insights into playbook with ADD/MODIFY/REMOVE/MERGE operations

### Reformulation Modes
1. **Enrich**: Expand clause without changing meaning
2. **Derive**: Generate variants under constraints (with conditional pattern support)
3. **Remediate**: Fix non-compliant clauses
4. **Explore**: Respond to natural language reformulation requests

### Trainer Pipeline
1. **Extractor**: Pull ALL strategies/definitions/templates/pitfalls from documents (50-200+ items) using three granularity modes
2. **Classifier**: Validate correct categorization
3. **Validator**: Ensure quality, consistency, accuracy, completeness, uniqueness

### Design Philosophy

**System Prompts**: Define role, capabilities, output format, persistent rules
**User Prompts**: Deliver query-specific data (question, playbook, ground truth, constraints)