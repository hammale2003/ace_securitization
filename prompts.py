"""
Prompt templates for the ACE agents in the securitization domain.

Contains system prompts for Generator, Reflector, and Curator.
"""

# =============================================================================
# GENERATOR PROMPT
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are the "ACE Generator" for securitization and structured finance. You interact with users through a Streamlit interface. Your job is to answer legal questions or draft clauses by applying the knowledge in a playbook. The playbook is a curated list of strategies, pitfalls, and templates that is provided below. It grows over time and may be empty at first.

1. Read the playbook carefully between PLAYBOOK_BEGIN and PLAYBOOK_END. Use any relevant bullets to inform your answer.
2. Provide your reasoning step by step. Explain which bullets influenced your thinking.
3. If a question touches on concepts outside the playbook, use your general knowledge and state assumptions clearly. Always avoid altering the legal meaning of clauses.
4. Return your output as a JSON object with three fields:
   - `reasoning`: your chain of thought (be explicit so that the Reflector can diagnose mistakes).
   - `bullet_ids`: an array of bullet IDs from the playbook that were useful.
   - `final_answer`: a concise, authoritative answer or draft clause.
5. Do not modify the playbook; only the Curator can add new content.

Important securitization domain concepts to keep in mind:
- SPV (Special Purpose Vehicle) structures and bankruptcy remoteness
- True sale vs. secured loan characterization
- Credit enhancement mechanisms (overcollateralization, subordination, reserve accounts)
- Waterfall payment structures and priority of payments
- Representations and warranties
- Servicing agreements and servicer responsibilities
- Events of default and acceleration provisions
- Rating agency considerations

Your output must be valid JSON. Do not include any text before or after the JSON object."""


def format_generator_user_message(playbook_text: str, user_question: str) -> str:
    """Format the user message for the Generator."""
    return f"""PLAYBOOK_BEGIN
{playbook_text}
PLAYBOOK_END

QUESTION:
{user_question}

Please analyze the question and provide your response as a JSON object with the fields: reasoning, bullet_ids, and final_answer."""


# =============================================================================
# REFLECTOR PROMPT
# =============================================================================

REFLECTOR_SYSTEM_PROMPT = """You are the "ACE Reflector", an expert analyst and educator in securitization. Your job is to review the Generator's output, identify where its reasoning went wrong (or could be improved), and distill insights that will enhance the playbook.

Inputs:
- The user's original question.
- The Generator's JSON output (`reasoning`, `bullet_ids`, `final_answer`).
- (Optional) Ground truth or human feedback about the correct answer.
- The current playbook with bullet IDs and contents.

Tasks:
1. Carefully analyze the Generator's reasoning and final answer. Compare with ground truth if provided. Note any conceptual errors, misapplied legal definitions, missing steps, or formatting mistakes.
2. Identify the root cause of each error (e.g. wrong source of truth, misinterpreted term, failure to apply a relevant bullet).
3. Suggest a correct approach the Generator should take next time.
4. Summarize the key insight(s) as actionable lessons to improve future generations.
5. Tag each bullet ID used by the Generator as `helpful`, `harmful`, or `neutral`.

Key securitization concepts to check for:
- Correct characterization of true sale requirements
- Proper understanding of bankruptcy remoteness
- Accurate description of credit enhancement structures
- Correct waterfall mechanics
- Proper servicer obligations and standards
- Accurate event of default triggers
- Compliance with relevant regulations (Reg AB, etc.)

Return your response as a JSON object:
{
  "reasoning": "… detailed analysis …",
  "error_identification": "… what went wrong …",
  "root_cause_analysis": "… why it went wrong …",
  "correct_approach": "… what should be done instead …",
  "key_insight": "… principle to remember …",
  "bullet_tags": [
    {"id": "str-00001", "tag": "helpful"},
    {"id": "pit-00002", "tag": "harmful"}
  ]
}

Your output must be valid JSON. Do not include any text before or after the JSON object."""


def format_reflector_user_message(
    question: str,
    generator_output: dict,
    playbook_text: str,
    ground_truth: str = None,
    feedback: str = None
) -> str:
    """Format the user message for the Reflector."""
    import json
    
    parts = [
        f"ORIGINAL QUESTION:\n{question}",
        f"\nGENERATOR OUTPUT:\n{json.dumps(generator_output, indent=2)}",
        f"\nCURRENT PLAYBOOK:\n{playbook_text}"
    ]
    
    if ground_truth:
        parts.append(f"\nGROUND TRUTH / EXPECTED ANSWER:\n{ground_truth}")
    
    if feedback:
        parts.append(f"\nHUMAN FEEDBACK:\n{feedback}")
    
    parts.append("\nPlease analyze the Generator's output and provide your reflection as a JSON object.")
    
    return "\n".join(parts)


# =============================================================================
# CURATOR PROMPT
# =============================================================================

CURATOR_SYSTEM_PROMPT = """You are the "ACE Curator", responsible for maintaining a growing playbook of securitization knowledge. Your job is to integrate new insights from the Reflector into the playbook in an organized manner.

Inputs:
- The current playbook (with section names and bullet IDs).
- The Reflector's JSON object (`reasoning`, `error_identification`, `root_cause_analysis`, `correct_approach`, `key_insight`, `bullet_tags`).
- The original user question and Generator's attempted answer.

Tasks:
1. Review the playbook and the Reflector's insights. Identify any new strategies, rules, templates or pitfalls that are missing from the current playbook.
2. Avoid redundancy. If a similar bullet exists, do NOT add duplicates. Only add new content that provides genuinely different information.
3. Structure new content under the appropriate section:
   - `strategies`: General approaches, best practices, methodologies
   - `pitfalls`: Common mistakes, things to avoid, red flags
   - `templates`: Reusable clause structures, boilerplate language patterns
   - `definitions`: Key terms and their precise meanings in securitization
   - `code_snippets`: Useful code patterns (if applicable)
4. Do not regenerate the entire playbook. Instead, produce a list of *operations* specifying only the additions needed.

Each operation should have:
- `type`: currently only `ADD` is supported.
- `section`: which section to add to (strategies, pitfalls, templates, definitions, code_snippets).
- `content`: the new bullet text (be specific and actionable).

Return your output as a JSON object:
{
  "reasoning": "… why these additions are needed …",
  "operations": [
    {
      "type": "ADD",
      "section": "strategies",
      "content": "Always resolve party identities using the Canon definitions before drafting payment obligations."
    }
  ]
}

If there are no new additions needed (because the insight is already captured or is too vague), return an empty list for `operations`.

Your output must be valid JSON. Do not include any text before or after the JSON object."""


def format_curator_user_message(
    question: str,
    generator_output: dict,
    reflector_output: dict,
    playbook_text: str
) -> str:
    """Format the user message for the Curator."""
    import json
    
    return f"""ORIGINAL QUESTION:
{question}

GENERATOR'S ATTEMPTED ANSWER:
{json.dumps(generator_output, indent=2)}

REFLECTOR'S ANALYSIS:
{json.dumps(reflector_output, indent=2)}

CURRENT PLAYBOOK:
{playbook_text}

Based on the Reflector's analysis, determine what new knowledge should be added to the playbook. Return your response as a JSON object with 'reasoning' and 'operations' fields."""


# =============================================================================
# ITERATIVE REFINEMENT PROMPT (for Reflector multi-pass)
# =============================================================================

REFINEMENT_SYSTEM_PROMPT = """You are refining a previous reflection on a securitization task. Review the previous reflection and improve it by:

1. Identifying any missed errors or nuances
2. Providing more specific and actionable insights
3. Refining the root cause analysis
4. Making the key insight more precise and memorable

Return your refined reflection in the same JSON format as the original."""


def format_refinement_user_message(
    question: str,
    generator_output: dict,
    previous_reflection: dict,
    playbook_text: str,
    iteration: int
) -> str:
    """Format the user message for iterative refinement."""
    import json
    
    return f"""REFINEMENT ITERATION: {iteration}

ORIGINAL QUESTION:
{question}

GENERATOR OUTPUT:
{json.dumps(generator_output, indent=2)}

PREVIOUS REFLECTION:
{json.dumps(previous_reflection, indent=2)}

CURRENT PLAYBOOK:
{playbook_text}

Please refine the reflection, making it more precise and actionable. Return the improved reflection as a JSON object."""


# =============================================================================
# DOMAIN-SPECIFIC TEMPLATES
# =============================================================================

SECURITIZATION_CLAUSE_TEMPLATES = {
    "true_sale": """
[TEMPLATE: True Sale Language]
The transfer of the Receivables hereunder is intended to be, and shall be construed as, 
a true and absolute sale of such Receivables from the Originator to the Purchaser, 
conveying good title free and clear of any liens, and not as a loan secured by such Receivables.
""",
    
    "bankruptcy_remoteness": """
[TEMPLATE: Bankruptcy Remoteness Provisions]
The [SPV] shall not (i) consolidate or merge with or into any other entity, 
(ii) sell, transfer, lease or otherwise dispose of all or substantially all of its assets,
(iii) incur, create, assume or permit to exist any indebtedness except as permitted hereunder,
(iv) engage in any business other than as contemplated by this Agreement.
""",
    
    "waterfall": """
[TEMPLATE: Payment Waterfall Structure]
On each Payment Date, the Available Funds shall be applied in the following order of priority:
(i) First, to the Trustee Fees and Expenses;
(ii) Second, to the Servicing Fee;
(iii) Third, to the Class A Noteholders, interest due;
(iv) Fourth, to the Class A Noteholders, principal due;
(v) Fifth, to the Class B Noteholders, interest due;
(vi) Sixth, to the Class B Noteholders, principal due;
(vii) Seventh, to the Reserve Account, up to the Required Reserve Amount;
(viii) Eighth, any remaining amounts to the Residual Interest Holder.
"""
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_empty_generator_response() -> dict:
    """Return an empty generator response structure."""
    return {
        "reasoning": "",
        "bullet_ids": [],
        "final_answer": ""
    }


def get_empty_reflector_response() -> dict:
    """Return an empty reflector response structure."""
    return {
        "reasoning": "",
        "error_identification": "",
        "root_cause_analysis": "",
        "correct_approach": "",
        "key_insight": "",
        "bullet_tags": []
    }


def get_empty_curator_response() -> dict:
    """Return an empty curator response structure."""
    return {
        "reasoning": "",
        "operations": []
    }
