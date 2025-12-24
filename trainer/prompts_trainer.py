"""
Prompts for Trainer Mode agents.

Contains system and user prompts for knowledge extraction,
classification, and validation.
"""

# Knowledge Extractor Prompts are defined in knowledge_extractor.py
# This file can be extended with additional prompts if needed

EXTRACTION_GUIDELINES = """
EXTRACTION QUALITY GUIDELINES:

1. **Reusability**: Extract knowledge that applies across multiple transactions
2. **Actionability**: Focus on what practitioners should DO, not just what exists
3. **Specificity**: Include concrete examples and references
4. **Context**: Preserve jurisdictional and structural context
5. **Clarity**: Write in clear, professional language

AVOID:
- Transaction-specific details (party names, amounts, dates)
- Overly generic statements without supporting evidence
- Verbatim copying without synthesis
- Ambiguous or vague language
"""

CLASSIFICATION_GUIDELINES = """
CLASSIFICATION CRITERIA:

**Strategies**: How-to knowledge, patterns, approaches
- Example: "Security Trustee appointment requires explicit authorization..."

**Definitions**: What things mean, terminology
- Example: "Secured Property: All assets held by Security Trustee..."

**Templates**: Reusable structures with placeholders
- Example: "[Party A] appoints [Party B] to act as [Role]..."

**Pitfalls**: What to avoid, risks, anti-patterns
- Example: "Missing indemnification creates enforcement risk..."

**Code Snippets**: Formulas, calculations, logic
- Example: "Effectiveness = (helpful - harmful) / max(total, 1)"
"""

VALIDATION_GUIDELINES = """
VALIDATION CHECKS:

1. **Quality**: Minimum length, proper formatting, no placeholders
2. **Consistency**: No contradictions with existing knowledge
3. **Accuracy**: Domain-appropriate terminology and concepts
4. **Completeness**: Sufficient context and examples
5. **Uniqueness**: Not duplicate of existing bullets

QUALITY GATES:
- Confidence >= 0.6
- Length >= 20 characters
- No placeholder text (TODO, TBD, etc.)
- Proper sentence structure
"""

