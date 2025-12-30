# Playbook Enricher

Document-based knowledge extraction integrated with ACE framework for intelligent playbook enrichment.

## Architecture

```
Document → Extractor → ACE Evaluation → Playbook
                       (Generator, Reflector, Curator)
```

## Key Features

1. **ACE Integration**: Uses Generator, Reflector, and Curator for quality control
2. **Noise Reduction**: Only adds transaction-specific, reusable knowledge
3. **Intelligent Deduplication**: ACE framework checks for existing coverage
4. **Two Granularity Levels**:
   - OPERATIVE_CLAUSE_BY_CLAUSE: Highest accuracy
   - FULL_DOCUMENT: Faster processing

## Usage

### Basic Enrichment

```python
from ace_securitization import (
    EnrichmentPipeline,
    EnrichmentConfig,
    PlaybookManager,
    GranularityLevel
)

# Configure
config = EnrichmentConfig(
    granularity_level=GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE
)

# Initialize
playbook_manager = PlaybookManager()
enricher = EnrichmentPipeline(config, playbook_manager)

# Run enrichment
result = enricher.run_from_file("master_framework_agreement.json")

print(result.get_summary())
# Output:
# Playbook Enrichment Complete
# 
# Document: Master Framework Agreement
# Type: Legal Agreement
# 
# Results:
#   Extracted: 45 items
#   Processed: 45 items
#   Added: 12 bullets
#   Skipped: 33 items (duplicates/low quality)
```

### Preview Before Enrichment

```python
# Parse document
from ace_securitization import DocumentParser

parser = DocumentParser()
document = parser.parse_json_file("document.json")

# Preview
preview = enricher.get_extraction_preview(document, max_items=10)

print(f"Would extract {preview['total_extracted']} items")
for item in preview['preview_items']:
    print(f"- {item['content'][:100]}...")
```

### Granularity Comparison

```python
# Clause-by-clause (highest quality)
config = EnrichmentConfig(
    granularity_level=GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE
)
# Processes each operative clause individually
# Best for: Critical documents, maximum accuracy

# Full document (faster)
config = EnrichmentConfig(
    granularity_level=GranularityLevel.FULL_DOCUMENT
)
# Processes entire document in one pass
# Best for: Quick overviews, smaller documents
```

## How It Works

### 1. Extraction Phase

The extractor identifies transaction-specific knowledge:

```
Input: Document clause
Output: Raw knowledge items

Criteria:
- Specific to this transaction structure
- Reusable for similar transactions
- Not generic legal principles
```

### 2. ACE Evaluation Phase

Each extracted item goes through ACE:

```python
# Generator evaluates specificity
question = "Is this knowledge specific and reusable?"
generator_output = generator.generate(question, playbook)

# Reflector assesses quality
reflector_output = reflector.reflect(
    question, generator_output, playbook
)

# Curator decides whether to add
curator_output = curator.curate(
    question, generator_output, reflector_output, playbook
)
```

### 3. Application Phase

Only curator-approved knowledge is added:

```python
if curator_output.operations:
    for op in curator_output.operations:
        if op["type"] == "ADD":
            playbook.add_bullet(op["section"], op["content"])
```

## Differences from Old Trainer

| Aspect | Old Trainer | New Enricher |
|--------|-------------|--------------|
| Quality Control | Simple validation | ACE framework |
| Deduplication | Embedding similarity | Semantic understanding |
| Decision Making | Rule-based | AI-driven evaluation |
| Noise Level | High (bulk adds) | Low (selective) |
| Granularity | 3 levels | 2 levels (simplified) |
| Prompts | Separated | Integrated with ACE |

## Document Format

Expected JSON structure:

```json
{
  "document": {
    "uid": "DOC_UID-0",
    "document_type": "Master Framework Agreement",
    "title_text": "Master Definitions Agreement",
    "clauses": [
      {
        "uid": "CLAUSE_UID-1",
        "level": 0,
        "position": 1,
        "title_text": "Definitions",
        "body_text": "...",
        "metadata": {
          "type": "operative_clause"
        },
        "clauses": [...]
      }
    ]
  }
}
```

Clause types:
- `operative_clause`: Substantive provisions
- `definition_clause`: Term definitions
- `structural_clause`: Headers (ignored)
- `subsidiary_clause`: Sub-provisions

## Configuration

```python
@dataclass
class EnrichmentConfig:
    granularity_level: GranularityLevel = GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE
    llm_config: Optional[LLMConfig] = None
```

## Integration with ACE Pipeline

The enricher complements the interactive ACE pipeline:

```python
# Phase 1: Bootstrap with enricher
enricher.run_from_file("master_agreement.json")
# Result: 50-100 transaction-specific bullets

# Phase 2: Refine with interactive training
from ace_securitization import ACEPipeline

pipeline = ACEPipeline()
for qa in training_questions:
    pipeline.run(question=qa['question'], ground_truth=qa['answer'])
# Result: Playbook learns from usage
```

## Best Practices

1. **Start with clause-by-clause** for critical documents
2. **Preview first** to estimate extraction volume
3. **Review added bullets** after enrichment
4. **Use with retriever** for large playbooks
5. **Combine with interactive training** for best results

