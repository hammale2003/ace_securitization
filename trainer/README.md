# Trainer Mode

Trainer Mode is a knowledge extraction and enrichment pipeline for the ACE Securitization System. It automatically extracts structured knowledge from real securitization documents (Master Framework Agreements, etc.) and enriches the playbook.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINER MODE PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DOCUMENT INGESTION (document_parser.py)                 │
│     ├─ Parse JSON structure                                 │
│     ├─ Extract clauses hierarchy                            │
│     └─ Identify key sections                                │
│                                                             │
│  2. KNOWLEDGE EXTRACTOR (knowledge_extractor.py)            │
│     ├─ Identify strategies (patterns, approaches)           │
│     ├─ Extract definitions (terms, concepts)                │
│     ├─ Capture templates (clause structures)                │
│     ├─ Detect pitfalls (anti-patterns, risks)               │
│     └─ Find code snippets (formulas, calculations)          │
│                                                             │
│  3. KNOWLEDGE CLASSIFIER (knowledge_classifier.py)          │
│     ├─ Categorize extracted knowledge                       │
│     ├─ Assign confidence scores                             │
│     ├─ Detect duplicates with existing playbook             │
│     └─ Prioritize by importance                             │
│                                                             │
│  4. KNOWLEDGE VALIDATOR (knowledge_validator.py)            │
│     ├─ Check consistency with existing playbook             │
│     ├─ Validate legal/domain accuracy                       │
│     ├─ Resolve conflicts                                    │
│     └─ Quality assurance                                    │
│                                                             │
│  5. PLAYBOOK ENRICHER (playbook_enricher.py)                │
│     ├─ Bulk ADD operations                                  │
│     ├─ MERGE with existing knowledge                        │
│     ├─ Update definitions                                   │
│     └─ Cross-reference bullets                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Document Parser (`document_parser.py`)
Parses minified JSON documents and extracts hierarchical clause structure.

**Key Classes:**
- `ParsedDocument`: Represents a parsed document with metadata
- `ParsedClause`: Represents a single clause with sub-clauses
- `DocumentParser`: Main parser class

### 2. Knowledge Extractor (`knowledge_extractor.py`)
Extracts structured knowledge using LLM analysis.

**Extraction Types:**
- **Strategies**: Patterns and best practices
- **Definitions**: Terms and concepts
- **Templates**: Reusable clause structures
- **Pitfalls**: Anti-patterns and risks
- **Code Snippets**: Formulas and calculations

### 3. Knowledge Classifier (`knowledge_classifier.py`)
Classifies and scores extracted knowledge.

**Functions:**
- Section validation
- Confidence adjustment
- Importance scoring
- Duplicate detection

### 4. Knowledge Validator (`knowledge_validator.py`)
Validates knowledge before enrichment.

**Validation Checks:**
- Quality checks (length, formatting, completeness)
- Consistency checks (no contradictions)
- Accuracy checks (domain terminology)
- Conflict detection

### 5. Playbook Enricher (`playbook_enricher.py`)
Applies validated knowledge to playbook.

**Operations:**
- Bulk ADD for new knowledge
- MERGE for similar knowledge
- Metadata enrichment
- Cross-referencing

### 6. Trainer Pipeline (`trainer_pipeline.py`)
Main orchestrator that coordinates all agents.

## Usage

### Via API

```python
import requests

# Initialize pipeline first
requests.post("http://localhost:8000/initialize", json={
    "llm_config": {"provider": "openai", "model": "gpt-4"},
    "playbook_path": "playbook.json"
})

# Enrich from document
response = requests.post("http://localhost:8000/trainer/enrich", json={
    "json_document": '{"uid":"DOC_UID-0",...}',  # Minified JSON
    "extraction_types": ["strategies", "definitions", "templates", "pitfalls"],
    "min_confidence": 0.5,
    "preview_only": False
})

print(response.json())
```

### Via Streamlit App

1. Initialize pipeline in sidebar
2. Navigate to "Trainer Mode" page
3. Paste minified JSON document
4. Configure extraction settings
5. Click "Preview" to see what will be extracted
6. Click "Enrich Playbook" to apply changes

### Programmatic Usage

```python
from trainer import TrainerPipeline, TrainerConfig
from playbook import PlaybookManager
from config import LLMConfig

# Setup
llm_config = LLMConfig(provider="openai", model="gpt-4")
trainer_config = TrainerConfig(
    llm_config=llm_config,
    extraction_types=["strategies", "definitions", "templates", "pitfalls"],
    min_extraction_confidence=0.5
)

playbook_manager = PlaybookManager()
trainer = TrainerPipeline(trainer_config, playbook_manager)

# Run enrichment
result = trainer.run_from_file("master_framework_agreement.json")

print(result.get_summary())
print(f"Added {result.enrichment_result.total_added} bullets")
```

## Configuration

### Granularity Levels

Trainer Mode supports three granularity levels for document processing:

#### 1. **OPERATIVE_CLAUSE_BY_CLAUSE** 🔬
- **Description**: Process each operative clause individually
- **Accuracy**: ⭐⭐⭐ (Highest)
- **Cost**: 💰💰💰 (Highest)
- **Best For**: Critical documents requiring maximum accuracy, legal compliance checks
- **API Calls**: N clauses × M extraction types

```python
trainer_config.granularity_level = GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE
```

#### 2. **BATCH** ⚡ (Default)
- **Description**: Process clauses in batches (default: 15 per batch)
- **Accuracy**: ⭐⭐ (Balanced)
- **Cost**: 💰💰 (Moderate)
- **Best For**: Most use cases - good balance of accuracy and cost
- **API Calls**: (N clauses / batch_size) × M extraction types

```python
trainer_config.granularity_level = GranularityLevel.BATCH
trainer_config.batch_size = 15  # Configurable
```

#### 3. **FULL_DOCUMENT** 🚀
- **Description**: Process entire document in one LLM call
- **Accuracy**: ⭐ (Lower - may miss details)
- **Cost**: 💰 (Lowest)
- **Best For**: Quick overviews, small documents, initial exploration
- **API Calls**: M extraction types (one per type)

```python
trainer_config.granularity_level = GranularityLevel.FULL_DOCUMENT
```

### TrainerConfig

```python
@dataclass
class TrainerConfig:
    # Extraction settings
    extraction_types: List[str] = ["strategies", "definitions", "templates", "pitfalls", "code_snippets"]
    min_extraction_confidence: float = 0.5
    
    # Granularity settings
    granularity_level: GranularityLevel = GranularityLevel.BATCH  # Processing granularity
    batch_size: int = 15  # For BATCH mode only
    
    # Classification settings
    duplicate_threshold: float = 0.85
    
    # Validation settings
    min_validation_score: float = 0.6
    skip_invalid: bool = True
    
    # Enrichment settings
    auto_merge_threshold: float = 0.75
```

## Document Format

Trainer Mode expects minified JSON documents with this structure:

```json
{
  "uid": "DOC_UID-0",
  "document_type": "Master Framework Agreement",
  "title_text": "Master Framework Agreement",
  "clauses": [
    {
      "uid": "CLAUSE_UID-1",
      "level": 1,
      "position": 1,
      "title_text": "Definitions",
      "body_text": "In this Agreement...",
      "metadata": {"type": "definitions"},
      "clauses": [...]
    }
  ]
}
```

## Output Format

Enriched bullets follow the standard playbook format:

```json
{
  "id": "str-00001",
  "content": "Security Trustee appointment requires explicit authorization...",
  "helpful_count": 0,
  "harmful_count": 0,
  "neutral_count": 0,
  "created_at": "2025-12-19T22:00:00.000000",
  "updated_at": "2025-12-19T22:00:00.000000",
  "revision_count": 0,
  "archived": false
}
```

## Best Practices

1. **Start with Preview**: Always preview before enriching to see what will be extracted
2. **Adjust Confidence**: Lower confidence threshold to extract more, raise to be more selective
3. **Review Skipped Items**: Check why items were skipped to tune extraction
4. **Incremental Enrichment**: Enrich one document at a time to maintain control
5. **Monitor Duplicates**: High duplicate rate may indicate over-extraction

## Troubleshooting

### No Knowledge Extracted
- Check document format (must be valid JSON)
- Lower `min_extraction_confidence` threshold
- Verify document contains substantive clauses (not just structure)

### Too Many Duplicates
- Increase `duplicate_threshold` in config
- Review existing playbook for redundancy
- Consider using MERGE instead of ADD

### Low Quality Extractions
- Increase `min_extraction_confidence`
- Increase `min_validation_score`
- Review extraction prompts in `knowledge_extractor.py`

## Future Enhancements

- [ ] Multi-document synthesis
- [ ] Incremental learning from updates
- [ ] Knowledge graph construction
- [ ] Active learning for gap identification
- [ ] Human-in-the-loop review queue
- [ ] PDF document support
- [ ] Cross-document conflict detection

