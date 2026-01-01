# ACORD: ACET Evaluation Framework for Clause Extraction

A self-contained implementation of the **ACET (Agentic Context Engineering)** framework specifically designed for evaluating legal clause extraction on the **ACORD (Atticus Clause Retrieval Dataset)** benchmark.

## Overview

This framework implements the three-agent ACET architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     ACET PIPELINE                           │
│                                                             │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │ GENERATOR │───▶│ REFLECTOR │───▶│  CURATOR  │          │
│  │           │    │           │    │           │          │
│  │ Extracts  │    │ Analyzes  │    │ Updates   │          │
│  │ clauses   │    │ results   │    │ playbook  │          │
│  │ using     │    │ vs ground │    │ with new  │          │
│  │ playbook  │    │ truth     │    │ knowledge │          │
│  └───────────┘    └───────────┘    └───────────┘          │
│        │                                  │                │
│        ▼                                  ▼                │
│  ┌───────────┐                    ┌───────────┐          │
│  │ EXTRACTED │                    │ EVOLVING  │          │
│  │  CLAUSE   │                    │ PLAYBOOK  │          │
│  └───────────┘                    └───────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Self-contained**: No external dependencies on other ACET modules
- **Multiple LLM Providers**: OpenAI, Anthropic, Google, or Mock (for testing)
- **Evolving Playbook**: Learns strategies, definitions, pitfalls from training
- **Comprehensive Metrics**: Token F1, Exact Match, Precision, Recall
- **Synthetic Data**: Built-in synthetic dataset for testing without real data
- **CUAD Compatible**: Can load real CUAD/ACORD datasets

## Installation

```bash
# Clone/copy the ACORD folder to your project
cd your_project
cp -r /path/to/ACORD .

# Install dependencies
pip install numpy

# For LLM providers (install as needed):
pip install openai      # For OpenAI
pip install anthropic   # For Anthropic
pip install google-generativeai  # For Google
```

## Quick Start

### Demo with Mock LLM (No API Key Required)

```bash
python run.py --synthetic --provider mock --epochs 2
```

### With OpenAI

```bash
export OPENAI_API_KEY="your-key-here"
python run.py --synthetic --provider openai --model gpt-4 --epochs 3
```

### With Real CUAD Data

```bash
python run.py --cuad path/to/CUADv1.json --provider openai --epochs 5
```

## Usage in Code

### Basic Clause Extraction

```python
from ACORD import ACORDPipeline

# Create pipeline
pipeline = ACORDPipeline()

# Extract a clause
contract = """
This Agreement may be terminated by either party 
upon thirty (30) days prior written notice...
"""

result = pipeline.generate(contract, "Termination For Convenience")
print(result.final_answer)
print(f"Confidence: {result.confidence}")
```

### Full Evaluation Pipeline

```python
from ACORD import (
    ACORDConfig, LLMConfig,
    ACORDPipeline, ACORDEvaluator,
    create_synthetic_dataset
)

# Configure
config = ACORDConfig(
    llm=LLMConfig(provider="openai", model="gpt-4")
)

# Create data
dataset = create_synthetic_dataset(num_samples=100)
train, val, test = dataset.split()

# Initialize
pipeline = ACORDPipeline(config)
evaluator = ACORDEvaluator(config, pipeline)

# Baseline evaluation
baseline = evaluator.evaluate(test)
print(f"Baseline F1: {baseline.overall_token_f1:.4f}")

# Training
for epoch in range(3):
    metrics = evaluator.train_epoch(train, epoch)
    print(f"Epoch {epoch}: F1={metrics.avg_token_f1:.4f}")

# Final evaluation
final = evaluator.evaluate(test)
print(f"Final F1: {final.overall_token_f1:.4f}")
print(f"Improvement: {final.overall_token_f1 - baseline.overall_token_f1:+.4f}")
```

### Loading Real Data

```python
from ACORD import ACORDLoader

# From CUAD format
dataset = ACORDLoader.from_cuad_json("CUADv1.json")

# From custom JSON
dataset = ACORDLoader.from_json("my_data.json")

# From CSV
dataset = ACORDLoader.from_csv("data.csv", 
    text_col="contract", 
    clause_type_col="type",
    passage_col="answer"
)
```

## File Structure

```
ACORD/
├── __init__.py      # Package exports
├── config.py        # Configuration classes
├── llm_client.py    # LLM provider interfaces
├── playbook.py      # Evolving knowledge storage
├── agents.py        # Generator, Reflector, Curator
├── pipeline.py      # Main orchestration
├── dataset.py       # Data loading and synthetic generation
├── evaluator.py     # Metrics and evaluation
├── run.py           # CLI entry point
└── README.md        # This file
```

## Configuration Options

### LLM Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | "openai" | LLM provider: openai, anthropic, google, mock |
| `model` | "gpt-4" | Model name |
| `temperature` | 0.0 | Generation temperature |
| `max_tokens` | 2000 | Maximum tokens |

### Playbook Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | "playbook.json" | Playbook file path |
| `max_bullets_per_section` | 100 | Max bullets per section |
| `dedup_similarity_threshold` | 0.85 | Duplicate detection threshold |

### Evaluation Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_contract_length` | 8000 | Max characters to include |
| `include_negative_samples` | True | Include samples without target clause |

## Supported Clause Types

The framework supports all 41 CUAD clause types:

- Termination For Convenience
- Governing Law
- Cap On Liability
- License Grant
- IP Ownership Assignment
- Non-Compete
- Change Of Control
- And 34 more...

## Output Files

After running evaluation:

```
results/
├── baseline.json         # Baseline metrics
├── final.json           # Final metrics with learning curve
└── learned_playbook.json # Updated playbook
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Token F1** | Token-level overlap F1 score |
| **Exact Match** | Exact string match rate |
| **Precision** | Token precision |
| **Recall** | Token recall |

## API Reference

### ACORDPipeline

```python
pipeline = ACORDPipeline(config)

# Generation only (no learning)
output = pipeline.generate(contract_text, clause_type)

# Full pipeline with learning
result = pipeline.run(contract_text, clause_type, ground_truth="...")

# Playbook management
stats = pipeline.get_playbook_stats()
pipeline.reset_playbook(use_initial=True)
pipeline.save_playbook("path.json")
```

### ACORDEvaluator

```python
evaluator = ACORDEvaluator(config, pipeline)

# Evaluate
report = evaluator.evaluate(test_dataset)

# Train
metrics = evaluator.train_epoch(train_dataset, epoch=0)

# Reset
evaluator.reset(reset_playbook=True)
```

## License

MIT License - See LICENSE file for details.
