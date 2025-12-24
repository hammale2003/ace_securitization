# ACE Securitization System

**Agentic Context Engineering for Securitization and Structured Finance**

This system implements the ACE (Agentic Context Engineering) framework for building self-improving AI systems in the securitization/legal domain. Based on the research paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models".

## Features

- **Three-Agent Architecture**: Generator, Reflector, and Curator work together to process questions and evolve knowledge
- **Evolving Playbook**: Accumulates domain-specific knowledge over time
- **Multiple Interfaces**: Streamlit web UI and FastAPI REST API
- **Multi-Provider Support**: Works with OpenAI, Anthropic Claude, and Google Gemini
- **Streaming Support**: Real-time token streaming for responsive UX
- **Incremental Updates**: Delta-based playbook updates prevent context collapse

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Question                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Generator                               │
│  • Reads playbook                                                │
│  • Produces answer with reasoning                                │
│  • References relevant bullets                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Reflector                               │
│  • Analyzes Generator output                                     │
│  • Identifies errors and root causes                            │
│  • Extracts key insights                                        │
│  • Tags bullets as helpful/harmful                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                           Curator                                │
│  • Reviews reflection insights                                   │
│  • Determines new knowledge to add                              │
│  • Avoids redundancy                                            │
│  • Updates playbook with delta operations                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Playbook                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Strategies│  │ Pitfalls │  │Templates │  │Definitions│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key (choose your provider)
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
# or
export GOOGLE_API_KEY="your-key-here"
```

## Quick Start

### Python API

```python
from ace_securitization import ACEPipeline, ACEConfig, LLMConfig

# Initialize with default settings
pipeline = ACEPipeline()

# Or customize configuration
config = ACEConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.0
    )
)
pipeline = ACEPipeline(config)

# Ask a question (full pipeline with learning)
result = pipeline.run("What are the key requirements for a true sale in securitization?")
print(result.generator_output.final_answer)

# Or just generate without learning
output = pipeline.generate_only("Define overcollateralization")
print(output.final_answer)
```

### Streamlit Web UI

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### FastAPI REST API

```bash
python api.py
```

API will be available at http://localhost:8000. See `/docs` for Swagger UI.

**Example API calls:**

```bash
# Initialize pipeline
curl -X POST http://localhost:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{"llm_config": {"provider": "openai", "model": "gpt-4"}}'

# Ask a question
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a waterfall structure?"}'

# Get playbook
curl http://localhost:8000/playbook
```

## Project Structure

```
ace_framwork/
├── __init__.py          # Package exports
├── config.py            # Configuration classes
├── playbook.py          # Playbook data model and manager
├── llm_client.py        # LLM provider abstraction
├── prompts.py           # Agent prompts
├── agents.py            # Generator, Reflector, Curator agents
├── app.py               # Streamlit web interface
├── api.py               # FastAPI REST API
├── utils.py             # Utility functions
├── tests.py             # Unit tests
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |

### LLM Configuration

```python
from ace_securitization import LLMConfig

config = LLMConfig(
    provider="openai",      # openai, anthropic, google, mock
    model="gpt-4",          # Model name
    temperature=0.0,        # 0-1, lower = more deterministic
    max_tokens=4096,        # Max response tokens
    stream=True,            # Enable streaming
    api_key=None            # Override env variable
)
```

### ACE Configuration

```python
from ace_securitization import ACEConfig, LLMConfig, PlaybookConfig

config = ACEConfig(
    llm=LLMConfig(...),
    playbook=PlaybookConfig(
        path="playbook.json",
        max_bullets_per_section=100,
        dedup_similarity_threshold=0.85
    ),
    max_reflector_iterations=5,
    enable_streaming=True,
    verbose=True
)
```

## Playbook Structure

The playbook is a JSON file with the following structure:

```json
{
  "strategies": [
    {
      "id": "str-00001",
      "content": "Always verify true sale requirements...",
      "helpful_count": 5,
      "harmful_count": 0,
      "neutral_count": 2
    }
  ],
  "pitfalls": [...],
  "templates": [...],
  "definitions": [...],
  "code_snippets": [...],
  "metadata": {
    "version": "1.0",
    "created_at": "...",
    "updated_at": "...",
    "total_updates": 42
  }
}
```

## Training

To train the system on a dataset:

1. Create a JSON file with questions and answers:

```json
[
  {
    "question": "What is a true sale?",
    "answer": "A true sale is a legal characterization..."
  },
  {
    "question": "Define overcollateralization",
    "answer": "Overcollateralization is..."
  }
]
```

2. Run training using the Python API:

```python
from ace_securitization import ACEPipeline
import json

pipeline = ACEPipeline()

# Load training data
with open('training_data.json', 'r') as f:
    training_data = json.load(f)

# Train on each question
for item in training_data:
    result = pipeline.run(item['question'])
    print(f"Learned from: {item['question']}")
```

The playbook will grow as the system learns from each question.

## Testing

```bash
# Run all tests
pytest tests.py -v

# Run specific test class
pytest tests.py::TestPlaybook -v

# Run with coverage
pytest tests.py --cov=. --cov-report=html
```

## API Reference

### ACEPipeline

Main class for running the ACE system.

```python
pipeline = ACEPipeline(config=None)

# Full pipeline (generates, reflects, curates)
result = pipeline.run(
    question="...",
    ground_truth=None,    # Optional correct answer
    feedback=None,        # Optional human feedback
    stream_callbacks={}   # Optional streaming callbacks
)

# Generate only (no learning)
output = pipeline.generate_only(question, stream_callback=None)

# Get playbook
playbook = pipeline.get_playbook()
stats = pipeline.get_playbook_stats()
```

### ACEPipelineResult

Result from running the full pipeline.

```python
result.question              # Original question
result.generator_output      # GeneratorOutput
result.reflector_output      # ReflectorOutput  
result.curator_output        # CuratorOutput
result.added_bullets         # List of new Bullet objects
result.playbook_stats        # Dict with stats
result.timestamp             # ISO timestamp
```

### GeneratorOutput

```python
output.reasoning      # Chain of thought
output.bullet_ids     # List of bullet IDs used
output.final_answer   # The generated answer
```

### ReflectorOutput

```python
output.reasoning             # Analysis
output.error_identification  # What went wrong
output.root_cause_analysis   # Why it went wrong
output.correct_approach      # What should be done
output.key_insight          # Lesson learned
output.bullet_tags          # List of {id, tag} dicts
```

### CuratorOutput

```python
output.reasoning    # Why additions are needed
output.operations   # List of {type, section, content} dicts
```

## References

- Paper: "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"
- Dynamic Cheatsheet: https://github.com/suzgunmirac/dynamic-cheatsheet
