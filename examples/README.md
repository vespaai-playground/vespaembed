# VespaEmbed Examples

This folder contains examples for each training task supported by VespaEmbed.

## Tasks Overview

| Task | Description | Required Columns |
|------|-------------|------------------|
| `pairs` | Query-document pairs with in-batch negatives (MNR loss) | `anchor`, `positive` |
| `triplets` | Explicit anchor-positive-negative triplets | `anchor`, `positive`, `negative` |
| `similarity` | Sentence pairs with continuous similarity scores | `sentence1`, `sentence2`, `score` |
| `tsdae` | Unsupervised domain adaptation with denoising auto-encoder | `text` |

**Note:** Matryoshka embeddings are an option (`--matryoshka-dims`) that can be combined with any task.

## Quick Start

### Using the Web UI

```bash
# Start the web UI
vespaembed

# Open http://localhost:8000 in your browser
```

### Using the CLI

```bash
# Train with a local file
vespaembed train --task pairs --data examples/data/pairs.csv --base-model sentence-transformers/all-MiniLM-L6-v2

# Train with a HuggingFace dataset
vespaembed train --task pairs --data sentence-transformers/gooaq --split train[:10000] --base-model sentence-transformers/all-MiniLM-L6-v2

# Train with YAML config
vespaembed train --config examples/configs/pairs.yaml
```

### Using Python API

```python
from vespaembed.core.config import TrainingConfig, DataConfig
from vespaembed.core.trainer import VespaEmbedTrainer

config = TrainingConfig(
    task="pairs",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    data=DataConfig(train="examples/data/pairs.csv"),
)

trainer = VespaEmbedTrainer(config)
trainer.train()
```

## Example Files

| File | Task | Description |
|------|------|-------------|
| `pairs_example.py` | pairs | Query-document pairs for semantic search |
| `triplets_example.py` | triplets | Anchor-positive-negative triplet training |
| `similarity_example.py` | similarity | Semantic textual similarity (STS) |
| `tsdae_example.py` | tsdae | Unsupervised domain adaptation |
| `matryoshka_example.py` | pairs + matryoshka | Multi-dimensional embeddings |

Each example includes:
- CLI usage examples
- Python API examples with minimal, advanced, LoRA, and Unsloth configurations
- Evaluation code

## Sample Data

The `data/` folder contains sample CSV files for each task:

| File | Task | Columns |
|------|------|---------|
| `pairs.csv` | pairs | `anchor`, `positive` |
| `triplets.csv` | triplets | `anchor`, `positive`, `negative` |
| `similarity.csv` | similarity | `sentence1`, `sentence2`, `score` |
| `tsdae.csv` | tsdae | `text` |

## YAML Configurations

See `configs/` folder for YAML configuration examples:
- Basic configs for each task
- LoRA configurations
- Unsloth acceleration configs
- HuggingFace dataset configs

```bash
# List all configs
ls examples/configs/

# Train with a config
vespaembed train --config examples/configs/pairs.yaml
```

## Column Aliases

VespaEmbed supports flexible column naming:

### Pairs / Triplets
- `anchor`: query, question, sent1, sentence1, text1
- `positive`: document, answer, pos, sent2, sentence2, text2
- `negative`: neg, hard_negative, sent3, sentence3, text3

### Similarity
- `sentence1`: sent1, text1, anchor, query
- `sentence2`: sent2, text2, positive, document
- `score`: similarity, label, sim_score

### TSDAE
- `text`: sentence, sentences, content, input
