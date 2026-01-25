# VespaEmbed Examples

This folder contains examples for each training task supported by VespaEmbed.

## Tasks Overview

| Task | Description | Required Columns |
|------|-------------|------------------|
| `mnr` | Multiple Negatives Ranking - semantic search with in-batch negatives | `anchor`, `positive` |
| `triplet` | Triplet training - learn from anchor/positive/negative triplets | `anchor`, `positive`, `negative` |
| `contrastive` | Contrastive learning - pairs with binary similarity labels | `sentence1`, `sentence2`, `label` |
| `sts` | Semantic Textual Similarity - pairs with continuous scores | `sentence1`, `sentence2`, `score` |
| `nli` | Sentence pair classification with SoftmaxLoss | `sentence1`, `sentence2`, `label` |
| `tsdae` | TSDAE - unsupervised domain adaptation with denoising auto-encoder | `text` |
| `matryoshka` | Matryoshka embeddings - multi-dimensional representations | `anchor`, `positive` |

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
vespaembed train --task mnr --data examples/data/mnr.csv --base-model sentence-transformers/all-MiniLM-L6-v2

# Train with a HuggingFace dataset
vespaembed train --task mnr --data sentence-transformers/all-nli --subset pair --base-model sentence-transformers/all-MiniLM-L6-v2
```

### Using Python API

```python
from vespaembed.core.config import TrainingConfig, DataConfig
from vespaembed.core.trainer import VespaEmbedTrainer

config = TrainingConfig(
    task="mnr",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    data=DataConfig(train="examples/data/mnr.csv"),
)

trainer = VespaEmbedTrainer(config)
trainer.train()
```

## Example Files

- `mnr_example.py` - Multiple Negatives Ranking
- `triplet_example.py` - Triplet Loss Training
- `contrastive_example.py` - Contrastive Learning
- `sts_example.py` - Semantic Textual Similarity
- `nli_example.py` - Sentence Pair Classification (NLI)
- `tsdae_example.py` - TSDAE Domain Adaptation
- `matryoshka_example.py` - Matryoshka Embeddings

## Sample Data

The `data/` folder contains sample CSV files for each task:

- `data/mnr.csv` - Query-document pairs
- `data/triplet.csv` - Anchor-positive-negative triplets
- `data/contrastive.csv` - Sentence pairs with binary labels
- `data/sts.csv` - Sentence pairs with similarity scores
- `data/nli.csv` - Sentence pairs with class labels (auto-detected)
- `data/tsdae.csv` - Unlabeled text for domain adaptation
