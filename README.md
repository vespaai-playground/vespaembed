# VespaEmbed

No-code training for embedding models. Train custom embedding models with a web UI or CLI.

<p align="center">
  <a href="https://huggingface.co/spaces/vespa-engine/vespaembed?duplicate=true">
    <img src="assets/hf-spaces-badge.svg" alt="Deploy on HF Spaces">
  </a>
</p>


## Features

- **Web UI** - Visual interface for configuring and monitoring training
- **CLI** - Command-line interface for scripting and automation
- **Multiple Tasks** - Support for pairs, triplets, similarity scoring, and unsupervised learning
- **Loss Variants** - Choose from multiple loss functions per task
- **Matryoshka Embeddings** - Train multi-dimensional embeddings for flexible retrieval
- **LoRA Support** - Parameter-efficient fine-tuning with LoRA adapters
- **Unsloth Integration** - Faster training with Unsloth optimizations
- **HuggingFace Integration** - Load datasets, models from HuggingFace Hub, push models to Hub

## Installation

> **Note:** VespaEmbed is in experimental phase. Install from source.

```bash
git clone https://github.com/vespaai-playground/vespaembed.git
cd vespaembed
uv sync
```

### Optional Dependencies

```bash
# For Unsloth acceleration (requires NVIDIA/AMD GPU)
uv sync --extra unsloth

# For TensorBoard metrics visualization
uv sync --extra tensorboard

# For ONNX export
uv sync --extra onnx

# For development
uv sync --extra dev
```

## Quick Start

### Web UI

Launch the web interface:

```bash
vespaembed
```

Open http://localhost:8000 in your browser. The UI lets you:
- Upload training data (CSV or JSONL)
- Select task type and base model
- Configure hyperparameters
- Monitor training progress
- Download trained models

### CLI

Train a model from the command line:

```bash
vespaembed train \
  --data examples/data/pairs.csv \
  --task pairs \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 3
```

Or use a YAML config file:

```bash
vespaembed train --config config.yaml
```

## Tasks

VespaEmbed supports 4 training tasks based on your data format:

### Pairs

Text pairs for semantic search. Use when you have query-document pairs without explicit negatives.

**Data format:**
```csv
anchor,positive
What is machine learning?,Machine learning is a subset of AI...
How does photosynthesis work?,Photosynthesis converts sunlight...
```

**Loss variants:** `mnr` (default), `mnr_symmetric`, `gist`, `cached_mnr`, `cached_gist`

### Triplets

Text triplets with hard negatives. Use when you have explicit negative examples.

**Data format:**
```csv
anchor,positive,negative
What is Python?,Python is a programming language...,A python is a large snake...
```

**Loss variants:** `mnr` (default), `mnr_symmetric`, `gist`, `cached_mnr`, `cached_gist`

### Similarity

Text pairs with similarity scores (STS-style). Use when you have continuous similarity labels.

**Data format:**
```csv
sentence1,sentence2,score
A man is playing guitar,A person plays music,0.85
The cat is sleeping,A dog is running,0.12
```

**Loss variants:** `cosine` (default), `cosent`, `angle`

### TSDAE

Unsupervised learning with denoising auto-encoder. Use when you only have unlabeled text for domain adaptation.

**Data format:**
```csv
text
Machine learning is transforming how we analyze data.
Natural language processing enables computers to understand human language.
```

## Configuration

### CLI Arguments

```bash
vespaembed train \
  --data <path>              # Training data (CSV, JSONL, or HF dataset)
  --task <task>              # Task type: pairs, triplets, similarity, tsdae
  --base-model <model>       # Base model name or path
  --project <name>           # Project name (optional)
  --eval-data <path>         # Evaluation data (optional)
  --epochs <n>               # Number of epochs (default: 3)
  --batch-size <n>           # Batch size (default: 32)
  --learning-rate <lr>       # Learning rate (default: 2e-5)
  --optimizer <opt>          # Optimizer (default: adamw_torch)
  --scheduler <sched>        # LR scheduler (default: linear)
  --matryoshka               # Enable Matryoshka embeddings
  --matryoshka-dims <dims>   # Dimensions (default: 768,512,256,128,64)
  --unsloth                  # Use Unsloth for faster training
  --subset <name>            # HuggingFace dataset subset
  --split <name>             # HuggingFace dataset split
```

### Optimizers

| Option | Description |
|--------|-------------|
| `adamw_torch` | AdamW (default) |
| `adamw_torch_fused` | Fused AdamW (faster on CUDA) |
| `adamw_8bit` | 8-bit AdamW (memory efficient) |
| `adafactor` | Adafactor (memory efficient, no momentum) |
| `sgd` | SGD with momentum |

### Schedulers

| Option | Description |
|--------|-------------|
| `linear` | Linear decay (default) |
| `cosine` | Cosine annealing |
| `cosine_with_restarts` | Cosine with warm restarts |
| `constant` | Constant learning rate |
| `constant_with_warmup` | Constant after warmup |
| `polynomial` | Polynomial decay |

### YAML Configuration

```yaml
task: pairs
base_model: sentence-transformers/all-MiniLM-L6-v2

data:
  train: train.csv
  eval: eval.csv            # optional

training:
  epochs: 3
  batch_size: 32
  learning_rate: 2e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true
  eval_steps: 500
  save_steps: 500
  logging_steps: 100
  optimizer: adamw_torch    # adamw_torch, adamw_8bit, adafactor, sgd
  scheduler: linear         # linear, cosine, constant, polynomial

output:
  dir: ./output
  push_to_hub: false
  hf_username: null

# Optional: LoRA configuration
lora:
  enabled: false
  r: 64
  alpha: 128
  dropout: 0.1
  target_modules: [query, key, value, dense]

# Optional: Matryoshka dimensions
matryoshka_dims: [768, 512, 256, 128, 64]

# Optional: Loss variant (uses task default if not specified)
loss_variant: mnr
```

### HuggingFace Datasets

Load datasets directly from HuggingFace Hub:

```bash
vespaembed train \
  --data sentence-transformers/all-nli \
  --subset triplet \
  --split train \
  --task triplets \
  --base-model sentence-transformers/all-MiniLM-L6-v2
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `vespaembed` | Launch web UI (default) |
| `vespaembed serve` | Launch web UI |
| `vespaembed train` | Train a model |
| `vespaembed evaluate` | Evaluate a model |
| `vespaembed export` | Export model to ONNX |
| `vespaembed info` | Show task information |

## Output

Trained models are saved to `~/.vespaembed/projects/<project-name>/`:

```
~/.vespaembed/projects/my-project/
├── final/              # Final trained model
├── checkpoint-500/     # Training checkpoints
├── checkpoint-1000/
└── logs/               # TensorBoard logs
```

## Column Aliases

VespaEmbed automatically recognizes common column name variations:

| Task | Expected | Also Accepts |
|------|----------|--------------|
| pairs | `anchor` | `query`, `question`, `sent1`, `sentence1`, `text1` |
| pairs | `positive` | `document`, `answer`, `pos`, `sent2`, `sentence2`, `text2` |
| triplets | `negative` | `neg`, `hard_negative`, `sent3`, `sentence3`, `text3` |
| similarity | `sentence1` | `sent1`, `text1`, `anchor`, `query` |
| similarity | `sentence2` | `sent2`, `text2`, `positive`, `document` |
| similarity | `score` | `similarity`, `label`, `sim_score` |
| tsdae | `text` | `sentence`, `sentences`, `content`, `input` |

**Important:** Columns are matched by **name** (or alias), not by position. For example, with a pairs task:
- `[anchor, positive]` or `[query, document]` → works ✓
- `[document, query]` → still works (names identify roles, not position) ✓
- `[foo, bar]` → fails (no matching column names or aliases) ✗

Columns named `score`, `scores`, `label`, or `labels` (and aliases like `similarity`) are treated as labels/targets.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=vespaembed

# Format code
make format

# Lint
make lint
```

## License

Apache 2.0
