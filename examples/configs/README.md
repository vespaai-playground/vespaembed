# Example Configurations

Example YAML configurations for training embedding models with VespaEmbed.

## Usage

```bash
vespaembed train --config examples/configs/pairs.yaml
```

## Available Configs

### Basic Configs (Local Data)
| Config | Task | Description |
|--------|------|-------------|
| `pairs.yaml` | pairs | Query-document pairs |
| `triplets.yaml` | triplets | Anchor-positive-negative triplets |
| `similarity.yaml` | similarity | Sentence pairs with similarity scores |
| `tsdae.yaml` | tsdae | Unsupervised domain adaptation |

### LoRA Configs
| Config | Task | Description |
|--------|------|-------------|
| `pairs_lora.yaml` | pairs | Pairs with LoRA adapters |
| `triplets_lora.yaml` | triplets | Triplets with LoRA |
| `similarity_lora.yaml` | similarity | Similarity with LoRA |
| `tsdae_lora.yaml` | tsdae | TSDAE with LoRA |
| `pairs_bge_lora.yaml` | pairs | BGE model with LoRA + gradient checkpointing |

### Unsloth Configs
| Config | Task | Description |
|--------|------|-------------|
| `pairs_unsloth.yaml` | pairs | Pairs with Unsloth acceleration |
| `triplets_unsloth.yaml` | triplets | Triplets with Unsloth |
| `similarity_unsloth.yaml` | similarity | Similarity with Unsloth |
| `tsdae_unsloth.yaml` | tsdae | TSDAE with Unsloth |

### Unsloth + LoRA Configs
| Config | Task | Description |
|--------|------|-------------|
| `pairs_unsloth_lora.yaml` | pairs | Pairs with Unsloth + LoRA |
| `triplets_unsloth_lora.yaml` | triplets | Triplets with Unsloth + LoRA |
| `similarity_unsloth_lora.yaml` | similarity | Similarity with Unsloth + LoRA |
| `tsdae_unsloth_lora.yaml` | tsdae | TSDAE with Unsloth + LoRA |

### HuggingFace Dataset Configs
| Config | Dataset | Description |
|--------|---------|-------------|
| `pairs_msmarco.yaml` | msmarco-bm25 | MS MARCO passage retrieval |
| `pairs_quora.yaml` | quora-duplicates | Quora duplicate questions |
| `pairs_nq.yaml` | natural-questions | Google Natural Questions |
| `pairs_gooaq.yaml` | gooaq | Google Answer Questions |
| `triplets_allnli.yaml` | all-nli | SNLI + MultiNLI with eval split |
| `triplets_allnli_lora.yaml` | all-nli | AllNLI with BGE + LoRA |
| `similarity_stsb.yaml` | stsb | STS Benchmark |
| `tsdae_wikipedia.yaml` | simple-wiki | Wikipedia domain adaptation |

### Special Configs
| Config | Description |
|--------|-------------|
| `pairs_matryoshka.yaml` | Matryoshka multi-dimensional embeddings |

---

## Configuration Reference

### Required Fields

```yaml
task: pairs                    # Task type: pairs, triplets, similarity, tsdae
base_model: sentence-transformers/all-MiniLM-L6-v2  # Base model
data:
  train: ./train.csv           # Training data path or HuggingFace dataset
```

### Optional Fields & Defaults

When a field is omitted, the default value is used:

#### Data Configuration
```yaml
data:
  train: ...                   # Required
  eval: null                   # Evaluation data path (optional)
  subset: null                 # HuggingFace dataset subset
  split: train                 # HuggingFace training split
  eval_split: null             # HuggingFace evaluation split
```

#### Training Hyperparameters
```yaml
training:
  epochs: 3                    # Number of training epochs
  batch_size: 32               # Batch size per device
  learning_rate: 2e-5          # Learning rate
  warmup_ratio: 0.1            # Warmup ratio (0-1)
  weight_decay: 0.01           # Weight decay
  fp16: true                   # Use FP16 mixed precision
  bf16: false                  # Use BF16 mixed precision
  optimizer: adamw_torch       # Optimizer type
  scheduler: linear            # LR scheduler type
  eval_steps: 500              # Evaluate every N steps
  save_steps: 500              # Save checkpoint every N steps
  logging_steps: 100           # Log every N steps
  gradient_accumulation_steps: 1  # Gradient accumulation
```

#### Output Configuration
```yaml
output:
  dir: ./output                # Output directory (default: ~/.vespaembed/projects/)
  save_total_limit: 3          # Max checkpoints to keep
  push_to_hub: false           # Push to HuggingFace Hub
  hf_username: null            # HuggingFace username (required if push_to_hub)
```

#### Optional Features
```yaml
loss_variant: null             # Loss function variant (uses task default)
max_seq_length: null           # Max sequence length (auto-detect if null)
gradient_checkpointing: false  # Enable gradient checkpointing
matryoshka_dims: null          # Matryoshka dimensions, e.g., [768, 512, 256]
```

#### LoRA Configuration
```yaml
lora:
  enabled: false               # Enable LoRA
  r: 64                        # LoRA rank
  alpha: 128                   # LoRA alpha
  dropout: 0.1                 # LoRA dropout (use 0.0 for Unsloth)
  target_modules:              # Target modules for LoRA
    - query
    - key
    - value
    - dense
```

#### Unsloth Configuration
```yaml
unsloth:
  enabled: false               # Enable Unsloth acceleration
  save_method: merged_16bit    # Save method: lora, merged_16bit, merged_4bit
```

---

## Pushing to HuggingFace Hub

To push your trained model to HuggingFace Hub:

### 1. Set up authentication

```bash
# Option A: Environment variable
export HF_TOKEN=your_huggingface_token

# Option B: Login via CLI
huggingface-cli login
```

### 2. Add hub configuration to your YAML

```yaml
output:
  push_to_hub: true
  hf_username: your-username   # Your HuggingFace username
```

### 3. Train

```bash
vespaembed train --config your_config.yaml
```

The model will be pushed to: `https://huggingface.co/{hf_username}/{project_name}`

### Example config with Hub push

```yaml
task: pairs
base_model: sentence-transformers/all-MiniLM-L6-v2

data:
  train: sentence-transformers/all-nli
  subset: triplet
  split: train

training:
  epochs: 1
  batch_size: 64
  learning_rate: 2e-5
  fp16: true

output:
  push_to_hub: true
  hf_username: your-username   # <-- Change this
```

### Notes

- Models are pushed as **private** by default
- The repository name will be the project name (auto-generated or specified)
- Make sure you have write access to your HuggingFace account

---

## Tips

### Using local data
```yaml
data:
  train: ./path/to/train.csv
  eval: ./path/to/eval.csv     # Optional
```

### Using HuggingFace datasets
```yaml
data:
  train: sentence-transformers/all-nli
  subset: triplet
  split: train
  eval_split: dev              # Use different split for eval
```

### Memory-efficient training
```yaml
gradient_checkpointing: true   # Reduces VRAM usage
training:
  batch_size: 16
  gradient_accumulation_steps: 4  # Effective batch = 64
```

### Faster training with Unsloth
```yaml
training:
  bf16: true                   # BF16 recommended for Unsloth
gradient_checkpointing: true   # Uses Unsloth's optimized GC
unsloth:
  enabled: true
  save_method: merged_16bit
```
