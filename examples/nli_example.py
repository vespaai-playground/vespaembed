"""
Natural Language Inference (NLI) Example

NLI training uses sentence pairs with entailment labels to learn embeddings
that capture semantic relationships between sentences.

Data format:
- sentence1: The premise/first sentence
- sentence2: The hypothesis/second sentence
- label: Entailment label (0=entailment, 1=neutral, 2=contradiction)

Column aliases supported:
- sentence1: premise, sent1, text1, anchor
- sentence2: hypothesis, sent2, text2, positive
- label: gold_label, class
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train with local CSV file
vespaembed train \\
    --task nli \\
    --data examples/data/nli.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project my-nli-model \\
    --epochs 3 \\
    --batch-size 32

# Train with HuggingFace dataset
vespaembed train \\
    --task nli \\
    --data sentence-transformers/all-nli \\
    --subset pair-class \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project nli-model
"""

# =============================================================================
# Python API Example
# =============================================================================

from vespaembed.core.config import (
    DataConfig,
    OutputConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
from vespaembed.core.trainer import VespaEmbedTrainer


def train_nli_basic():
    """Basic NLI training with minimal configuration."""
    config = TrainingConfig(
        task="nli",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/nli.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_nli_huggingface():
    """NLI training with HuggingFace dataset."""
    config = TrainingConfig(
        task="nli",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/all-nli",
            subset="pair-class",
            split="train[:10000]",  # Use subset for faster training
        ),
        training=TrainingHyperparameters(
            epochs=1,
            batch_size=32,
        ),
        output=OutputConfig(
            dir="./output/nli-model",
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


if __name__ == "__main__":
    print("Training NLI model...")
    train_nli_basic()
    print("Training complete!")
