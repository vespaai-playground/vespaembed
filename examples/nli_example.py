"""
Sentence Pair Classification (NLI) Example

Uses SoftmaxLoss to classify sentence pairs into categories.
Common use cases include:
- NLI (entailment/neutral/contradiction)
- Paraphrase detection
- Semantic similarity classification
- Any multi-class sentence pair task

Data format:
- sentence1: First sentence
- sentence2: Second sentence
- label: Class label (integer or string - auto-converted)

Column aliases supported:
- sentence1: premise, sent1, text1, anchor
- sentence2: hypothesis, sent2, text2, positive
- label: gold_label, class, category

Note: Labels can be strings (auto-converted) or integers.
The number of classes is auto-detected from the data.
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

# Train with HuggingFace dataset (classic NLI)
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

from vespaembed.core.config import (  # noqa: E402
    DataConfig,
    OutputConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
from vespaembed.core.trainer import VespaEmbedTrainer  # noqa: E402


def train_nli_basic():
    """Basic sentence pair classification training."""
    config = TrainingConfig(
        task="nli",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/nli.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_nli_huggingface():
    """Training with HuggingFace NLI dataset."""
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
    print("Training sentence pair classification model...")
    train_nli_basic()
    print("Training complete!")
