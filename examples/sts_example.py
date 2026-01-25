"""
Semantic Textual Similarity (STS) Example

STS training uses pairs of sentences with continuous similarity scores (typically 0-1)
to learn embeddings that capture semantic similarity.

Data format:
- sentence1: First text
- sentence2: Second text
- score: Similarity score between 0 and 1 (or 0-5, will be normalized)

Column aliases supported:
- sentence1: sent1, text1, anchor, query
- sentence2: sent2, text2, positive, document
- score: label, similarity
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train with local CSV file
vespaembed train \\
    --task sts \\
    --data examples/data/sts.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project my-sts-model \\
    --epochs 3

# Train with HuggingFace STS Benchmark dataset
vespaembed train \\
    --task sts \\
    --data sentence-transformers/stsb \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project stsb-model

# Train with evaluation data
vespaembed train \\
    --task sts \\
    --data examples/data/sts.csv \\
    --eval-data examples/data/sts.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --epochs 5 \\
    --batch-size 16
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


def train_sts_basic():
    """Basic STS training with minimal configuration."""
    config = TrainingConfig(
        task="sts",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/sts.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_sts_advanced():
    """STS training with advanced configuration."""
    config = TrainingConfig(
        task="sts",
        base_model="BAAI/bge-small-en-v1.5",
        data=DataConfig(
            train="examples/data/sts.csv",
            eval="examples/data/sts.csv",
        ),
        training=TrainingHyperparameters(
            epochs=5,
            batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=True,
            eval_steps=50,
            save_steps=100,
        ),
        output=OutputConfig(
            dir="./output/sts-model",
            save_total_limit=2,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_sts_huggingface():
    """STS training with HuggingFace STS Benchmark dataset."""
    config = TrainingConfig(
        task="sts",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/stsb",
            split="train",
            eval="sentence-transformers/stsb",
            eval_split="validation",
        ),
        training=TrainingHyperparameters(
            epochs=3,
            batch_size=32,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def evaluate_sts_model():
    """Example of evaluating STS model quality."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer("./output/sts-model/final")

    # Test pairs with expected similarity
    test_pairs = [
        ("A man is playing a guitar", "A person is playing a musical instrument", 0.85),
        ("The cat is sleeping peacefully", "A feline is resting quietly", 0.92),
        ("A man is reading a newspaper", "A woman is watching television", 0.15),
        ("The sun is setting over the ocean", "The sun is rising over the mountains", 0.35),
    ]

    print("STS Evaluation:")
    print("-" * 70)
    for s1, s2, expected in test_pairs:
        embeddings = model.encode([s1, s2])
        predicted = cos_sim(embeddings[0:1], embeddings[1:2]).item()
        diff = abs(predicted - expected)
        print(f"Expected: {expected:.2f} | Predicted: {predicted:.2f} | Diff: {diff:.2f}")
        print(f"  '{s1}' <-> '{s2}'")
        print()


if __name__ == "__main__":
    print("Training STS model...")
    train_sts_basic()
    print("Training complete!")
