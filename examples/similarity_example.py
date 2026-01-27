"""
Similarity Task Example

Similarity training uses sentence pairs with continuous similarity scores (0-1).
Ideal for semantic textual similarity (STS) tasks.

Data format:
- sentence1: First text
- sentence2: Second text
- score: Similarity score (0-1, or 0-5 which will be normalized)

Column aliases supported:
- sentence1: sent1, text1, anchor, query
- sentence2: sent2, text2, positive, document
- score: similarity, label, sim_score

Loss variants available:
- cosine (default): Cosine Similarity Loss
- cosent: CoSENT (often outperforms cosine)
- angle: AnglE Loss
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train with local CSV file
vespaembed train \\
    --task similarity \\
    --data examples/data/similarity.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --epochs 3

# Train with HuggingFace STS Benchmark
vespaembed train \\
    --task similarity \\
    --data sentence-transformers/stsb \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2

# Train with YAML config
vespaembed train --config examples/configs/similarity_stsb.yaml
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


def train_similarity_basic():
    """Basic similarity training with minimal configuration."""
    config = TrainingConfig(
        task="similarity",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/similarity.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_similarity_cosent():
    """Similarity training with CoSENT loss (often better than cosine)."""
    config = TrainingConfig(
        task="similarity",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        loss_variant="cosent",  # CoSENT loss
        data=DataConfig(train="examples/data/similarity.csv"),
        training=TrainingHyperparameters(
            epochs=4,
            batch_size=32,
            learning_rate=2e-5,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_similarity_stsb():
    """Similarity training with STS Benchmark from HuggingFace."""
    config = TrainingConfig(
        task="similarity",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        loss_variant="cosent",
        data=DataConfig(
            train="sentence-transformers/stsb",
            split="train",
            eval_split="validation",
        ),
        training=TrainingHyperparameters(
            epochs=4,
            batch_size=32,
            learning_rate=2e-5,
            optimizer="adamw_torch",
            scheduler="linear",
            eval_steps=100,
            save_steps=500,
        ),
        output=OutputConfig(dir="./output/similarity-model"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def evaluate_similarity_model():
    """Example of evaluating similarity model quality."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer("./output/similarity-model/final")

    # Test pairs with expected similarity
    test_pairs = [
        ("A man is playing a guitar", "A person is playing a musical instrument", 0.85),
        ("The cat is sleeping peacefully", "A feline is resting quietly", 0.92),
        ("A man is reading a newspaper", "A woman is watching television", 0.15),
        ("The sun is setting over the ocean", "The sun is rising over the mountains", 0.35),
    ]

    print("Similarity Evaluation:")
    print("-" * 70)
    for s1, s2, expected in test_pairs:
        embeddings = model.encode([s1, s2])
        predicted = cos_sim(embeddings[0:1], embeddings[1:2]).item()
        diff = abs(predicted - expected)
        print(f"Expected: {expected:.2f} | Predicted: {predicted:.2f} | Diff: {diff:.2f}")
        print(f"  '{s1}'")
        print(f"  '{s2}'")
        print()


if __name__ == "__main__":
    print("Training similarity model...")
    train_similarity_basic()
    print("Training complete!")
