"""
Contrastive Learning Example

Contrastive learning trains embeddings using pairs of sentences with binary labels
indicating whether they are similar (1) or dissimilar (0).

Data format:
- sentence1: First text
- sentence2: Second text
- label: 1 for similar pairs, 0 for dissimilar pairs

Column aliases supported:
- sentence1: sent1, text1, anchor, query
- sentence2: sent2, text2, positive, document
- label: score, similarity
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train with local CSV file
vespaembed train \\
    --task contrastive \\
    --data examples/data/contrastive.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project my-contrastive-model \\
    --epochs 3

# Train with HuggingFace dataset
vespaembed train \\
    --task contrastive \\
    --data sentence-transformers/all-nli \\
    --subset pair-class \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project nli-contrastive-model

# Train with custom hyperparameters
vespaembed train \\
    --task contrastive \\
    --data examples/data/contrastive.csv \\
    --base-model BAAI/bge-small-en-v1.5 \\
    --epochs 5 \\
    --batch-size 64 \\
    --learning-rate 2e-5
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


def train_contrastive_basic():
    """Basic contrastive training with minimal configuration."""
    config = TrainingConfig(
        task="contrastive",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/contrastive.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_contrastive_advanced():
    """Contrastive training with advanced configuration."""
    config = TrainingConfig(
        task="contrastive",
        base_model="BAAI/bge-small-en-v1.5",
        data=DataConfig(
            train="examples/data/contrastive.csv",
            eval="examples/data/contrastive.csv",
        ),
        training=TrainingHyperparameters(
            epochs=5,
            batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=True,
            eval_steps=50,
            save_steps=100,
        ),
        output=OutputConfig(
            dir="./output/contrastive-model",
            save_total_limit=2,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_contrastive_huggingface():
    """Contrastive training with HuggingFace dataset."""
    config = TrainingConfig(
        task="contrastive",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/all-nli",
            subset="pair-class",
            split="train[:5000]",
            eval="sentence-transformers/all-nli",
            eval_split="dev[:1000]",
        ),
        training=TrainingHyperparameters(
            epochs=2,
            batch_size=32,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def evaluate_contrastive_model():
    """Example of evaluating contrastive model quality."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer("./output/contrastive-model/final")

    # Test with similar and dissimilar pairs
    similar_pairs = [
        ("A man is playing a guitar", "A person is playing a musical instrument"),
        ("The cat is sleeping on the couch", "A feline is resting on the sofa"),
    ]

    dissimilar_pairs = [
        ("A man is playing a guitar", "The weather is sunny today"),
        ("The cat is sleeping on the couch", "Stock prices rose sharply yesterday"),
    ]

    print("Similar pairs:")
    for s1, s2 in similar_pairs:
        embeddings = model.encode([s1, s2])
        sim = cos_sim(embeddings[0:1], embeddings[1:2]).item()
        print(f"  '{s1[:30]}...' <-> '{s2[:30]}...': {sim:.4f}")

    print("\nDissimilar pairs:")
    for s1, s2 in dissimilar_pairs:
        embeddings = model.encode([s1, s2])
        sim = cos_sim(embeddings[0:1], embeddings[1:2]).item()
        print(f"  '{s1[:30]}...' <-> '{s2[:30]}...': {sim:.4f}")


if __name__ == "__main__":
    print("Training contrastive model...")
    train_contrastive_basic()
    print("Training complete!")
