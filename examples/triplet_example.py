"""
Triplet Loss Training Example

Triplet training explicitly uses anchor-positive-negative triplets to learn
embeddings where anchors are closer to positives than negatives.

Data format:
- anchor: The reference text
- positive: Text similar to anchor
- negative: Text dissimilar to anchor

Column aliases supported:
- anchor: query, question, sent1, sentence1, text1
- positive: document, answer, pos, sent2, sentence2, text2
- negative: neg, hard_negative, sent3, sentence3, text3
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train with local CSV file
vespaembed train \\
    --task triplet \\
    --data examples/data/triplet.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project my-triplet-model \\
    --epochs 3

# Train with HuggingFace dataset
vespaembed train \\
    --task triplet \\
    --data sentence-transformers/all-nli \\
    --subset triplet \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project nli-triplet-model

# Train with custom hyperparameters
vespaembed train \\
    --task triplet \\
    --data examples/data/triplet.csv \\
    --base-model BAAI/bge-small-en-v1.5 \\
    --epochs 5 \\
    --batch-size 64 \\
    --learning-rate 1e-5
"""

# =============================================================================
# Python API Example
# =============================================================================

from vespaembed.core.config import DataConfig, OutputConfig, TrainingConfig, TrainingHyperparameters  # noqa: E402
from vespaembed.core.trainer import VespaEmbedTrainer  # noqa: E402


def train_triplet_basic():
    """Basic triplet training with minimal configuration."""
    config = TrainingConfig(
        task="triplet",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/triplet.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_triplet_advanced():
    """Triplet training with advanced configuration."""
    config = TrainingConfig(
        task="triplet",
        base_model="BAAI/bge-small-en-v1.5",
        data=DataConfig(
            train="examples/data/triplet.csv",
            eval="examples/data/triplet.csv",
        ),
        training=TrainingHyperparameters(
            epochs=5,
            batch_size=32,
            learning_rate=1e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=True,
            eval_steps=50,
            save_steps=100,
        ),
        output=OutputConfig(
            dir="./output/triplet-model",
            save_total_limit=2,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_triplet_huggingface():
    """Triplet training with HuggingFace dataset."""
    config = TrainingConfig(
        task="triplet",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/all-nli",
            subset="triplet",
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


def evaluate_triplet_model():
    """Example of evaluating triplet model quality."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer("./output/triplet-model/final")

    # Test with triplet examples
    anchor = "What is Python programming?"
    positive = "Python is a high-level programming language"
    negative = "A python is a large snake"

    embeddings = model.encode([anchor, positive, negative])

    pos_sim = cos_sim(embeddings[0:1], embeddings[1:2]).item()
    neg_sim = cos_sim(embeddings[0:1], embeddings[2:3]).item()

    print(f"Anchor-Positive similarity: {pos_sim:.4f}")
    print(f"Anchor-Negative similarity: {neg_sim:.4f}")
    print(f"Margin (should be positive): {pos_sim - neg_sim:.4f}")


if __name__ == "__main__":
    print("Training triplet model...")
    train_triplet_basic()
    print("Training complete!")
