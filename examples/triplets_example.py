"""
Triplets Task Example

Triplets training uses anchor-positive-negative triplets with explicit hard negatives.
More effective than pairs when you have good negative examples.

Data format:
- anchor: The reference text
- positive: Text similar to anchor
- negative: Text dissimilar to anchor (hard negative)

Column aliases supported:
- anchor: query, question, sent1, sentence1, text1
- positive: document, answer, pos, sent2, sentence2, text2
- negative: neg, hard_negative, sent3, sentence3, text3

Loss variants available:
- mnr (default): Multiple Negatives Ranking
- mnr_symmetric: Symmetric MNR
- gist: Guided In-Sample Triplet
- cached_mnr: Cached MNR for large batches
- cached_gist: Cached GIST
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train with local CSV file
vespaembed train \\
    --task triplets \\
    --data examples/data/triplets.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --epochs 3

# Train with HuggingFace AllNLI dataset
vespaembed train \\
    --task triplets \\
    --data sentence-transformers/all-nli \\
    --subset triplet \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2

# Train with YAML config
vespaembed train --config examples/configs/triplets_allnli.yaml
"""

# =============================================================================
# Python API Example
# =============================================================================

from vespaembed.core.config import (  # noqa: E402
    DataConfig,
    LoraConfig,
    OutputConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
from vespaembed.core.trainer import VespaEmbedTrainer  # noqa: E402


def train_triplets_basic():
    """Basic triplets training with minimal configuration."""
    config = TrainingConfig(
        task="triplets",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/triplets.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_triplets_with_eval():
    """Triplets training with evaluation on AllNLI dev split."""
    config = TrainingConfig(
        task="triplets",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/all-nli",
            subset="triplet",
            split="train",
            eval_split="dev",
        ),
        training=TrainingHyperparameters(
            epochs=1,
            batch_size=64,
            learning_rate=2e-5,
            optimizer="adamw_torch",
            scheduler="linear",
            eval_steps=1000,
            save_steps=1000,
        ),
        output=OutputConfig(dir="./output/triplets-model"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_triplets_with_lora():
    """Triplets training with LoRA on larger model."""
    config = TrainingConfig(
        task="triplets",
        base_model="BAAI/bge-base-en-v1.5",
        data=DataConfig(
            train="sentence-transformers/all-nli",
            subset="triplet",
            split="train[:50000]",
        ),
        training=TrainingHyperparameters(
            epochs=1,
            batch_size=32,
            learning_rate=1e-5,
            fp16=True,
            gradient_accumulation_steps=2,
        ),
        gradient_checkpointing=True,
        lora=LoraConfig(
            enabled=True,
            r=32,
            alpha=64,
            dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def evaluate_triplets_model():
    """Example of evaluating triplets model quality."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer("./output/triplets-model/final")

    # Test with triplet examples
    test_triplets = [
        (
            "What is Python programming?",
            "Python is a high-level programming language",
            "A python is a large snake",
        ),
        (
            "How to make coffee?",
            "Brew coffee by adding hot water to ground beans",
            "Coffee tables are furniture items",
        ),
    ]

    print("Triplets Evaluation:")
    print("-" * 70)
    for anchor, positive, negative in test_triplets:
        embeddings = model.encode([anchor, positive, negative])
        pos_sim = cos_sim(embeddings[0:1], embeddings[1:2]).item()
        neg_sim = cos_sim(embeddings[0:1], embeddings[2:3]).item()
        margin = pos_sim - neg_sim

        print(f"Anchor: {anchor}")
        print(f"  Positive sim: {pos_sim:.4f}")
        print(f"  Negative sim: {neg_sim:.4f}")
        print(f"  Margin: {margin:.4f} (should be positive)")
        print()


if __name__ == "__main__":
    print("Training triplets model...")
    train_triplets_basic()
    print("Training complete!")
