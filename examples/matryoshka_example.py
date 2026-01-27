"""
Matryoshka Representation Learning Example

Matryoshka training creates embeddings that work at multiple dimensions,
allowing you to trade off between embedding size and quality at inference time.

This is useful when you need:
- Flexible storage: Use smaller dimensions for storage-constrained environments
- Speed vs accuracy: Use fewer dimensions for faster retrieval, more for accuracy
- Progressive refinement: Start with coarse search, refine with full dimensions

IMPORTANT: Matryoshka is NOT a task type - it's an option that can be combined
with any task (pairs, triplets, similarity). Use matryoshka_dims to enable it.

Reference: https://arxiv.org/abs/2205.13147
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train pairs with Matryoshka dimensions
vespaembed train \\
    --task pairs \\
    --data examples/data/pairs.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --matryoshka-dims 384,256,128,64 \\
    --epochs 3

# Train triplets with Matryoshka dimensions
vespaembed train \\
    --task triplets \\
    --data examples/data/triplets.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --matryoshka-dims 384,256,128,64 \\
    --epochs 3

# Train with HuggingFace dataset
vespaembed train \\
    --task pairs \\
    --data sentence-transformers/gooaq \\
    --split train[:10000] \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --matryoshka-dims 384,192,96,48

# Train with YAML config
vespaembed train --config examples/configs/pairs_matryoshka.yaml
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


def train_pairs_with_matryoshka():
    """Pairs training with Matryoshka embeddings."""
    config = TrainingConfig(
        task="pairs",  # Use pairs task
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/pairs.csv"),
        matryoshka_dims=[384, 256, 128, 64],  # Enable Matryoshka
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_triplets_with_matryoshka():
    """Triplets training with Matryoshka embeddings."""
    config = TrainingConfig(
        task="triplets",  # Use triplets task
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/triplets.csv"),
        matryoshka_dims=[384, 256, 128, 64],  # Enable Matryoshka
        training=TrainingHyperparameters(
            epochs=3,
            batch_size=32,
            learning_rate=2e-5,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_matryoshka_huggingface():
    """Matryoshka training with HuggingFace dataset."""
    config = TrainingConfig(
        task="pairs",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/gooaq",
            split="train[:10000]",
        ),
        matryoshka_dims=[384, 192, 96, 48],
        training=TrainingHyperparameters(
            epochs=1,
            batch_size=64,
            fp16=True,
        ),
        output=OutputConfig(dir="./output/matryoshka-model"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_matryoshka_with_lora():
    """Matryoshka training with LoRA for memory efficiency."""
    config = TrainingConfig(
        task="pairs",
        base_model="BAAI/bge-base-en-v1.5",
        data=DataConfig(train="examples/data/pairs.csv"),
        matryoshka_dims=[768, 512, 256, 128],
        training=TrainingHyperparameters(
            epochs=3,
            batch_size=16,
            learning_rate=1e-5,
            fp16=True,
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


def evaluate_matryoshka_model():
    """Example of using Matryoshka embeddings at different dimensions."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("./output/matryoshka-model/final")

    # Test query and documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with many layers.",
        "The weather forecast predicts rain tomorrow.",
    ]

    # Get full embeddings
    query_embedding = model.encode([query])[0]
    doc_embeddings = model.encode(documents)

    # Test at different dimensions
    dimensions = [384, 256, 128, 64]

    print("Matryoshka Evaluation - Similarity at Different Dimensions:")
    print("=" * 70)

    for dim in dimensions:
        # Truncate embeddings to dimension
        q_trunc = query_embedding[:dim]
        d_trunc = doc_embeddings[:, :dim]

        # Normalize truncated embeddings
        q_norm = q_trunc / np.linalg.norm(q_trunc)
        d_norm = d_trunc / np.linalg.norm(d_trunc, axis=1, keepdims=True)

        print(f"\nDimension: {dim}")
        print("-" * 40)
        for i, doc in enumerate(documents):
            sim = np.dot(q_norm, d_norm[i])
            print(f"  Doc {i + 1}: {sim:.4f} - {doc[:50]}...")

    # Show ranking consistency across dimensions
    print("\n\nRanking Consistency:")
    print("=" * 70)
    for dim in dimensions:
        q_trunc = query_embedding[:dim]
        d_trunc = doc_embeddings[:, :dim]
        q_norm = q_trunc / np.linalg.norm(q_trunc)
        d_norm = d_trunc / np.linalg.norm(d_trunc, axis=1, keepdims=True)
        sims = np.dot(d_norm, q_norm)
        ranking = np.argsort(sims)[::-1] + 1
        print(f"  Dim {dim:3d}: Ranking = {list(ranking)}")


if __name__ == "__main__":
    print("Training Matryoshka model...")
    train_pairs_with_matryoshka()
    print("Training complete!")
