"""
Matryoshka Representation Learning Example

Matryoshka training creates embeddings that work at multiple dimensions,
allowing you to trade off between embedding size and quality at inference time.

This is useful when you need:
- Flexible storage: Use smaller dimensions for storage-constrained environments
- Speed vs accuracy: Use fewer dimensions for faster retrieval, more for accuracy
- Progressive refinement: Start with coarse search, refine with full dimensions

Data format: Same as MNR (anchor-positive pairs)
- anchor: The query or question
- positive: The relevant document or answer

Column aliases supported:
- anchor: query, question, sent1, sentence1, text1
- positive: document, answer, pos, sent2, sentence2, text2
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train with default Matryoshka dimensions [768, 512, 256, 128, 64]
vespaembed train \\
    --task matryoshka \\
    --data examples/data/mnr.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project my-matryoshka-model \\
    --epochs 3

# Train with custom dimensions
vespaembed train \\
    --task matryoshka \\
    --data examples/data/mnr.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --matryoshka-dims 384,256,128,64,32 \\
    --project custom-matryoshka

# Train with HuggingFace dataset
vespaembed train \\
    --task matryoshka \\
    --data sentence-transformers/all-nli \\
    --subset pair \\
    --split train[:10000] \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --matryoshka-dims 384,192,96,48 \\
    --project nli-matryoshka
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


def train_matryoshka_basic():
    """Basic Matryoshka training with default dimensions."""
    config = TrainingConfig(
        task="matryoshka",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/mnr.csv"),
        # Default dimensions: [768, 512, 256, 128, 64]
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_matryoshka_custom_dims():
    """Matryoshka training with custom dimensions."""
    config = TrainingConfig(
        task="matryoshka",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/mnr.csv"),
        matryoshka_dims=[384, 256, 128, 64, 32],  # Custom dimensions
        training=TrainingHyperparameters(
            epochs=3,
            batch_size=32,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_matryoshka_advanced():
    """Matryoshka training with advanced configuration."""
    config = TrainingConfig(
        task="matryoshka",
        base_model="BAAI/bge-small-en-v1.5",
        data=DataConfig(
            train="examples/data/mnr.csv",
            eval="examples/data/mnr.csv",
        ),
        matryoshka_dims=[384, 256, 128, 64],
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
            dir="./output/matryoshka-model",
            save_total_limit=2,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_matryoshka_huggingface():
    """Matryoshka training with HuggingFace dataset."""
    config = TrainingConfig(
        task="matryoshka",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/all-nli",
            subset="pair",
            split="train[:10000]",
        ),
        matryoshka_dims=[384, 192, 96, 48],
        training=TrainingHyperparameters(
            epochs=2,
            batch_size=32,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def evaluate_matryoshka_model():
    """Example of using Matryoshka embeddings at different dimensions."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    import numpy as np

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
    dimensions = [384, 256, 128, 64, 32]

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
            print(f"  Doc {i+1}: {sim:.4f} - {doc[:50]}...")

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
    train_matryoshka_basic()
    print("Training complete!")
