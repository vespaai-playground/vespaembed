"""
Multiple Negatives Ranking (MNR) Example

MNR is ideal for semantic search and retrieval tasks. It uses in-batch negatives
to learn embeddings where similar items are close together.

Data format:
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
# Train with local CSV file
vespaembed train \\
    --task mnr \\
    --data examples/data/mnr.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project my-mnr-model \\
    --epochs 3 \\
    --batch-size 32

# Train with HuggingFace dataset
vespaembed train \\
    --task mnr \\
    --data sentence-transformers/all-nli \\
    --subset pair \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project nli-mnr-model

# Train with evaluation data
vespaembed train \\
    --task mnr \\
    --data examples/data/mnr.csv \\
    --eval-data examples/data/mnr.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --epochs 5
"""

# =============================================================================
# Python API Example
# =============================================================================

from vespaembed.core.config import DataConfig, OutputConfig, TrainingConfig, TrainingHyperparameters  # noqa: E402
from vespaembed.core.trainer import VespaEmbedTrainer  # noqa: E402


def train_mnr_basic():
    """Basic MNR training with minimal configuration."""
    config = TrainingConfig(
        task="mnr",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/mnr.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_mnr_with_eval():
    """MNR training with evaluation data."""
    config = TrainingConfig(
        task="mnr",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="examples/data/mnr.csv",
            eval="examples/data/mnr.csv",
        ),
        training=TrainingHyperparameters(
            epochs=5,
            batch_size=16,
            learning_rate=2e-5,
            eval_steps=100,
        ),
        output=OutputConfig(
            dir="./output/mnr-model",
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_mnr_huggingface():
    """MNR training with HuggingFace dataset."""
    config = TrainingConfig(
        task="mnr",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/all-nli",
            subset="pair",
            split="train[:10000]",  # Use subset for faster training
        ),
        training=TrainingHyperparameters(
            epochs=1,
            batch_size=32,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def use_trained_model():
    """Example of using a trained model for inference."""
    from sentence_transformers import SentenceTransformer

    # Load the trained model
    model = SentenceTransformer("./output/mnr-model/final")

    # Encode queries and documents
    queries = ["What is machine learning?", "How does AI work?"]
    documents = [
        "Machine learning is a subset of AI.",
        "Artificial intelligence mimics human cognition.",
        "Deep learning uses neural networks.",
    ]

    query_embeddings = model.encode(queries)
    doc_embeddings = model.encode(documents)

    # Compute similarity
    from sentence_transformers.util import cos_sim

    similarities = cos_sim(query_embeddings, doc_embeddings)
    print("Similarity matrix:")
    print(similarities)


if __name__ == "__main__":
    # Run basic training
    print("Training MNR model...")
    train_mnr_basic()
    print("Training complete!")
