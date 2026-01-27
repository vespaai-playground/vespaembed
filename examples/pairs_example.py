"""
Pairs Task Example

Pairs training uses query-document pairs with in-batch negatives (MNR loss).
Ideal for semantic search and retrieval tasks.

Data format:
- anchor: The query or question
- positive: The relevant document or answer

Column aliases supported:
- anchor: query, question, sent1, sentence1, text1
- positive: document, answer, pos, sent2, sentence2, text2

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
    --task pairs \\
    --data examples/data/pairs.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --epochs 3 \\
    --batch-size 32

# Train with HuggingFace dataset
vespaembed train \\
    --task pairs \\
    --data sentence-transformers/gooaq \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2

# Train with YAML config
vespaembed train --config examples/configs/pairs.yaml
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
    UnslothConfig,
)
from vespaembed.core.trainer import VespaEmbedTrainer  # noqa: E402


def train_pairs_basic():
    """Basic pairs training with minimal configuration."""
    config = TrainingConfig(
        task="pairs",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/pairs.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_pairs_with_eval():
    """Pairs training with evaluation data."""
    config = TrainingConfig(
        task="pairs",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="examples/data/pairs.csv",
            eval="examples/data/pairs.csv",
        ),
        training=TrainingHyperparameters(
            epochs=3,
            batch_size=32,
            learning_rate=2e-5,
            optimizer="adamw_torch",
            scheduler="linear",
            eval_steps=100,
        ),
        output=OutputConfig(dir="./output/pairs-model"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_pairs_huggingface():
    """Pairs training with HuggingFace dataset."""
    config = TrainingConfig(
        task="pairs",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/gooaq",
            split="train[:10000]",  # Use subset for faster training
        ),
        training=TrainingHyperparameters(
            epochs=1,
            batch_size=64,
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_pairs_with_lora():
    """Pairs training with LoRA for memory efficiency."""
    config = TrainingConfig(
        task="pairs",
        base_model="BAAI/bge-base-en-v1.5",
        data=DataConfig(train="examples/data/pairs.csv"),
        training=TrainingHyperparameters(
            epochs=3,
            batch_size=16,
            learning_rate=1e-5,
            fp16=True,
        ),
        gradient_checkpointing=True,
        lora=LoraConfig(
            enabled=True,
            r=64,
            alpha=128,
            dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_pairs_with_unsloth():
    """Pairs training with Unsloth acceleration."""
    config = TrainingConfig(
        task="pairs",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/pairs.csv"),
        training=TrainingHyperparameters(
            epochs=3,
            batch_size=32,
            bf16=True,  # BF16 recommended for Unsloth
        ),
        gradient_checkpointing=True,
        unsloth=UnslothConfig(
            enabled=True,
            save_method="merged_16bit",
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def use_trained_model():
    """Example of using a trained model for inference."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    # Load the trained model
    model = SentenceTransformer("./output/pairs-model/final")

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
    similarities = cos_sim(query_embeddings, doc_embeddings)
    print("Similarity matrix:")
    print(similarities)


if __name__ == "__main__":
    print("Training pairs model...")
    train_pairs_basic()
    print("Training complete!")
