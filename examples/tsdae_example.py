"""
TSDAE Task Example

TSDAE (Transformer-based Sequential Denoising Auto-Encoder) is an unsupervised
training method for domain adaptation. It learns embeddings by reconstructing
corrupted input sentences, making it ideal when you only have unlabeled text
from your target domain.

Data format:
- text: Raw text sentences (no labels needed)

Column aliases supported:
- text: sentence, sentences, content, input

Reference: https://arxiv.org/abs/2104.06979
"""

# =============================================================================
# CLI Examples
# =============================================================================

"""
# Train with local CSV file
vespaembed train \\
    --task tsdae \\
    --data examples/data/tsdae.csv \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --epochs 3

# Train with HuggingFace dataset (any dataset with text column)
vespaembed train \\
    --task tsdae \\
    --data sentence-transformers/simple-wiki \\
    --subset simple \\
    --split train[:10000] \\
    --base-model sentence-transformers/all-MiniLM-L6-v2

# Train with YAML config
vespaembed train --config examples/configs/tsdae.yaml
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


def train_tsdae_basic():
    """Basic TSDAE training with minimal configuration."""
    config = TrainingConfig(
        task="tsdae",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/tsdae.csv"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_tsdae_domain_adaptation():
    """TSDAE for domain adaptation with custom settings."""
    config = TrainingConfig(
        task="tsdae",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/tsdae.csv"),
        training=TrainingHyperparameters(
            epochs=5,
            batch_size=16,
            learning_rate=3e-5,
            optimizer="adamw_torch",
            scheduler="cosine",
        ),
        output=OutputConfig(dir="./output/domain-adapted-model"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_tsdae_wikipedia():
    """TSDAE training with Wikipedia for general domain adaptation."""
    config = TrainingConfig(
        task="tsdae",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(
            train="sentence-transformers/simple-wiki",
            subset="simple",
            split="train[:50000]",  # Use subset for faster training
        ),
        training=TrainingHyperparameters(
            epochs=1,
            batch_size=32,
            learning_rate=3e-5,
            optimizer="adamw_torch",
            scheduler="linear",
            logging_steps=100,
            save_steps=1000,
        ),
        output=OutputConfig(dir="./output/wiki-adapted-model"),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


def train_tsdae_with_lora():
    """TSDAE training with LoRA for memory efficiency."""
    config = TrainingConfig(
        task="tsdae",
        base_model="BAAI/bge-base-en-v1.5",
        data=DataConfig(train="examples/data/tsdae.csv"),
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


def train_tsdae_with_unsloth():
    """TSDAE training with Unsloth acceleration."""
    config = TrainingConfig(
        task="tsdae",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        data=DataConfig(train="examples/data/tsdae.csv"),
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


def use_domain_adapted_model():
    """Example of using a domain-adapted model."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    # Load the domain-adapted model
    model = SentenceTransformer("./output/domain-adapted-model/final")

    # Test with domain-specific texts
    texts = [
        "The transformer architecture uses self-attention mechanisms.",
        "Neural networks learn patterns from training data.",
        "Deep learning models require large datasets.",
    ]

    # Encode and compute similarities
    embeddings = model.encode(texts)
    similarities = cos_sim(embeddings, embeddings)

    print("Domain-adapted embeddings similarity matrix:")
    print(similarities)


if __name__ == "__main__":
    print("Training TSDAE model for domain adaptation...")
    train_tsdae_basic()
    print("Training complete!")
