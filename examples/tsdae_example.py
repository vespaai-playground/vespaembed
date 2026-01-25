"""
TSDAE (Transformer-based Sequential Denoising Auto-Encoder) Example

TSDAE is an unsupervised training method for domain adaptation. It learns
embeddings by reconstructing corrupted input sentences, making it ideal
when you only have unlabeled text from your target domain.

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
    --project my-domain-adapted-model \\
    --epochs 3 \\
    --batch-size 32

# Train with HuggingFace dataset (any dataset with text column)
vespaembed train \\
    --task tsdae \\
    --data sentence-transformers/simple-wiki \\
    --subset simple \\
    --split train \\
    --base-model sentence-transformers/all-MiniLM-L6-v2 \\
    --project wiki-adapted-model
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
        data=DataConfig(
            train="examples/data/tsdae.csv",
        ),
        training=TrainingHyperparameters(
            epochs=5,
            batch_size=16,
            learning_rate=3e-5,
        ),
        output=OutputConfig(
            dir="./output/domain-adapted-model",
        ),
    )

    trainer = VespaEmbedTrainer(config)
    model = trainer.train()
    return model


if __name__ == "__main__":
    print("Training TSDAE model for domain adaptation...")
    train_tsdae_basic()
    print("Training complete!")
