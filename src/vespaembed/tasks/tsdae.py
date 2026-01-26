import random

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


def _add_noise(text: str, del_ratio: float = 0.6) -> str:
    """Add noise to text by randomly deleting words.

    Args:
        text: Input text string
        del_ratio: Probability of keeping each word (default 0.6 = 60% kept)

    Returns:
        Noisy version of the text with some words randomly deleted
    """
    words = text.split()
    if not words:
        return text
    # Keep words with probability del_ratio
    kept_words = [word for word in words if random.random() < del_ratio]
    if len(kept_words) == 0:
        # Keep at least one random word
        return random.choice(words)
    return " ".join(kept_words)


@Registry.register_task("tsdae")
class TSDAETask(BaseTask):
    """TSDAE (Transformer-based Sequential Denoising Auto-Encoder) task.

    Unsupervised training that learns embeddings by reconstructing
    corrupted input sentences. Useful for domain adaptation when
    you only have unlabeled text.

    Data format:
    - text/sentence: Raw text sentences (no labels needed)

    The task automatically adds noise by randomly deleting ~40% of words
    from the input text. The model learns to reconstruct the original
    text from the corrupted version.

    Reference: https://arxiv.org/abs/2104.06979
    """

    name = "tsdae"
    description = "TSDAE - unsupervised domain adaptation with denoising auto-encoder"

    expected_columns = ["text"]
    column_aliases = {
        "text": ["sentence", "sentences", "content", "input"],
    }

    batch_sampler = BatchSamplers.BATCH_SAMPLER

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for TSDAE training with noise.

        Adds noise to text by randomly deleting words (40% deletion rate).
        The model learns to reconstruct the original text from the noisy input.

        Args:
            dataset: Input dataset with 'text' column

        Returns:
            Dataset with 'anchor' (noisy) and 'positive' (original) columns
        """
        # First, apply base class normalization (handles column aliases)
        dataset = super().prepare_dataset(dataset)

        # Create noisy versions by randomly deleting words
        # anchor = noisy text (input), positive = original text (target)
        dataset = dataset.map(
            lambda x: {"anchor": _add_noise(x["text"]), "positive": x["text"]},
            desc="Adding noise to text",
        )

        # Remove the original text column as we now have anchor/positive
        dataset = dataset.remove_columns(["text"])

        return dataset

    def get_loss(self, model: SentenceTransformer, **kwargs) -> DenoisingAutoEncoderLoss:
        """Return DenoisingAutoEncoderLoss for unsupervised training."""
        return DenoisingAutoEncoderLoss(
            model=model,
            tie_encoder_decoder=True,
            **kwargs,
        )

    def get_evaluator(self, eval_dataset: Dataset):
        """Return None - TSDAE has no intrinsic evaluator.

        Evaluation should be done on downstream tasks (e.g., STS, retrieval)
        that match your target use case.
        """
        return None
