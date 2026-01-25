from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("tsdae")
class TSDAETask(BaseTask):
    """TSDAE (Transformer-based Sequential Denoising Auto-Encoder) task.

    Unsupervised training that learns embeddings by reconstructing
    corrupted input sentences. Useful for domain adaptation when
    you only have unlabeled text.

    Data format:
    - text/sentence: Raw text sentences (no labels needed)

    Reference: https://arxiv.org/abs/2104.06979
    """

    name = "tsdae"
    description = "TSDAE - unsupervised domain adaptation with denoising auto-encoder"

    expected_columns = ["text"]
    column_aliases = {
        "text": ["sentence", "sentences", "content", "input"],
    }

    batch_sampler = BatchSamplers.BATCH_SAMPLER

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
