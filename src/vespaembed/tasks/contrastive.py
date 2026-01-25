from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("contrastive")
class ContrastiveTask(BaseTask):
    """Contrastive learning task with binary labels.

    Uses pairs of sentences with 0/1 labels indicating similarity.
    Good for duplicate detection and paraphrase identification.
    """

    name = "contrastive"
    description = "Contrastive learning - pairs with binary similarity labels"

    expected_columns = ["sentence1", "sentence2", "label"]
    column_aliases = {
        "sentence1": ["sent1", "text1", "anchor", "query"],
        "sentence2": ["sent2", "text2", "positive", "document"],
        "label": ["similarity", "is_similar", "is_duplicate"],
    }

    batch_sampler = BatchSamplers.BATCH_SAMPLER

    def get_loss(self, model: SentenceTransformer, **kwargs) -> ContrastiveLoss:
        """Return ContrastiveLoss."""
        return ContrastiveLoss(model, **kwargs)

    def get_evaluator(self, eval_dataset: Dataset) -> BinaryClassificationEvaluator:
        """Return BinaryClassificationEvaluator."""
        return BinaryClassificationEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            labels=eval_dataset["label"],
            name="contrastive-eval",
        )
