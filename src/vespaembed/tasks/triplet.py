from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("triplet")
class TripletTask(BaseTask):
    """Triplet training task.

    Uses anchor, positive, negative triplets for training.
    Effective for learning fine-grained similarity.
    """

    name = "triplet"
    description = "Triplet training - learn from anchor/positive/negative examples"

    expected_columns = ["anchor", "positive", "negative"]
    column_aliases = {
        "anchor": ["query", "question", "sent1", "sentence1", "text1"],
        "positive": ["pos", "document", "answer", "sent2", "sentence2", "text2"],
        "negative": ["neg", "hard_negative", "sent3", "sentence3", "text3"],
    }

    batch_sampler = BatchSamplers.NO_DUPLICATES

    def get_loss(self, model: SentenceTransformer, **kwargs) -> MultipleNegativesRankingLoss:
        """Return MultipleNegativesRankingLoss (handles triplets natively)."""
        return MultipleNegativesRankingLoss(model, **kwargs)

    def get_evaluator(self, eval_dataset: Dataset) -> TripletEvaluator:
        """Return TripletEvaluator."""
        return TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="triplet-eval",
        )
