from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("sts")
class STSTask(BaseTask):
    """Semantic Textual Similarity task.

    Uses pairs of sentences with continuous similarity scores.
    Best for learning nuanced similarity relationships.
    """

    name = "sts"
    description = "Semantic Textual Similarity - pairs with continuous similarity scores"

    expected_columns = ["sentence1", "sentence2", "score"]
    column_aliases = {
        "sentence1": ["sent1", "text1", "anchor", "query"],
        "sentence2": ["sent2", "text2", "positive", "document"],
        "score": ["similarity", "label", "sim_score"],
    }

    batch_sampler = BatchSamplers.BATCH_SAMPLER

    def get_loss(self, model: SentenceTransformer, **kwargs) -> CoSENTLoss:
        """Return CoSENTLoss."""
        return CoSENTLoss(model, **kwargs)

    def get_evaluator(self, eval_dataset: Dataset) -> EmbeddingSimilarityEvaluator:
        """Return EmbeddingSimilarityEvaluator."""
        return EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            name="sts-eval",
        )
