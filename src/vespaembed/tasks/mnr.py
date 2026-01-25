from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("mnr")
class MNRTask(BaseTask):
    """Multiple Negatives Ranking task.

    Uses in-batch negatives for contrastive learning.
    Ideal for semantic search and retrieval tasks.
    """

    name = "mnr"
    description = "Multiple Negatives Ranking - semantic search with in-batch negatives"

    expected_columns = ["anchor", "positive"]
    column_aliases = {
        "anchor": ["query", "question", "sent1", "sentence1", "text1"],
        "positive": ["positive", "document", "answer", "pos", "sent2", "sentence2", "text2"],
    }

    batch_sampler = BatchSamplers.NO_DUPLICATES

    def get_loss(self, model: SentenceTransformer, **kwargs) -> MultipleNegativesRankingLoss:
        """Return MultipleNegativesRankingLoss."""
        return MultipleNegativesRankingLoss(model, **kwargs)

    def get_evaluator(self, eval_dataset: Dataset) -> InformationRetrievalEvaluator:
        """Return InformationRetrievalEvaluator."""
        # Build corpus and queries
        queries = {str(i): text for i, text in enumerate(eval_dataset["anchor"])}
        corpus = {str(i): text for i, text in enumerate(eval_dataset["positive"])}

        # Each query maps to its corresponding positive (same index)
        relevant_docs = {str(i): {str(i)} for i in range(len(eval_dataset))}

        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="mnr-eval",
        )
