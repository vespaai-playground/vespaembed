from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.losses import MatryoshkaLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask
from vespaembed.tasks.mnr import MNRTask


@Registry.register_task("matryoshka")
class MatryoshkaTask(BaseTask):
    """Matryoshka Representation Learning task.

    Trains embeddings that work at multiple dimensions.
    Wraps MNR loss with MatryoshkaLoss and evaluates at each dimension.
    """

    name = "matryoshka"
    description = "Matryoshka embeddings - multi-dimensional representation learning"

    # Default to MNR-style columns
    expected_columns = ["anchor", "positive"]
    column_aliases = {
        "anchor": ["query", "question", "sent1", "sentence1", "text1"],
        "positive": ["positive", "document", "answer", "pos", "sent2", "sentence2", "text2"],
    }

    batch_sampler = BatchSamplers.NO_DUPLICATES

    def __init__(self, matryoshka_dims: list[int] = None):
        """Initialize with optional dimension list.

        Args:
            matryoshka_dims: List of embedding dimensions (e.g., [768, 512, 256, 128])
        """
        super().__init__()
        self.matryoshka_dims = matryoshka_dims or [768, 512, 256, 128, 64]
        self._inner_task = MNRTask()

    def get_loss(self, model: SentenceTransformer, **kwargs) -> MatryoshkaLoss:
        """Return MatryoshkaLoss wrapping MNR loss."""
        inner_loss = self._inner_task.get_loss(model, **kwargs)
        return MatryoshkaLoss(
            model=model,
            loss=inner_loss,
            matryoshka_dims=self.matryoshka_dims,
        )

    def get_evaluator(self, eval_dataset: Dataset) -> SequentialEvaluator:
        """Return evaluators for each matryoshka dimension.

        Creates an InformationRetrievalEvaluator for each dimension to verify
        that the model performs well at all truncated dimensions.
        """
        # Build corpus and queries
        queries = {str(i): text for i, text in enumerate(eval_dataset["anchor"])}
        corpus = {str(i): text for i, text in enumerate(eval_dataset["positive"])}
        relevant_docs = {str(i): {str(i)} for i in range(len(eval_dataset))}

        # Create evaluator for each dimension
        evaluators = []
        for dim in self.matryoshka_dims:
            evaluators.append(
                InformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name=f"matryoshka-{dim}",
                    truncate_dim=dim,
                )
            )

        return SequentialEvaluator(evaluators)
