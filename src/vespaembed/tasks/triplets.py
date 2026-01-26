from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import (
    CachedGISTEmbedLoss,
    CachedMultipleNegativesRankingLoss,
    GISTEmbedLoss,
    MultipleNegativesRankingLoss,
    MultipleNegativesSymmetricRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("triplets")
class TripletsTask(BaseTask):
    """Text triplets task for semantic search and retrieval.

    Use this when you have query-document pairs WITH explicit hard negatives.
    The loss uses both explicit negatives and in-batch negatives.

    Data format:
    - anchor: Query/question text
    - positive: Relevant document/answer
    - negative: Hard negative (irrelevant but similar document)

    Loss variants:
    - mnr: MultipleNegativesRankingLoss (default) - explicit + in-batch negatives
    - mnr_symmetric: Bidirectional ranking - use if you need "given answer, find query"
    - gist: GISTEmbedLoss - uses guide model to filter false negatives
    - cached_mnr: Cached version - allows larger effective batch sizes
    - cached_gist: Cached GIST - combines both benefits

    Tip: Hard negatives significantly improve model quality. Use mine_hard_negatives()
    to generate them from pairs data.
    """

    name = "triplets"
    description = "Text triplets for semantic search (anchor, positive, negative)"

    expected_columns = ["anchor", "positive", "negative"]
    column_aliases = {
        "anchor": ["query", "question", "sent1", "sentence1", "text1", "premise"],
        "positive": ["document", "answer", "pos", "sent2", "sentence2", "text2", "entailment"],
        "negative": ["neg", "hard_negative", "sent3", "sentence3", "text3", "contradiction"],
    }

    # Loss variants (same as pairs)
    loss_options = ["mnr", "mnr_symmetric", "gist", "cached_mnr", "cached_gist"]
    default_loss = "mnr"

    batch_sampler = BatchSamplers.NO_DUPLICATES

    def get_loss(self, model: SentenceTransformer, **kwargs) -> MultipleNegativesRankingLoss:
        """Return the selected loss variant."""
        guide_model = kwargs.pop("guide_model", None)
        mini_batch_size = kwargs.pop("mini_batch_size", 32)

        if self._loss_variant == "mnr":
            return MultipleNegativesRankingLoss(model, **kwargs)

        elif self._loss_variant == "mnr_symmetric":
            return MultipleNegativesSymmetricRankingLoss(model, **kwargs)

        elif self._loss_variant == "gist":
            if guide_model is None:
                guide_model = model
            return GISTEmbedLoss(model, guide=guide_model, **kwargs)

        elif self._loss_variant == "cached_mnr":
            return CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size, **kwargs)

        elif self._loss_variant == "cached_gist":
            if guide_model is None:
                guide_model = model
            return CachedGISTEmbedLoss(model, guide=guide_model, mini_batch_size=mini_batch_size, **kwargs)

        else:
            return MultipleNegativesRankingLoss(model, **kwargs)

    def get_evaluator(self, eval_dataset: Dataset):
        """Return TripletEvaluator for triplet data."""
        return TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="triplets-eval",
        )
