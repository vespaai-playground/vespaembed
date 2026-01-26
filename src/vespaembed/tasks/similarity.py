from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import AnglELoss, CoSENTLoss, CosineSimilarityLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("similarity")
class SimilarityTask(BaseTask):
    """Text pairs with similarity scores task.

    Use this when you have pairs of sentences with continuous similarity
    scores (e.g., 0.0 to 1.0 or 0 to 5). Common for STS (Semantic Textual
    Similarity) benchmarks.

    Data format:
    - sentence1: First sentence
    - sentence2: Second sentence
    - score: Similarity score (will be normalized to 0-1 range)

    Loss variants (similar performance, pick one):
    - cosine: CosineSimilarityLoss (default) - simple and effective
    - cosent: CoSENTLoss - ranking-based, from CoSENT paper
    - angle: AnglELoss - angle-optimized, from AnglE paper

    According to papers: AnglE >= CoSENT >= Cosine, but results are often similar.
    """

    name = "similarity"
    description = "Text pairs with similarity scores (STS-style)"

    expected_columns = ["sentence1", "sentence2", "score"]
    column_aliases = {
        "sentence1": ["sent1", "text1", "anchor", "query"],
        "sentence2": ["sent2", "text2", "positive", "document"],
        "score": ["similarity", "label", "sim_score"],
    }

    # Loss variants
    loss_options = ["cosine", "cosent", "angle"]
    default_loss = "cosine"

    batch_sampler = BatchSamplers.BATCH_SAMPLER

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset and normalize scores to 0-1 range if needed."""
        dataset = super().prepare_dataset(dataset)

        # Check if scores need normalization (e.g., 0-5 scale to 0-1)
        scores = dataset["score"]
        max_score = max(scores)

        if max_score > 1.0:
            # Normalize to 0-1 range
            dataset = dataset.map(lambda x: {"score": x["score"] / max_score})

        return dataset

    def get_loss(self, model: SentenceTransformer, **kwargs):
        """Return the selected loss variant."""
        if self._loss_variant == "cosine":
            return CosineSimilarityLoss(model, **kwargs)

        elif self._loss_variant == "cosent":
            return CoSENTLoss(model, **kwargs)

        elif self._loss_variant == "angle":
            return AnglELoss(model, **kwargs)

        else:
            # Fallback to default
            return CosineSimilarityLoss(model, **kwargs)

    def get_evaluator(self, eval_dataset: Dataset) -> EmbeddingSimilarityEvaluator:
        """Return EmbeddingSimilarityEvaluator."""
        return EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            name="similarity-eval",
        )
