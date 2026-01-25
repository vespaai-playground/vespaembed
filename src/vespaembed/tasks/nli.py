from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import SoftmaxLoss
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("nli")
class NLITask(BaseTask):
    """Sentence pair classification task using SoftmaxLoss.

    Learns embeddings by classifying sentence pairs into categories.
    Common use cases include NLI (entailment/neutral/contradiction),
    paraphrase detection, or any multi-class sentence pair classification.

    Data format:
    - sentence1: First sentence
    - sentence2: Second sentence
    - label: Class label (integer or string - auto-converted)

    Evaluation uses EmbeddingSimilarityEvaluator with labels converted to
    similarity scores (following official sentence-transformers approach).
    """

    name = "nli"
    description = "Sentence pair classification with SoftmaxLoss"

    expected_columns = ["sentence1", "sentence2", "label"]
    column_aliases = {
        "sentence1": ["premise", "sent1", "text1", "anchor"],
        "sentence2": ["hypothesis", "sent2", "text2", "positive"],
        "label": ["gold_label", "class", "category"],
    }

    batch_sampler = BatchSamplers.BATCH_SAMPLER

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset and encode labels to integers."""
        from datasets import Features, Value

        dataset = super().prepare_dataset(dataset)

        # Check if labels need conversion from strings
        sample_label = dataset["label"][0]
        if isinstance(sample_label, str):
            # Build label mappings from unique labels
            unique_labels = sorted(set(dataset["label"]))
            self._label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            self._idx_to_label = {i: label for label, i in self._label_to_idx.items()}

            new_features = Features(
                {
                    "sentence1": dataset.features["sentence1"],
                    "sentence2": dataset.features["sentence2"],
                    "label": Value("int64"),
                }
            )
            dataset = dataset.map(
                lambda x: {"label": self._label_to_idx[x["label"]]},
                features=new_features,
            )
        else:
            # Labels are already integers - create mappings from existing values
            unique_labels = sorted(set(dataset["label"]))
            self._label_to_idx = {str(label): label for label in unique_labels}
            self._idx_to_label = {label: str(label) for label in unique_labels}

        return dataset

    def get_loss(self, model: SentenceTransformer, **kwargs) -> SoftmaxLoss:
        """Return SoftmaxLoss for sentence pair classification."""
        if self.num_labels is None:
            raise ValueError("prepare_dataset must be called before get_loss")

        return SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=self.num_labels,
            **kwargs,
        )

    def _label_to_score(self, label: int) -> float:
        """Convert NLI label to similarity score.

        Standard NLI mapping:
        - 0 (entailment) -> 1.0 (high similarity)
        - 1 (neutral) -> 0.5 (medium similarity)
        - 2 (contradiction) -> 0.0 (low similarity)

        For other label counts, distribute evenly from 1.0 to 0.0.
        """
        if self.num_labels == 3:
            # Standard NLI labels
            return {0: 1.0, 1: 0.5, 2: 0.0}.get(label, 0.5)
        elif self.num_labels == 2:
            # Binary classification (e.g., similar/dissimilar)
            return 1.0 if label == 1 else 0.0
        else:
            # Distribute evenly for other label counts
            return 1.0 - (label / (self.num_labels - 1)) if self.num_labels > 1 else 0.5

    def get_evaluator(self, eval_dataset: Dataset) -> EmbeddingSimilarityEvaluator:
        """Return EmbeddingSimilarityEvaluator with labels converted to scores.

        Following the official sentence-transformers approach, NLI models are
        evaluated using embedding similarity on sentence pairs.
        """
        # Convert labels to similarity scores
        scores = [self._label_to_score(label) for label in eval_dataset["label"]]

        return EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=scores,
            main_similarity=SimilarityFunction.COSINE,
            name="nli-eval",
        )
