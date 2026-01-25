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
            # Labels are already integers (0/1) - create mappings
            self._label_to_idx = {"dissimilar": 0, "similar": 1}
            self._idx_to_label = {0: "dissimilar", 1: "similar"}

        return dataset

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
