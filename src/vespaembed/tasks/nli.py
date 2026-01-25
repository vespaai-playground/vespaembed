from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import SoftmaxLoss
from sentence_transformers.training_args import BatchSamplers

from vespaembed.core.registry import Registry
from vespaembed.tasks.base import BaseTask


@Registry.register_task("nli")
class NLITask(BaseTask):
    """Natural Language Inference training task.

    Uses sentence pairs with entailment labels (entailment, neutral, contradiction)
    to learn embeddings. This is the classic SBERT training approach.

    Data format:
    - premise/sentence1: First sentence
    - hypothesis/sentence2: Second sentence
    - label: 0 (entailment), 1 (neutral), 2 (contradiction)
    """

    name = "nli"
    description = "Natural Language Inference - sentence pairs with entailment labels"

    expected_columns = ["sentence1", "sentence2", "label"]
    column_aliases = {
        "sentence1": ["premise", "sent1", "text1", "anchor"],
        "sentence2": ["hypothesis", "sent2", "text2", "positive"],
        "label": ["gold_label", "class"],
    }

    batch_sampler = BatchSamplers.BATCH_SAMPLER

    def __init__(self):
        super().__init__()
        self._num_labels = 3  # entailment, neutral, contradiction

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset and ensure labels are integers."""
        from datasets import Features, Value

        dataset = super().prepare_dataset(dataset)

        # Map string labels to integers if needed
        label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }

        # Check if labels need conversion
        sample_label = dataset["label"][0]
        if isinstance(sample_label, str):
            new_features = Features({
                "sentence1": dataset.features["sentence1"],
                "sentence2": dataset.features["sentence2"],
                "label": Value("int64"),
            })
            dataset = dataset.map(
                lambda x: {"label": label_map.get(x["label"].lower(), int(x["label"]))},
                features=new_features,
            )

        return dataset

    def get_loss(self, model: SentenceTransformer, **kwargs) -> SoftmaxLoss:
        """Return SoftmaxLoss for NLI classification."""
        return SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=self._num_labels,
            **kwargs,
        )

    def get_evaluator(self, eval_dataset: Dataset) -> EmbeddingSimilarityEvaluator:
        """Return EmbeddingSimilarityEvaluator.

        Note: NLI models are typically evaluated on STS benchmarks
        to measure embedding quality, not on NLI accuracy.
        """
        return EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            # Use label as similarity proxy (entailment=1.0, neutral=0.5, contradiction=0.0)
            scores=[1.0 if l == 0 else (0.5 if l == 1 else 0.0) for l in eval_dataset["label"]],
            name="nli-eval",
        )
