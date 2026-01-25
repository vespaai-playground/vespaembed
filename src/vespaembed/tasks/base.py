from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import BatchSamplers


class BaseTask(ABC):
    """Base class for all training tasks."""

    # Task metadata
    name: str = ""
    description: str = ""

    # Column configuration
    expected_columns: list[str] = []
    column_aliases: dict[str, list[str]] = {}

    # Batch sampler (NO_DUPLICATES for in-batch negative losses)
    batch_sampler: BatchSamplers = BatchSamplers.BATCH_SAMPLER

    def __init__(self):
        # Label encoding mappings (set by prepare_dataset if task has labels)
        self._label_to_idx: dict[str, int] | None = None
        self._idx_to_label: dict[int, str] | None = None

    @property
    def label_to_idx(self) -> dict[str, int] | None:
        """Return label to index mapping, or None if task has no labels."""
        return self._label_to_idx

    @property
    def idx_to_label(self) -> dict[int, str] | None:
        """Return index to label mapping, or None if task has no labels."""
        return self._idx_to_label

    @property
    def num_labels(self) -> int | None:
        """Return number of labels, or None if task has no labels."""
        if self._label_to_idx is None:
            return None
        return len(self._label_to_idx)

    def get_label_config(self) -> dict | None:
        """Return label configuration in HuggingFace format.

        Returns:
            Dict with id2label, label2id, and num_labels, or None if no labels
        """
        if self._label_to_idx is None:
            return None
        return {
            "id2label": {str(k): v for k, v in self._idx_to_label.items()},
            "label2id": self._label_to_idx,
            "num_labels": len(self._label_to_idx),
        }

    @abstractmethod
    def get_loss(self, model: SentenceTransformer, **kwargs) -> Any:
        """Return configured loss function from sentence-transformers.

        Args:
            model: The SentenceTransformer model
            **kwargs: Additional loss configuration

        Returns:
            Loss function instance
        """
        raise NotImplementedError

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Normalize column names and reorder for the loss function.

        Args:
            dataset: Input dataset

        Returns:
            Prepared dataset with normalized columns
        """
        # 1. Rename aliased columns to canonical names
        for canonical, aliases in self.column_aliases.items():
            for alias in aliases:
                if alias in dataset.column_names and canonical not in dataset.column_names:
                    dataset = dataset.rename_column(alias, canonical)
                    break

        # 2. Validate required columns exist
        missing = set(self.expected_columns) - set(dataset.column_names)
        if missing:
            available = ", ".join(sorted(dataset.column_names))
            raise ValueError(
                f"Missing required columns for task '{self.name}': {missing}. " f"Available columns: {available}"
            )

        # 3. Select and reorder columns
        return dataset.select_columns(self.expected_columns)

    @abstractmethod
    def get_evaluator(self, eval_dataset: Dataset) -> Any:
        """Return appropriate sentence-transformers evaluator for this task.

        Args:
            eval_dataset: Evaluation dataset (already prepared)

        Returns:
            Evaluator instance
        """
        raise NotImplementedError
