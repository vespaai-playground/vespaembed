from pathlib import Path
from typing import Optional

from datasets import Dataset

from vespaembed.datasets.formats.csv import load_csv
from vespaembed.datasets.formats.huggingface import load_hf_dataset
from vespaembed.datasets.formats.jsonl import load_jsonl


def load_dataset(
    path: str,
    subset: Optional[str] = None,
    split: Optional[str] = None,
) -> Dataset:
    """Load a dataset from various sources.

    Supports:
    - CSV files (.csv)
    - JSONL files (.jsonl)
    - HuggingFace datasets (org/dataset-name)

    Args:
        path: Path to file or HuggingFace dataset name
        subset: HuggingFace dataset subset (optional)
        split: HuggingFace dataset split (optional, defaults to "train")

    Returns:
        HuggingFace Dataset object

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    path_obj = Path(path)

    # Check if it's a local file
    if path_obj.exists():
        suffix = path_obj.suffix.lower()

        if suffix == ".csv":
            return load_csv(path)
        elif suffix in (".jsonl", ".json"):
            return load_jsonl(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. " "Supported formats: .csv, .jsonl")

    # Check if it looks like a HuggingFace dataset
    if "/" in path or not path_obj.suffix:
        return load_hf_dataset(path, subset=subset, split=split or "train")

    # File doesn't exist and doesn't look like HF dataset
    raise FileNotFoundError(
        f"File not found: {path}. " "Provide a valid file path or HuggingFace dataset name (e.g., 'org/dataset-name')."
    )


def preview_dataset(dataset: Dataset, num_samples: int = 5) -> list[dict]:
    """Preview a dataset by returning the first N samples.

    Args:
        dataset: Dataset to preview
        num_samples: Number of samples to return

    Returns:
        List of sample dictionaries
    """
    return [dataset[i] for i in range(min(num_samples, len(dataset)))]


def get_columns(dataset: Dataset) -> list[str]:
    """Get column names from a dataset.

    Args:
        dataset: Dataset to inspect

    Returns:
        List of column names
    """
    return dataset.column_names


def split_dataset(dataset: Dataset, eval_pct: float) -> tuple[Dataset, Dataset]:
    """Split a dataset into train and eval sets.

    Args:
        dataset: Dataset to split
        eval_pct: Percentage of data to use for evaluation (0.1-50)

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    if eval_pct < 0.1 or eval_pct > 50:
        raise ValueError("eval_pct must be between 0.1 and 50")

    # Convert percentage to ratio
    eval_ratio = eval_pct / 100.0

    # Use train_test_split
    split_data = dataset.train_test_split(test_size=eval_ratio, seed=42)
    return split_data["train"], split_data["test"]
