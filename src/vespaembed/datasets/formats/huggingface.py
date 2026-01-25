import os
from typing import Optional

from datasets import Dataset
from datasets import load_dataset as hf_load_dataset

# Get HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")


def load_hf_dataset(
    name: str,
    subset: Optional[str] = None,
    split: str = "train",
) -> Dataset:
    """Load a dataset from the HuggingFace Hub.

    Args:
        name: Dataset name (e.g., "sentence-transformers/all-nli")
        subset: Dataset subset/configuration (optional)
        split: Dataset split (default: "train")

    Returns:
        HuggingFace Dataset

    Note:
        Uses HF_TOKEN environment variable for authentication if set.
    """
    if subset:
        dataset = hf_load_dataset(name, subset, split=split, token=HF_TOKEN)
    else:
        dataset = hf_load_dataset(name, split=split, token=HF_TOKEN)

    return dataset
