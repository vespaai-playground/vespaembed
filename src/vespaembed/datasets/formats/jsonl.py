import json
from pathlib import Path

from datasets import Dataset


def load_jsonl(path: str) -> Dataset:
    """Load a JSONL file as a HuggingFace Dataset.

    Args:
        path: Path to JSONL file

    Returns:
        HuggingFace Dataset
    """
    records = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}")

    return Dataset.from_list(records)
