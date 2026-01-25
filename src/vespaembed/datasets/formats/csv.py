import pandas as pd
from datasets import Dataset


def load_csv(path: str) -> Dataset:
    """Load a CSV file as a HuggingFace Dataset.

    Args:
        path: Path to CSV file

    Returns:
        HuggingFace Dataset
    """
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)
