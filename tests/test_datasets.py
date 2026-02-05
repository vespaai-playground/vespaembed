"""Tests for the datasets module."""

import pytest
from datasets import Dataset

from vespaembed.datasets.loader import get_columns, preview_dataset, split_dataset


class TestSplitDataset:
    """Test split_dataset function."""

    def test_split_dataset_1_percent(self):
        """Test splitting dataset with 1% for eval."""
        # Create a sample dataset with 1000 rows
        data = {"text": [f"text_{i}" for i in range(1000)]}
        dataset = Dataset.from_dict(data)

        train, eval_set = split_dataset(dataset, eval_pct=1.0)

        # Check sizes (1% = 10 rows for eval, 990 for train)
        assert len(train) == 990
        assert len(eval_set) == 10
        assert len(train) + len(eval_set) == len(dataset)

    def test_split_dataset_10_percent(self):
        """Test splitting dataset with 10% for eval."""
        data = {"text": [f"text_{i}" for i in range(100)]}
        dataset = Dataset.from_dict(data)

        train, eval_set = split_dataset(dataset, eval_pct=10.0)

        # Check sizes (10% = 10 rows for eval, 90 for train)
        assert len(train) == 90
        assert len(eval_set) == 10

    def test_split_dataset_50_percent(self):
        """Test splitting dataset with 50% for eval (maximum)."""
        data = {"text": [f"text_{i}" for i in range(100)]}
        dataset = Dataset.from_dict(data)

        train, eval_set = split_dataset(dataset, eval_pct=50.0)

        # Check sizes (50% = 50 rows for eval, 50 for train)
        assert len(train) == 50
        assert len(eval_set) == 50

    def test_split_dataset_preserves_columns(self):
        """Test that split preserves all columns."""
        data = {
            "anchor": ["a1", "a2", "a3", "a4"],
            "positive": ["p1", "p2", "p3", "p4"],
        }
        dataset = Dataset.from_dict(data)

        train, eval_set = split_dataset(dataset, eval_pct=25.0)

        # Check columns are preserved
        assert train.column_names == ["anchor", "positive"]
        assert eval_set.column_names == ["anchor", "positive"]

    def test_split_dataset_invalid_pct_too_low(self):
        """Test that eval_pct must be greater than 0."""
        data = {"text": ["text1", "text2"]}
        dataset = Dataset.from_dict(data)

        with pytest.raises(ValueError, match="must be between 0.1 and 50"):
            split_dataset(dataset, eval_pct=0.0)

    def test_split_dataset_invalid_pct_too_high(self):
        """Test that eval_pct must be at most 50."""
        data = {"text": ["text1", "text2"]}
        dataset = Dataset.from_dict(data)

        with pytest.raises(ValueError, match="must be between 0.1 and 50"):
            split_dataset(dataset, eval_pct=51.0)

    def test_split_dataset_deterministic(self):
        """Test that split is deterministic (same seed produces same split)."""
        data = {"text": [f"text_{i}" for i in range(100)]}
        dataset = Dataset.from_dict(data)

        # Split twice
        train1, eval1 = split_dataset(dataset, eval_pct=10.0)
        train2, eval2 = split_dataset(dataset, eval_pct=10.0)

        # Check that splits are identical
        assert train1["text"] == train2["text"]
        assert eval1["text"] == eval2["text"]


class TestPreviewDataset:
    """Test preview_dataset function."""

    def test_preview_default_samples(self):
        """Test preview with default number of samples."""
        data = {"text": [f"text_{i}" for i in range(10)]}
        dataset = Dataset.from_dict(data)

        preview = preview_dataset(dataset)

        assert len(preview) == 5  # Default is 5
        assert preview[0]["text"] == "text_0"
        assert preview[4]["text"] == "text_4"

    def test_preview_custom_samples(self):
        """Test preview with custom number of samples."""
        data = {"text": [f"text_{i}" for i in range(10)]}
        dataset = Dataset.from_dict(data)

        preview = preview_dataset(dataset, num_samples=3)

        assert len(preview) == 3
        assert preview[0]["text"] == "text_0"
        assert preview[2]["text"] == "text_2"

    def test_preview_fewer_samples_than_requested(self):
        """Test preview when dataset has fewer samples than requested."""
        data = {"text": ["text_0", "text_1"]}
        dataset = Dataset.from_dict(data)

        preview = preview_dataset(dataset, num_samples=5)

        assert len(preview) == 2  # Only 2 available
        assert preview[0]["text"] == "text_0"
        assert preview[1]["text"] == "text_1"


class TestGetColumns:
    """Test get_columns function."""

    def test_get_columns_single(self):
        """Test getting columns from a dataset with one column."""
        data = {"text": ["text1", "text2"]}
        dataset = Dataset.from_dict(data)

        columns = get_columns(dataset)

        assert columns == ["text"]

    def test_get_columns_multiple(self):
        """Test getting columns from a dataset with multiple columns."""
        data = {
            "anchor": ["a1", "a2"],
            "positive": ["p1", "p2"],
            "negative": ["n1", "n2"],
        }
        dataset = Dataset.from_dict(data)

        columns = get_columns(dataset)

        assert columns == ["anchor", "positive", "negative"]
