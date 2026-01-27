"""Tests for the task registry and task classes."""

import pytest

# Import tasks to register them
import vespaembed.tasks  # noqa: F401
from vespaembed.core.registry import DEFAULT_HYPERPARAMETERS, TASK_SAMPLE_DATA, Registry


class TestRegistry:
    """Test the Registry class."""

    def test_list_tasks(self):
        """Test listing all registered tasks."""
        tasks = Registry.list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 4
        assert "pairs" in tasks
        assert "triplets" in tasks
        assert "similarity" in tasks
        assert "tsdae" in tasks

    def test_get_task(self):
        """Test getting a task class by name."""
        task_cls = Registry.get_task("pairs")
        assert task_cls is not None
        assert task_cls.name == "pairs"

    def test_get_unknown_task_raises(self):
        """Test that getting an unknown task raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Registry.get_task("unknown_task")
        assert "Unknown task" in str(exc_info.value)
        assert "unknown_task" in str(exc_info.value)

    def test_get_task_info_all(self):
        """Test getting info for all tasks."""
        tasks = Registry.get_task_info()
        assert isinstance(tasks, list)
        assert len(tasks) == 4

        for task in tasks:
            assert "name" in task
            assert "description" in task
            assert "expected_columns" in task
            assert "hyperparameters" in task
            assert "sample_data" in task

    def test_get_task_info_single(self):
        """Test getting info for a single task."""
        task = Registry.get_task_info("pairs")
        assert isinstance(task, dict)
        assert task["name"] == "pairs"
        assert task["expected_columns"] == ["anchor", "positive"]

    def test_get_task_info_unknown_raises(self):
        """Test that getting info for unknown task raises ValueError."""
        with pytest.raises(ValueError):
            Registry.get_task_info("unknown_task")


class TestDefaultHyperparameters:
    """Test default hyperparameters."""

    def test_has_all_required_params(self):
        """Test that all required hyperparameters are defined."""
        required = [
            "epochs",
            "batch_size",
            "learning_rate",
            "warmup_ratio",
            "weight_decay",
            "fp16",
            "bf16",
            "eval_steps",
            "save_steps",
            "logging_steps",
            "gradient_accumulation_steps",
            "optimizer",
            "scheduler",
        ]
        for param in required:
            assert param in DEFAULT_HYPERPARAMETERS

    def test_sensible_defaults(self):
        """Test that default values are sensible."""
        assert DEFAULT_HYPERPARAMETERS["epochs"] == 3
        assert DEFAULT_HYPERPARAMETERS["batch_size"] == 32
        assert DEFAULT_HYPERPARAMETERS["learning_rate"] == 2e-5
        assert DEFAULT_HYPERPARAMETERS["fp16"] is True
        assert DEFAULT_HYPERPARAMETERS["bf16"] is False


class TestTaskSampleData:
    """Test task sample data."""

    def test_all_tasks_have_sample_data(self):
        """Test that all tasks have sample data."""
        tasks = ["pairs", "triplets", "similarity", "tsdae"]
        for task in tasks:
            assert task in TASK_SAMPLE_DATA
            assert len(TASK_SAMPLE_DATA[task]) > 0

    def test_pairs_sample_data(self):
        """Test pairs sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["pairs"]
        assert len(samples) >= 1
        for sample in samples:
            assert "anchor" in sample
            assert "positive" in sample

    def test_triplets_sample_data(self):
        """Test triplets sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["triplets"]
        assert len(samples) >= 1
        for sample in samples:
            assert "anchor" in sample
            assert "positive" in sample
            assert "negative" in sample

    def test_similarity_sample_data(self):
        """Test similarity sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["similarity"]
        assert len(samples) >= 1
        for sample in samples:
            assert "sentence1" in sample
            assert "sentence2" in sample
            assert "score" in sample
            assert 0 <= sample["score"] <= 1

    def test_tsdae_sample_data(self):
        """Test TSDAE sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["tsdae"]
        assert len(samples) >= 1
        for sample in samples:
            assert "text" in sample


class TestPairsTask:
    """Test pairs task class."""

    def test_task_attributes(self):
        """Test pairs task has correct attributes."""
        task_cls = Registry.get_task("pairs")
        assert task_cls.name == "pairs"
        assert task_cls.expected_columns == ["anchor", "positive"]
        assert "anchor" in task_cls.column_aliases
        assert "positive" in task_cls.column_aliases

    def test_column_aliases(self):
        """Test pairs task column aliases."""
        task_cls = Registry.get_task("pairs")
        assert "query" in task_cls.column_aliases["anchor"]
        assert "question" in task_cls.column_aliases["anchor"]
        assert "document" in task_cls.column_aliases["positive"]

    def test_loss_options(self):
        """Test pairs task has loss options."""
        task_cls = Registry.get_task("pairs")
        assert "mnr" in task_cls.loss_options
        assert "mnr_symmetric" in task_cls.loss_options
        assert "gist" in task_cls.loss_options
        assert task_cls.default_loss == "mnr"

    def test_loss_variant_selection(self):
        """Test pairs task loss variant selection."""
        task_cls = Registry.get_task("pairs")
        # Default loss
        task = task_cls()
        assert task.loss_variant == "mnr"
        # Custom loss
        task = task_cls(loss_variant="mnr_symmetric")
        assert task.loss_variant == "mnr_symmetric"

    def test_invalid_loss_variant_raises(self):
        """Test that invalid loss variant raises ValueError."""
        task_cls = Registry.get_task("pairs")
        with pytest.raises(ValueError) as exc_info:
            task_cls(loss_variant="invalid_loss")
        assert "Unknown loss variant" in str(exc_info.value)


class TestTripletsTask:
    """Test triplets task class."""

    def test_task_attributes(self):
        """Test triplets task has correct attributes."""
        task_cls = Registry.get_task("triplets")
        assert task_cls.name == "triplets"
        assert task_cls.expected_columns == ["anchor", "positive", "negative"]

    def test_column_aliases(self):
        """Test triplets task column aliases."""
        task_cls = Registry.get_task("triplets")
        assert "query" in task_cls.column_aliases["anchor"]
        assert "document" in task_cls.column_aliases["positive"]
        assert "hard_negative" in task_cls.column_aliases["negative"]

    def test_loss_options(self):
        """Test triplets task has loss options."""
        task_cls = Registry.get_task("triplets")
        assert "mnr" in task_cls.loss_options
        assert "mnr_symmetric" in task_cls.loss_options
        assert task_cls.default_loss == "mnr"


class TestSimilarityTask:
    """Test similarity task class."""

    def test_task_attributes(self):
        """Test similarity task has correct attributes."""
        task_cls = Registry.get_task("similarity")
        assert task_cls.name == "similarity"
        assert task_cls.expected_columns == ["sentence1", "sentence2", "score"]

    def test_column_aliases(self):
        """Test similarity task column aliases."""
        task_cls = Registry.get_task("similarity")
        assert "sent1" in task_cls.column_aliases["sentence1"]
        assert "sent2" in task_cls.column_aliases["sentence2"]
        assert "similarity" in task_cls.column_aliases["score"]

    def test_loss_options(self):
        """Test similarity task has loss options."""
        task_cls = Registry.get_task("similarity")
        assert "cosine" in task_cls.loss_options
        assert "cosent" in task_cls.loss_options
        assert "angle" in task_cls.loss_options
        assert task_cls.default_loss == "cosine"

    def test_score_normalization(self):
        """Test that similarity task normalizes scores."""
        from datasets import Dataset

        task = Registry.get_task("similarity")()

        # Create dataset with 0-5 scale scores
        dataset = Dataset.from_dict(
            {
                "sentence1": ["A man is eating", "A woman is running", "A cat sleeps"],
                "sentence2": ["A person is eating food", "A person is exercising", "A dog runs"],
                "score": [5.0, 2.5, 1.0],  # 0-5 scale, max is 5.0
            }
        )

        prepared = task.prepare_dataset(dataset)

        # Scores should be normalized to 0-1 range (divided by max_score)
        assert all(0 <= score <= 1 for score in prepared["score"])
        assert prepared["score"][0] == pytest.approx(1.0)  # 5.0/5.0
        assert prepared["score"][1] == pytest.approx(0.5)  # 2.5/5.0
        assert prepared["score"][2] == pytest.approx(0.2)  # 1.0/5.0


class TestTSDAETask:
    """Test TSDAE task class."""

    def test_task_attributes(self):
        """Test TSDAE task has correct attributes."""
        task_cls = Registry.get_task("tsdae")
        assert task_cls.name == "tsdae"
        assert task_cls.expected_columns == ["text"]

    def test_column_aliases(self):
        """Test TSDAE task column aliases."""
        task_cls = Registry.get_task("tsdae")
        assert "sentence" in task_cls.column_aliases["text"]
        assert "content" in task_cls.column_aliases["text"]

    def test_no_loss_options(self):
        """Test TSDAE task has no loss variants (only TSDAE loss)."""
        task_cls = Registry.get_task("tsdae")
        assert task_cls.loss_options == []

    def test_prepare_dataset_adds_noise(self):
        """Test that TSDAE prepare_dataset adds noisy anchor column."""
        from datasets import Dataset

        task = Registry.get_task("tsdae")()

        dataset = Dataset.from_dict(
            {
                "text": ["This is a test sentence with many words."],
            }
        )

        prepared = task.prepare_dataset(dataset)

        # Should have anchor (noisy) and positive (original) columns
        assert "anchor" in prepared.column_names
        assert "positive" in prepared.column_names
        assert "text" not in prepared.column_names

        # Positive should be original text
        assert prepared["positive"][0] == "This is a test sentence with many words."

        # Anchor should be shorter (words deleted)
        assert len(prepared["anchor"][0].split()) <= len(prepared["positive"][0].split())


class TestBaseTaskFunctionality:
    """Test common BaseTask functionality across all tasks."""

    def test_all_tasks_have_required_attributes(self):
        """Test that all tasks have required attributes."""
        for task_name in Registry.list_tasks():
            task_cls = Registry.get_task(task_name)
            assert hasattr(task_cls, "name")
            assert hasattr(task_cls, "description")
            assert hasattr(task_cls, "expected_columns")
            assert hasattr(task_cls, "column_aliases")
            assert hasattr(task_cls, "get_loss")
            assert hasattr(task_cls, "get_evaluator")
            assert hasattr(task_cls, "prepare_dataset")

    def test_tasks_have_no_labels(self):
        """Test that current tasks don't use label encoding."""
        for task_name in Registry.list_tasks():
            task = Registry.get_task(task_name)()
            assert task.label_to_idx is None
            assert task.idx_to_label is None
            assert task.num_labels is None
            assert task.get_label_config() is None

    def test_column_alias_resolution(self):
        """Test that column aliases are resolved correctly."""
        from datasets import Dataset

        task = Registry.get_task("pairs")()

        # Create dataset with aliased column names
        dataset = Dataset.from_dict(
            {
                "query": ["What is AI?", "How does ML work?"],
                "document": ["AI is artificial intelligence", "ML learns from data"],
            }
        )

        prepared = task.prepare_dataset(dataset)

        # Columns should be renamed to canonical names
        assert "anchor" in prepared.column_names
        assert "positive" in prepared.column_names
        assert "query" not in prepared.column_names
        assert "document" not in prepared.column_names

    def test_missing_required_columns_raises(self):
        """Test that missing required columns raise ValueError."""
        from datasets import Dataset

        task = Registry.get_task("pairs")()

        # Create dataset missing required columns
        dataset = Dataset.from_dict(
            {
                "text": ["Some text"],
            }
        )

        with pytest.raises(ValueError) as exc_info:
            task.prepare_dataset(dataset)
        assert "Missing required columns" in str(exc_info.value)
