"""Tests for the task registry and task classes."""

import pytest

# Import tasks to register them
import vespaembed.tasks  # noqa: F401
from vespaembed.core.registry import DEFAULT_HYPERPARAMETERS, TASK_SAMPLE_DATA, TASK_SPECIFIC_PARAMS, Registry


class TestRegistry:
    """Test the Registry class."""

    def test_list_tasks(self):
        """Test listing all registered tasks."""
        tasks = Registry.list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 7
        assert "mnr" in tasks
        assert "triplet" in tasks
        assert "contrastive" in tasks
        assert "sts" in tasks
        assert "nli" in tasks
        assert "tsdae" in tasks
        assert "matryoshka" in tasks

    def test_get_task(self):
        """Test getting a task class by name."""
        task_cls = Registry.get_task("mnr")
        assert task_cls is not None
        assert task_cls.name == "mnr"

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
        assert len(tasks) == 7

        for task in tasks:
            assert "name" in task
            assert "description" in task
            assert "expected_columns" in task
            assert "hyperparameters" in task
            assert "sample_data" in task

    def test_get_task_info_single(self):
        """Test getting info for a single task."""
        task = Registry.get_task_info("mnr")
        assert isinstance(task, dict)
        assert task["name"] == "mnr"
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


class TestTaskSpecificParams:
    """Test task-specific parameters."""

    def test_matryoshka_has_dims_param(self):
        """Test that matryoshka task has dims parameter."""
        assert "matryoshka" in TASK_SPECIFIC_PARAMS
        assert "matryoshka_dims" in TASK_SPECIFIC_PARAMS["matryoshka"]

    def test_matryoshka_dims_config(self):
        """Test matryoshka dims parameter configuration."""
        param = TASK_SPECIFIC_PARAMS["matryoshka"]["matryoshka_dims"]
        assert param["type"] == "text"
        assert param["default"] == "768,512,256,128,64"
        assert "label" in param
        assert "description" in param


class TestTaskSampleData:
    """Test task sample data."""

    def test_all_tasks_have_sample_data(self):
        """Test that all tasks have sample data."""
        tasks = ["mnr", "triplet", "contrastive", "sts", "nli", "tsdae", "matryoshka"]
        for task in tasks:
            assert task in TASK_SAMPLE_DATA
            assert len(TASK_SAMPLE_DATA[task]) > 0

    def test_mnr_sample_data(self):
        """Test MNR sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["mnr"]
        assert len(samples) >= 1
        for sample in samples:
            assert "anchor" in sample
            assert "positive" in sample

    def test_triplet_sample_data(self):
        """Test triplet sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["triplet"]
        assert len(samples) >= 1
        for sample in samples:
            assert "anchor" in sample
            assert "positive" in sample
            assert "negative" in sample

    def test_sts_sample_data(self):
        """Test STS sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["sts"]
        assert len(samples) >= 1
        for sample in samples:
            assert "sentence1" in sample
            assert "sentence2" in sample
            assert "score" in sample
            assert 0 <= sample["score"] <= 1

    def test_contrastive_sample_data(self):
        """Test contrastive sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["contrastive"]
        assert len(samples) >= 1
        for sample in samples:
            assert "sentence1" in sample
            assert "sentence2" in sample
            assert "label" in sample
            assert sample["label"] in [0, 1]

    def test_nli_sample_data(self):
        """Test NLI sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["nli"]
        assert len(samples) >= 1
        for sample in samples:
            assert "sentence1" in sample
            assert "sentence2" in sample
            assert "label" in sample
            assert sample["label"] in [0, 1, 2]

    def test_tsdae_sample_data(self):
        """Test TSDAE sample data has correct columns."""
        samples = TASK_SAMPLE_DATA["tsdae"]
        assert len(samples) >= 1
        for sample in samples:
            assert "text" in sample


class TestMNRTask:
    """Test MNR task class."""

    def test_task_attributes(self):
        """Test MNR task has correct attributes."""
        task_cls = Registry.get_task("mnr")
        assert task_cls.name == "mnr"
        assert task_cls.expected_columns == ["anchor", "positive"]
        assert "anchor" in task_cls.column_aliases
        assert "positive" in task_cls.column_aliases

    def test_column_aliases(self):
        """Test MNR task column aliases."""
        task_cls = Registry.get_task("mnr")
        assert "query" in task_cls.column_aliases["anchor"]
        assert "question" in task_cls.column_aliases["anchor"]
        assert "document" in task_cls.column_aliases["positive"]


class TestTripletTask:
    """Test triplet task class."""

    def test_task_attributes(self):
        """Test triplet task has correct attributes."""
        task_cls = Registry.get_task("triplet")
        assert task_cls.name == "triplet"
        assert task_cls.expected_columns == ["anchor", "positive", "negative"]


class TestSTSTask:
    """Test STS task class."""

    def test_task_attributes(self):
        """Test STS task has correct attributes."""
        task_cls = Registry.get_task("sts")
        assert task_cls.name == "sts"
        assert task_cls.expected_columns == ["sentence1", "sentence2", "score"]


class TestContrastiveTask:
    """Test contrastive task class."""

    def test_task_attributes(self):
        """Test contrastive task has correct attributes."""
        task_cls = Registry.get_task("contrastive")
        assert task_cls.name == "contrastive"
        assert task_cls.expected_columns == ["sentence1", "sentence2", "label"]


class TestNLITask:
    """Test NLI task class."""

    def test_task_attributes(self):
        """Test NLI task has correct attributes."""
        task_cls = Registry.get_task("nli")
        assert task_cls.name == "nli"
        assert task_cls.expected_columns == ["sentence1", "sentence2", "label"]

    def test_column_aliases(self):
        """Test NLI task column aliases."""
        task_cls = Registry.get_task("nli")
        assert "premise" in task_cls.column_aliases["sentence1"]
        assert "hypothesis" in task_cls.column_aliases["sentence2"]


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


class TestMatryoshkaTask:
    """Test matryoshka task class."""

    def test_task_attributes(self):
        """Test matryoshka task has correct attributes."""
        task_cls = Registry.get_task("matryoshka")
        assert task_cls.name == "matryoshka"
        assert task_cls.expected_columns == ["anchor", "positive"]

    def test_default_dims(self):
        """Test matryoshka task default dimensions."""
        task_cls = Registry.get_task("matryoshka")
        task = task_cls()
        assert task.matryoshka_dims == [768, 512, 256, 128, 64]

    def test_custom_dims(self):
        """Test matryoshka task with custom dimensions."""
        task_cls = Registry.get_task("matryoshka")
        task = task_cls(matryoshka_dims=[512, 256, 128])
        assert task.matryoshka_dims == [512, 256, 128]
