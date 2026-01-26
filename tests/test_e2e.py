"""End-to-end tests for training with all tasks and configurations.

These tests verify that training works correctly with:
- All task types
- Standard SentenceTransformer training
- SentenceTransformer + LoRA training

Note: Unsloth tests are skipped as they require NVIDIA/AMD GPU.
"""

import tempfile
from pathlib import Path

import pytest

from vespaembed.core.config import (
    DataConfig,
    LoraConfig,
    OutputConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
from vespaembed.core.trainer import VespaEmbedTrainer

# Import tasks to register them
import vespaembed.tasks  # noqa: F401

# Path to example data
EXAMPLES_DATA_DIR = Path(__file__).parent.parent / "examples" / "data"

# Small model for fast testing
TEST_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def assert_model_saved(final_path: Path):
    """Assert that a model was saved correctly."""
    assert final_path.exists(), f"Final path {final_path} does not exist"
    files = list(final_path.iterdir())
    assert len(files) > 0, f"No files saved in {final_path}"
    # Check for at least one expected file type
    file_names = [f.name for f in files]
    expected_files = ["config.json", "config_sentence_transformers.json", "adapter_config.json", "model.safetensors"]
    has_expected = any(ef in file_names for ef in expected_files)
    assert has_expected, f"No expected model files found. Files present: {file_names}"

# Minimal training parameters for fast tests
MINIMAL_TRAINING = TrainingHyperparameters(
    epochs=1,
    batch_size=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    eval_steps=10,
    save_steps=10,
    logging_steps=5,
    gradient_accumulation_steps=1,
)

# Task to data file mapping
TASK_DATA_MAP = {
    "mnr": "mnr.csv",
    "triplet": "triplet.csv",
    "sts": "sts.csv",
    "contrastive": "contrastive.csv",
    "nli": "nli.csv",
    "tsdae": "tsdae.csv",
    "matryoshka": "mnr.csv",  # Matryoshka uses same format as MNR
}

# All tasks to test
ALL_TASKS = list(TASK_DATA_MAP.keys())

# Tasks that are NOT compatible with LoRA
# TSDAE uses a decoder architecture that conflicts with LoRA adapters
LORA_INCOMPATIBLE_TASKS = ["tsdae"]

# Tasks compatible with LoRA
LORA_COMPATIBLE_TASKS = [t for t in ALL_TASKS if t not in LORA_INCOMPATIBLE_TASKS]


def get_data_path(task: str) -> str:
    """Get the path to the example data file for a task."""
    data_file = TASK_DATA_MAP[task]
    return str(EXAMPLES_DATA_DIR / data_file)


def create_config(
    task: str,
    output_dir: str,
    lora_enabled: bool = False,
    matryoshka_dims: list[int] = None,
) -> TrainingConfig:
    """Create a minimal training config for testing."""
    config = TrainingConfig(
        task=task,
        base_model=TEST_MODEL,
        data=DataConfig(train=get_data_path(task)),
        training=MINIMAL_TRAINING,
        output=OutputConfig(
            dir=output_dir,
            save_total_limit=1,
            push_to_hub=False,
        ),
        lora=LoraConfig(
            enabled=lora_enabled,
            r=8,  # Small rank for faster tests
            alpha=16,
            dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
        ),
        max_seq_length=128,  # Small for faster tests
        gradient_checkpointing=False,
        matryoshka_dims=matryoshka_dims,
    )
    return config


class TestStandardTraining:
    """Test standard SentenceTransformer training for all tasks."""

    @pytest.mark.slow
    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_task_training(self, task):
        """Test that training completes successfully for each task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Special handling for matryoshka - use smaller dims for test model
            matryoshka_dims = [384, 256, 128, 64] if task == "matryoshka" else None

            config = create_config(
                task=task,
                output_dir=tmpdir,
                lora_enabled=False,
                matryoshka_dims=matryoshka_dims,
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            # Verify model was returned
            assert model is not None

            # Verify final model was saved
            final_path = Path(tmpdir) / "final"
            assert_model_saved(final_path)


class TestLoRATraining:
    """Test SentenceTransformer + LoRA training for all compatible tasks.

    Note: TSDAE is excluded as its decoder architecture is incompatible with LoRA.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("task", LORA_COMPATIBLE_TASKS)
    def test_task_lora_training(self, task):
        """Test that LoRA training completes successfully for each task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Special handling for matryoshka - use smaller dims for test model
            matryoshka_dims = [384, 256, 128, 64] if task == "matryoshka" else None

            config = create_config(
                task=task,
                output_dir=tmpdir,
                lora_enabled=True,
                matryoshka_dims=matryoshka_dims,
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            # Verify model was returned
            assert model is not None

            # Verify final model was saved
            final_path = Path(tmpdir) / "final"
            assert_model_saved(final_path)


class TestLoRARankVariations:
    """Test LoRA with different rank values."""

    @pytest.mark.slow
    @pytest.mark.parametrize("lora_r", [8, 16, 32])
    def test_lora_ranks(self, lora_r):
        """Test that LoRA training works with different ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                task="mnr",
                base_model=TEST_MODEL,
                data=DataConfig(train=get_data_path("mnr")),
                training=MINIMAL_TRAINING,
                output=OutputConfig(
                    dir=tmpdir,
                    save_total_limit=1,
                    push_to_hub=False,
                ),
                lora=LoraConfig(
                    enabled=True,
                    r=lora_r,
                    alpha=lora_r * 2,  # Common practice: alpha = 2 * r
                    dropout=0.1,
                    target_modules=["query", "key", "value", "dense"],
                ),
                max_seq_length=128,
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            assert model is not None
            final_path = Path(tmpdir) / "final"
            assert_model_saved(final_path)


class TestTargetModuleVariations:
    """Test LoRA with different target module configurations."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "target_modules",
        [
            ["query", "key", "value", "dense"],  # BERT-style
            ["query", "value"],  # Minimal
            ["dense"],  # Single module
        ],
    )
    def test_target_modules(self, target_modules):
        """Test that LoRA training works with different target modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                task="mnr",
                base_model=TEST_MODEL,
                data=DataConfig(train=get_data_path("mnr")),
                training=MINIMAL_TRAINING,
                output=OutputConfig(
                    dir=tmpdir,
                    save_total_limit=1,
                    push_to_hub=False,
                ),
                lora=LoraConfig(
                    enabled=True,
                    r=8,
                    alpha=16,
                    dropout=0.1,
                    target_modules=target_modules,
                ),
                max_seq_length=128,
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            assert model is not None
            final_path = Path(tmpdir) / "final"
            assert_model_saved(final_path)


class TestPrecisionModes:
    """Test training with different precision settings."""

    @pytest.mark.slow
    def test_fp32_training(self):
        """Test training with FP32 precision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_params = TrainingHyperparameters(
                epochs=1,
                batch_size=2,
                learning_rate=2e-5,
                fp16=False,
                bf16=False,
                eval_steps=10,
                save_steps=10,
                logging_steps=5,
            )

            config = TrainingConfig(
                task="mnr",
                base_model=TEST_MODEL,
                data=DataConfig(train=get_data_path("mnr")),
                training=training_params,
                output=OutputConfig(dir=tmpdir, push_to_hub=False),
                max_seq_length=128,
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            assert model is not None


class TestMatryoshkaDimensions:
    """Test Matryoshka training with different dimension configurations."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "dims",
        [
            [384, 256, 128],
            [384, 192, 96],
            [256, 128, 64, 32],
        ],
    )
    def test_matryoshka_dims(self, dims):
        """Test Matryoshka training with different dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_config(
                task="matryoshka",
                output_dir=tmpdir,
                lora_enabled=False,
                matryoshka_dims=dims,
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            assert model is not None
            final_path = Path(tmpdir) / "final"
            assert_model_saved(final_path)


class TestGradientCheckpointing:
    """Test training with gradient checkpointing enabled."""

    @pytest.mark.slow
    def test_gradient_checkpointing_standard(self):
        """Test standard training with gradient checkpointing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                task="mnr",
                base_model=TEST_MODEL,
                data=DataConfig(train=get_data_path("mnr")),
                training=MINIMAL_TRAINING,
                output=OutputConfig(dir=tmpdir, push_to_hub=False),
                gradient_checkpointing=True,
                max_seq_length=128,
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            assert model is not None

    @pytest.mark.slow
    def test_gradient_checkpointing_lora(self):
        """Test LoRA training with gradient checkpointing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                task="mnr",
                base_model=TEST_MODEL,
                data=DataConfig(train=get_data_path("mnr")),
                training=MINIMAL_TRAINING,
                output=OutputConfig(dir=tmpdir, push_to_hub=False),
                lora=LoraConfig(
                    enabled=True,
                    r=8,
                    alpha=16,
                    dropout=0.1,
                    target_modules=["query", "key", "value", "dense"],
                ),
                gradient_checkpointing=True,
                max_seq_length=128,
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            assert model is not None


class TestCombinedConfigurations:
    """Test combined configurations (LoRA + Matryoshka, etc.)."""

    @pytest.mark.slow
    def test_lora_with_matryoshka(self):
        """Test LoRA training combined with Matryoshka."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_config(
                task="matryoshka",
                output_dir=tmpdir,
                lora_enabled=True,
                matryoshka_dims=[384, 256, 128],
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            assert model is not None
            final_path = Path(tmpdir) / "final"
            assert_model_saved(final_path)

    @pytest.mark.slow
    def test_lora_with_gradient_checkpointing_and_matryoshka(self):
        """Test LoRA + gradient checkpointing + Matryoshka."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                task="matryoshka",
                base_model=TEST_MODEL,
                data=DataConfig(train=get_data_path("matryoshka")),
                training=MINIMAL_TRAINING,
                output=OutputConfig(dir=tmpdir, push_to_hub=False),
                lora=LoraConfig(
                    enabled=True,
                    r=8,
                    alpha=16,
                    dropout=0.1,
                    target_modules=["query", "key", "value", "dense"],
                ),
                gradient_checkpointing=True,
                max_seq_length=128,
                matryoshka_dims=[384, 256, 128],
            )

            trainer = VespaEmbedTrainer(config=config)
            model = trainer.train()

            assert model is not None
