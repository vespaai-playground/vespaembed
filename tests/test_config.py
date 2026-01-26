"""Tests for the configuration module."""

import tempfile
from pathlib import Path

import pytest

from vespaembed.core.config import (
    DataConfig,
    LoraConfig,
    OutputConfig,
    TrainingConfig,
    TrainingHyperparameters,
    UnslothConfig,
    load_config_from_dict,
    load_config_from_yaml,
)


class TestDataConfig:
    """Test DataConfig model."""

    def test_required_train_field(self):
        """Test that train is required."""
        with pytest.raises(ValueError):
            DataConfig()

    def test_minimal_config(self):
        """Test minimal data config."""
        config = DataConfig(train="data.csv")
        assert config.train == "data.csv"
        assert config.eval is None
        assert config.subset is None
        assert config.split is None

    def test_full_config(self):
        """Test full data config."""
        config = DataConfig(
            train="train.csv",
            eval="eval.csv",
            subset="triplet",
            split="train",
            eval_split="validation",
        )
        assert config.train == "train.csv"
        assert config.eval == "eval.csv"
        assert config.subset == "triplet"
        assert config.split == "train"
        assert config.eval_split == "validation"


class TestTrainingHyperparameters:
    """Test TrainingHyperparameters model."""

    def test_default_values(self):
        """Test default hyperparameter values."""
        config = TrainingHyperparameters()
        assert config.epochs == 3
        assert config.batch_size == 32
        assert config.learning_rate == 2e-5
        assert config.warmup_ratio == 0.1
        assert config.weight_decay == 0.01
        assert config.fp16 is True
        assert config.bf16 is False
        assert config.eval_steps == 500
        assert config.save_steps == 500
        assert config.logging_steps == 100
        assert config.gradient_accumulation_steps == 1

    def test_custom_values(self):
        """Test custom hyperparameter values."""
        config = TrainingHyperparameters(
            epochs=10,
            batch_size=64,
            learning_rate=1e-4,
            fp16=False,
            bf16=True,
        )
        assert config.epochs == 10
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.fp16 is False
        assert config.bf16 is True

    def test_validation_epochs_positive(self):
        """Test that epochs must be positive."""
        with pytest.raises(ValueError):
            TrainingHyperparameters(epochs=0)

    def test_validation_batch_size_positive(self):
        """Test that batch_size must be positive."""
        with pytest.raises(ValueError):
            TrainingHyperparameters(batch_size=0)

    def test_validation_learning_rate_positive(self):
        """Test that learning_rate must be positive."""
        with pytest.raises(ValueError):
            TrainingHyperparameters(learning_rate=0)

    def test_validation_warmup_ratio_range(self):
        """Test that warmup_ratio is between 0 and 1."""
        with pytest.raises(ValueError):
            TrainingHyperparameters(warmup_ratio=1.5)


class TestOutputConfig:
    """Test OutputConfig model."""

    def test_default_values(self):
        """Test default output config values."""
        config = OutputConfig()
        assert config.dir == "./output"
        assert config.save_total_limit == 3
        assert config.push_to_hub is False
        assert config.hf_username is None

    def test_custom_values(self):
        """Test custom output config values."""
        config = OutputConfig(
            dir="/custom/path",
            save_total_limit=5,
            push_to_hub=True,
            hf_username="testuser",
        )
        assert config.dir == "/custom/path"
        assert config.save_total_limit == 5
        assert config.push_to_hub is True
        assert config.hf_username == "testuser"


class TestLoraConfig:
    """Test LoraConfig model."""

    def test_default_values(self):
        """Test default LoRA config values."""
        config = LoraConfig()
        assert config.enabled is False
        assert config.r == 64
        assert config.alpha == 128
        assert config.dropout == 0.1
        assert config.target_modules == ["query", "key", "value", "dense"]

    def test_custom_values(self):
        """Test custom LoRA config values."""
        config = LoraConfig(
            enabled=True,
            r=32,
            alpha=64,
            dropout=0.05,
            target_modules=["Wqkv", "Wi", "Wo"],
        )
        assert config.enabled is True
        assert config.r == 32
        assert config.alpha == 64
        assert config.dropout == 0.05
        assert config.target_modules == ["Wqkv", "Wi", "Wo"]

    def test_validation_r_positive(self):
        """Test that r must be positive."""
        with pytest.raises(ValueError):
            LoraConfig(r=0)

    def test_validation_dropout_range(self):
        """Test that dropout is between 0 and 1."""
        with pytest.raises(ValueError):
            LoraConfig(dropout=1.5)


class TestUnslothConfig:
    """Test UnslothConfig model."""

    def test_default_values(self):
        """Test default Unsloth config values."""
        config = UnslothConfig()
        assert config.enabled is False
        assert config.save_method == "merged_16bit"

    def test_custom_values(self):
        """Test custom Unsloth config values."""
        config = UnslothConfig(
            enabled=True,
            save_method="lora",
        )
        assert config.enabled is True
        assert config.save_method == "lora"

    def test_validation_save_method(self):
        """Test that save_method must be valid literal."""
        with pytest.raises(ValueError):
            UnslothConfig(save_method="invalid")


class TestTrainingConfig:
    """Test TrainingConfig model."""

    def test_minimal_config(self):
        """Test minimal training config."""
        config = TrainingConfig(
            task="mnr",
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            data=DataConfig(train="train.csv"),
        )
        assert config.task == "mnr"
        assert config.base_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.data.train == "train.csv"
        assert config.unsloth.enabled is False
        assert config.lora.enabled is False
        assert config.max_seq_length is None  # Auto-detect by default
        assert config.gradient_checkpointing is False  # Disabled by default
        assert config.matryoshka_dims is None

    def test_full_config(self):
        """Test full training config."""
        config = TrainingConfig(
            task="matryoshka",
            base_model="BAAI/bge-small-en-v1.5",
            data=DataConfig(train="train.csv", eval="eval.csv"),
            training=TrainingHyperparameters(epochs=5, batch_size=16),
            output=OutputConfig(dir="/output", push_to_hub=True, hf_username="user"),
            lora=LoraConfig(enabled=True, r=32, alpha=64),
            unsloth=UnslothConfig(enabled=True),
            max_seq_length=256,
            gradient_checkpointing=True,
            matryoshka_dims=[512, 256, 128],
        )
        assert config.task == "matryoshka"
        assert config.training.epochs == 5
        assert config.training.batch_size == 16
        assert config.output.push_to_hub is True
        assert config.lora.enabled is True
        assert config.lora.r == 32
        assert config.unsloth.enabled is True
        assert config.max_seq_length == 256
        assert config.gradient_checkpointing is True
        assert config.matryoshka_dims == [512, 256, 128]

    def test_nested_dict_config(self):
        """Test config with nested dicts (as from YAML)."""
        config = TrainingConfig(
            task="mnr",
            base_model="model",
            data={"train": "train.csv", "eval": "eval.csv"},
            training={"epochs": 10},
            output={"dir": "/out"},
        )
        assert config.data.train == "train.csv"
        assert config.data.eval == "eval.csv"
        assert config.training.epochs == 10
        assert config.output.dir == "/out"


class TestLoadConfigFromDict:
    """Test load_config_from_dict function."""

    def test_load_minimal(self):
        """Test loading minimal config from dict."""
        data = {
            "task": "mnr",
            "base_model": "model",
            "data": {"train": "train.csv"},
        }
        config = load_config_from_dict(data)
        assert config.task == "mnr"
        assert config.base_model == "model"
        assert config.data.train == "train.csv"

    def test_load_full(self):
        """Test loading full config from dict."""
        data = {
            "task": "triplet",
            "base_model": "model",
            "data": {"train": "train.csv", "eval": "eval.csv"},
            "training": {"epochs": 5, "batch_size": 64},
            "output": {"dir": "/output"},
        }
        config = load_config_from_dict(data)
        assert config.task == "triplet"
        assert config.training.epochs == 5
        assert config.training.batch_size == 64


class TestLoadConfigFromYaml:
    """Test load_config_from_yaml function."""

    def test_load_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
task: mnr
base_model: sentence-transformers/all-MiniLM-L6-v2
data:
  train: train.csv
  eval: eval.csv
training:
  epochs: 5
  batch_size: 32
output:
  dir: ./output
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config_from_yaml(f.name)

        assert config.task == "mnr"
        assert config.base_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.data.train == "train.csv"
        assert config.data.eval == "eval.csv"
        assert config.training.epochs == 5
        assert config.output.dir == "./output"

        # Cleanup
        Path(f.name).unlink()

    def test_load_yaml_with_matryoshka(self):
        """Test loading config with matryoshka dims from YAML."""
        yaml_content = """
task: matryoshka
base_model: model
data:
  train: train.csv
matryoshka_dims: [768, 512, 256, 128]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config_from_yaml(f.name)

        assert config.task == "matryoshka"
        assert config.matryoshka_dims == [768, 512, 256, 128]

        # Cleanup
        Path(f.name).unlink()
