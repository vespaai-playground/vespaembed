from typing import Optional

from pydantic import BaseModel, Field

from vespaembed.enums import TaskType


class DataConfig(BaseModel):
    """Data configuration."""

    train: str = Field(..., description="Path to training data (CSV, JSONL, or HF dataset)")
    eval: Optional[str] = Field(None, description="Path to evaluation data (or HF dataset name)")
    subset: Optional[str] = Field(None, description="HuggingFace dataset subset")
    split: Optional[str] = Field(None, description="HuggingFace dataset split for training")
    eval_split: Optional[str] = Field(None, description="HuggingFace dataset split for evaluation")


class TrainingHyperparameters(BaseModel):
    """Training hyperparameters."""

    epochs: int = Field(3, description="Number of training epochs", ge=1)
    batch_size: int = Field(32, description="Batch size", ge=1)
    learning_rate: float = Field(2e-5, description="Learning rate", gt=0)
    warmup_ratio: float = Field(0.1, description="Warmup ratio", ge=0, le=1)
    weight_decay: float = Field(0.01, description="Weight decay", ge=0)
    fp16: bool = Field(True, description="Use FP16 training")
    bf16: bool = Field(False, description="Use BF16 training")
    eval_steps: int = Field(500, description="Evaluate every N steps", ge=1)
    save_steps: int = Field(500, description="Save checkpoint every N steps", ge=1)
    logging_steps: int = Field(100, description="Log every N steps", ge=1)
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps", ge=1)


class OutputConfig(BaseModel):
    """Output configuration."""

    dir: str = Field("./output", description="Output directory")
    save_total_limit: int = Field(3, description="Maximum checkpoints to keep", ge=1)
    push_to_hub: bool = Field(False, description="Push model to HuggingFace Hub")
    hub_model_id: Optional[str] = Field(None, description="HuggingFace Hub model ID")


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    # Required
    task: TaskType = Field(..., description="Training task type")
    base_model: str = Field(..., description="Base model name or path")
    data: DataConfig = Field(..., description="Data configuration")

    # Optional
    training: TrainingHyperparameters = Field(
        default_factory=TrainingHyperparameters,
        description="Training hyperparameters",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration",
    )

    # Unsloth
    unsloth: bool = Field(False, description="Use Unsloth for faster training")

    # Matryoshka dimensions (optional)
    matryoshka_dims: Optional[list[int]] = Field(
        None,
        description="Matryoshka embedding dimensions (e.g., [768, 512, 256, 128])",
    )

    class Config:
        use_enum_values = True


def load_config_from_yaml(path: str) -> TrainingConfig:
    """Load configuration from a YAML file."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    return TrainingConfig(**data)


def load_config_from_dict(data: dict) -> TrainingConfig:
    """Load configuration from a dictionary."""
    return TrainingConfig(**data)
