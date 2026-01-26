from typing import Literal, Optional

from pydantic import BaseModel, Field

from vespaembed.enums import TaskType


class DataConfig(BaseModel):
    """Data configuration."""

    train: str = Field(..., description="Path to training data (CSV, JSONL, or HF dataset)")
    eval: Optional[str] = Field(None, description="Path to evaluation data (or HF dataset name)")
    subset: Optional[str] = Field(None, description="HuggingFace dataset subset")
    split: Optional[str] = Field(None, description="HuggingFace dataset split for training")
    eval_split: Optional[str] = Field(None, description="HuggingFace dataset split for evaluation")


class LoraConfig(BaseModel):
    """LoRA/PEFT configuration - works with both standard and Unsloth training."""

    enabled: bool = Field(False, description="Enable LoRA training")
    r: int = Field(64, description="LoRA rank (common values: 8, 16, 32, 64, 128)", ge=1)
    alpha: int = Field(128, description="LoRA alpha (typically 2x rank)")
    dropout: float = Field(0.1, description="LoRA dropout (use 0 for Unsloth optimization)", ge=0, le=1)
    target_modules: list[str] = Field(
        default=["query", "key", "value", "dense"],
        description="Target modules for LoRA",
    )


class UnslothConfig(BaseModel):
    """Unsloth-specific configuration for faster training."""

    enabled: bool = Field(False, description="Enable Unsloth for faster training")
    save_method: Literal["lora", "merged_16bit", "merged_4bit"] = Field(
        "merged_16bit",
        description="How to save the model (lora=adapters only, merged=full model)",
    )


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
    hf_username: Optional[str] = Field(None, description="HuggingFace username")


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

    # LoRA/PEFT configuration
    lora: LoraConfig = Field(
        default_factory=LoraConfig,
        description="LoRA/PEFT configuration",
    )

    # Unsloth configuration
    unsloth: UnslothConfig = Field(
        default_factory=UnslothConfig,
        description="Unsloth configuration for faster training",
    )

    # Model configuration
    max_seq_length: Optional[int] = Field(
        None,
        description="Maximum sequence length (auto-detect from model if not specified)",
        ge=1,
    )
    gradient_checkpointing: bool = Field(
        False,
        description="Enable gradient checkpointing (saves VRAM, uses Unsloth optimization when Unsloth is enabled)",
    )

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
