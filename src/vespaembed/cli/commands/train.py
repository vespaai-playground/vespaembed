import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

# Import tasks to register them
import vespaembed.tasks  # noqa: F401
from vespaembed.cli import BaseCommand
from vespaembed.core.config import TrainingConfig, load_config_from_yaml
from vespaembed.core.trainer import VespaEmbedTrainer
from vespaembed.utils.logging import logger

# Projects directory
PROJECTS_DIR = Path.home() / ".vespaembed" / "projects"


def train_command_factory(args: Namespace) -> "TrainCommand":
    """Factory function for TrainCommand."""
    return TrainCommand(
        config_path=args.config,
        data=args.data,
        task=args.task,
        base_model=args.base_model,
        project=args.project,
        eval_data=args.eval_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        unsloth=args.unsloth,
        matryoshka=args.matryoshka,
        matryoshka_dims=args.matryoshka_dims,
        subset=args.subset,
        split=args.split,
    )


class TrainCommand(BaseCommand):
    """Train or fine-tune an embedding model."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """Register the train subcommand."""
        train_parser = parser.add_parser(
            "train",
            help="Train or fine-tune an embedding model",
        )

        # Config file (alternative to CLI args)
        train_parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to YAML configuration file",
        )

        # Required arguments (unless using config file)
        train_parser.add_argument(
            "--data",
            type=str,
            default=None,
            help="Path to training data (CSV, JSONL, or HF dataset)",
        )
        train_parser.add_argument(
            "--task",
            type=str,
            default=None,
            choices=["pairs", "triplets", "similarity", "tsdae"],
            help="Training task type",
        )
        train_parser.add_argument(
            "--base-model",
            type=str,
            default=None,
            help="Base model name or path",
        )

        # Optional arguments
        train_parser.add_argument(
            "--project",
            type=str,
            default=None,
            help="Project name. Output saved to ~/.vespaembed/projects/<name>/",
        )
        train_parser.add_argument(
            "--eval-data",
            type=str,
            default=None,
            help="Path to evaluation data",
        )
        train_parser.add_argument(
            "--epochs",
            type=int,
            default=3,
            help="Number of training epochs (default: 3)",
        )
        train_parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size (default: 32)",
        )
        train_parser.add_argument(
            "--learning-rate",
            type=float,
            default=2e-5,
            help="Learning rate (default: 2e-5)",
        )
        train_parser.add_argument(
            "--unsloth",
            action="store_true",
            help="Use Unsloth for faster training",
        )
        train_parser.add_argument(
            "--matryoshka",
            action="store_true",
            help="Enable Matryoshka embeddings (multi-dimensional)",
        )
        train_parser.add_argument(
            "--matryoshka-dims",
            type=str,
            default="768,512,256,128,64",
            help="Matryoshka dimensions, comma-separated (default: 768,512,256,128,64)",
        )

        # HuggingFace dataset options
        train_parser.add_argument(
            "--subset",
            type=str,
            default=None,
            help="HuggingFace dataset subset",
        )
        train_parser.add_argument(
            "--split",
            type=str,
            default=None,
            help="HuggingFace dataset split",
        )

        train_parser.set_defaults(func=train_command_factory)

    def __init__(
        self,
        config_path: Optional[str] = None,
        data: Optional[str] = None,
        task: Optional[str] = None,
        base_model: Optional[str] = None,
        project: Optional[str] = None,
        eval_data: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        unsloth: bool = False,
        matryoshka: bool = False,
        matryoshka_dims: str = "768,512,256,128,64",
        subset: Optional[str] = None,
        split: Optional[str] = None,
    ):
        self.config_path = config_path
        self.data = data
        self.task = task
        self.base_model = base_model
        self.project = project
        self.eval_data = eval_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.unsloth = unsloth
        self.matryoshka = matryoshka
        self.matryoshka_dims = matryoshka_dims
        self.subset = subset
        self.split = split

    def _generate_project_name(self) -> str:
        """Generate a random project name."""
        import random
        import string

        return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

    def _resolve_output_dir(self, project_name: str) -> Path:
        """Resolve output directory from project name."""
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = PROJECTS_DIR / project_name
        if output_dir.exists():
            # Append timestamp to make unique
            timestamp = int(time.time())
            output_dir = PROJECTS_DIR / f"{project_name}-{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def execute(self):
        """Execute the train command."""
        # Load config from file or build from CLI args
        if self.config_path:
            logger.info(f"Loading config from: {self.config_path}")
            config = load_config_from_yaml(self.config_path)
        else:
            # Validate required arguments
            if not self.data:
                raise ValueError("--data is required (or use --config)")
            if not self.task:
                raise ValueError("--task is required (or use --config)")
            if not self.base_model:
                raise ValueError("--base-model is required (or use --config)")

            # Generate project name if not provided
            project_name = self.project or self._generate_project_name()
            output_dir = self._resolve_output_dir(project_name)

            logger.info(f"Project: {project_name}")
            logger.info(f"Output: {output_dir}")

            # Parse matryoshka dimensions if enabled
            matryoshka_dims = None
            if self.matryoshka:
                if self.task == "tsdae":
                    raise ValueError("Matryoshka is not supported with TSDAE task")
                matryoshka_dims = [int(d.strip()) for d in self.matryoshka_dims.split(",") if d.strip()]
                logger.info(f"Matryoshka enabled with dimensions: {matryoshka_dims}")

            # Build config from CLI args
            config = TrainingConfig(
                task=self.task,
                base_model=self.base_model,
                data={
                    "train": self.data,
                    "eval": self.eval_data,
                    "subset": self.subset,
                    "split": self.split,
                },
                training={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                },
                output={
                    "dir": str(output_dir),
                },
                unsloth=self.unsloth,
                matryoshka_dims=matryoshka_dims,
            )

        # Create and run trainer
        trainer = VespaEmbedTrainer(config)
        trainer.train()
