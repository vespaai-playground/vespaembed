from argparse import ArgumentParser, Namespace

from vespaembed.cli import BaseCommand
from vespaembed.core.registry import Registry
from vespaembed.datasets.loader import load_dataset
from vespaembed.models.loader import load_model
from vespaembed.utils.logging import logger


def evaluate_command_factory(args: Namespace) -> "EvaluateCommand":
    """Factory function for EvaluateCommand."""
    return EvaluateCommand(
        model_path=args.model,
        data_path=args.data,
        task=args.task,
    )


class EvaluateCommand(BaseCommand):
    """Evaluate a trained embedding model."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """Register the evaluate subcommand."""
        eval_parser = parser.add_parser(
            "evaluate",
            help="Evaluate a trained embedding model",
        )

        eval_parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Path to trained model",
        )
        eval_parser.add_argument(
            "--data",
            type=str,
            required=True,
            help="Path to evaluation data",
        )
        eval_parser.add_argument(
            "--task",
            type=str,
            required=True,
            choices=["mnr", "triplet", "contrastive", "sts", "nli", "tsdae", "matryoshka"],
            help="Task type (determines evaluator)",
        )

        eval_parser.set_defaults(func=evaluate_command_factory)

    def __init__(self, model_path: str, data_path: str, task: str):
        self.model_path = model_path
        self.data_path = data_path
        self.task = task

    def execute(self):
        """Execute the evaluate command."""
        # Load model
        logger.info(f"Loading model: {self.model_path}")
        model = load_model(self.model_path)

        # Load and prepare data
        logger.info(f"Loading evaluation data: {self.data_path}")
        task_cls = Registry.get_task(self.task)
        task = task_cls()

        eval_data = load_dataset(self.data_path)
        eval_data = task.prepare_dataset(eval_data)

        # Create evaluator
        evaluator = task.get_evaluator(eval_data)

        if evaluator is None:
            logger.warning(f"No evaluator available for task: {self.task}")
            return

        # Run evaluation
        logger.info("Running evaluation...")
        results = evaluator(model)

        # Print results
        logger.success("Evaluation Results:")
        for key, value in results.items():
            logger.print(f"  {key}: {value:.4f}")
