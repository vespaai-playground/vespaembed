from argparse import ArgumentParser, Namespace

from vespaembed.cli import BaseCommand
from vespaembed.models.export import export_model, push_to_hub
from vespaembed.models.loader import load_model
from vespaembed.utils.logging import logger


def export_command_factory(args: Namespace) -> "ExportCommand":
    """Factory function for ExportCommand."""
    return ExportCommand(
        model_path=args.model,
        output_path=args.output,
        format=args.format,
        hub_id=args.hub_id,
    )


class ExportCommand(BaseCommand):
    """Export a trained model to different formats."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """Register the export subcommand."""
        export_parser = parser.add_parser(
            "export",
            help="Export a trained model to different formats",
        )

        export_parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Path to trained model",
        )
        export_parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output path for exported model",
        )
        export_parser.add_argument(
            "--format",
            type=str,
            default="onnx",
            choices=["onnx"],
            help="Export format (default: onnx)",
        )
        export_parser.add_argument(
            "--hub-id",
            type=str,
            default=None,
            help="Push to HuggingFace Hub with this repo ID",
        )

        export_parser.set_defaults(func=export_command_factory)

    def __init__(
        self,
        model_path: str,
        output_path: str = None,
        format: str = "onnx",
        hub_id: str = None,
    ):
        self.model_path = model_path
        self.output_path = output_path or f"{model_path}_exported"
        self.format = format
        self.hub_id = hub_id

    def execute(self):
        """Execute the export command."""
        # Load model
        logger.info(f"Loading model: {self.model_path}")
        model = load_model(self.model_path)

        # Export
        if self.format:
            logger.info(f"Exporting to {self.format}: {self.output_path}")
            export_path = export_model(model, self.output_path, self.format)
            logger.success(f"Model exported to: {export_path}")

        # Push to Hub
        if self.hub_id:
            logger.info(f"Pushing to HuggingFace Hub: {self.hub_id}")
            url = push_to_hub(model, self.hub_id)
            logger.success(f"Model pushed to: {url}")
