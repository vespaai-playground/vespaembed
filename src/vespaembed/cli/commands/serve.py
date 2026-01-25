from argparse import ArgumentParser, Namespace

from vespaembed.cli import BaseCommand
from vespaembed.utils.logging import logger


def serve_command_factory(args: Namespace) -> "ServeCommand":
    """Factory function for ServeCommand."""
    return ServeCommand(host=args.host, port=args.port)


class ServeCommand(BaseCommand):
    """Start the web UI server."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """Register the serve subcommand."""
        serve_parser = parser.add_parser(
            "serve",
            help="Start the web UI server",
        )

        serve_parser.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Host to bind to (default: 127.0.0.1)",
        )
        serve_parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to bind to (default: 8000)",
        )

        serve_parser.set_defaults(func=serve_command_factory)

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port

    def execute(self):
        """Execute the serve command."""
        import uvicorn

        from vespaembed.web.app import app

        logger.info(f"Starting VespaEmbed web UI at http://{self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)
