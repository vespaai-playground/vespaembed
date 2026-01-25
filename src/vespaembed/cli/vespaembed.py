import argparse
import sys

from vespaembed.cli.commands.evaluate import EvaluateCommand, evaluate_command_factory
from vespaembed.cli.commands.export import ExportCommand, export_command_factory
from vespaembed.cli.commands.info import InfoCommand, info_command_factory
from vespaembed.cli.commands.serve import ServeCommand, serve_command_factory
from vespaembed.cli.commands.train import TrainCommand, train_command_factory


def main():
    """Main entry point for vespaembed CLI."""
    parser = argparse.ArgumentParser(
        prog="vespaembed",
        description="VespaEmbed - No-code training for embedding models",
        usage="vespaembed [<command>] [<args>]",
    )

    # Global arguments
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web UI (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web UI (default: 8000)",
    )

    # Subcommands
    commands_parser = parser.add_subparsers(dest="command")

    # Register all commands
    TrainCommand.register_subcommand(commands_parser)
    EvaluateCommand.register_subcommand(commands_parser)
    ExportCommand.register_subcommand(commands_parser)
    ServeCommand.register_subcommand(commands_parser)
    InfoCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    # If no command specified, launch web UI
    if not hasattr(args, "func") or args.func is None:
        command = ServeCommand(host=args.host, port=args.port)
        command.execute()
    else:
        # Execute the specified command
        command = args.func(args)
        command.execute()


if __name__ == "__main__":
    main()
