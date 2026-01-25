from argparse import ArgumentParser, Namespace

# Import tasks to register them
import vespaembed.tasks  # noqa: F401
from vespaembed.cli import BaseCommand
from vespaembed.core.registry import Registry
from vespaembed.utils.logging import logger


def info_command_factory(args: Namespace) -> "InfoCommand":
    """Factory function for InfoCommand."""
    return InfoCommand(show_tasks=args.tasks)


class InfoCommand(BaseCommand):
    """Show information about available tasks."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """Register the info subcommand."""
        info_parser = parser.add_parser(
            "info",
            help="Show information about available tasks",
        )

        info_parser.add_argument(
            "--tasks",
            action="store_true",
            help="List available tasks",
        )

        info_parser.set_defaults(func=info_command_factory)

    def __init__(self, show_tasks: bool = True):
        self.show_tasks = show_tasks

    def execute(self):
        """Execute the info command."""
        if self.show_tasks or True:  # Default to showing tasks
            self._show_tasks()

    def _show_tasks(self):
        """Display available tasks."""
        logger.print("\n[bold]Available Tasks:[/bold]\n")

        tasks_info = Registry.get_task_info()

        for task in tasks_info:
            logger.print(f"  [cyan]{task['name']}[/cyan]")
            logger.print(f"    {task['description']}")
            logger.print(f"    Expected columns: {', '.join(task['expected_columns'])}")
            logger.print("")
