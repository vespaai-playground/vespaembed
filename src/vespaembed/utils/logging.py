import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme matching vespatune
VESPAEMBED_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "success": "green",
        "highlight": "magenta",
    }
)

console = Console(theme=VESPAEMBED_THEME)


def get_logger(name: str = "vespaembed", level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger with rich output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


class VespaEmbedLogger:
    """Logger with convenience methods for vespaembed."""

    def __init__(self, name: str = "vespaembed"):
        self._logger = get_logger(name)
        self._console = console

    def info(self, message: str):
        self._logger.info(message)

    def debug(self, message: str):
        self._logger.debug(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str):
        self._logger.error(message)

    def success(self, message: str):
        self._console.print(f"[success]{message}[/success]")

    def highlight(self, message: str):
        self._console.print(f"[highlight]{message}[/highlight]")

    def print(self, message: str):
        self._console.print(message)


logger = VespaEmbedLogger()
