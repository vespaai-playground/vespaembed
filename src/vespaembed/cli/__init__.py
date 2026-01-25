from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseCommand(ABC):
    """Base class for all CLI commands."""

    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        """Register the subcommand with argparse."""
        raise NotImplementedError

    @abstractmethod
    def execute(self):
        """Execute the command."""
        raise NotImplementedError
