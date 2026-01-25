"""Tests for the CLI commands."""


class TestCLIImports:
    """Test that CLI modules can be imported."""

    def test_import_main(self):
        """Test importing the main CLI module."""
        from vespaembed.cli.vespaembed import main

        assert main is not None

    def test_import_train_command(self):
        """Test importing the train command."""
        from vespaembed.cli.commands.train import TrainCommand, train_command_factory

        assert TrainCommand is not None
        assert train_command_factory is not None

    def test_import_serve_command(self):
        """Test importing the serve command."""
        from vespaembed.cli.commands.serve import ServeCommand, serve_command_factory

        assert ServeCommand is not None
        assert serve_command_factory is not None

    def test_import_evaluate_command(self):
        """Test importing the evaluate command."""
        from vespaembed.cli.commands.evaluate import EvaluateCommand, evaluate_command_factory

        assert EvaluateCommand is not None
        assert evaluate_command_factory is not None

    def test_import_export_command(self):
        """Test importing the export command."""
        from vespaembed.cli.commands.export import ExportCommand, export_command_factory

        assert ExportCommand is not None
        assert export_command_factory is not None

    def test_import_info_command(self):
        """Test importing the info command."""
        from vespaembed.cli.commands.info import InfoCommand, info_command_factory

        assert InfoCommand is not None
        assert info_command_factory is not None


class TestTrainCommand:
    """Test TrainCommand class."""

    def test_train_command_init(self):
        """Test TrainCommand initialization."""
        from vespaembed.cli.commands.train import TrainCommand

        cmd = TrainCommand(
            data="train.csv",
            task="mnr",
            base_model="model",
        )
        assert cmd.data == "train.csv"
        assert cmd.task == "mnr"
        assert cmd.base_model == "model"
        assert cmd.epochs == 3  # default
        assert cmd.batch_size == 32  # default

    def test_train_command_custom_params(self):
        """Test TrainCommand with custom parameters."""
        from vespaembed.cli.commands.train import TrainCommand

        cmd = TrainCommand(
            data="train.csv",
            task="triplet",
            base_model="model",
            epochs=10,
            batch_size=64,
            learning_rate=1e-4,
            project="my-project",
        )
        assert cmd.epochs == 10
        assert cmd.batch_size == 64
        assert cmd.learning_rate == 1e-4
        assert cmd.project == "my-project"

    def test_generate_project_name(self):
        """Test project name generation."""
        from vespaembed.cli.commands.train import TrainCommand

        cmd = TrainCommand(data="train.csv", task="mnr", base_model="model")
        name = cmd._generate_project_name()

        assert len(name) == 8
        assert name.isalnum()


class TestServeCommand:
    """Test ServeCommand class."""

    def test_serve_command_defaults(self):
        """Test ServeCommand default values."""
        from vespaembed.cli.commands.serve import ServeCommand

        cmd = ServeCommand()
        assert cmd.host == "127.0.0.1"
        assert cmd.port == 8000

    def test_serve_command_custom(self):
        """Test ServeCommand with custom values."""
        from vespaembed.cli.commands.serve import ServeCommand

        cmd = ServeCommand(host="0.0.0.0", port=9000)
        assert cmd.host == "0.0.0.0"
        assert cmd.port == 9000
