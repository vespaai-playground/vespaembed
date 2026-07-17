"""Tests for the CLI commands."""

from unittest.mock import MagicMock, patch


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

    def test_build_config_from_cli_unsloth_default(self, tmp_path, monkeypatch):
        """The --unsloth flag default (False) must build a valid config."""
        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(data="train.csv", task="pairs", base_model="model", project="p1")
        config = cmd._build_config_from_cli()

        assert config.unsloth.enabled is False

    def test_build_config_from_cli_unsloth_enabled(self, tmp_path, monkeypatch):
        """--unsloth must map onto UnslothConfig.enabled."""
        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(data="train.csv", task="pairs", base_model="model", project="p2", unsloth=True)
        config = cmd._build_config_from_cli()

        assert config.unsloth.enabled is True

    def test_build_config_from_cli_matryoshka_dims_empty(self, tmp_path, monkeypatch):
        """--matryoshka with an empty dims list must fail fast, not silently disable."""
        import pytest

        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(
            data="train.csv", task="pairs", base_model="model", project="p3", matryoshka=True, matryoshka_dims=","
        )
        with pytest.raises(ValueError, match="positive integers"):
            cmd._build_config_from_cli()

    def test_build_config_from_cli_matryoshka_dims_negative(self, tmp_path, monkeypatch):
        """Non-positive matryoshka dimensions must be rejected."""
        import pytest

        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(
            data="train.csv",
            task="pairs",
            base_model="model",
            project="p4",
            matryoshka=True,
            matryoshka_dims="256,-64",
        )
        with pytest.raises(ValueError, match="positive integers"):
            cmd._build_config_from_cli()

    def test_build_config_from_cli_matryoshka_dims_non_integer(self, tmp_path, monkeypatch):
        """Non-integer dims get the same clear error, and no project dir is left behind."""
        import pytest

        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(
            data="train.csv",
            task="pairs",
            base_model="model",
            project="p5",
            matryoshka=True,
            matryoshka_dims="256,abc",
        )
        with pytest.raises(ValueError, match="positive integers"):
            cmd._build_config_from_cli()
        assert not list(tmp_path.iterdir())

    def test_build_config_from_cli_fp16_default_off(self, tmp_path, monkeypatch):
        """CLI-built configs must not enable FP16 by default (crashes on CPU)."""
        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(data="train.csv", task="pairs", base_model="model", project="p6")
        config = cmd._build_config_from_cli()

        assert config.training.fp16 is False
        assert config.training.bf16 is False

    def test_build_config_from_cli_fp16_flag(self, tmp_path, monkeypatch):
        """--fp16 must map onto training.fp16."""
        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(data="train.csv", task="pairs", base_model="model", project="p7", fp16=True)
        config = cmd._build_config_from_cli()

        assert config.training.fp16 is True

    def test_train_auto_exports_onnx(self, tmp_path, monkeypatch):
        """After training, the CLI must export the final model to ONNX like the UI worker."""
        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(data="train.csv", task="pairs", base_model="model", project="p8")

        with patch.object(train_module, "VespaEmbedTrainer") as mock_trainer_cls:
            with patch.object(train_module, "export_model") as mock_export:
                mock_trainer_cls.return_value = MagicMock()
                cmd.execute()

        mock_export.assert_called_once()
        args, kwargs = mock_export.call_args
        assert args[0].endswith("/final")
        assert args[1].endswith("/onnx")

    def test_train_skips_onnx_export_for_unsloth(self, tmp_path, monkeypatch):
        """Unsloth runs must not attempt ONNX export."""
        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(data="train.csv", task="pairs", base_model="model", project="p9", unsloth=True)

        with patch.object(train_module, "VespaEmbedTrainer") as mock_trainer_cls:
            with patch.object(train_module, "export_model") as mock_export:
                mock_trainer_cls.return_value = MagicMock()
                cmd.execute()

        mock_export.assert_not_called()

    def test_train_onnx_export_failure_does_not_fail_run(self, tmp_path, monkeypatch):
        """A failing ONNX export must not turn a successful training into an error."""
        from vespaembed.cli.commands import train as train_module
        from vespaembed.cli.commands.train import TrainCommand

        monkeypatch.setattr(train_module, "PROJECTS_DIR", tmp_path)
        cmd = TrainCommand(data="train.csv", task="pairs", base_model="model", project="p10")

        with patch.object(train_module, "VespaEmbedTrainer") as mock_trainer_cls:
            with patch.object(train_module, "export_model", side_effect=RuntimeError("onnx broke")):
                mock_trainer_cls.return_value = MagicMock()
                cmd.execute()  # must not raise

    def test_generate_project_name(self):
        """Test project name generation."""
        from vespaembed.cli.commands.train import TrainCommand

        cmd = TrainCommand(data="train.csv", task="mnr", base_model="model")
        name = cmd._generate_project_name()

        assert len(name) == 8
        assert name.isalnum()


class TestExportCommand:
    """Test ExportCommand class."""

    def test_export_passes_model_path_not_model_object(self, tmp_path):
        """export must hand the saved-model PATH to export_model, not a loaded model."""
        from vespaembed.cli.commands import export as export_module
        from vespaembed.cli.commands.export import ExportCommand

        cmd = ExportCommand(model_path=str(tmp_path / "final"), output_path=str(tmp_path / "out"))

        with patch.object(export_module, "export_model", return_value=str(tmp_path / "out")) as mock_export:
            with patch.object(export_module, "load_model") as mock_load:
                cmd.execute()

        mock_export.assert_called_once_with(str(tmp_path / "final"), str(tmp_path / "out"), "onnx")
        # No hub push requested, so the model should never be loaded
        mock_load.assert_not_called()

    def test_export_loads_model_only_for_hub_push(self, tmp_path):
        """Hub push still loads the model and pushes it."""
        from vespaembed.cli.commands import export as export_module
        from vespaembed.cli.commands.export import ExportCommand

        cmd = ExportCommand(model_path=str(tmp_path / "final"), hub_id="user/repo")

        with patch.object(export_module, "export_model", return_value="out"):
            with patch.object(export_module, "load_model") as mock_load:
                with patch.object(export_module, "push_to_hub", return_value="url") as mock_push:
                    cmd.execute()

        mock_load.assert_called_once_with(str(tmp_path / "final"))
        mock_push.assert_called_once()


class TestEvaluateCommand:
    """Test EvaluateCommand class."""

    def test_task_choices_match_registry(self):
        """evaluate --task choices must be real registered task names."""
        import argparse

        import vespaembed.tasks  # noqa: F401
        from vespaembed.cli.commands.evaluate import EvaluateCommand
        from vespaembed.core.registry import Registry

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        EvaluateCommand.register_subcommand(subparsers)

        args = parser.parse_args(["evaluate", "--model", "m", "--data", "d", "--task", "pairs"])
        assert args.task == "pairs"

        # Every accepted choice must resolve in the registry
        for task_name in ("pairs", "triplets", "similarity", "tsdae"):
            parsed = parser.parse_args(["evaluate", "--model", "m", "--data", "d", "--task", task_name])
            assert Registry.get_task(parsed.task) is not None


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
