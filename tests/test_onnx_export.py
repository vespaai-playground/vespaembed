"""Tests for ONNX export functionality."""

from unittest.mock import MagicMock, patch

import pytest

from vespaembed.enums import RunStatus


class TestExportModel:
    """Tests for vespaembed.models.export.export_model."""

    def test_unsupported_format_raises(self):
        """Test that an unsupported format raises ValueError."""
        from vespaembed.models.export import export_model

        with pytest.raises(ValueError, match="Unsupported export format"):
            export_model("/tmp/model", "/tmp/out", format="tflite")

    def test_export_creates_onnx_file(self, tmp_path):
        """Test that export produces an onnx/model.onnx file."""
        from sentence_transformers import SentenceTransformer

        from vespaembed.models.export import export_model

        # Save a model to disk first
        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        model_dir = tmp_path / "saved_model"
        model.save_pretrained(str(model_dir))

        # Export to ONNX
        output_dir = tmp_path / "onnx_output"
        result = export_model(str(model_dir), str(output_dir), format="onnx")

        # Should return the output directory path
        assert result == str(output_dir)

        # model.onnx should exist inside onnx/ subdirectory
        onnx_file = output_dir / "onnx" / "model.onnx"
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 0

    def test_export_includes_tokenizer(self, tmp_path):
        """Test that export includes tokenizer files for inference."""
        from sentence_transformers import SentenceTransformer

        from vespaembed.models.export import export_model

        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        model_dir = tmp_path / "saved_model"
        model.save_pretrained(str(model_dir))

        output_dir = tmp_path / "onnx_output"
        export_model(str(model_dir), str(output_dir))

        # Tokenizer files should be present
        assert (output_dir / "tokenizer.json").exists() or (output_dir / "vocab.txt").exists()
        assert (output_dir / "tokenizer_config.json").exists()

    def test_onnx_model_is_valid(self, tmp_path):
        """Test that the exported ONNX model passes onnx.checker."""
        import onnx
        from sentence_transformers import SentenceTransformer

        from vespaembed.models.export import export_model

        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        model_dir = tmp_path / "saved_model"
        model.save_pretrained(str(model_dir))

        output_dir = tmp_path / "onnx_valid"
        export_model(str(model_dir), str(output_dir))

        onnx_file = output_dir / "onnx" / "model.onnx"
        onnx_model = onnx.load(str(onnx_file))
        onnx.checker.check_model(onnx_model)

    def test_onnx_inference_produces_embeddings(self, tmp_path):
        """Test that the exported ONNX model can be loaded and used for inference."""
        from sentence_transformers import SentenceTransformer

        from vespaembed.models.export import export_model

        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        model_dir = tmp_path / "saved_model"
        model.save_pretrained(str(model_dir))

        output_dir = tmp_path / "onnx_infer"
        export_model(str(model_dir), str(output_dir))

        # Reload the exported model with ONNX backend and run inference (force CPU to avoid CoreML issues on macOS)
        onnx_model = SentenceTransformer(
            str(output_dir), backend="onnx", model_kwargs={"provider": "CPUExecutionProvider"}
        )
        embeddings = onnx_model.encode(["Hello world", "Test sentence"])

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Has embedding dimensions


class TestWorkerOnnxExport:
    """Tests for ONNX export integration in TrainingWorker."""

    @patch("vespaembed.worker.update_run_status")
    @patch("vespaembed.worker.VespaEmbedTrainer")
    def test_onnx_export_called_after_training(self, mock_trainer_cls, mock_update_status, tmp_path):
        """Test that ONNX export runs after training for non-Unsloth runs."""
        from vespaembed.worker import TrainingWorker

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = self._make_config(tmp_path, output_dir, unsloth_enabled=False)
        worker = TrainingWorker(run_id=1, config=config)

        with patch.object(worker, "_send_update") as mock_send:
            with patch("vespaembed.worker.export_model") as mock_export:
                mock_trainer = MagicMock()
                mock_trainer_cls.return_value = mock_trainer

                worker.run()

                mock_export.assert_called_once()
                call_args = mock_export.call_args
                assert call_args[0][0] == str(output_dir / "final")
                assert call_args[0][1] == str(output_dir / "onnx")

                log_calls = [c for c in mock_send.call_args_list if c[0][0] == "log"]
                log_messages = [c[0][1]["message"] for c in log_calls]
                assert any("Exporting model to ONNX" in m for m in log_messages)
                assert any("ONNX export complete" in m for m in log_messages)

    @patch("vespaembed.worker.update_run_status")
    @patch("vespaembed.worker.VespaEmbedTrainer")
    def test_onnx_export_skipped_for_unsloth(self, mock_trainer_cls, mock_update_status, tmp_path):
        """Test that ONNX export is skipped when unsloth is enabled."""
        from vespaembed.worker import TrainingWorker

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = self._make_config(tmp_path, output_dir, unsloth_enabled=True)
        worker = TrainingWorker(run_id=1, config=config)

        with patch.object(worker, "_send_update"):
            with patch("vespaembed.worker.export_model") as mock_export:
                mock_trainer = MagicMock()
                mock_trainer_cls.return_value = mock_trainer

                worker.run()

                mock_export.assert_not_called()

    @patch("vespaembed.worker.update_run_status")
    @patch("vespaembed.worker.VespaEmbedTrainer")
    def test_onnx_export_failure_does_not_fail_training(self, mock_trainer_cls, mock_update_status, tmp_path):
        """Test that ONNX export failure still results in completed training."""
        from vespaembed.worker import TrainingWorker

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = self._make_config(tmp_path, output_dir, unsloth_enabled=False)
        worker = TrainingWorker(run_id=1, config=config)

        with patch.object(worker, "_send_update") as mock_send:
            with patch("vespaembed.worker.export_model", side_effect=RuntimeError("onnx broke")):
                mock_trainer = MagicMock()
                mock_trainer_cls.return_value = mock_trainer

                worker.run()

                # Training should still complete
                mock_update_status.assert_called_with(1, RunStatus.COMPLETED)

                log_calls = [c for c in mock_send.call_args_list if c[0][0] == "log"]
                log_messages = [c[0][1]["message"] for c in log_calls]
                assert any("ONNX export failed" in m for m in log_messages)
                assert any("training still succeeded" in m for m in log_messages)

    def _make_config(self, tmp_path, output_dir, unsloth_enabled=False):
        """Build a minimal worker config dict."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        train_file = data_dir / "train.csv"
        train_file.write_text("anchor,positive\nhello,world\n")
        return {
            "base_model": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "task": "pairs",
            "train_filename": str(train_file),
            "output_dir": str(output_dir),
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "unsloth_enabled": unsloth_enabled,
        }


class TestArtifactsOnnx:
    """Tests for ONNX artifact discovery in the API."""

    def test_onnx_artifact_listed(self, client, tmp_path):
        """Test that ONNX directory is returned as an artifact."""
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"\x00" * 1024)
        (onnx_dir / "tokenizer.json").write_text("{}")

        run = {"output_dir": str(tmp_path), "id": 999}

        with patch("vespaembed.web.app.get_run", return_value=run):
            resp = client.get("/runs/999/artifacts")

        assert resp.status_code == 200
        artifacts = resp.json()["artifacts"]
        onnx_artifacts = [a for a in artifacts if a["name"] == "onnx"]
        assert len(onnx_artifacts) == 1
        assert onnx_artifacts[0]["label"] == "ONNX Model"
        assert onnx_artifacts[0]["category"] == "model"
        assert onnx_artifacts[0]["is_directory"] is True
        assert onnx_artifacts[0]["size"] == 1024 + 2  # model.onnx + tokenizer.json

    def test_no_onnx_artifact_when_missing(self, client, tmp_path):
        """Test that no ONNX artifact is returned when directory doesn't exist."""
        run = {"output_dir": str(tmp_path), "id": 999}

        with patch("vespaembed.web.app.get_run", return_value=run):
            resp = client.get("/runs/999/artifacts")

        assert resp.status_code == 200
        artifacts = resp.json()["artifacts"]
        onnx_artifacts = [a for a in artifacts if a["name"] == "onnx"]
        assert len(onnx_artifacts) == 0

    def test_onnx_artifact_size_sums_all_files(self, client, tmp_path):
        """Test that ONNX artifact size sums all files in the directory."""
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"\x00" * 500)
        (onnx_dir / "tokenizer.json").write_bytes(b"\x00" * 200)
        (onnx_dir / "vocab.txt").write_bytes(b"\x00" * 100)

        run = {"output_dir": str(tmp_path), "id": 999}

        with patch("vespaembed.web.app.get_run", return_value=run):
            resp = client.get("/runs/999/artifacts")

        artifacts = resp.json()["artifacts"]
        onnx_artifact = [a for a in artifacts if a["name"] == "onnx"][0]
        assert onnx_artifact["size"] == 800
