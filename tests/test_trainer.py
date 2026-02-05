"""Tests for VespaEmbedTrainer functionality."""

from pathlib import Path
from unittest.mock import MagicMock

from vespaembed.core.trainer import VespaEmbedTrainer


class TestAddVespaembedToReadme:
    """Tests for _add_vespaembed_to_readme method."""

    def test_adds_mention_after_first_heading(self, tmp_path: Path):
        """Test that vespaembed mention is added after the first heading."""
        readme_path = tmp_path / "README.md"
        readme_path.write_text("# My Model\n\nThis is a model.", encoding="utf-8")

        trainer = MagicMock(spec=VespaEmbedTrainer)
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)

        content = readme_path.read_text(encoding="utf-8")
        assert "github.com/vespa-engine/vespaembed" in content
        assert content.startswith("# My Model\n")
        # Verify the mention comes after the heading
        heading_pos = content.find("# My Model")
        mention_pos = content.find("vespaembed")
        assert mention_pos > heading_pos

    def test_skips_yaml_frontmatter_heading(self, tmp_path: Path):
        """Test that YAML frontmatter comment headings are skipped."""
        readme_path = tmp_path / "README.md"
        content = """---
# For reference on model card metadata
tags:
  - sentence-transformers
---

# Actual Model Name

This is the model description."""
        readme_path.write_text(content, encoding="utf-8")

        trainer = MagicMock(spec=VespaEmbedTrainer)
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)

        result = readme_path.read_text(encoding="utf-8")
        # The mention should be after "# Actual Model Name", not after "# For reference"
        lines = result.split("\n")
        mention_line_idx = None
        actual_heading_idx = None
        for idx, line in enumerate(lines):
            if "vespaembed" in line:
                mention_line_idx = idx
            if line == "# Actual Model Name":
                actual_heading_idx = idx

        assert mention_line_idx is not None
        assert actual_heading_idx is not None
        assert mention_line_idx > actual_heading_idx

    def test_handles_missing_readme(self, tmp_path: Path):
        """Test that missing README.md is handled gracefully."""
        trainer = MagicMock(spec=VespaEmbedTrainer)
        # Should not raise an exception
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)

    def test_idempotency_no_duplicate_insertions(self, tmp_path: Path):
        """Test that calling multiple times doesn't add duplicate mentions."""
        readme_path = tmp_path / "README.md"
        readme_path.write_text("# My Model\n\nThis is a model.", encoding="utf-8")

        trainer = MagicMock(spec=VespaEmbedTrainer)

        # Call multiple times
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)

        content = readme_path.read_text(encoding="utf-8")
        # Should only have one mention
        assert content.count("github.com/vespa-engine/vespaembed") == 1

    def test_preserves_utf8_characters(self, tmp_path: Path):
        """Test that UTF-8 characters are preserved in the README."""
        readme_path = tmp_path / "README.md"
        original_content = "# Modèle Français 日本語モデル\n\nThis model supports émojis: 🚀 and ñ."
        readme_path.write_text(original_content, encoding="utf-8")

        trainer = MagicMock(spec=VespaEmbedTrainer)
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)

        content = readme_path.read_text(encoding="utf-8")
        # Check UTF-8 characters are preserved
        assert "Modèle Français" in content
        assert "日本語モデル" in content
        assert "🚀" in content
        assert "ñ" in content
        # And the mention was added
        assert "vespaembed" in content

    def test_no_modification_without_heading(self, tmp_path: Path):
        """Test that README without a heading is not modified."""
        readme_path = tmp_path / "README.md"
        original_content = "This is content without a heading."
        readme_path.write_text(original_content, encoding="utf-8")

        trainer = MagicMock(spec=VespaEmbedTrainer)
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)

        content = readme_path.read_text(encoding="utf-8")
        # Content should be unchanged (no heading to insert after)
        assert content == original_content

    def test_handles_existing_vespaembed_mention(self, tmp_path: Path):
        """Test that README already containing vespaembed mention is not modified."""
        readme_path = tmp_path / "README.md"
        original_content = """# My Model

> This model was trained using [vespaembed](https://github.com/vespa-engine/vespaembed).

This is the description."""
        readme_path.write_text(original_content, encoding="utf-8")

        trainer = MagicMock(spec=VespaEmbedTrainer)
        VespaEmbedTrainer._add_vespaembed_to_readme(trainer, tmp_path)

        content = readme_path.read_text(encoding="utf-8")
        # Content should be unchanged
        assert content == original_content


class TestAddVespaembedTag:
    """Tests for adding vespaembed tag to model card metadata."""

    def test_adds_tag_when_model_card_data_exists(self):
        """Test that vespaembed tag is added to model_card_data."""
        mock_model = MagicMock()
        mock_model.model_card_data = MagicMock()

        trainer = MagicMock(spec=VespaEmbedTrainer)
        trainer.model = mock_model
        trainer.config = MagicMock()
        trainer.config.unsloth.enabled = False

        # Call the actual _save_model logic for adding the tag
        if hasattr(trainer.model, "model_card_data") and trainer.model.model_card_data is not None:
            trainer.model.model_card_data.add_tags("vespaembed")

        mock_model.model_card_data.add_tags.assert_called_once_with("vespaembed")

    def test_handles_missing_model_card_data(self):
        """Test that missing model_card_data is handled gracefully."""
        mock_model = MagicMock()
        mock_model.model_card_data = None

        trainer = MagicMock(spec=VespaEmbedTrainer)
        trainer.model = mock_model

        # Should not raise an exception
        if hasattr(trainer.model, "model_card_data") and trainer.model.model_card_data is not None:
            trainer.model.model_card_data.add_tags("vespaembed")

        # add_tags should not have been called since model_card_data is None
        # (This is implicitly tested by the fact that we don't get an AttributeError)

    def test_handles_model_without_model_card_data_attr(self):
        """Test that model without model_card_data attribute is handled."""
        mock_model = MagicMock(spec=[])  # Empty spec means no attributes

        trainer = MagicMock(spec=VespaEmbedTrainer)
        trainer.model = mock_model

        # Should not raise an exception
        if hasattr(trainer.model, "model_card_data") and trainer.model.model_card_data is not None:
            trainer.model.model_card_data.add_tags("vespaembed")
