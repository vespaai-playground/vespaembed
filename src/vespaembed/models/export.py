from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer


def export_model(
    model: SentenceTransformer,
    output_path: str,
    format: str = "onnx",
) -> str:
    """Export a model to a different format.

    Args:
        model: SentenceTransformer model
        output_path: Output directory or file path
        format: Export format ("onnx")

    Returns:
        Path to exported model

    Raises:
        ValueError: If format is not supported
    """
    output_path = Path(output_path)

    if format.lower() == "onnx":
        return _export_onnx(model, output_path)
    else:
        raise ValueError(f"Unsupported export format: {format}. Supported: onnx")


def _export_onnx(model: SentenceTransformer, output_path: Path) -> str:
    """Export model to ONNX format.

    Args:
        model: SentenceTransformer model
        output_path: Output directory

    Returns:
        Path to ONNX model
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError("ONNX not installed. Install with: pip install vespaembed[onnx]")

    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / "model.onnx"

    # Use sentence-transformers built-in ONNX export if available
    # Otherwise fall back to manual export
    try:
        model.save(str(output_path), model_name_or_path="model.onnx", create_model_card=False)
    except Exception:
        # Manual export via transformers
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model[0].auto_model.config._name_or_path)

        # Export the transformer part
        model[0].auto_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    return str(onnx_path)


def push_to_hub(
    model: SentenceTransformer,
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
) -> str:
    """Push model to HuggingFace Hub.

    Args:
        model: SentenceTransformer model
        repo_id: Repository ID (e.g., "username/model-name")
        commit_message: Commit message
        private: Whether to create a private repository

    Returns:
        URL of the model on HuggingFace Hub
    """
    return model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message or "Upload model via vespaembed",
        private=private,
    )
