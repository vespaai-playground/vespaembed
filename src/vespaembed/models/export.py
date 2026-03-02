from pathlib import Path
from typing import Optional, Union

from sentence_transformers import SentenceTransformer


def export_model(
    model_path: Union[str, Path],
    output_path: str,
    format: str = "onnx",
) -> str:
    """Export a saved model to a different format.

    Args:
        model_path: Path to saved SentenceTransformer model directory
        output_path: Output directory for the exported model
        format: Export format ("onnx")

    Returns:
        Path to exported model directory

    Raises:
        ValueError: If format is not supported
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    if format.lower() == "onnx":
        return _export_onnx(model_path, output_path)
    else:
        raise ValueError(f"Unsupported export format: {format}. Supported: onnx")


def _export_onnx(model_path: Path, output_path: Path) -> str:
    """Export model to ONNX format using sentence-transformers' built-in ONNX backend.

    Reloads the saved model with backend="onnx" (which triggers optimum-based
    conversion) and saves the result.

    Args:
        model_path: Path to saved SentenceTransformer model
        output_path: Output directory

    Returns:
        Path to exported model directory
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Reload the saved model with ONNX backend — this triggers automatic conversion
    onnx_model = SentenceTransformer(str(model_path), backend="onnx")
    onnx_model.save_pretrained(str(output_path))

    return str(output_path)


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
