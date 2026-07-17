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
    conversion) and saves the result. The ONNX graph ends up at
    <output_path>/model.onnx alongside the tokenizer files.

    Args:
        model_path: Path to saved SentenceTransformer model
        output_path: Output directory

    Returns:
        Path to exported model directory
    """
    import shutil

    output_path.mkdir(parents=True, exist_ok=True)

    # Reload the saved model with ONNX backend — this triggers automatic conversion
    onnx_model = SentenceTransformer(str(model_path), backend="onnx")
    onnx_model.save_pretrained(str(output_path))

    # sentence-transformers saves the graph under an onnx/ subdirectory; flatten it
    # so the file sits at <output_path>/model.onnx (the layout optimum-cli produces,
    # and what deployment targets expect). Otherwise exporting into a directory
    # named "onnx" yields a confusing onnx/onnx/model.onnx.
    nested_dir = output_path / "onnx"
    if nested_dir.is_dir():
        for item in nested_dir.iterdir():
            target = output_path / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        if not any(nested_dir.iterdir()):
            nested_dir.rmdir()

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
