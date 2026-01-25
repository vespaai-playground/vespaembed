from sentence_transformers import SentenceTransformer


def load_model(model_name_or_path: str, use_unsloth: bool = False) -> SentenceTransformer:
    """Load a sentence transformer model.

    Args:
        model_name_or_path: Model name from HuggingFace Hub or local path
        use_unsloth: Whether to use Unsloth for faster inference

    Returns:
        SentenceTransformer model
    """
    if use_unsloth:
        try:
            from unsloth import FastSentenceTransformer

            return FastSentenceTransformer.from_pretrained(
                model_name_or_path,
                for_inference=True,
            )
        except ImportError:
            raise ImportError("Unsloth not installed. Install with: pip install vespaembed[unsloth]")

    return SentenceTransformer(model_name_or_path)
