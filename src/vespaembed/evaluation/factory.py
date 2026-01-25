from typing import Any, Optional

from datasets import Dataset
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
    InformationRetrievalEvaluator,
    TripletEvaluator,
)


def create_evaluator(
    task: str,
    eval_dataset: Dataset,
    name: str = "eval",
) -> Optional[Any]:
    """Create an appropriate evaluator based on task type.

    Args:
        task: Task name (mnr, triplet, contrastive, sts, nli, tsdae, matryoshka)
        eval_dataset: Prepared evaluation dataset
        name: Evaluator name

    Returns:
        Evaluator instance or None
    """
    if task == "mnr":
        return _create_ir_evaluator(eval_dataset, name)
    elif task == "triplet":
        return _create_triplet_evaluator(eval_dataset, name)
    elif task == "contrastive":
        return _create_binary_evaluator(eval_dataset, name)
    elif task == "sts":
        return _create_similarity_evaluator(eval_dataset, name)
    elif task == "nli":
        return _create_similarity_evaluator(eval_dataset, name)
    elif task == "tsdae":
        return None  # TSDAE has no intrinsic evaluator
    elif task == "matryoshka":
        return _create_ir_evaluator(eval_dataset, name)
    else:
        return None


def _create_ir_evaluator(dataset: Dataset, name: str) -> InformationRetrievalEvaluator:
    """Create Information Retrieval evaluator."""
    queries = {str(i): text for i, text in enumerate(dataset["anchor"])}
    corpus = {str(i): text for i, text in enumerate(dataset["positive"])}
    relevant_docs = {str(i): {str(i)} for i in range(len(dataset))}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
    )


def _create_triplet_evaluator(dataset: Dataset, name: str) -> TripletEvaluator:
    """Create Triplet evaluator."""
    return TripletEvaluator(
        anchors=dataset["anchor"],
        positives=dataset["positive"],
        negatives=dataset["negative"],
        name=name,
    )


def _create_binary_evaluator(dataset: Dataset, name: str) -> BinaryClassificationEvaluator:
    """Create Binary Classification evaluator."""
    return BinaryClassificationEvaluator(
        sentences1=dataset["sentence1"],
        sentences2=dataset["sentence2"],
        labels=dataset["label"],
        name=name,
    )


def _create_similarity_evaluator(dataset: Dataset, name: str) -> EmbeddingSimilarityEvaluator:
    """Create Embedding Similarity evaluator."""
    return EmbeddingSimilarityEvaluator(
        sentences1=dataset["sentence1"],
        sentences2=dataset["sentence2"],
        scores=dataset["score"],
        name=name,
    )
