# Tasks module - imports are performed when this module is loaded
# This registers all tasks with the Registry
#
# Available tasks (by data format):
# 1. pairs - Text pairs for semantic search (anchor, positive)
# 2. triplets - Text triplets with hard negatives (anchor, positive, negative)
# 3. similarity - Text pairs with similarity scores (sentence1, sentence2, score)
# 4. tsdae - Unlabeled text for unsupervised learning (text)
#
# Matryoshka is a training option (--matryoshka flag) that can be enabled for any task except TSDAE.
from vespaembed.tasks.base import BaseTask
from vespaembed.tasks.pairs import PairsTask
from vespaembed.tasks.similarity import SimilarityTask
from vespaembed.tasks.triplets import TripletsTask
from vespaembed.tasks.tsdae import TSDAETask

__all__ = [
    "BaseTask",
    "PairsTask",
    "SimilarityTask",
    "TripletsTask",
    "TSDAETask",
]
