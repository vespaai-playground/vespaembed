# Tasks module - imports are performed when this module is loaded
# This registers all tasks with the Registry
from vespaembed.tasks.base import BaseTask
from vespaembed.tasks.contrastive import ContrastiveTask
from vespaembed.tasks.matryoshka import MatryoshkaTask
from vespaembed.tasks.mnr import MNRTask
from vespaembed.tasks.nli import NLITask
from vespaembed.tasks.sts import STSTask
from vespaembed.tasks.triplet import TripletTask
from vespaembed.tasks.tsdae import TSDAETask

__all__ = [
    "BaseTask",
    "ContrastiveTask",
    "MatryoshkaTask",
    "MNRTask",
    "NLITask",
    "STSTask",
    "TripletTask",
    "TSDAETask",
]
