from enum import Enum


class TaskType(str, Enum):
    """Supported training tasks."""

    MNR = "mnr"
    TRIPLET = "triplet"
    CONTRASTIVE = "contrastive"
    STS = "sts"
    NLI = "nli"
    TSDAE = "tsdae"
    MATRYOSHKA = "matryoshka"


class RunStatus(str, Enum):
    """Training run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
