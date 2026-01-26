from enum import Enum


class TaskType(str, Enum):
    """Supported training tasks.

    Tasks are organized by data format:
    - pairs: Text pairs for semantic search (anchor, positive)
    - triplets: Text triplets with hard negatives (anchor, positive, negative)
    - similarity: Text pairs with similarity scores
    - tsdae: Unlabeled text for unsupervised learning

    Note: Matryoshka is a training option (--matryoshka flag), not a separate task.
    """

    PAIRS = "pairs"
    TRIPLETS = "triplets"
    SIMILARITY = "similarity"
    TSDAE = "tsdae"


class LossVariant(str, Enum):
    """Available loss function variants.

    For pairs task:
    - mnr: MultipleNegativesRankingLoss (default, recommended)
    - mnr_symmetric: Bidirectional ranking
    - gist: GISTEmbedLoss with guide model
    - cached_mnr: Cached version for larger batches
    - cached_gist: Cached GIST

    For similarity task:
    - cosine: CosineSimilarityLoss (default)
    - cosent: CoSENTLoss
    - angle: AnglELoss
    """

    # Pairs task variants
    MNR = "mnr"
    MNR_SYMMETRIC = "mnr_symmetric"
    GIST = "gist"
    CACHED_MNR = "cached_mnr"
    CACHED_GIST = "cached_gist"

    # Similarity task variants
    COSINE = "cosine"
    COSENT = "cosent"
    ANGLE = "angle"


class RunStatus(str, Enum):
    """Training run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
