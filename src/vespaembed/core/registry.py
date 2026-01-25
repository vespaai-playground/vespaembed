from typing import TYPE_CHECKING, Any, Type


if TYPE_CHECKING:
    from vespaembed.tasks.base import BaseTask


# Default hyperparameters for all tasks
DEFAULT_HYPERPARAMETERS = {
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    "bf16": False,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,
    "gradient_accumulation_steps": 1,
}

# Task-specific parameter definitions
TASK_SPECIFIC_PARAMS = {
    "matryoshka": {
        "matryoshka_dims": {
            "type": "text",
            "label": "Matryoshka Dimensions",
            "description": "Comma-separated embedding dimensions (e.g., 768,512,256,128,64)",
            "default": "768,512,256,128,64",
        }
    }
}

# Sample data for each task
TASK_SAMPLE_DATA = {
    "mnr": [
        {"anchor": "What is machine learning?", "positive": "Machine learning is a subset of AI that enables systems to learn from data."},
        {"anchor": "How does photosynthesis work?", "positive": "Photosynthesis converts sunlight into chemical energy in plants."},
    ],
    "triplet": [
        {"anchor": "What is Python?", "positive": "Python is a programming language", "negative": "A python is a large snake"},
        {"anchor": "Apple stock price", "positive": "AAPL is trading at $150", "negative": "Apples are a healthy fruit"},
    ],
    "contrastive": [
        {"sentence1": "The cat sat on the mat", "sentence2": "A feline rested on the rug", "label": 1},
        {"sentence1": "I love programming", "sentence2": "The weather is sunny", "label": 0},
    ],
    "sts": [
        {"sentence1": "A man is playing guitar", "sentence2": "A person plays a musical instrument", "score": 0.85},
        {"sentence1": "A dog is running", "sentence2": "The cat sleeps peacefully", "score": 0.12},
    ],
    "nli": [
        {"sentence1": "A man is eating pizza", "sentence2": "A man is eating food", "label": 0},
        {"sentence1": "A woman is playing guitar", "sentence2": "A man is playing piano", "label": 2},
    ],
    "tsdae": [
        {"text": "Machine learning is transforming how we analyze data."},
        {"text": "Natural language processing enables computers to understand human language."},
    ],
    "matryoshka": [
        {"anchor": "What is deep learning?", "positive": "Deep learning uses neural networks with many layers."},
        {"anchor": "Explain quantum computing", "positive": "Quantum computing uses quantum bits for computation."},
    ],
}


class Registry:
    """Central registry for tasks."""

    _tasks: dict[str, Type["BaseTask"]] = {}

    @classmethod
    def register_task(cls, name: str):
        """Decorator to register a task.

        Usage:
            @Registry.register_task("mnr")
            class MNRTask(BaseTask):
                ...
        """

        def decorator(task_cls: Type["BaseTask"]):
            cls._tasks[name] = task_cls
            return task_cls

        return decorator

    @classmethod
    def get_task(cls, name: str) -> Type["BaseTask"]:
        """Get a task class by name.

        Args:
            name: Task name (e.g., "mnr", "triplet")

        Returns:
            Task class

        Raises:
            ValueError: If task is not found
        """
        if name not in cls._tasks:
            available = ", ".join(sorted(cls._tasks.keys()))
            raise ValueError(f"Unknown task: '{name}'. Available tasks: {available}")
        return cls._tasks[name]

    @classmethod
    def list_tasks(cls) -> list[str]:
        """List all registered task names."""
        return sorted(cls._tasks.keys())

    @classmethod
    def get_task_info(cls, name: str = None) -> dict | list[dict]:
        """Get information about registered tasks.

        Args:
            name: If provided, get info for a specific task. Otherwise, get all tasks.

        Returns:
            Task info dict or list of task info dicts
        """
        if name:
            if name not in cls._tasks:
                available = ", ".join(sorted(cls._tasks.keys()))
                raise ValueError(f"Unknown task: '{name}'. Available tasks: {available}")
            return cls._build_task_info(name, cls._tasks[name])

        return [
            cls._build_task_info(task_name, task_cls)
            for task_name, task_cls in sorted(cls._tasks.items())
        ]

    @classmethod
    def _build_task_info(cls, name: str, task_cls: Type["BaseTask"]) -> dict:
        """Build task info dictionary."""
        info = {
            "name": name,
            "description": getattr(task_cls, "description", ""),
            "expected_columns": getattr(task_cls, "expected_columns", []),
            "column_aliases": getattr(task_cls, "column_aliases", {}),
            "hyperparameters": DEFAULT_HYPERPARAMETERS.copy(),
            "task_specific_params": TASK_SPECIFIC_PARAMS.get(name, {}),
            "sample_data": TASK_SAMPLE_DATA.get(name, []),
        }
        return info
