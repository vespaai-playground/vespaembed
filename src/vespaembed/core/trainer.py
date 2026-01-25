import os
from pathlib import Path
from typing import Callable, Optional

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback

from vespaembed.core.config import TrainingConfig
from vespaembed.core.registry import Registry
from vespaembed.datasets.loader import load_dataset
from vespaembed.utils.logging import logger


# Get HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")


class ProgressCallback(TrainerCallback):
    """Callback to report training progress."""

    def __init__(self, callback: Callable[[dict], None]):
        self.callback = callback

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics."""
        if logs and self.callback:
            progress = {
                "epoch": state.epoch,
                "step": state.global_step,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
            }
            self.callback(progress)


class VespaEmbedTrainer:
    """High-level trainer that wraps SentenceTransformerTrainer."""

    def __init__(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.model = None
        self.task = None
        self.progress_callback = progress_callback

    def _load_model(self) -> SentenceTransformer:
        """Load the model, using Unsloth if enabled."""
        if self.config.unsloth:
            try:
                from unsloth import FastSentenceTransformer

                logger.info(f"Loading model with Unsloth: {self.config.base_model}")
                return FastSentenceTransformer.from_pretrained(
                    self.config.base_model,
                    token=HF_TOKEN,
                )
            except ImportError:
                raise ImportError(
                    "Unsloth not installed. Install with: pip install vespaembed[unsloth]"
                )
        else:
            logger.info(f"Loading model: {self.config.base_model}")
            return SentenceTransformer(self.config.base_model, token=HF_TOKEN)

    def train(self) -> SentenceTransformer:
        """Run the training process.

        Returns:
            Trained SentenceTransformer model
        """
        # 1. Load model
        self.model = self._load_model()

        # 2. Get task (pass task-specific params if applicable)
        task_cls = Registry.get_task(self.config.task)
        if self.config.task == "matryoshka" and self.config.matryoshka_dims:
            self.task = task_cls(matryoshka_dims=self.config.matryoshka_dims)
        else:
            self.task = task_cls()
        logger.info(f"Using task: {self.task.name} - {self.task.description}")

        # 3. Load and prepare training data
        logger.info(f"Loading training data: {self.config.data.train}")
        train_data = load_dataset(
            self.config.data.train,
            subset=self.config.data.subset,
            split=self.config.data.split or "train",
        )
        train_data = self.task.prepare_dataset(train_data)
        logger.info(f"Training samples: {len(train_data)}")

        # 4. Load and prepare evaluation data (optional)
        eval_data = None
        evaluator = None
        if self.config.data.eval:
            logger.info(f"Loading evaluation data: {self.config.data.eval}")
            # Use eval_split if specified (for HF datasets), otherwise use default split detection
            eval_split = self.config.data.eval_split
            eval_data = load_dataset(
                self.config.data.eval,
                subset=self.config.data.subset,  # Use same subset as training
                split=eval_split,
            )
            eval_data = self.task.prepare_dataset(eval_data)
            evaluator = self.task.get_evaluator(eval_data)
            logger.info(f"Evaluation samples: {len(eval_data)}")

        # 5. Create loss function
        loss = self.task.get_loss(self.model)

        # Wrap with MatryoshkaLoss if dimensions specified (but not for matryoshka task which handles it internally)
        if self.config.matryoshka_dims and self.config.task != "matryoshka":
            from sentence_transformers.losses import MatryoshkaLoss

            logger.info(f"Wrapping with MatryoshkaLoss: {self.config.matryoshka_dims}")
            loss = MatryoshkaLoss(self.model, loss, matryoshka_dims=self.config.matryoshka_dims)

        # 6. Create output directory
        output_dir = Path(self.config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 7. Training arguments
        args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            batch_sampler=self.task.batch_sampler,
            eval_strategy="steps" if evaluator else "no",
            eval_steps=self.config.training.eval_steps if evaluator else None,
            save_strategy="steps",
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.output.save_total_limit,
            logging_steps=self.config.training.logging_steps,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            load_best_model_at_end=True if evaluator else False,
            report_to="tensorboard",
            logging_dir=str(output_dir / "logs"),
        )

        # 8. Create trainer
        callbacks = []
        if self.progress_callback:
            callbacks.append(ProgressCallback(self.progress_callback))

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            loss=loss,
            evaluator=evaluator,
            callbacks=callbacks if callbacks else None,
        )

        # 9. Train
        logger.info("Starting training...")
        trainer.train()

        # 10. Save final model
        final_path = output_dir / "final"
        logger.info(f"Saving model to: {final_path}")

        if self.config.unsloth:
            self.model.save_pretrained_merged(str(final_path))
        else:
            self.model.save_pretrained(str(final_path))

        # 11. Push to hub if configured (always private)
        if self.config.output.push_to_hub and self.config.output.hub_model_id:
            logger.info(f"Pushing to HuggingFace Hub (private): {self.config.output.hub_model_id}")
            self.model.push_to_hub(
                self.config.output.hub_model_id,
                token=HF_TOKEN,
                private=True,
            )

        logger.success("Training completed!")
        return self.model
