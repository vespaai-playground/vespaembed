import json
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

# Get HuggingFace token from environment (fallback for loading private models)
HF_TOKEN_ENV = os.environ.get("HF_TOKEN")


class ProgressCallback(TrainerCallback):
    """Callback to report training progress like tqdm."""

    def __init__(self, callback: Callable[[dict], None]):
        self.callback = callback
        self.start_time = None
        self.total_steps = 0
        self.total_epochs = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        import time

        self.start_time = time.time()
        self.total_steps = state.max_steps
        self.total_epochs = args.num_train_epochs

        if self.callback:
            self.callback(
                {
                    "type": "train_start",
                    "total_steps": self.total_steps,
                    "total_epochs": self.total_epochs,
                }
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics."""
        import time

        if logs and self.callback:
            current_step = state.global_step
            elapsed = time.time() - self.start_time if self.start_time else 0

            # Calculate progress
            progress_pct = (current_step / self.total_steps * 100) if self.total_steps > 0 else 0
            steps_per_sec = current_step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - current_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            progress = {
                "type": "progress",
                "epoch": state.epoch,
                "total_epochs": self.total_epochs,
                "step": current_step,
                "total_steps": self.total_steps,
                "progress_pct": progress_pct,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "steps_per_sec": steps_per_sec,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
            }
            self.callback(progress)

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        import time

        if self.callback:
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.callback(
                {
                    "type": "train_end",
                    "total_steps": state.global_step,
                    "elapsed_seconds": elapsed,
                }
            )


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
        """Load the model with optional LoRA and Unsloth support.

        Supports four modes:
        1. Standard: SentenceTransformer only
        2. Standard + LoRA: SentenceTransformer with PEFT adapter
        3. Unsloth: FastSentenceTransformer (faster training)
        4. Unsloth + LoRA: FastSentenceTransformer with LoRA via get_peft_model
        """
        use_unsloth = self.config.unsloth.enabled

        if use_unsloth:
            return self._load_unsloth_model()
        else:
            return self._load_standard_model()

    def _load_standard_model(self) -> SentenceTransformer:
        """Load model with standard SentenceTransformer, optionally with LoRA."""
        logger.info(f"Loading model: {self.config.base_model}")
        model = SentenceTransformer(self.config.base_model, token=HF_TOKEN_ENV)

        # Set max_seq_length if specified
        if self.config.max_seq_length:
            model.max_seq_length = self.config.max_seq_length
            logger.info(f"Set max_seq_length: {self.config.max_seq_length}")
        else:
            logger.info(f"Using model's default max_seq_length: {model.max_seq_length}")

        # Add LoRA adapter if enabled
        if self.config.lora.enabled:
            try:
                from peft import LoraConfig, TaskType

                logger.info(
                    f"Adding LoRA adapter: r={self.config.lora.r}, "
                    f"alpha={self.config.lora.alpha}, dropout={self.config.lora.dropout}, "
                    f"target_modules={self.config.lora.target_modules}"
                )

                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=self.config.lora.r,
                    lora_alpha=self.config.lora.alpha,
                    lora_dropout=self.config.lora.dropout,
                    target_modules=self.config.lora.target_modules,
                )
                model.add_adapter(peft_config)

            except ImportError:
                raise ImportError("PEFT not installed. Install with: pip install peft")

        return model

    def _load_unsloth_model(self) -> SentenceTransformer:
        """Load model with Unsloth for faster training."""
        try:
            from unsloth import FastSentenceTransformer
        except ImportError:
            raise ImportError("Unsloth not installed. Install with: pip install unsloth")

        # Auto-detect max_seq_length from model if not specified
        max_seq_length = self.config.max_seq_length
        if max_seq_length is None:
            # Load model temporarily to get max_seq_length
            temp_model = SentenceTransformer(self.config.base_model, token=HF_TOKEN_ENV)
            max_seq_length = temp_model.max_seq_length
            del temp_model
            logger.info(f"Auto-detected max_seq_length: {max_seq_length}")
        else:
            logger.info(f"Using specified max_seq_length: {max_seq_length}")

        # Full finetuning when Unsloth is enabled but LoRA is not
        full_finetuning = not self.config.lora.enabled

        logger.info(f"Loading model with Unsloth: {self.config.base_model}")
        model = FastSentenceTransformer.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=max_seq_length,
            full_finetuning=full_finetuning,
            token=HF_TOKEN_ENV,
        )

        # Add LoRA if enabled
        if self.config.lora.enabled:
            # Use Unsloth's optimized gradient checkpointing when enabled
            use_gc = "unsloth" if self.config.gradient_checkpointing else False

            logger.info(
                f"Applying Unsloth LoRA: r={self.config.lora.r}, "
                f"alpha={self.config.lora.alpha}, target_modules={self.config.lora.target_modules}"
            )

            model = FastSentenceTransformer.get_peft_model(
                model,
                r=self.config.lora.r,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules,
                bias="none",
                use_gradient_checkpointing=use_gc,
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
                task_type="FEATURE_EXTRACTION",
            )

        return model

    def _log_training_config(self):
        """Log training configuration parameters."""
        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info("=" * 60)

        # Model & Task
        logger.info(f"  Base Model:      {self.config.base_model}")
        logger.info(f"  Task:            {self.config.task}")
        if self.config.loss_variant:
            logger.info(f"  Loss Variant:    {self.config.loss_variant}")

        # Data
        logger.info(f"  Training Data:   {self.config.data.train}")
        if self.config.data.eval:
            logger.info(f"  Eval Data:       {self.config.data.eval}")

        # Training hyperparameters
        t = self.config.training
        logger.info(f"  Epochs:          {t.epochs}")
        logger.info(f"  Batch Size:      {t.batch_size}")
        logger.info(f"  Learning Rate:   {t.learning_rate}")
        logger.info(f"  Optimizer:       {t.optimizer}")
        logger.info(f"  Scheduler:       {t.scheduler}")
        logger.info(f"  Warmup Ratio:    {t.warmup_ratio}")
        logger.info(f"  Weight Decay:    {t.weight_decay}")

        # Precision
        if t.bf16:
            logger.info("  Precision:       BF16")
        elif t.fp16:
            logger.info("  Precision:       FP16")
        else:
            logger.info("  Precision:       FP32")

        # Optional features
        if self.config.max_seq_length:
            logger.info(f"  Max Seq Length:  {self.config.max_seq_length}")
        if self.config.gradient_checkpointing:
            logger.info("  Grad Checkpoint: Enabled")
        if t.gradient_accumulation_steps > 1:
            logger.info(f"  Grad Accum:      {t.gradient_accumulation_steps}")

        # LoRA
        if self.config.lora.enabled:
            logger.info(f"  LoRA:            r={self.config.lora.r}, alpha={self.config.lora.alpha}")

        # Unsloth
        if self.config.unsloth.enabled:
            logger.info(f"  Unsloth:         Enabled (save: {self.config.unsloth.save_method})")

        # Matryoshka
        if self.config.matryoshka_dims:
            logger.info(f"  Matryoshka:      {self.config.matryoshka_dims}")

        # Output
        logger.info(f"  Output Dir:      {self.config.output.dir}")
        if self.config.output.push_to_hub:
            logger.info(f"  Push to Hub:     {self.config.output.hf_username}")

        logger.info("=" * 60)

    def train(self) -> SentenceTransformer:
        """Run the training process.

        Returns:
            Trained SentenceTransformer model
        """
        # Log configuration
        self._log_training_config()

        # 1. Load model
        self.model = self._load_model()

        # 2. Get task (pass task-specific params if applicable)
        task_cls = Registry.get_task(self.config.task)
        # Handle loss_variant as either enum or string
        loss_variant = self.config.loss_variant
        if loss_variant is not None:
            loss_variant = loss_variant.value if hasattr(loss_variant, "value") else loss_variant

        if loss_variant:
            self.task = task_cls(loss_variant=loss_variant)
        else:
            self.task = task_cls()

        loss_info = f" (loss: {self.task.loss_variant})" if self.task.loss_variant else ""
        logger.info(f"Using task: {self.task.name} - {self.task.description}{loss_info}")

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

        # Wrap with MatryoshkaLoss if dimensions specified (not supported for TSDAE)
        if self.config.matryoshka_dims:
            if self.config.task == "tsdae":
                raise ValueError("Matryoshka is not supported with TSDAE (uses decoder architecture)")
            from sentence_transformers.losses import MatryoshkaLoss

            logger.info(f"Wrapping with MatryoshkaLoss: {self.config.matryoshka_dims}")
            loss = MatryoshkaLoss(self.model, loss, matryoshka_dims=self.config.matryoshka_dims)

        # 6. Create output directory
        output_dir = Path(self.config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 7. Training arguments
        logger.info(f"Optimizer: {self.config.training.optimizer}, Scheduler: {self.config.training.scheduler}")
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
            optim=self.config.training.optimizer,
            lr_scheduler_type=self.config.training.scheduler,
            gradient_checkpointing=self.config.gradient_checkpointing,
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
        self._save_model(final_path)

        # 11. Add label mappings to config.json if task has labels (HuggingFace convention)
        label_config = self.task.get_label_config()
        if label_config:
            config_path = final_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                config.update(label_config)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Added label mappings to config.json ({label_config['num_labels']} labels)")

        # 12. Push to hub if configured (always private)
        if self.config.output.push_to_hub and self.config.output.hf_username:
            if not HF_TOKEN_ENV:
                logger.warning("HF_TOKEN environment variable not set, skipping push to hub")
            else:
                # Construct repo name from username and project directory name
                project_name = Path(self.config.output.dir).name
                repo_id = f"{self.config.output.hf_username}/{project_name}"
                logger.info(f"Pushing to HuggingFace Hub (private): {repo_id}")
                self._push_to_hub(repo_id)

        logger.success("Training completed!")
        return self.model

    def _save_model(self, path: Path) -> None:
        """Save the model based on configuration.

        Handles different save methods for standard, LoRA, and Unsloth models.
        """
        path_str = str(path)

        if self.config.unsloth.enabled:
            save_method = self.config.unsloth.save_method

            if save_method == "lora":
                # Save only LoRA adapters
                logger.info("Saving LoRA adapters only")
                self.model.save_pretrained(path_str)
            elif save_method == "merged_16bit":
                # Merge and save as FP16
                logger.info("Saving merged model (FP16)")
                self.model.save_pretrained_merged(
                    path_str,
                    tokenizer=self.model.tokenizer,
                    save_method="merged_16bit",
                )
            elif save_method == "merged_4bit":
                # Merge and save as 4-bit
                logger.info("Saving merged model (4-bit)")
                self.model.save_pretrained_merged(
                    path_str,
                    tokenizer=self.model.tokenizer,
                    save_method="merged_4bit",
                )
        else:
            # Standard or LoRA (PEFT) save
            self.model.save_pretrained(path_str)

    def _push_to_hub(self, repo_id: str) -> None:
        """Push the model to HuggingFace Hub.

        Handles different push methods for standard, LoRA, and Unsloth models.
        """
        if self.config.unsloth.enabled:
            save_method = self.config.unsloth.save_method

            if save_method == "lora":
                # Push only LoRA adapters
                self.model.push_to_hub(repo_id, token=HF_TOKEN_ENV, private=True)
            else:
                # Push merged model
                self.model.push_to_hub_merged(
                    repo_id,
                    tokenizer=self.model.tokenizer,
                    save_method=save_method,
                    token=HF_TOKEN_ENV,
                    private=True,
                )
        else:
            # Standard or LoRA (PEFT) save
            self.model.push_to_hub(repo_id, token=HF_TOKEN_ENV, private=True)
