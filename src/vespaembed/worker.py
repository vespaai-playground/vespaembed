"""Background worker for training runs."""

import argparse
import json
import signal
import sys
import traceback
from pathlib import Path

# Import tasks to register them
import vespaembed.tasks  # noqa: F401
from vespaembed.core.config import DataConfig, OutputConfig, TrainingConfig, TrainingHyperparameters
from vespaembed.core.trainer import VespaEmbedTrainer
from vespaembed.db import update_run_status
from vespaembed.enums import RunStatus
from vespaembed.utils.logging import logger


class TrainingWorker:
    """Worker that executes training in a subprocess."""

    def __init__(self, run_id: int, config: dict):
        self.run_id = run_id
        self.config = config
        self.trainer = None
        self.stopped = False

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, stopping training...")
        self.stopped = True
        if self.trainer:
            # Trainer will check for stop flag
            pass
        sys.exit(0)

    def _send_update(self, update_type: str, data: dict):
        """Send update to the web server via file."""
        update_dir = Path.home() / ".vespaembed" / "updates"
        update_dir.mkdir(parents=True, exist_ok=True)

        update_file = update_dir / f"run_{self.run_id}.jsonl"
        update = {"type": update_type, "run_id": self.run_id, **data}

        with open(update_file, "a") as f:
            f.write(json.dumps(update) + "\n")

    def run(self):
        """Execute the training run."""
        try:
            logger.info(f"Starting training run {self.run_id}")
            self._send_update("status", {"status": "running"})

            # Determine data source - file upload or HuggingFace dataset
            train_source = self.config.get("train_filename") or self.config.get("hf_dataset")
            eval_source = self.config.get("eval_filename")
            eval_split = None

            # For HuggingFace datasets, eval can come from a different split
            hf_eval_split = self.config.get("hf_eval_split")
            if self.config.get("hf_dataset") and hf_eval_split:
                # Use the same dataset but different split for eval
                eval_source = self.config.get("hf_dataset")
                eval_split = hf_eval_split

            # Build data config
            data_config = DataConfig(
                train=train_source,
                eval=eval_source,
                subset=self.config.get("hf_subset"),
                split=self.config.get("hf_train_split", "train"),
                eval_split=eval_split,
            )

            # Parse matryoshka_dims if present (comes as comma-separated string from UI)
            matryoshka_dims = None
            if self.config.get("matryoshka_dims"):
                dims_str = self.config["matryoshka_dims"]
                if isinstance(dims_str, str):
                    matryoshka_dims = [int(d.strip()) for d in dims_str.split(",") if d.strip()]
                elif isinstance(dims_str, list):
                    matryoshka_dims = dims_str

            # Build training config with nested structure
            training_config = TrainingConfig(
                task=self.config["task"],
                base_model=self.config["base_model"],
                data=data_config,
                training=TrainingHyperparameters(
                    epochs=self.config.get("epochs", 3),
                    batch_size=self.config.get("batch_size", 32),
                    learning_rate=self.config.get("learning_rate", 2e-5),
                    warmup_ratio=self.config.get("warmup_ratio", 0.1),
                    weight_decay=self.config.get("weight_decay", 0.01),
                    fp16=self.config.get("fp16", True),
                    bf16=self.config.get("bf16", False),
                    eval_steps=self.config.get("eval_steps", 500),
                    save_steps=self.config.get("save_steps", 500),
                    logging_steps=self.config.get("logging_steps", 100),
                    gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
                ),
                output=OutputConfig(
                    dir=self.config["output_dir"],  # Required - set by web app
                    push_to_hub=self.config.get("push_to_hub", False),
                    hub_model_id=self.config.get("hub_model_id"),
                ),
                unsloth=self.config.get("use_unsloth", False),
                matryoshka_dims=matryoshka_dims,
            )

            # Create trainer with progress callback
            self.trainer = VespaEmbedTrainer(
                config=training_config,
                progress_callback=self._progress_callback,
            )

            # Run training
            self.trainer.train()

            # Update status on completion
            update_run_status(self.run_id, RunStatus.COMPLETED)
            self._send_update("complete", {"output_dir": training_config.output.dir})
            logger.info(f"Training run {self.run_id} completed successfully")

        except Exception as e:
            logger.error(f"Training run {self.run_id} failed: {e}")
            logger.error(traceback.format_exc())
            update_run_status(self.run_id, RunStatus.ERROR, error_message=str(e))
            self._send_update("error", {"message": str(e)})
            sys.exit(1)

    def _progress_callback(self, progress: dict):
        """Callback for training progress updates."""
        self._send_update("progress", progress)
        self._send_update("log", {"message": self._format_progress(progress)})

    def _format_progress(self, progress: dict) -> str:
        """Format progress as a log message."""
        parts = []
        if progress.get("epoch") is not None:
            parts.append(f"Epoch: {progress['epoch']}")
        if progress.get("step") is not None:
            parts.append(f"Step: {progress['step']}")
        if progress.get("loss") is not None:
            parts.append(f"Loss: {progress['loss']:.4f}")
        if progress.get("learning_rate") is not None:
            parts.append(f"LR: {progress['learning_rate']:.2e}")
        return " | ".join(parts)


def main():
    """Main entry point for the worker."""
    parser = argparse.ArgumentParser(description="VespaEmbed Training Worker")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID")
    parser.add_argument("--config", type=str, required=True, help="Training config JSON")

    args = parser.parse_args()

    config = json.loads(args.config)
    worker = TrainingWorker(run_id=args.run_id, config=config)
    worker.run()


if __name__ == "__main__":
    main()
