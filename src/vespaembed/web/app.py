import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request

from vespaembed.db import create_run, delete_run, get_active_run, get_all_runs, get_run, update_run_status
from vespaembed.enums import RunStatus


# Helper: Check if process is alive
def is_process_alive(pid: int | None) -> bool:
    """Check if a process with the given PID is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # Signal 0 checks if process exists without killing it
        return True
    except (ProcessLookupError, OSError):
        return False


# Helper: Check if final model exists for a run
def has_final_model(output_dir: str | None) -> bool:
    """Check if training completed successfully by looking for final model."""
    if not output_dir:
        return False
    final_path = Path(output_dir) / "final"
    return final_path.exists() and final_path.is_dir()


# Startup: Sync run statuses with actual process states
def sync_run_statuses():
    """Check all running/pending runs and update status if process is dead.

    This handles cases where:
    - Server was restarted while training was in progress
    - Training process crashed unexpectedly
    - Training completed but status wasn't updated
    """
    runs = get_all_runs()
    for run in runs:
        if run["status"] in [RunStatus.RUNNING.value, RunStatus.PENDING.value]:
            pid = run.get("pid")
            run_id = run["id"]

            # Check if process is still alive
            if is_process_alive(pid):
                continue  # Still running, leave it

            # Process is dead - determine final status
            if has_final_model(run.get("output_dir")):
                update_run_status(run_id, RunStatus.COMPLETED)
                print(f"[sync] Run {run_id}: Marked as completed (final model exists)")
            else:
                update_run_status(run_id, RunStatus.ERROR, error_message="Process terminated unexpectedly")
                print(f"[sync] Run {run_id}: Marked as error (no final model)")


# Paths
PACKAGE_DIR = Path(__file__).parent.parent
STATIC_DIR = PACKAGE_DIR / "static"
TEMPLATES_DIR = PACKAGE_DIR / "templates"
BASE_DIR = Path.home() / ".vespaembed"
UPLOAD_DIR = BASE_DIR / "uploads"
UPDATE_DIR = BASE_DIR / "updates"
PROJECTS_DIR = BASE_DIR / "projects"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPDATE_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI app
app = FastAPI(title="VespaEmbed", version="0.0.1")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Sync run statuses on startup
sync_run_statuses()


# Pydantic models
class TrainRequest(BaseModel):
    # Project name (required) - used to create output directory
    project_name: str = Field(..., description="Project name (alphanumeric and hyphens only)")

    # Data source - either file upload OR HuggingFace dataset
    train_filename: Optional[str] = Field(None, description="Path to uploaded training file")
    eval_filename: Optional[str] = Field(None, description="Path to uploaded evaluation file (optional)")

    # HuggingFace dataset (alternative to file upload)
    hf_dataset: Optional[str] = Field(
        None, description="HuggingFace dataset name (e.g., 'sentence-transformers/all-nli')"
    )
    hf_subset: Optional[str] = Field(None, description="Dataset subset/config name")
    hf_train_split: str = Field("train", description="Training split name")
    hf_eval_split: Optional[str] = Field(None, description="Evaluation split name (optional)")

    # Required
    task: str
    base_model: str

    # Basic hyperparameters
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5

    # Advanced hyperparameters
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = False
    bf16: bool = False
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    gradient_accumulation_steps: int = 1

    # LoRA/PEFT parameters
    lora_enabled: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: str = Field("query, key, value, dense", description="Comma-separated list of target modules")

    # Model configuration
    max_seq_length: Optional[int] = None  # Auto-detect from model if not specified
    gradient_checkpointing: bool = False  # Saves VRAM, uses Unsloth optimization when Unsloth is enabled

    # Unsloth parameters
    unsloth_enabled: bool = False
    unsloth_save_method: str = "merged_16bit"  # "lora", "merged_16bit", "merged_4bit"

    # Hub push
    push_to_hub: bool = False
    hf_username: Optional[str] = None

    # Task-specific parameters
    matryoshka_dims: Optional[str] = Field(
        None, description="Matryoshka dimensions as comma-separated string (e.g., '768,512,256,128')"
    )
    loss_variant: Optional[str] = Field(
        None, description="Loss function variant (task-specific, uses default if not specified)"
    )

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9-]*$", v):
            raise ValueError(
                "Project name must start with alphanumeric and contain only alphanumeric characters and hyphens"
            )
        return v


class StopRequest(BaseModel):
    run_id: int


class UploadResponse(BaseModel):
    filename: str
    filepath: str
    columns: list[str]
    preview: list[dict]
    row_count: int


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form("train"),  # "train" or "eval"
):
    """Upload a training or evaluation data file.

    Args:
        file: The file to upload (CSV or JSONL)
        file_type: Either "train" or "eval" to indicate file purpose
    """
    if file_type not in ("train", "eval"):
        raise HTTPException(status_code=400, detail="file_type must be 'train' or 'eval'")

    # Save file with prefix to distinguish train/eval
    original_filename = file.filename
    filename = f"{file_type}_{original_filename}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Get preview and row count
    preview = []
    columns = []
    row_count = 0

    try:
        if original_filename.endswith(".csv"):
            import pandas as pd

            # Get row count
            df_full = pd.read_csv(filepath)
            row_count = len(df_full)

            # Get preview
            df = df_full.head(5)
            columns = df.columns.tolist()
            preview = df.to_dict("records")

        elif original_filename.endswith(".jsonl"):
            with open(filepath) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    row_count += 1
                    if len(preview) < 5:
                        preview.append(record)
                        if not columns:
                            columns = list(record.keys())
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV or JSONL files.",
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSONL format: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    return UploadResponse(
        filename=original_filename,
        filepath=str(filepath),
        columns=columns,
        preview=preview,
        row_count=row_count,
    )


@app.post("/train")
async def train(config: TrainRequest):
    """Start a training run."""
    # Check for active run
    active = get_active_run()
    if active:
        raise HTTPException(
            status_code=400,
            detail="A training run is already in progress. Stop it first.",
        )

    # Validate data source - must have either file or HF dataset
    has_file = config.train_filename is not None
    has_hf = config.hf_dataset is not None

    if not has_file and not has_hf:
        raise HTTPException(
            status_code=400,
            detail="Must provide either train_filename or hf_dataset",
        )

    if has_file and has_hf:
        raise HTTPException(
            status_code=400,
            detail="Cannot specify both train_filename and hf_dataset. Choose one.",
        )

    # Validate file exists if using file upload
    if has_file and not Path(config.train_filename).exists():
        raise HTTPException(status_code=400, detail="Training data file not found")

    if config.eval_filename and not Path(config.eval_filename).exists():
        raise HTTPException(status_code=400, detail="Evaluation data file not found")

    # Validate hub config
    if config.push_to_hub and not config.hf_username:
        raise HTTPException(
            status_code=400,
            detail="hf_username is required when push_to_hub is enabled",
        )

    # Create output directory in ~/.vespaembed/projects/
    output_dir = PROJECTS_DIR / config.project_name
    if output_dir.exists():
        # Append timestamp to make unique
        timestamp = int(time.time())
        output_dir = PROJECTS_DIR / f"{config.project_name}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare config with resolved output_dir
    config_dict = config.model_dump()
    config_dict["output_dir"] = str(output_dir)

    # Create run record
    run_id = create_run(
        config=config_dict,
        project_name=config.project_name,
        output_dir=str(output_dir),
    )

    # Clear any existing update file for this run
    update_file = UPDATE_DIR / f"run_{run_id}.jsonl"
    if update_file.exists():
        update_file.unlink()

    # Start worker process
    cmd = [
        sys.executable,
        "-m",
        "vespaembed.worker",
        "--run-id",
        str(run_id),
        "--config",
        json.dumps(config_dict),
    ]

    # Don't capture stdout/stderr - let them flow to terminal for visibility
    process = subprocess.Popen(
        cmd,
        stdout=None,  # Inherit from parent - logs visible in terminal
        stderr=None,  # Inherit from parent - errors visible in terminal
        start_new_session=True,
    )

    update_run_status(run_id, RunStatus.RUNNING, pid=process.pid)

    return {"message": "Training started", "run_id": run_id, "output_dir": str(output_dir)}


@app.post("/stop")
async def stop(request: StopRequest):
    """Stop a training run."""
    run = get_run(request.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run["status"] != RunStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="Run is not active")

    # Send SIGTERM to process
    if run.get("pid"):
        try:
            os.kill(run["pid"], 15)  # SIGTERM
        except ProcessLookupError:
            pass

    update_run_status(request.run_id, RunStatus.STOPPED)

    return {"message": "Training stopped"}


@app.get("/runs")
async def list_runs():
    """List all training runs."""
    return get_all_runs()


@app.get("/runs/{run_id}")
async def get_run_details(run_id: int):
    """Get details of a specific run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.delete("/runs/{run_id}")
async def delete_run_endpoint(run_id: int):
    """Delete a training run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Stop if running
    if run["status"] == RunStatus.RUNNING.value and run.get("pid"):
        try:
            os.kill(run["pid"], 15)
        except ProcessLookupError:
            pass

    delete_run(run_id, delete_files=True)
    return {"message": "Run deleted"}


@app.get("/active_run_id")
async def get_active_run_id():
    """Get the currently active run ID."""
    run = get_active_run()
    return {"run_id": run["id"] if run else None}


@app.get("/runs/{run_id}/updates")
async def get_run_updates(run_id: int, since_line: int = 0):
    """Poll for updates from a training run.

    Args:
        run_id: The run ID to get updates for
        since_line: Return updates after this line number (0-indexed).
                    Client should track this and pass it on subsequent requests.

    Returns:
        updates: List of update objects (progress, log, status, etc.)
        next_line: The line number to use for the next poll request
        has_more: Whether there might be more updates (run still active)
    """
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    update_file = UPDATE_DIR / f"run_{run_id}.jsonl"
    updates = []
    next_line = since_line

    if update_file.exists():
        with open(update_file) as f:
            for i, line in enumerate(f):
                if i < since_line:
                    continue
                line = line.strip()
                if line:
                    try:
                        updates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                next_line = i + 1

    # Check if run is still active
    is_active = run["status"] == RunStatus.RUNNING.value

    return {
        "updates": updates,
        "next_line": next_line,
        "has_more": is_active,
        "run_status": run["status"],
    }


@app.get("/api/tasks")
async def get_all_tasks():
    """Get information about all available training tasks."""
    # Import tasks to ensure they're registered
    import vespaembed.tasks  # noqa: F401
    from vespaembed.core.registry import Registry

    return Registry.get_task_info()


@app.get("/api/tasks/{task_name}")
async def get_task(task_name: str):
    """Get information about a specific training task."""
    # Import tasks to ensure they're registered
    import vespaembed.tasks  # noqa: F401
    from vespaembed.core.registry import Registry

    try:
        return Registry.get_task_info(task_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: int):
    """Get training metrics from TensorBoard event files.

    Returns metrics like loss, learning_rate, etc. parsed from tfevents files.
    """
    import math

    def sanitize_value(value):
        """Convert NaN/Inf to None for JSON serialization."""
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    log_dir = Path(run["output_dir"]) / "logs"
    if not log_dir.exists():
        return {"metrics": {}, "steps": []}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        # Find all event files
        metrics = {}
        ea = EventAccumulator(str(log_dir))
        ea.Reload()

        # Get all scalar tags
        scalar_tags = ea.Tags().get("scalars", [])

        for tag in scalar_tags:
            events = ea.Scalars(tag)
            # Clean up tag name (remove "train/" prefix if present)
            clean_tag = tag.replace("train/", "").replace("eval/", "eval_")
            # Sanitize values to handle NaN/Inf
            metrics[clean_tag] = [{"step": e.step, "value": sanitize_value(e.value)} for e in events]

        return {"metrics": metrics}

    except Exception as e:
        # If tensorboard parsing fails, return empty metrics
        return {"metrics": {}, "error": str(e)}


@app.get("/runs/{run_id}/artifacts")
async def get_run_artifacts(run_id: int):
    """Get list of downloadable artifacts for a training run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    output_dir = Path(run["output_dir"])
    if not output_dir.exists():
        return {"artifacts": []}

    artifacts = []

    # Check for final model
    final_path = output_dir / "final"
    if final_path.exists() and final_path.is_dir():
        # Get total size of final directory
        total_size = sum(f.stat().st_size for f in final_path.rglob("*") if f.is_file())
        artifacts.append(
            {
                "name": "final",
                "label": "Final Model",
                "category": "model",
                "path": str(final_path),
                "size": total_size,
                "is_directory": True,
            }
        )

    # Check for config file
    config_files = list(output_dir.glob("*.json"))
    for cf in config_files:
        if cf.name in ["config.json", "training_config.json"]:
            artifacts.append(
                {
                    "name": cf.name,
                    "label": "Training Config",
                    "category": "config",
                    "path": str(cf),
                    "size": cf.stat().st_size,
                    "is_directory": False,
                }
            )

    # Check for checkpoints
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    for ckpt in checkpoints[:3]:  # Limit to 3 most recent
        if ckpt.is_dir():
            total_size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
            artifacts.append(
                {
                    "name": ckpt.name,
                    "label": f"Checkpoint ({ckpt.name})",
                    "category": "checkpoint",
                    "path": str(ckpt),
                    "size": total_size,
                    "is_directory": True,
                }
            )

    # Check for logs
    logs_path = output_dir / "logs"
    if logs_path.exists():
        total_size = sum(f.stat().st_size for f in logs_path.rglob("*") if f.is_file())
        artifacts.append(
            {
                "name": "logs",
                "label": "TensorBoard Logs",
                "category": "logs",
                "path": str(logs_path),
                "size": total_size,
                "is_directory": True,
            }
        )

    return {"artifacts": artifacts, "output_dir": str(output_dir)}
