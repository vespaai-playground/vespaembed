import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from vespaembed.enums import RunStatus

# Default database location
DEFAULT_DB_DIR = Path.home() / ".vespaembed"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "vespaembed.db"


def get_db_path() -> Path:
    """Get the database path, creating directory if needed."""
    DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'pending',
            pid INTEGER,
            config TEXT,
            project_name TEXT,
            output_dir TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT
        )
    """)

    conn.commit()
    conn.close()


def create_run(config: dict, project_name: str, output_dir: str) -> int:
    """Create a new training run."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO runs (status, config, project_name, output_dir)
        VALUES (?, ?, ?, ?)
    """,
        (RunStatus.PENDING.value, json.dumps(config), project_name, output_dir),
    )

    run_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return run_id


def update_run_status(
    run_id: int,
    status: RunStatus,
    pid: Optional[int] = None,
    error_message: Optional[str] = None,
):
    """Update a run's status."""
    conn = get_connection()
    cursor = conn.cursor()

    if pid is not None:
        cursor.execute(
            """
            UPDATE runs SET status = ?, pid = ?, updated_at = ?
            WHERE id = ?
        """,
            (status.value, pid, datetime.now(), run_id),
        )
    elif error_message is not None:
        cursor.execute(
            """
            UPDATE runs SET status = ?, error_message = ?, updated_at = ?
            WHERE id = ?
        """,
            (status.value, error_message, datetime.now(), run_id),
        )
    else:
        cursor.execute(
            """
            UPDATE runs SET status = ?, updated_at = ?
            WHERE id = ?
        """,
            (status.value, datetime.now(), run_id),
        )

    conn.commit()
    conn.close()


def get_run(run_id: int) -> Optional[dict]:
    """Get a run by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def get_all_runs() -> list[dict]:
    """Get all runs."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM runs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_active_run() -> Optional[dict]:
    """Get the currently active (running or pending) run."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM runs
        WHERE status IN (?, ?)
        ORDER BY created_at DESC
        LIMIT 1
    """,
        (RunStatus.PENDING.value, RunStatus.RUNNING.value),
    )

    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def delete_run(run_id: int, delete_files: bool = True):
    """Delete a run and optionally its output files."""
    run = get_run(run_id)

    if run and delete_files and run.get("output_dir"):
        output_path = Path(run["output_dir"])
        if output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()


# Initialize database on import
init_db()
