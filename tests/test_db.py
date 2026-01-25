"""Tests for the database module."""

import json
from pathlib import Path

import pytest

from vespaembed import db
from vespaembed.enums import RunStatus


class TestDatabase:
    """Test database operations."""

    @pytest.fixture(autouse=True)
    def cleanup_test_runs(self):
        """Clean up test runs before and after each test."""
        # Clean up any test runs from previous tests
        self._cleanup_test_runs()
        yield
        # Clean up after test
        self._cleanup_test_runs()

    def _cleanup_test_runs(self):
        """Delete all runs with test project names."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM runs WHERE project_name LIKE 'test%' OR project_name LIKE 'project-%'"
        )
        conn.commit()
        conn.close()

    def test_init_db_creates_table(self):
        """Test that init_db creates the runs table."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
        result = cursor.fetchone()
        conn.close()
        assert result is not None

    def test_create_run(self):
        """Test creating a new run."""
        config = {"task": "mnr", "base_model": "model"}
        run_id = db.create_run(
            config=config,
            project_name="test-create",
            output_dir="/output",
        )
        assert run_id is not None
        assert run_id > 0

    def test_get_run(self):
        """Test getting a run by ID."""
        config = {"task": "mnr", "base_model": "model"}
        run_id = db.create_run(
            config=config,
            project_name="test-get",
            output_dir="/output",
        )

        run = db.get_run(run_id)
        assert run is not None
        assert run["id"] == run_id
        assert run["project_name"] == "test-get"
        assert run["output_dir"] == "/output"
        assert run["status"] == RunStatus.PENDING.value

        # Config is stored as JSON string
        stored_config = json.loads(run["config"])
        assert stored_config["task"] == "mnr"

    def test_get_nonexistent_run(self):
        """Test getting a run that doesn't exist."""
        run = db.get_run(999999)
        assert run is None

    def test_update_run_status(self):
        """Test updating a run's status."""
        run_id = db.create_run(
            config={"task": "mnr"},
            project_name="test-update",
            output_dir="/output",
        )

        db.update_run_status(run_id, RunStatus.RUNNING, pid=12345)
        run = db.get_run(run_id)
        assert run["status"] == RunStatus.RUNNING.value
        assert run["pid"] == 12345

    def test_update_run_status_completed(self):
        """Test updating a run to completed status."""
        run_id = db.create_run(
            config={"task": "mnr"},
            project_name="test-completed",
            output_dir="/output",
        )

        db.update_run_status(run_id, RunStatus.COMPLETED)
        run = db.get_run(run_id)
        assert run["status"] == RunStatus.COMPLETED.value

    def test_update_run_status_error(self):
        """Test updating a run with error message."""
        run_id = db.create_run(
            config={"task": "mnr"},
            project_name="test-error",
            output_dir="/output",
        )

        db.update_run_status(
            run_id, RunStatus.ERROR, error_message="Something went wrong"
        )
        run = db.get_run(run_id)
        assert run["status"] == RunStatus.ERROR.value
        assert run["error_message"] == "Something went wrong"

    def test_delete_run(self, temp_dir):
        """Test deleting a run."""
        # Create output directory
        output_dir = temp_dir / "output-delete"
        output_dir.mkdir()
        (output_dir / "model.bin").touch()

        run_id = db.create_run(
            config={"task": "mnr"},
            project_name="test-delete",
            output_dir=str(output_dir),
        )

        db.delete_run(run_id, delete_files=True)

        # Run should be gone
        assert db.get_run(run_id) is None
        # Files should be deleted
        assert not output_dir.exists()

    def test_delete_run_keep_files(self, temp_dir):
        """Test deleting a run while keeping files."""
        output_dir = temp_dir / "output-keep"
        output_dir.mkdir()
        (output_dir / "model.bin").touch()

        run_id = db.create_run(
            config={"task": "mnr"},
            project_name="test-keep",
            output_dir=str(output_dir),
        )

        db.delete_run(run_id, delete_files=False)

        # Run should be gone
        assert db.get_run(run_id) is None
        # Files should still exist
        assert output_dir.exists()


class TestRunStatus:
    """Test RunStatus enum."""

    def test_status_values(self):
        """Test RunStatus enum values."""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.ERROR.value == "error"
        assert RunStatus.STOPPED.value == "stopped"
