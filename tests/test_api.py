"""Tests for the FastAPI web application endpoints."""

import pytest


class TestHealthAndIndex:
    """Test basic endpoints."""

    def test_index_returns_html(self, client):
        """Test that the index page returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "VespaEmbed" in response.text

    def test_index_contains_modal(self, client):
        """Test that the index page contains the new project modal with wizard layout."""
        response = client.get("/")
        assert "new-training-modal" in response.text
        assert "modal-wizard" in response.text


class TestTasksAPI:
    """Test the /api/tasks endpoints."""

    def test_get_all_tasks(self, client):
        """Test getting all tasks."""
        response = client.get("/api/tasks")
        assert response.status_code == 200

        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) == 4  # pairs, triplets, similarity, tsdae

        task_names = [t["name"] for t in tasks]
        assert "pairs" in task_names
        assert "triplets" in task_names
        assert "similarity" in task_names
        assert "tsdae" in task_names

    def test_task_has_required_fields(self, client):
        """Test that each task has all required fields."""
        response = client.get("/api/tasks")
        tasks = response.json()

        required_fields = [
            "name",
            "description",
            "expected_columns",
            "column_aliases",
            "hyperparameters",
            "task_specific_params",
            "sample_data",
        ]

        for task in tasks:
            for field in required_fields:
                assert field in task, f"Task {task['name']} missing field: {field}"

    def test_get_single_task(self, client):
        """Test getting a single task by name."""
        response = client.get("/api/tasks/pairs")
        assert response.status_code == 200

        task = response.json()
        assert task["name"] == "pairs"
        assert task["expected_columns"] == ["anchor", "positive"]
        assert "hyperparameters" in task
        assert "sample_data" in task

    def test_get_nonexistent_task(self, client):
        """Test getting a task that doesn't exist."""
        response = client.get("/api/tasks/nonexistent")
        assert response.status_code == 404

    def test_pairs_task_columns(self, client):
        """Test pairs task has correct columns."""
        response = client.get("/api/tasks/pairs")
        task = response.json()
        assert task["expected_columns"] == ["anchor", "positive"]

    def test_triplets_task_columns(self, client):
        """Test triplets task has correct columns."""
        response = client.get("/api/tasks/triplets")
        task = response.json()
        assert task["expected_columns"] == ["anchor", "positive", "negative"]

    def test_triplets_task_has_loss_options(self, client):
        """Test triplets task has loss variant options."""
        response = client.get("/api/tasks/triplets")
        task = response.json()

        assert "loss_options" in task
        assert "mnr" in task["loss_options"]
        assert "mnr_symmetric" in task["loss_options"]
        assert task["default_loss"] == "mnr"

    def test_similarity_task_columns(self, client):
        """Test similarity task has correct columns."""
        response = client.get("/api/tasks/similarity")
        task = response.json()
        assert task["expected_columns"] == ["sentence1", "sentence2", "score"]

    def test_tsdae_task_columns(self, client):
        """Test TSDAE task has correct columns."""
        response = client.get("/api/tasks/tsdae")
        task = response.json()
        assert task["expected_columns"] == ["text"]

    def test_pairs_task_has_loss_options(self, client):
        """Test pairs task has loss variant options."""
        response = client.get("/api/tasks/pairs")
        task = response.json()

        assert "loss_options" in task
        assert "mnr" in task["loss_options"]
        assert "mnr_symmetric" in task["loss_options"]
        assert "gist" in task["loss_options"]
        assert task["default_loss"] == "mnr"

    def test_similarity_task_has_loss_options(self, client):
        """Test similarity task has loss variant options."""
        response = client.get("/api/tasks/similarity")
        task = response.json()

        assert "loss_options" in task
        assert "cosine" in task["loss_options"]
        assert "cosent" in task["loss_options"]
        assert "angle" in task["loss_options"]
        assert task["default_loss"] == "cosine"

    def test_hyperparameters_have_defaults(self, client):
        """Test that hyperparameters have sensible defaults."""
        response = client.get("/api/tasks/pairs")
        task = response.json()
        hyper = task["hyperparameters"]

        assert hyper["epochs"] == 3
        assert hyper["batch_size"] == 32
        assert hyper["learning_rate"] == 2e-5
        assert hyper["warmup_ratio"] == 0.1
        assert hyper["fp16"] is True
        assert hyper["bf16"] is False

    def test_sample_data_matches_columns(self, client):
        """Test that sample data has the expected columns."""
        response = client.get("/api/tasks")
        tasks = response.json()

        for task in tasks:
            if task["sample_data"]:
                sample = task["sample_data"][0]
                for col in task["expected_columns"]:
                    assert col in sample, f"Task {task['name']} sample missing column: {col}"


class TestUploadEndpoint:
    """Test the /upload endpoint."""

    def test_upload_csv_file(self, client, sample_csv_file):
        """Test uploading a CSV file."""
        with open(sample_csv_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("train.csv", f, "text/csv")},
                data={"file_type": "train"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "train.csv"
        assert "anchor" in data["columns"]
        assert "positive" in data["columns"]
        assert data["row_count"] == 2
        assert len(data["preview"]) == 2

    def test_upload_jsonl_file(self, client, sample_jsonl_file):
        """Test uploading a JSONL file."""
        with open(sample_jsonl_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("train.jsonl", f, "application/jsonl")},
                data={"file_type": "train"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "train.jsonl"
        assert data["row_count"] == 2

    def test_upload_eval_file(self, client, sample_csv_file):
        """Test uploading an evaluation file."""
        with open(sample_csv_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("eval.csv", f, "text/csv")},
                data={"file_type": "eval"},
            )

        assert response.status_code == 200
        data = response.json()
        # Filename should be prefixed in the stored path
        assert "filepath" in data

    def test_upload_invalid_file_type(self, client, sample_csv_file):
        """Test uploading with invalid file_type."""
        with open(sample_csv_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("train.csv", f, "text/csv")},
                data={"file_type": "invalid"},
            )

        assert response.status_code == 400
        assert "file_type must be" in response.json()["detail"]


class TestRunsEndpoint:
    """Test the /runs endpoints."""

    def test_list_runs_empty(self, client):
        """Test listing runs when none exist."""
        response = client.get("/runs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_nonexistent_run(self, client):
        """Test getting a run that doesn't exist."""
        response = client.get("/runs/99999")
        assert response.status_code == 404

    def test_get_active_run_none(self, client):
        """Test getting active run when none is running."""
        response = client.get("/active_run_id")
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data


class TestTrainEndpoint:
    """Test the /train endpoint validation."""

    @pytest.fixture(autouse=True)
    def cleanup_active_runs(self):
        """Clean up any active runs before each test."""
        from vespaembed import db
        from vespaembed.enums import RunStatus

        # Mark any running/pending runs as stopped so they don't block tests
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE runs SET status = ? WHERE status IN (?, ?)",
            (RunStatus.STOPPED.value, RunStatus.PENDING.value, RunStatus.RUNNING.value),
        )
        conn.commit()
        conn.close()
        yield

    def test_train_requires_data_source(self, client):
        """Test that training requires either file or HF dataset."""
        response = client.post(
            "/train",
            json={
                "project_name": "test-project",
                "task": "pairs",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        )

        assert response.status_code == 400
        assert "Must provide either" in response.json()["detail"]

    def test_train_rejects_both_data_sources(self, client):
        """Test that training rejects both file and HF dataset."""
        response = client.post(
            "/train",
            json={
                "project_name": "test-project",
                "task": "pairs",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "train_filename": "/some/file.csv",
                "hf_dataset": "some/dataset",
            },
        )

        assert response.status_code == 400
        assert "Cannot specify both" in response.json()["detail"]

    def test_train_validates_project_name(self, client):
        """Test that project name is validated."""
        response = client.post(
            "/train",
            json={
                "project_name": "invalid name with spaces",
                "task": "pairs",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "hf_dataset": "some/dataset",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_train_validates_hub_config(self, client):
        """Test that hf_username is required when push_to_hub is true."""
        response = client.post(
            "/train",
            json={
                "project_name": "test-project",
                "task": "pairs",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "hf_dataset": "sentence-transformers/all-nli",
                "hf_subset": "triplet",
                "push_to_hub": True,
            },
        )

        assert response.status_code == 400
        assert "hf_username is required" in response.json()["detail"]

    def test_train_rejects_nonexistent_file(self, client):
        """Test that training rejects non-existent file."""
        response = client.post(
            "/train",
            json={
                "project_name": "test-project",
                "task": "pairs",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "train_filename": "/nonexistent/file.csv",
            },
        )

        assert response.status_code == 400
        assert "not found" in response.json()["detail"]


class TestStopEndpoint:
    """Test the /stop endpoint."""

    def test_stop_nonexistent_run(self, client):
        """Test stopping a run that doesn't exist."""
        response = client.post("/stop", json={"run_id": 99999})
        assert response.status_code == 404
