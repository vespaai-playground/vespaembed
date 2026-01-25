"""Shared test fixtures for vespaembed tests."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from vespaembed.web.app import app
    return TestClient(app)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    csv_path = temp_dir / "train.csv"
    csv_path.write_text(
        "anchor,positive\n"
        "What is Python?,Python is a programming language\n"
        "How does ML work?,Machine learning learns from data\n"
    )
    return csv_path


@pytest.fixture
def sample_jsonl_file(temp_dir):
    """Create a sample JSONL file for testing."""
    jsonl_path = temp_dir / "train.jsonl"
    jsonl_path.write_text(
        '{"anchor": "What is Python?", "positive": "Python is a programming language"}\n'
        '{"anchor": "How does ML work?", "positive": "Machine learning learns from data"}\n'
    )
    return jsonl_path


@pytest.fixture
def sample_triplet_csv(temp_dir):
    """Create a sample triplet CSV file for testing."""
    csv_path = temp_dir / "triplet.csv"
    csv_path.write_text(
        "anchor,positive,negative\n"
        "What is Python?,Python is a programming language,A python is a snake\n"
        "Apple stock,AAPL trading info,Apples are fruits\n"
    )
    return csv_path


@pytest.fixture
def sample_sts_csv(temp_dir):
    """Create a sample STS CSV file for testing."""
    csv_path = temp_dir / "sts.csv"
    csv_path.write_text(
        "sentence1,sentence2,score\n"
        "A man plays guitar,A person plays music,0.85\n"
        "A dog runs,A cat sleeps,0.1\n"
    )
    return csv_path


@pytest.fixture
def sample_nli_csv(temp_dir):
    """Create a sample NLI CSV file for testing."""
    csv_path = temp_dir / "nli.csv"
    csv_path.write_text(
        "sentence1,sentence2,label\n"
        "A man is eating pizza,A man is eating food,0\n"
        "A woman is playing guitar,A man is playing piano,2\n"
    )
    return csv_path


@pytest.fixture
def sample_tsdae_csv(temp_dir):
    """Create a sample TSDAE CSV file for testing."""
    csv_path = temp_dir / "tsdae.csv"
    csv_path.write_text(
        "text\n"
        "Machine learning is transforming how we analyze data.\n"
        "Natural language processing enables computers to understand human language.\n"
    )
    return csv_path
