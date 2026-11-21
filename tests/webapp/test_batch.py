"""Unit and integration tests for Batch file processing with mocked inference."""

import io
import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from webapp.config import settings
from webapp.main import app
from webapp.services.batch_jobs import batch_manager
from webapp.services.model_manager import model_manager


@pytest.fixture(autouse=True)
def clean_batch_environment(tmp_path):
    """Isolate job store to a temporary directory for each test."""
    original_jobs_dir = batch_manager.jobs_dir
    test_jobs_dir = tmp_path / "jobs"
    test_jobs_dir.mkdir(parents=True, exist_ok=True)
    batch_manager.jobs_dir = test_jobs_dir
    model_manager.state = "unloaded"
    model_manager.is_training = False
    yield
    batch_manager.jobs_dir = original_jobs_dir
    model_manager.state = "unloaded"
    model_manager.is_training = False


@pytest.mark.anyio
async def test_batch_parsing_txt():
    """Test .txt file upload, parsing, and execution with mocked model."""
    content = b"Line one: 12 x 4 = 48\nLine two: roorn 2O2\nLine three: modern art\n"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("sample.txt", io.BytesIO(content), "text/plain")}
        res = await client.post("/api/batch", files=files, data={"adapter": "deep1104/ocr-lora-large"})
        assert res.status_code == 200
        data = res.json()
        assert data["filename"] == "sample.txt"
        assert data["total_rows"] == 3
        assert data["status"] == "queued"

        # Verify job files on disk
        job_dir = batch_manager.get_job_dir(data["job_id"])
        assert (job_dir / "input.jsonl").exists()
        assert (job_dir / "meta.json").exists()


@pytest.mark.anyio
async def test_batch_parsing_csv_with_bom():
    """Test .csv file with UTF-8 BOM, auto-column detection, and configuration."""
    # Create CSV with UTF-8 BOM (\xef\xbb\xbf)
    csv_text = "id,text_col,other\n1,Mu1tiply 76S by 17,foo\n2,Room 2O2,bar\n"
    bom_content = b"\xef\xbb\xbf" + csv_text.encode("utf-8")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First upload without column name -> needs_column
        files = {"file": ("test.csv", io.BytesIO(bom_content), "text/csv")}
        res = await client.post("/api/batch", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "needs_column"
        assert "text_col" in data["detected_columns"]
        assert "other" in data["detected_columns"]

        # Configure column
        job_id = data["job_id"]
        res_cfg = await client.post(f"/api/batch/{job_id}/configure", data={"column_name": "text_col"})
        assert res_cfg.status_code == 200
        data_cfg = res_cfg.json()
        assert data_cfg["status"] == "queued"
        assert data_cfg["total_rows"] == 2


@pytest.mark.anyio
async def test_batch_parsing_jsonl():
    """Test .jsonl upload with field selection."""
    lines = [
        json.dumps({"input": "Sentence l04", "meta": "A"}),
        json.dumps({"input": "Sentence 2O2", "meta": "B"}),
    ]
    content = "\n".join(lines).encode("utf-8")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("data.jsonl", io.BytesIO(content), "application/x-ndjson")}
        res = await client.post("/api/batch", files=files, data={"column_name": "input"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "queued"
        assert data["total_rows"] == 2


@pytest.mark.anyio
async def test_batch_resume_after_crash():
    """Verify that an interrupted job resumes without redoing finished rows."""
    raw_content = b"Row 0\nRow 1\nRow 2\nRow 3\nRow 4\n"
    meta = batch_manager.create_job("resume_test.txt", raw_content, adapter="default")
    job_id = meta["job_id"]
    job_dir = batch_manager.get_job_dir(job_id)

    # Simulate that Row 0 and Row 1 were already processed and fsynced before a crash
    results_path = job_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"index": 0, "original": "Row 0", "corrected": "Row 0 corrected"}) + "\n")
        f.write(json.dumps({"index": 1, "original": "Row 1", "corrected": "Row 1 corrected"}) + "\n")

    # Mock the sync infer batch method to observe how many rows are passed to the model
    evaluated_texts = []

    def mock_infer(texts, adapter):
        evaluated_texts.extend(texts)
        return [f"{t} corrected" for t in texts]

    with patch.object(batch_manager, "_sync_infer_batch", side_effect=mock_infer):
        await batch_manager._execute_job(job_id)

    # Should only evaluate remaining 3 rows (Row 2, Row 3, Row 4)
    assert len(evaluated_texts) == 3
    assert "Row 0" not in evaluated_texts
    assert "Row 1" not in evaluated_texts

    # Final results.jsonl must contain all 5 rows
    final_meta = batch_manager.load_meta(job_id)
    assert final_meta["status"] == "completed"
    assert final_meta["done_rows"] == 5

    # Check export preserves original index order
    exported_bytes, _, _ = batch_manager.export_results(job_id, fmt="csv")
    csv_str = exported_bytes.decode("utf-8-sig")
    lines = csv_str.strip().split("\r\n")
    # Header + 5 rows
    assert len(lines) == 6
    assert lines[1].startswith("0,")
    assert lines[2].startswith("1,")
    assert lines[5].startswith("4,")


@pytest.mark.anyio
async def test_batch_cancel():
    """Verify cancellation marks status as cancelled."""
    content = b"Row 1\nRow 2\n"
    meta = batch_manager.create_job("cancel.txt", content, adapter="default")
    job_id = meta["job_id"]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(f"/api/batch/{job_id}/cancel")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    updated = batch_manager.load_meta(job_id)
    assert updated["status"] == "cancelled"


@pytest.mark.anyio
async def test_batch_size_limits_413():
    """Verify oversized files trigger HTTP 413."""
    # Test file exceeding MAX_UPLOAD_MB
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch.object(settings, "MAX_UPLOAD_MB", 0.0001):  # ~100 bytes limit
            big_content = b"A" * 1024
            files = {"file": ("big.txt", io.BytesIO(big_content), "text/plain")}
            res = await client.post("/api/batch", files=files)
            assert res.status_code == 413
            assert "exceeds maximum allowed limit" in res.json()["detail"]


@pytest.mark.anyio
async def test_batch_training_in_progress_503():
    """Verify batch upload is rejected with 503 while training subprocess is active."""
    model_manager.is_training = True
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("sample.txt", io.BytesIO(b"Hello world\n"), "text/plain")}
        res = await client.post("/api/batch", files=files)
        assert res.status_code == 503
        assert res.json()["detail"] == {"reason": "training_in_progress"}
