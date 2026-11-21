"""Tests for training runner, process management, parameter validation, and 503 inference guard."""

import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from webapp.main import app
from webapp.services.training_runner import training_runner
from webapp.services.model_manager import model_manager
from webapp.schemas import TrainRequest


@pytest.fixture(autouse=True)
def reset_training_state():
    """Ensure training runner state is clean before and after each test."""
    model_manager.is_training = False
    training_runner._proc = None
    training_runner._current_run = None
    training_runner._log_buffer.clear()
    training_runner._cancelled = False
    yield
    model_manager.is_training = False
    training_runner._proc = None
    training_runner._current_run = None


@pytest.mark.anyio
async def test_training_parameter_validation():
    """Verify validation bounds on epochs, batch size, learning rate, and regex."""
    from pydantic import ValidationError

    # 1. Invalid epochs (< 1 or > 10)
    with pytest.raises(ValidationError):
        TrainRequest(output_name="valid_name", epochs=0)
    with pytest.raises(ValidationError):
        TrainRequest(output_name="valid_name", epochs=11)

    # 2. Invalid batch size (< 1 or > 4)
    with pytest.raises(ValidationError):
        TrainRequest(output_name="valid_name", batch_size=0)
    with pytest.raises(ValidationError):
        TrainRequest(output_name="valid_name", batch_size=5)

    # 3. Invalid learning rate
    with pytest.raises(ValidationError):
        TrainRequest(output_name="valid_name", learning_rate=1e-7)
    with pytest.raises(ValidationError):
        TrainRequest(output_name="valid_name", learning_rate=2e-3)

    # 4. Invalid output name regex
    with pytest.raises(ValidationError):
        TrainRequest(output_name="invalid name with spaces!")


@pytest.mark.anyio
async def test_training_output_collision(tmp_path: Path, monkeypatch):
    """Verify that targeting an existing non-empty directory is refused."""
    from webapp.services import training_runner as tr_module
    fake_models_dir = tmp_path / "models"
    existing_adapter = fake_models_dir / "already_exists"
    existing_adapter.mkdir(parents=True)
    (existing_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(tr_module.settings, "MODELS_DIR", fake_models_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        await training_runner.preflight_check(TrainRequest(output_name="already_exists"))


@pytest.mark.anyio
async def test_training_singleton_enforcement():
    """Verify that launching a second run while one is running raises training_in_progress."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Process is running
    training_runner._proc = mock_proc

    with pytest.raises(RuntimeError, match="training_in_progress"):
        await training_runner.preflight_check(TrainRequest(output_name="second_run"))


@pytest.mark.anyio
async def test_inference_and_batch_return_503_during_training():
    """Verify that while model_manager.is_training is True, inference returns 503."""
    client = TestClient(app)

    # When training is active
    model_manager.is_training = True

    # 1. /api/correct returns 503
    resp_correct = client.post("/api/correct", json={"text": "Te5t sample"})
    assert resp_correct.status_code == 503
    assert resp_correct.json()["detail"]["reason"] == "training_in_progress"

    # 2. /api/batch returns 503
    resp_batch = client.post(
        "/api/batch",
        data={"adapter": "default"},
        files={"file": ("test.txt", b"sample\n", "text/plain")},
    )
    assert resp_batch.status_code == 503
    assert resp_batch.json()["detail"]["reason"] == "training_in_progress"

    # Re-enable
    model_manager.is_training = False
    resp_correct_after = client.post("/api/correct", json={"text": "Te5t sample"})
    # Not 503 anymore (returns either 200 or whatever normal status)
    assert resp_correct_after.status_code != 503


@pytest.mark.anyio
async def test_training_cancel(monkeypatch):
    """Verify process cancellation properly invokes taskkill and marks cancelled."""
    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_proc.poll.side_effect = [None, 0]  # First poll running, next poll exited
    training_runner._proc = mock_proc
    training_runner._current_run = {
        "run_id": "test_run_123",
        "status": "running",
    }
    model_manager.is_training = True

    with patch("subprocess.run") as mock_sub_run:
        res = await training_runner.cancel_training()
        assert res["status"] == "cancelled"
        assert res["pid"] == 99999
        assert not model_manager.is_training
        assert training_runner._current_run["status"] == "cancelled"


def test_training_api_endpoints():
    """Test HTTP contracts of /api/train endpoints."""
    client = TestClient(app)

    # 1. GET /api/train/status
    res_status = client.get("/api/train/status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert "status" in data
    assert "run_id" in data

    # 2. GET /api/train/runs
    res_runs = client.get("/api/train/runs")
    assert res_runs.status_code == 200
    assert isinstance(res_runs.json(), list)

    # 3. POST /api/train/cancel when not running
    res_cancel = client.post("/api/train/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] == "not_running"
