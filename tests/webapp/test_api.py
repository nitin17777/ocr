"""Integration and contract tests for FastAPI backend with mocked model inference."""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from webapp.main import app
from webapp.services.model_manager import model_manager
from webapp.schemas import CorrectResponse, DiffOp


@pytest.fixture(autouse=True)
def reset_model_manager():
    """Ensure clean model manager state before each test."""
    model_manager.state = "unloaded"
    model_manager.is_training = False
    model_manager.active_adapter = None
    yield
    model_manager.state = "unloaded"
    model_manager.is_training = False


@pytest.mark.anyio
async def test_health_starts_unloaded():
    """Acceptance criterion: /api/health shows 'unloaded' without loading model."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_state"] == "unloaded"
        assert data["active_adapter"] is None
        assert "gpu" in data


@pytest.mark.anyio
async def test_gpu_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/gpu")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        assert "free_mb" in data
        assert "used_mb" in data


@pytest.mark.anyio
async def test_adapters_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/adapters")
        assert response.status_code == 200
        adapters = response.json()
        assert isinstance(adapters, list)
        names = [a["name"] for a in adapters]
        assert any("ocr-lora-large" in n for n in names)


@pytest.mark.anyio
async def test_correct_endpoint_contract():
    """Verify /api/correct contract with mocked model inference."""
    mock_response = CorrectResponse(
        corrected="Multiply 765 by 17 to get 13005.",
        diff=[
            DiffOp(tag="equal", a_start=0, a_end=2, b_start=0, b_end=2, a_text="Mu", b_text="Mu"),
            DiffOp(tag="replace", a_start=2, a_end=3, b_start=2, b_end=3, a_text="1", b_text="l"),
        ],
        adapter="deep1104/ocr-lora-large",
        latency_ms=145.2,
        tokens=12,
        tokens_per_sec=82.64,
    )

    with patch.object(model_manager, "correct", new=AsyncMock(return_value=mock_response)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "text": "Mu1tiply 76S by 17 to get l3005.",
                "adapter": "deep1104/ocr-lora-large",
            }
            res = await client.post("/api/correct", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["corrected"] == "Multiply 765 by 17 to get 13005."
            assert len(data["diff"]) > 0
            assert data["adapter"] == "deep1104/ocr-lora-large"
            assert data["latency_ms"] == 145.2
            assert data["tokens"] == 12
            assert data["tokens_per_sec"] == 82.64


@pytest.mark.anyio
async def test_correct_503_during_training():
    """Acceptance criterion: inference returns 503 {"reason":"training_in_progress"} while training runs."""
    model_manager.is_training = True
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/correct", json={"text": "sample input"})
        assert res.status_code == 503
        data = res.json()
        assert data["detail"] == {"reason": "training_in_progress"}


@pytest.mark.anyio
async def test_correct_rejects_oversized_input():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        oversized = "A" * 3000
        res = await client.post("/api/correct", json={"text": oversized})
        assert res.status_code == 400
        assert "exceeds maximum allowed length" in res.json()["detail"]
