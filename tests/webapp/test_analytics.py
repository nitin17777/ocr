"""Unit and integration tests for analytics service and API endpoints."""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from webapp.main import app
from webapp.services.analytics import AnalyticsService


@pytest.fixture
def sample_checkpoint(tmp_path: Path) -> Path:
    """Create a temporary JSONL checkpoint file with known ground truth and failures."""
    file_path = tmp_path / "test_eval_checkpoint.jsonl"
    rows = [
        # Math correct (0)
        {"index": 0, "input": "2 + 2 = 4.", "expected": "2 + 2 = 4.", "prediction": "2 + 2 = 4."},
        # Math fail (1): expected '2', predicted 'S'
        {"index": 1, "input": "3 + S = 5.", "expected": "3 + 2 = 5.", "prediction": "3 + S = 5."},
        # Math correct (2)
        {"index": 2, "input": "10 × 2 = 20.", "expected": "10 × 2 = 20.", "prediction": "10 × 2 = 20."},
        # Text correct (3)
        {"index": 3, "input": "He1lo world.", "expected": "Hello world.", "prediction": "Hello world."},
        # Text fail (4): expected 'l', predicted '1'
        {"index": 4, "input": "Fu1l moon.", "expected": "Full moon.", "prediction": "Fu1l moon."},
        # Text fail (5): expected 'O', predicted '0'
        {"index": 5, "input": "0ctober sky.", "expected": "October sky.", "prediction": "0ctober sky."},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return file_path


def test_analytics_service_metrics(sample_checkpoint: Path):
    """Verify exact count, domain breakdown, and accuracy calculations on fixture."""
    svc = AnalyticsService()
    summary = svc.get_summary(path=sample_checkpoint)

    assert summary["total_rows"] == 6
    assert summary["exact_matches"] == 3
    assert summary["exact_match_acc"] == 50.0

    # Math: rows 0, 1, 2 (2 matches out of 3 = 66.67%)
    assert summary["math_total"] == 3
    assert summary["math_correct"] == 2
    assert summary["math_acc"] == 66.67

    # Text: rows 3, 4, 5 (1 match out of 3 = 33.33%)
    assert summary["text_total"] == 3
    assert summary["text_correct"] == 1
    assert summary["text_acc"] == 33.33

    assert summary["failures_count"] == 3


def test_analytics_service_caching_and_invalidation(sample_checkpoint: Path):
    """Test that mtime/size changes invalidate the in-memory cache."""
    svc = AnalyticsService()
    s1 = svc.get_summary(path=sample_checkpoint)
    assert s1["total_rows"] == 6

    # Append another record to file
    new_row = {"index": 6, "input": "7 ÷ 1 = 7.", "expected": "7 ÷ 1 = 7.", "prediction": "7 ÷ 1 = 7."}
    with open(sample_checkpoint, "a", encoding="utf-8") as f:
        f.write(json.dumps(new_row) + "\n")

    s2 = svc.get_summary(path=sample_checkpoint)
    assert s2["total_rows"] == 7
    assert s2["exact_matches"] == 4


def test_analytics_failures_search_and_diff(sample_checkpoint: Path):
    """Test keyword search, domain filtering, pagination, and diff generation."""
    svc = AnalyticsService()

    # Domain filter = math
    math_fails = svc.get_failures(domain="math", path=sample_checkpoint)
    assert math_fails["total"] == 1
    assert math_fails["failures"][0]["index"] == 1
    assert math_fails["failures"][0]["domain"] == "math"

    # Domain filter = text
    text_fails = svc.get_failures(domain="text", path=sample_checkpoint)
    assert text_fails["total"] == 2

    # Keyword search across input / target / prediction
    search_res = svc.get_failures(q="moon", path=sample_checkpoint)
    assert search_res["total"] == 1
    assert search_res["failures"][0]["index"] == 4
    # Verify diff ops exist
    diff = search_res["failures"][0]["diff"]
    assert len(diff) > 0
    # Expected: "Full moon." vs "Fu1l moon."
    assert any(op.tag == "replace" for op in diff)


def test_analytics_confusion_matrix(sample_checkpoint: Path):
    """Test NxN confusion matrix generation and top pairs."""
    svc = AnalyticsService()
    confusion = svc.get_confusion(top=10, path=sample_checkpoint)

    assert "characters" in confusion
    assert "grid" in confusion
    assert "max_count" in confusion
    assert "top_pairs" in confusion

    # Characters list should have corresponding grid dimensions
    n_chars = len(confusion["characters"])
    assert len(confusion["grid"]) == n_chars
    for row in confusion["grid"]:
        assert len(row) == n_chars


def test_api_analytics_endpoints(monkeypatch, sample_checkpoint: Path):
    """Verify HTTP API contracts for /api/analytics endpoints."""
    from webapp.services import analytics
    monkeypatch.setattr(analytics.settings, "CHECKPOINT_PATH", sample_checkpoint)

    client = TestClient(app)

    # 1. Summary
    resp = client.get("/api/analytics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_rows"] == 6
    assert data["exact_match_acc"] == 50.0

    # 2. Confusion
    resp = client.get("/api/analytics/confusion?top=8&domain=text")
    assert resp.status_code == 200
    c_data = resp.json()
    assert len(c_data["characters"]) <= 8

    # 3. Failures
    resp = client.get("/api/analytics/failures?page=1&page_size=2")
    assert resp.status_code == 200
    f_data = resp.json()
    assert f_data["total"] == 3
    assert len(f_data["failures"]) == 2
    assert f_data["total_pages"] == 2


def test_real_checkpoint_acceptance():
    """Verify that real checkpoint totals match outputs/report.md (14,549 rows, 90.70%)."""
    real_path = Path("data/ocr_eval_checkpoint.jsonl")
    if not real_path.exists():
        pytest.skip("data/ocr_eval_checkpoint.jsonl not found")

    svc = AnalyticsService()
    summary = svc.get_summary(path=real_path)

    assert summary["total_rows"] == 14549
    assert summary["exact_matches"] == 13196
    assert summary["exact_match_acc"] == 90.70
    assert summary["math_total"] == 12158
    assert summary["math_correct"] == 11965
    assert summary["math_acc"] == 98.41
    assert summary["text_total"] == 2391
    assert summary["text_correct"] == 1231
    assert summary["text_acc"] == 51.48
    assert summary["failures_count"] == 1353
