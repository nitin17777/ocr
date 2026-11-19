"""Unit tests for resumable checkpoint tracking."""

import json
from pathlib import Path
import pytest
from ocr_lora.evaluate import read_completed_indices


def test_read_completed_indices(tmp_path: Path):
    checkpoint_file = tmp_path / "test_eval_checkpoint.jsonl"
    
    # Write sample records with gaps and duplicates
    records = [
        {"index": 0, "input": "inp0", "expected": "exp0", "prediction": "exp0"},
        {"index": 1, "input": "inp1", "expected": "exp1", "prediction": "bad1"},
        {"index": 3, "input": "inp3", "expected": "exp3", "prediction": "exp3"},
        {"index": 1, "input": "dup1", "expected": "dup1", "prediction": "dup1"},  # duplicate
        "malformed line",  # invalid JSON
        {"index": 5, "input": "inp5", "expected": "exp5", "prediction": "exp5"},
    ]

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        for r in records:
            if isinstance(r, dict):
                f.write(json.dumps(r) + "\n")
            else:
                f.write(str(r) + "\n")

    completed = read_completed_indices(checkpoint_file)
    assert completed == {0, 1, 3, 5}
    assert 2 not in completed
    assert 4 not in completed
