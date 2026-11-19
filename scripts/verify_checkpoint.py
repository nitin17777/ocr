"""Verify integrity and accuracy of the OCR evaluation checkpoint.

Tolerates malformed lines; reports:
- Total valid lines
- Unique indices, duplicates, and gaps
- Last index
- Overall exact-match accuracy
- Sub-accuracy by input type (math vs text)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_lora.config import CHECKPOINT_INPUT_PATH, CHECKPOINT_OUTPUT_PATH, MATH_TEMPLATES


def is_math_input(text: str) -> bool:
    """Classify whether a sentence is an arithmetic problem or prose text."""
    math_indicators = ["+", "−", "×", "÷", "=", "sum", "add", "multiply", "subtract", "divide", "solve:"]
    lower = text.lower()
    return any(ind in lower for ind in math_indicators)


def verify_checkpoint(file_path: Path) -> Dict[str, Any]:
    """Parse and analyze checkpoint file."""
    if not file_path.exists():
        print(f"Error: Checkpoint file not found at: {file_path}")
        return {"error": "file_not_found"}

    lines_count = 0
    bad_lines_count = 0
    indices: List[int] = []
    duplicates = set()
    seen_indices = set()

    exact_matches = 0
    math_total = 0
    math_matches = 0
    text_total = 0
    text_matches = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            lines_count += 1
            try:
                record = json.loads(line)
            except Exception as e:
                bad_lines_count += 1
                continue

            idx = record.get("index")
            inp = record.get("input", "")
            exp = record.get("expected", "")
            pred = record.get("prediction", "")

            if idx is not None:
                idx_int = int(idx)
                if idx_int in seen_indices:
                    duplicates.add(idx_int)
                seen_indices.add(idx_int)
                indices.append(idx_int)

            is_match = (pred.strip() == exp.strip())
            if is_match:
                exact_matches += 1

            if is_math_input(inp) or is_math_input(exp):
                math_total += 1
                if is_match:
                    math_matches += 1
            else:
                text_total += 1
                if is_match:
                    text_matches += 1

    valid_records = len(indices)
    unique_indices = len(seen_indices)
    min_idx = min(indices) if indices else None
    max_idx = max(indices) if indices else None

    # Check for gaps in indices
    gaps = []
    if unique_indices > 0 and min_idx is not None and max_idx is not None:
        expected_range = set(range(min_idx, max_idx + 1))
        gaps = sorted(list(expected_range - seen_indices))

    overall_acc = (exact_matches / valid_records * 100) if valid_records else 0.0
    math_acc = (math_matches / math_total * 100) if math_total else 0.0
    text_acc = (text_matches / text_total * 100) if text_total else 0.0

    report = {
        "file_path": str(file_path),
        "total_lines": lines_count,
        "bad_lines": bad_lines_count,
        "valid_records": valid_records,
        "unique_indices": unique_indices,
        "duplicates_count": len(duplicates),
        "min_index": min_idx,
        "max_index": max_idx,
        "gaps_count": len(gaps),
        "gaps_sample": gaps[:10],
        "exact_matches": exact_matches,
        "overall_accuracy_pct": round(overall_acc, 2),
        "math_total": math_total,
        "math_matches": math_matches,
        "math_accuracy_pct": round(math_acc, 2),
        "text_total": text_total,
        "text_matches": text_matches,
        "text_accuracy_pct": round(text_acc, 2),
    }

    return report


def main():
    target = CHECKPOINT_INPUT_PATH
    if not target.exists() and CHECKPOINT_OUTPUT_PATH.exists():
        target = CHECKPOINT_OUTPUT_PATH

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])

    print(f"=== Checkpoint Verification ===")
    print(f"Target file: {target}")

    stats = verify_checkpoint(target)
    if "error" in stats:
        sys.exit(1)

    print(f"Total lines:          {stats['total_lines']}")
    print(f"Bad / unparsed lines: {stats['bad_lines']}")
    print(f"Valid records:        {stats['valid_records']}")
    print(f"Unique indices:       {stats['unique_indices']}")
    print(f"Index range:          [{stats['min_index']} .. {stats['max_index']}]")
    print(f"Duplicate indices:    {stats['duplicates_count']}")
    print(f"Index gaps:           {stats['gaps_count']}")
    if stats['gaps_sample']:
        print(f"  First gaps:         {stats['gaps_sample']}")

    print("\n--- Accuracy Breakdown ---")
    print(f"Overall Exact Match:  {stats['exact_matches']}/{stats['valid_records']} ({stats['overall_accuracy_pct']}%)")
    print(f"Math Sentences:       {stats['math_matches']}/{stats['math_total']} ({stats['math_accuracy_pct']}%)")
    print(f"Text Sentences:       {stats['text_matches']}/{stats['text_total']} ({stats['text_accuracy_pct']}%)")

    # Check expectation: 14,549 contiguous, ~90.70%
    if stats['valid_records'] == 14549 and stats['gaps_count'] == 0:
        print("\n✓ Checkpoint structure perfectly matches confirmed benchmark (14,549 contiguous rows).")
    else:
        print(f"\nNote: Valid records count is {stats['valid_records']} (expected 14,549 contiguous).")


if __name__ == "__main__":
    main()
