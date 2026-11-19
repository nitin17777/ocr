"""Report generation and failure analysis module for OCR LoRA evaluation.

Generates outputs/report.md containing:
- Executive summary with total rows and exact-match accuracy
- Stratified breakdown: Math vs Text
- Noise-level accuracy estimates
- Over-correction rate (failures when input == expected)
- Top 20 failure cases (input, expected, prediction)
- Common failure patterns (text swaps & math symbol swaps)
- Character-level confusion matrix
"""

from typing import List, Dict, Any, Tuple
import json
from collections import Counter, defaultdict
from pathlib import Path
import re

from ocr_lora.config import (
    CHECKPOINT_OUTPUT_PATH,
    CHECKPOINT_INPUT_PATH,
    REPORT_OUTPUT_PATH,
    TEXT_SWAPS,
    MATH_SWAPS,
)
from scripts.verify_checkpoint import is_math_input


def load_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from path."""
    records = []
    if not jsonl_path.exists():
        return records
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def estimate_corruption_level(inp: str, exp: str) -> str:
    """Heuristically categorize corruption intensity into Low, Medium, High."""
    diff_count = sum(1 for a, b in zip(inp, exp) if a != b) + abs(len(inp) - len(exp))
    ratio = diff_count / max(len(exp), 1)
    if ratio <= 0.15:
        return "Low (~0.10)"
    elif ratio <= 0.35:
        return "Medium (~0.25)"
    else:
        return "High (~0.45)"


def generate_report(
    checkpoint_path: Path = CHECKPOINT_OUTPUT_PATH,
    output_report_path: Path = REPORT_OUTPUT_PATH,
) -> str:
    """Analyze evaluation checkpoint and produce detailed markdown report."""
    if not checkpoint_path.exists() and CHECKPOINT_INPUT_PATH.exists():
        checkpoint_path = CHECKPOINT_INPUT_PATH

    records = load_records(checkpoint_path)
    if not records:
        raise FileNotFoundError(f"No records found at {checkpoint_path}")

    total_rows = len(records)
    exact_matches = 0

    # Stratified metrics
    math_total = 0
    math_correct = 0
    text_total = 0
    text_correct = 0

    level_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    # Over-correction tracking (input == expected, but prediction != expected)
    unaltered_inputs_total = 0
    over_corrections = 0

    failures = []
    confusion_pairs = Counter()

    target_pairs = list(TEXT_SWAPS.items()) + list(MATH_SWAPS.items())

    for r in records:
        inp = r.get("input", "")
        exp = r.get("expected", "")
        pred = r.get("prediction", "")

        is_correct = (pred.strip() == exp.strip())
        if is_correct:
            exact_matches += 1

        is_math = is_math_input(inp) or is_math_input(exp)
        if is_math:
            math_total += 1
            if is_correct:
                math_correct += 1
        else:
            text_total += 1
            if is_correct:
                text_correct += 1

        # Check noise level
        level_cat = estimate_corruption_level(inp, exp)
        level_stats[level_cat]["total"] += 1
        if is_correct:
            level_stats[level_cat]["correct"] += 1

        # Over-correction check
        if inp.strip() == exp.strip():
            unaltered_inputs_total += 1
            if not is_correct:
                over_corrections += 1

        # Failure analysis
        if not is_correct:
            failures.append({
                "index": r.get("index"),
                "input": inp,
                "expected": exp,
                "prediction": pred,
            })

            # Check for known symbol/char discrepancies between pred and exp
            for src, dst in target_pairs:
                if src in exp and dst in pred:
                    confusion_pairs[(src, dst)] += 1
                if dst in exp and src in pred:
                    confusion_pairs[(dst, src)] += 1

    overall_acc = (exact_matches / total_rows * 100) if total_rows else 0.0
    math_acc = (math_correct / math_total * 100) if math_total else 0.0
    text_acc = (text_correct / text_total * 100) if text_total else 0.0
    over_corr_rate = (over_corrections / unaltered_inputs_total * 100) if unaltered_inputs_total else 0.0

    # Build markdown report
    md = []
    md.append("# OCR LoRA Correction — Evaluation Report\n")
    md.append("## Executive Summary\n")
    md.append(f"- **Total Rows Evaluated**: {total_rows:,}")
    md.append(f"- **Exact Match Count**: {exact_matches:,}")
    md.append(f"- **Overall Exact Match Accuracy**: **{overall_acc:.2f}%**\n")

    md.append("## Accuracy Breakdown\n")
    md.append("### By Input Domain")
    md.append("| Domain | Total Samples | Correct | Accuracy |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Math Sentences** | {math_total:,} | {math_correct:,} | **{math_acc:.2f}%** |")
    md.append(f"| **Text Sentences (WikiText-2)** | {text_total:,} | {text_correct:,} | **{text_acc:.2f}%** |\n")

    md.append("### By Estimated Noise Level")
    md.append("| Noise Level | Total Samples | Correct | Accuracy |")
    md.append("| :--- | :--- | :--- | :--- |")
    for lvl in ["Low (~0.10)", "Medium (~0.25)", "High (~0.45)"]:
        t = level_stats[lvl]["total"]
        c = level_stats[lvl]["correct"]
        acc = (c / t * 100) if t else 0.0
        md.append(f"| {lvl} | {t:,} | {c:,} | {acc:.2f}% |")
    md.append("")

    md.append("### Over-Correction Analysis")
    md.append(f"- **Inputs already clean (`input == expected`)**: {unaltered_inputs_total:,}")
    md.append(f"- **Incorrectly modified by model**: {over_corrections:,}")
    md.append(f"- **Over-correction rate**: **{over_corr_rate:.2f}%**\n")

    md.append("## Character & Symbol Confusion Matrix (Most Frequent Errors)\n")
    md.append("| Expected Character | Model Predicted | Occurrences in Failures |")
    md.append("| :---: | :---: | :---: |")
    for (src, dst), count in confusion_pairs.most_common(12):
        md.append(f"| `{src}` | `{dst}` | {count:,} |")
    md.append("")

    md.append("## Top 20 Failure Cases\n")
    for i, fail in enumerate(failures[:20], 1):
        md.append(f"### Case {i} (Index {fail['index']})")
        md.append(f"- **Input**:    `{fail['input']}`")
        md.append(f"- **Expected**: `{fail['expected']}`")
        md.append(f"- **Model**:    `{fail['prediction']}`\n")

    content = "\n".join(md)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


if __name__ == "__main__":
    generate_report()
    print(f"Report written to {REPORT_OUTPUT_PATH}")
