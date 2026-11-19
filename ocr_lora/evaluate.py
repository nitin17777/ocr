"""Resumable evaluation loop for OCR LoRA model.

Features:
- Safe checkpoint handling: Never overwrites original data; copies to outputs/ first.
- Resumability: Inspects existing output checkpoint, records completed indices, and resumes seamlessly.
- Batching: Evaluates in batches of EVAL_BATCH_SIZE (default 8) with greedy decoding.
- Persistence: Flushes and executes os.fsync after every batch.
- Progress: Displays informative tqdm progress without terminal flood.
- Limit support: --limit N for fast verification/smoke testing.
"""

from typing import List, Dict, Set, Optional
import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

from ocr_lora.config import (
    CHECKPOINT_INPUT_PATH,
    CHECKPOINT_OUTPUT_PATH,
    EVAL_BATCH_SIZE,
    EVAL_MAX_LENGTH,
    EVAL_MAX_NEW_TOKENS,
    DEFAULT_ADAPTER_ID,
    BASE_MODEL_ID,
)
from ocr_lora.model import load_model_and_adapter
from ocr_lora.infer import correct_text_batch


def read_completed_indices(checkpoint_path: Path) -> Set[int]:
    """Read set of completed record indices from a JSONL file."""
    completed = set()
    if not checkpoint_path.exists():
        return completed

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "index" in record:
                    completed.add(int(record["index"]))
            except Exception:
                continue

    return completed


def prepare_output_checkpoint(
    source_path: Path = CHECKPOINT_INPUT_PATH,
    target_path: Path = CHECKPOINT_OUTPUT_PATH,
) -> None:
    """Ensure output checkpoint exists without modifying the original."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists() and source_path.exists():
        shutil.copy2(source_path, target_path)


def evaluate(
    dataset_rows: List[Dict[str, str]],
    output_checkpoint_path: Path = CHECKPOINT_OUTPUT_PATH,
    source_checkpoint_path: Optional[Path] = CHECKPOINT_INPUT_PATH,
    adapter_id: str = DEFAULT_ADAPTER_ID,
    base_model_id: str = BASE_MODEL_ID,
    batch_size: int = EVAL_BATCH_SIZE,
    limit: Optional[int] = None,
    print_samples_count: int = 0,
) -> Dict[str, float]:
    """Run resumable evaluation loop over dataset_rows."""
    if source_checkpoint_path and source_checkpoint_path.exists():
        prepare_output_checkpoint(source_checkpoint_path, output_checkpoint_path)
    else:
        output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    completed_indices = read_completed_indices(output_checkpoint_path)

    # Filter rows needing evaluation
    to_evaluate = []
    for idx, row in enumerate(dataset_rows):
        if idx not in completed_indices:
            to_evaluate.append((idx, row["input"], row["expected"]))

    if limit is not None and limit > 0:
        to_evaluate = to_evaluate[:limit]

    total_remaining = len(to_evaluate)
    print(f"Total dataset rows: {len(dataset_rows)}")
    print(f"Previously completed: {len(completed_indices)}")
    print(f"Remaining to evaluate in this run: {total_remaining}")

    if total_remaining == 0:
        print("All records already evaluated in checkpoint.")
        return {"evaluated_in_run": 0}

    # Load model and adapter
    print(f"Loading base model '{base_model_id}' and adapter '{adapter_id}'...")
    model, tokenizer = load_model_and_adapter(
        base_model_id=base_model_id,
        adapter_id=adapter_id,
    )

    sample_outputs = []

    # Open output checkpoint file in append mode
    with open(output_checkpoint_path, "a", encoding="utf-8") as out_f:
        pbar = tqdm(
            range(0, total_remaining, batch_size),
            desc="Evaluating",
            unit="batch",
        )

        for batch_start in pbar:
            batch_items = to_evaluate[batch_start : batch_start + batch_size]
            indices = [item[0] for item in batch_items]
            inputs = [item[1] for item in batch_items]
            expected = [item[2] for item in batch_items]

            predictions = correct_text_batch(
                model=model,
                tokenizer=tokenizer,
                input_texts=inputs,
                max_length=EVAL_MAX_LENGTH,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
            )

            # Write records to jsonl
            for idx, inp, exp, pred in zip(indices, inputs, expected, predictions):
                record = {
                    "index": idx,
                    "input": inp,
                    "expected": exp,
                    "prediction": pred,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if len(sample_outputs) < print_samples_count:
                    sample_outputs.append((inp, exp, pred))

            # Flush and fsync to guarantee persistence on disk
            out_f.flush()
            os.fsync(out_f.fileno())

    if sample_outputs:
        print("\n--- Sample Predictions ---")
        for i, (inp, exp, pred) in enumerate(sample_outputs, 1):
            match = "✓ MATCH" if pred.strip() == exp.strip() else "✗ MISMATCH"
            print(f"[{i}] [{match}]")
            print(f"  Input:    {inp}")
            print(f"  Expected: {exp}")
            print(f"  Pred:     {pred}")

    return {"evaluated_in_run": total_remaining}


def main():
    parser = argparse.ArgumentParser(description="OCR LoRA Evaluation Harness")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to evaluate")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample predictions to print")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER_ID, help="Adapter repository ID")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID, help="Base model repository ID")
    parser.add_argument("--smoke-test", action="store_true", help="Run 50-row smoke test with live inference")
    args = parser.parse_args()

    # Ensure output checkpoint is prepared from source
    if CHECKPOINT_INPUT_PATH.exists() and not CHECKPOINT_OUTPUT_PATH.exists():
        prepare_output_checkpoint(CHECKPOINT_INPUT_PATH, CHECKPOINT_OUTPUT_PATH)
        print(f"Copied {CHECKPOINT_INPUT_PATH} to {CHECKPOINT_OUTPUT_PATH}")

    # Determine input dataset
    if CHECKPOINT_INPUT_PATH.exists():
        rows = []
        with open(CHECKPOINT_INPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rows.append({"input": data["input"], "expected": data["expected"]})
                except Exception:
                    continue
    else:
        from ocr_lora.data import build_rows, load_wikitext_sentences, generate_math_sentences
        print("Checkpoint input not found; generating synthetic rows...")
        wikitext = load_wikitext_sentences()
        math = generate_math_sentences()
        rows = build_rows(wikitext + math)

    if args.smoke_test:
        limit = args.limit or 50
        print(f"\n=== Running Smoke Test on first {limit} rows ===")
        test_rows = rows[:limit]
        model, tokenizer = load_model_and_adapter(
            base_model_id=args.base_model,
            adapter_id=args.adapter,
        )
        inputs = [r["input"] for r in test_rows]
        expected = [r["expected"] for r in test_rows]
        
        preds = []
        for i in range(0, len(inputs), args.batch_size):
            b_in = inputs[i : i + args.batch_size]
            b_pred = correct_text_batch(model, tokenizer, b_in)
            preds.extend(b_pred)

        matches = sum(1 for p, e in zip(preds, expected) if p.strip() == e.strip())
        print(f"Smoke Test Accuracy: {matches}/{len(test_rows)} ({matches/len(test_rows)*100:.2f}%)")
        print("\n--- 5 Sample Predictions ---")
        for i in range(min(args.samples, len(test_rows))):
            m = "✓ MATCH" if preds[i].strip() == expected[i].strip() else "✗ MISMATCH"
            print(f"[{i+1}] [{m}]")
            print(f"  Input:    {inputs[i]}")
            print(f"  Expected: {expected[i]}")
            print(f"  Pred:     {preds[i]}")
        return

    evaluate(
        dataset_rows=rows,
        output_checkpoint_path=CHECKPOINT_OUTPUT_PATH,
        source_checkpoint_path=CHECKPOINT_INPUT_PATH,
        adapter_id=args.adapter,
        base_model_id=args.base_model,
        batch_size=args.batch_size,
        limit=args.limit,
        print_samples_count=args.samples,
    )


if __name__ == "__main__":
    main()
