"""Analytics service for OCR evaluation checkpoint analysis and failure indexing.

Stream-parses data/ocr_eval_checkpoint.jsonl with mtime/size caching.
Computes executive metrics, domain accuracy (math vs text), difflib character-level
confusion pairs, and indexes failures with search and pagination support.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import difflib
from collections import Counter, defaultdict

from webapp.config import settings
from webapp.services.diff import compute_char_diff


# Math indicators matching scripts/verify_checkpoint.py and ocr_lora/report.py
MATH_INDICATORS = ("+", "−", "×", "÷", "=", "sum", "add", "multiply", "subtract", "divide", "solve:")

# Canonical swaps from dataset generation (ocr_lora/config.py)
TEXT_SWAPS = {
    "l": "1",
    "1": "l",
    "O": "0",
    "0": "O",
    "S": "5",
    "5": "S",
    "rn": "m",
}

MATH_SWAPS = {
    "×": "x",
    "÷": "+",
    "−": "=",
    "+": "÷",
    "=": "−",
}

TARGET_PAIRS = list(TEXT_SWAPS.items()) + list(MATH_SWAPS.items())


def is_math_input(text: str) -> bool:
    """Classify whether a sentence is an arithmetic problem or prose text."""
    lower = text.lower()
    return any(ind in lower for ind in MATH_INDICATORS)


class AnalyticsService:
    """In-memory cached analytics and failure search service."""

    def __init__(self):
        self._cached_path: Optional[str] = None
        self._cached_mtime: float = -1.0
        self._cached_size: int = -1

        self._summary: Dict[str, Any] = {}
        self._failures: List[Dict[str, Any]] = []

        # Confusion counters by domain: "all", "math", "text"
        self._swap_counts: Dict[str, Counter] = {
            "all": Counter(),
            "math": Counter(),
            "text": Counter(),
        }
        self._difflib_counts: Dict[str, Counter] = {
            "all": Counter(),
            "math": Counter(),
            "text": Counter(),
        }

    def _should_invalidate(self, file_path: Path) -> bool:
        """Check if file has changed based on mtime and size."""
        if not file_path.exists():
            return True
        try:
            stat = file_path.stat()
            return (
                self._cached_path != str(file_path)
                or self._cached_mtime != stat.st_mtime
                or self._cached_size != stat.st_size
            )
        except OSError:
            return True

    def ensure_loaded(self, path: Optional[Path] = None) -> None:
        """Stream-parse checkpoint file if cache is missing or stale."""
        file_path = path or settings.CHECKPOINT_PATH
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {file_path}")

        if not self._should_invalidate(file_path):
            return

        stat = file_path.stat()
        mtime = stat.st_mtime
        size = stat.st_size

        total_rows = 0
        exact_matches = 0
        math_total = 0
        math_correct = 0
        text_total = 0
        text_correct = 0

        failures: List[Dict[str, Any]] = []
        swap_counts = {
            "all": Counter(),
            "math": Counter(),
            "text": Counter(),
        }
        difflib_counts = {
            "all": Counter(),
            "math": Counter(),
            "text": Counter(),
        }

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                total_rows += 1
                idx = record.get("index", total_rows - 1)
                inp = record.get("input", "")
                exp = record.get("expected", "")
                pred = record.get("prediction", "")

                is_match = (pred.strip() == exp.strip())
                if is_match:
                    exact_matches += 1

                is_math = is_math_input(inp) or is_math_input(exp)
                domain = "math" if is_math else "text"

                if is_math:
                    math_total += 1
                    if is_match:
                        math_correct += 1
                else:
                    text_total += 1
                    if is_match:
                        text_correct += 1

                if not is_match:
                    failure_item = {
                        "index": idx,
                        "input": inp,
                        "expected": exp,
                        "prediction": pred,
                        "domain": domain,
                    }
                    failures.append(failure_item)

                    # 1. Difflib character-level opcodes between target and prediction
                    matcher = difflib.SequenceMatcher(None, exp, pred, autojunk=False)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == "replace":
                            src = exp[i1:i2]
                            dst = pred[j1:j2]
                            difflib_counts["all"][(src, dst)] += 1
                            difflib_counts[domain][(src, dst)] += 1

                    # 2. Swap pairs mirroring report.py for verified baseline statistics
                    for src, dst in TARGET_PAIRS:
                        if src in exp and dst in pred:
                            swap_counts["all"][(src, dst)] += 1
                            swap_counts[domain][(src, dst)] += 1
                        if dst in exp and src in pred:
                            swap_counts["all"][(dst, src)] += 1
                            swap_counts[domain][(dst, src)] += 1

        overall_acc = round((exact_matches / total_rows * 100), 2) if total_rows else 0.0
        math_acc = round((math_correct / math_total * 100), 2) if math_total else 0.0
        text_acc = round((text_correct / text_total * 100), 2) if text_total else 0.0

        self._summary = {
            "total_rows": total_rows,
            "exact_matches": exact_matches,
            "exact_match_acc": overall_acc,
            "math_total": math_total,
            "math_correct": math_correct,
            "math_acc": math_acc,
            "text_total": text_total,
            "text_correct": text_correct,
            "text_acc": text_acc,
            "failures_count": len(failures),
            "checkpoint_file": file_path.name,
        }
        self._failures = failures
        self._swap_counts = swap_counts
        self._difflib_counts = difflib_counts

        self._cached_path = str(file_path)
        self._cached_mtime = mtime
        self._cached_size = size

    def get_summary(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Get high-level evaluation metrics."""
        self.ensure_loaded(path)
        return dict(self._summary)

    def get_confusion(
        self,
        top: int = 20,
        domain: Optional[str] = None,
        path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Compute confusion matrix heatmap and sorted top confusion pairs."""
        self.ensure_loaded(path)

        domain_key = domain.lower() if domain and domain.lower() in ("math", "text") else "all"

        # Combine report swap counts with difflib counts
        combined_counts = Counter(self._swap_counts[domain_key])
        for pair, cnt in self._difflib_counts[domain_key].items():
            if pair not in combined_counts:
                combined_counts[pair] = cnt

        sorted_pairs = combined_counts.most_common()

        # Build list of top pairs
        top_pairs_list = [
            {
                "expected": src,
                "predicted": dst,
                "count": count,
                "domain": domain_key,
            }
            for (src, dst), count in sorted_pairs[:50]
        ]

        # Extract top characters for NxN confusion matrix heatmap
        char_freq = Counter()
        for (src, dst), count in sorted_pairs:
            char_freq[src] += count
            char_freq[dst] += count

        top_chars = [c for c, _ in char_freq.most_common(max(4, min(top, 24)))]
        # Ensure canonical chars like 'l', '1', 'O', '0', 'S', '5' are present if observed
        preferred_order = ["l", "1", "0", "O", "S", "5", "rn", "m", "x", "×", "=", "−", "+", "÷", "9"]
        ordered_chars = []
        for ch in preferred_order:
            if ch in top_chars and ch not in ordered_chars:
                ordered_chars.append(ch)
        for ch in top_chars:
            if ch not in ordered_chars:
                ordered_chars.append(ch)
        top_chars = ordered_chars[:top]

        # Build NxN grid: grid[row][col] where row is Expected and col is Predicted
        grid = []
        max_count = 0
        for src in top_chars:
            row = []
            for dst in top_chars:
                val = combined_counts.get((src, dst), 0)
                row.append(val)
                if val > max_count:
                    max_count = val
            grid.append(row)

        return {
            "characters": top_chars,
            "grid": grid,
            "max_count": max_count,
            "top_pairs": top_pairs_list,
            "domain": domain_key,
        }

    def get_failures(
        self,
        q: Optional[str] = None,
        domain: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Search and paginate failed prediction instances with character diffs."""
        self.ensure_loaded(path)

        domain_filter = domain.lower() if domain and domain.lower() in ("math", "text") else None
        query = q.strip().lower() if q and q.strip() else None

        filtered = []
        for fail in self._failures:
            if domain_filter and fail["domain"] != domain_filter:
                continue

            if query:
                # Search across input, expected (target), and prediction
                inp_match = query in fail["input"].lower()
                exp_match = query in fail["expected"].lower()
                pred_match = query in fail["prediction"].lower()
                if not (inp_match or exp_match or pred_match):
                    continue

            filtered.append(fail)

        total = len(filtered)
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 1

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_slice = filtered[start_idx:end_idx]

        # Compute side-by-side character diffs between expected (target) and prediction
        results = []
        for item in page_slice:
            exp_text = item["expected"]
            pred_text = item["prediction"]
            diff_ops = compute_char_diff(exp_text, pred_text)

            results.append({
                "index": item["index"],
                "input": item["input"],
                "expected": exp_text,
                "prediction": pred_text,
                "domain": item["domain"],
                "diff": diff_ops,
            })

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "failures": results,
        }


analytics_service = AnalyticsService()
