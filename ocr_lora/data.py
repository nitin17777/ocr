"""Dataset generation and corruption utilities for OCR LoRA evaluation harness.

Handles:
1. Loading and filtering WikiText-2 raw train sentences.
2. Generating synthetic arithmetic sentences across 10 distinct templates.
3. Applying synthetic OCR text swaps and math swaps at specified noise levels.
4. Building deduplicated (input, expected) datasets with reproducible seeding.
"""

from typing import List, Dict, Tuple, Optional
import random
from pathlib import Path
from ocr_lora.config import (
    RANDOM_SEED,
    WIKITEXT_DATASET_ID,
    WIKITEXT_PARQUET_FILE,
    MIN_SENTENCE_LEN,
    MAX_SENTENCE_LEN,
    NUM_MATH_SENTENCES,
    MATH_TEMPLATES,
    CORRUPTION_LEVELS,
    TEXT_SWAPS,
    MATH_SWAPS,
)


def generate_math_sentences(
    count: int = NUM_MATH_SENTENCES,
    seed: int = RANDOM_SEED,
) -> List[str]:
    """Generate synthetic math sentences using confirmed templates and ranges.
    
    a in [2, 999], b in [2, 20].
    """
    rng = random.Random(seed)
    sentences: List[str] = []

    for _ in range(count):
        template = rng.choice(MATH_TEMPLATES)
        a = rng.randint(2, 999)
        b = rng.randint(2, 20)

        if "+" in template or "sum" in template.lower() or "add" in template.lower():
            c = a + b
            text = (
                template.replace("a", str(a))
                .replace("b", str(b))
                .replace("c", str(c))
            )
        elif "−" in template or "subtract" in template.lower():
            # Ensure non-negative or consistent subtraction
            c = a - b
            text = (
                template.replace("a", str(a))
                .replace("b", str(b))
                .replace("c", str(c))
            )
        elif "×" in template or "multiply" in template.lower():
            c = a * b
            text = (
                template.replace("a", str(a))
                .replace("b", str(b))
                .replace("c", str(c))
            )
        elif "÷" in template or "divide" in template.lower():
            c = a // b
            text = (
                template.replace("a", str(a))
                .replace("b", str(b))
                .replace("c", str(c))
            )
        else:
            c = a + b
            text = (
                template.replace("a", str(a))
                .replace("b", str(b))
                .replace("c", str(c))
            )

        sentences.append(text)

    return sentences


def load_wikitext_sentences(
    parquet_path: Optional[str | Path] = None,
    min_len: int = MIN_SENTENCE_LEN,
    max_len: int = MAX_SENTENCE_LEN,
) -> List[str]:
    """Load sentences from WikiText-2 raw train and filter by length."""
    from datasets import load_dataset
    
    if parquet_path and Path(parquet_path).exists():
        ds = load_dataset("parquet", data_files={"train": str(parquet_path)})["train"]
    else:
        ds = load_dataset(WIKITEXT_DATASET_ID, "wikitext-2-raw-v1", split="train")

    raw_texts = ds["text"]
    sentences: List[str] = []
    
    for row in raw_texts:
        cleaned = row.strip()
        if not cleaned:
            continue
        # Split on newline or period while preserving clean sentence candidates
        parts = [p.strip() for p in cleaned.split("\n") if p.strip()]
        for p in parts:
            if min_len <= len(p) <= max_len:
                sentences.append(p)

    # De-duplicate preserving order
    deduped = list(dict.fromkeys(sentences))
    return deduped


def corrupt_text(
    text: str,
    level: float,
    rng: Optional[random.Random] = None,
) -> str:
    """Apply OCR text swaps and math swaps to text at a given corruption level.
    
    1. Find all candidate occurrences for TEXT_SWAPS.
       Shuffle positions and change max(1, int(n * level)) of them.
    2. Apply MATH_SWAPS once each with probability = level.
    """
    if rng is None:
        rng = random.Random()

    corrupted = text

    # Step 1: Text swaps
    # Find all replacement candidates (including 2-char 'rn')
    candidates: List[Tuple[int, int, str, str]] = []  # (start, end, src, dst)
    i = 0
    n_len = len(corrupted)
    while i < n_len:
        if i + 1 < n_len and corrupted[i : i + 2] == "rn":
            candidates.append((i, i + 2, "rn", TEXT_SWAPS["rn"]))
            i += 2
            continue
        char = corrupted[i]
        if char in TEXT_SWAPS:
            candidates.append((i, i + 1, char, TEXT_SWAPS[char]))
        i += 1

    if candidates:
        n_candidates = len(candidates)
        k = max(1, int(n_candidates * level))
        chosen = rng.sample(candidates, min(k, n_candidates))
        # Sort in reverse order of start position so replacements don't shift earlier indices
        chosen.sort(key=lambda x: x[0], reverse=True)
        corrupted_chars = list(corrupted)
        for start, end, _, dst in chosen:
            corrupted_chars[start:end] = list(dst)
        corrupted = "".join(corrupted_chars)

    # Step 2: Math swaps
    # Applied once each with probability = level
    for src, dst in MATH_SWAPS.items():
        if src in corrupted and rng.random() < level:
            # Replace first occurrence once
            corrupted = corrupted.replace(src, dst, 1)

    return corrupted


def build_rows(
    sentences: List[str],
    levels: List[float] = CORRUPTION_LEVELS,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, str]]:
    """Build corrupted rows from clean sentences across corruption levels.
    
    Keep only rows where corrupted != clean, then de-duplicate on (input, expected).
    """
    rng = random.Random(seed)
    raw_pairs: List[Tuple[str, str]] = []

    for level in levels:
        for s in sentences:
            c = corrupt_text(s, level, rng=rng)
            if c != s:
                raw_pairs.append((c, s))

    # De-duplicate on (input, expected) preserving order
    seen = set()
    rows: List[Dict[str, str]] = []
    for inp, exp in raw_pairs:
        key = (inp, exp)
        if key not in seen:
            seen.add(key)
            rows.append({"input": inp, "expected": exp})

    return rows
