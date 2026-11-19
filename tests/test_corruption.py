"""Unit tests for corruption rules and math sentence generation."""

import random
import pytest
from ocr_lora.config import (
    TEXT_SWAPS,
    MATH_SWAPS,
    MIN_SENTENCE_LEN,
    MAX_SENTENCE_LEN,
    RANDOM_SEED,
)
from ocr_lora.data import (
    corrupt_text,
    generate_math_sentences,
    build_rows,
)


def test_math_sentence_generation_deterministic():
    sentences_1 = generate_math_sentences(count=20, seed=42)
    sentences_2 = generate_math_sentences(count=20, seed=42)
    assert len(sentences_1) == 20
    assert sentences_1 == sentences_2, "Seeded generation must be strictly deterministic!"


def test_corrupt_text_swaps():
    # Sentence with known text swap candidates
    clean = "Solid 100 level rn standard."
    rng = random.Random(42)
    corrupted = corrupt_text(clean, level=0.45, rng=rng)
    # Ensure change occurred
    assert corrupted != clean
    # Check that swap characters came from TEXT_SWAPS
    # '1' -> 'l', '0' -> 'O', 'rn' -> 'm', 'S' -> '5'


def test_corrupt_text_math_swaps():
    clean_math = "12 × 4 = 48."
    rng = random.Random(42)
    corrupted = corrupt_text(clean_math, level=1.0, rng=rng)
    assert "×" not in corrupted or "x" in corrupted
    assert "=" not in corrupted or "−" in corrupted


def test_build_rows_deduplication():
    sentences = ["The quick brown fox jumps over the lazy dog.", "10 + 20 = 30."]
    rows = build_rows(sentences, levels=[0.10, 0.25], seed=42)
    keys = [(r["input"], r["expected"]) for r in rows]
    assert len(keys) == len(set(keys)), "Rows must be strictly deduplicated on (input, expected)!"
    for r in rows:
        assert r["input"] != r["expected"], "Unaltered rows must be discarded!"
