"""Unit tests for character-level SequenceMatcher diff calculations."""

import pytest
from webapp.services.diff import compute_char_diff


def test_diff_o_to_zero():
    orig = "2O2"
    corr = "202"
    ops = compute_char_diff(orig, corr)
    assert len(ops) == 3
    # Check middle char replaced
    replace_ops = [op for op in ops if op.tag == "replace"]
    assert len(replace_ops) == 1
    assert replace_ops[0].a_text == "O"
    assert replace_ops[0].b_text == "0"
    assert replace_ops[0].a_start == 1


def test_diff_l_to_one():
    orig = "l04"
    corr = "104"
    ops = compute_char_diff(orig, corr)
    assert len(ops) == 2
    replace_ops = [op for op in ops if op.tag == "replace"]
    assert len(replace_ops) == 1
    assert replace_ops[0].a_text == "l"
    assert replace_ops[0].b_text == "1"


def test_diff_rn_to_m():
    orig = "rnodern"
    corr = "modern"
    ops = compute_char_diff(orig, corr)
    replace_ops = [op for op in ops if op.tag == "replace"]
    assert len(replace_ops) == 1
    assert replace_ops[0].a_text == "rn"
    assert replace_ops[0].b_text == "m"


def test_diff_x_multiplication_sign():
    orig = "12 x 4 = 48."
    corr = "12 × 4 = 48."
    ops = compute_char_diff(orig, corr)
    replace_ops = [op for op in ops if op.tag == "replace"]
    assert len(replace_ops) == 1
    assert replace_ops[0].a_text == "x"
    assert replace_ops[0].b_text == "×"


def test_diff_exact_match():
    text = "Clean sentence without errors."
    ops = compute_char_diff(text, text)
    assert len(ops) == 1
    assert ops[0].tag == "equal"
    assert ops[0].a_text == text
    assert ops[0].b_text == text
