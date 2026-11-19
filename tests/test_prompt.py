"""Unit tests for prompt formatting."""

import pytest
from ocr_lora.config import PROMPT_TEMPLATE
from ocr_lora.infer import format_prompt


def test_prompt_template_exact_match():
    expected = "### Instruction:\nFix the OCR errors: {input}\n\n### Response:\n"
    assert PROMPT_TEMPLATE == expected, "Prompt template has been altered from confirmed notebook format!"


def test_format_prompt_substitution():
    sample = "Test OCR sentence with 1 and 0."
    formatted = format_prompt(sample)
    expected = (
        "### Instruction:\n"
        "Fix the OCR errors: Test OCR sentence with 1 and 0.\n\n"
        "### Response:\n"
    )
    assert formatted == expected
