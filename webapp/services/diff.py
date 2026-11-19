"""Character-level SequenceMatcher diff generator."""

import difflib
from typing import List
from webapp.schemas import DiffOp


def compute_char_diff(original: str, corrected: str) -> List[DiffOp]:
    """Compute character-level diff operations between original and corrected strings.
    
    Uses difflib.SequenceMatcher with autojunk=False to prevent heuristics
    from skipping recurring symbols (e.g., spaces, common digits).
    """
    matcher = difflib.SequenceMatcher(None, original, corrected, autojunk=False)
    diff_ops: List[DiffOp] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        diff_ops.append(
            DiffOp(
                tag=tag,
                a_start=i1,
                a_end=i2,
                b_start=j1,
                b_end=j2,
                a_text=original[i1:i2],
                b_text=corrected[j1:j2],
            )
        )

    return diff_ops
