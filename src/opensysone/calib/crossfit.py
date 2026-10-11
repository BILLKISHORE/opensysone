"""Cross-fitted calibration: every decision is calibrated by a fit that never saw its case.

Split cases into k folds by group, fit on k-1 folds, apply to the held-out fold, rotate. The result
scores calibration on all decisions without a separate training set, which is how a calibrator can be
judged on someone else's model outputs.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .fit import fit_calibrator
from .records import DecisionRecord


def fold_of(group_id: str, folds: int) -> int:
    return int(hashlib.sha1(f"fold{group_id}".encode("utf-8")).hexdigest()[:8], 16) % folds


def crossfit(records: Sequence[DecisionRecord], *, folds: int = 5, target: str = "hard",
             seed: int = 0) -> tuple[list[list[float]], list[int]]:
    """Out-of-fold calibrated probabilities, aligned with `records`, and each record's fold."""
    assignment = [fold_of(r.group_id, folds) for r in records]
    out: list[list[float] | None] = [None] * len(records)
    for k in range(folds):
        fit_on = [r for r, f in zip(records, assignment) if f != k]
        cal = fit_calibrator(fit_on, seed=seed, target=target).calibrator
        for i, (r, f) in enumerate(zip(records, assignment)):
            if f == k:
                out[i] = cal.apply(r.kind, r.probs, r.prompt_tokens, signature=r.signature)
    return [list(p) for p in out], assignment
