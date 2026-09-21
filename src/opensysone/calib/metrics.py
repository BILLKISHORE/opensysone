"""Proper scores and calibration error for probability distributions."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .records import DecisionRecord

EPS = 1e-12
BINS = 10


def soft_nll(probs: Sequence[float], target: Sequence[float]) -> float:
    return -sum(t * math.log(max(p, EPS)) for p, t in zip(probs, target) if t > 0)


def brier(probs: Sequence[float], target: Sequence[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(probs, target))


def tvd(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def _bin(c: float, bins: int) -> int:
    return min(int(c * bins), bins - 1)


def reliability(conf: Sequence[float], correct: Sequence[bool], bins: int = BINS) -> list[dict[str, Any]]:
    buckets: dict[int, list[tuple[float, bool]]] = {}
    for c, ok in zip(conf, correct):
        buckets.setdefault(_bin(c, bins), []).append((c, ok))
    out = []
    for b in sorted(buckets):
        items = buckets[b]
        out.append({
            "lo": round(b / bins, 4),
            "hi": round((b + 1) / bins, 4),
            "n": len(items),
            "confidence": sum(c for c, _ in items) / len(items),
            "accuracy": sum(1 for _, ok in items if ok) / len(items),
        })
    return out


def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = BINS) -> float:
    total = len(conf)
    if total == 0:
        return 0.0
    return sum(b["n"] / total * abs(b["accuracy"] - b["confidence"]) for b in reliability(conf, correct, bins))


def summarize(records: Sequence[DecisionRecord], probs_list: Sequence[Sequence[float]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": None, "nll": None, "brier": None, "ece": None, "mean_tvd": None,
                "reliability": []}
    conf, correct, nlls, briers, tvds = [], [], [], [], []
    for rec, probs in zip(records, probs_list):
        top = max(range(len(probs)), key=lambda i: probs[i])
        conf.append(float(probs[top]))
        correct.append(top == rec.label_index)
        target = rec.target()
        nlls.append(soft_nll(probs, target))
        briers.append(brier(probs, target))
        if rec.label_distribution is not None:
            tvds.append(tvd(probs, rec.label_distribution))
    return {
        "n": n,
        "accuracy": sum(correct) / n,
        "nll": sum(nlls) / n,
        "brier": sum(briers) / n,
        "ece": ece(conf, correct),
        "mean_tvd": (sum(tvds) / len(tvds)) if len(tvds) == n else None,
        "reliability": reliability(conf, correct),
    }
