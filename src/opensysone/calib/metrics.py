"""Proper scores and calibration error for probability distributions."""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from .records import DecisionRecord

EPS = 1e-12
BINS = 10
MASS_BINS = 15  # equal-mass bins: every bin holds the same number of decisions
BOOTSTRAP = 1000


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


def _equal_mass(conf: Sequence[float], correct: Sequence[float], bins: int) -> list[tuple[int, float, float]]:
    """(n, mean confidence, accuracy) for `bins` contiguous runs of the decisions sorted by confidence.

    Equal confidences always share a bin, so a cut never splits a tie into fake error.
    """
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    n, bins = len(order), max(1, min(bins, len(order)))
    chunks: list[list[int]] = []
    start = 0
    for b in range(bins):
        end = start + n // bins + (1 if b < n % bins else 0)
        idx = order[start:end]
        if chunks and conf[chunks[-1][-1]] == conf[idx[0]]:
            chunks[-1].extend(idx)
        else:
            chunks.append(list(idx))
        start = end
    return [(len(idx), sum(conf[i] for i in idx) / len(idx), sum(float(correct[i]) for i in idx) / len(idx))
            for idx in chunks]


def _ece_of(groups: list[tuple[int, float, float]], total: int) -> float:
    return sum(n / total * abs(acc - c) for n, c, acc in groups)


def ece_equal_mass(conf: Sequence[float], correct: Sequence[float], bins: int = MASS_BINS) -> float:
    """Calibration error with equal-count bins; less biased than equal-width bins when confidences bunch up."""
    if not conf:
        return 0.0
    return _ece_of(_equal_mass(conf, correct, bins), len(conf))


def ece_sweep(conf: Sequence[float], correct: Sequence[float]) -> float:
    """Roelofs et al. 2022: the most equal-mass bins whose accuracies still rise with confidence."""
    if not conf:
        return 0.0
    best = _equal_mass(conf, correct, 1)
    for bins in range(2, len(conf) + 1):
        groups = _equal_mass(conf, correct, bins)
        if any(b[2] < a[2] for a, b in zip(groups, groups[1:])):
            break
        best = groups
    return _ece_of(best, len(conf))


def option_ece(probs_list: Sequence[Sequence[float]], label_indices: Sequence[int], bins: int = MASS_BINS) -> float:
    """Calibration of every option's probability, not only the top one (pooled, equal-mass)."""
    conf, hit = [], []
    for probs, y in zip(probs_list, label_indices):
        for k, p in enumerate(probs):
            conf.append(float(p))
            hit.append(1.0 if k == y else 0.0)
    return ece_equal_mass(conf, hit, bins)


def bootstrap_ci(groups: Sequence[str], metric: Callable[[list[int]], float], *, n: int = BOOTSTRAP,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile interval from resampling whole groups (one case's questions move together)."""
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    keys = list(members)
    rng = random.Random(seed)
    stats = sorted(metric([i for _ in keys for i in members[rng.choice(keys)]]) for _ in range(n))
    return stats[int(alpha / 2 * (n - 1))], stats[int((1 - alpha / 2) * (n - 1))]


def summarize(records: Sequence[DecisionRecord], probs_list: Sequence[Sequence[float]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": None, "nll": None, "nll_winner": None, "brier": None, "ece": None,
                "ece_em15": None,
                "ece_sweep": None, "option_ece": None, "mean_tvd": None, "reliability": []}
    conf, correct, nlls, winner_nlls, briers, tvds = [], [], [], [], [], []
    for rec, probs in zip(records, probs_list):
        top = max(range(len(probs)), key=lambda i: probs[i])
        conf.append(float(probs[top]))
        correct.append(top == rec.label_index)
        target = rec.target()
        nlls.append(soft_nll(probs, target))
        winner_nlls.append(-math.log(max(probs[rec.label_index], EPS)))
        briers.append(brier(probs, target))
        if rec.label_distribution is not None:
            tvds.append(tvd(probs, rec.label_distribution))
    return {
        "n": n,
        "accuracy": sum(correct) / n,
        "nll": sum(nlls) / n,  # against vote shares when the record has them
        "nll_winner": sum(winner_nlls) / n,
        "brier": sum(briers) / n,
        "ece": ece(conf, correct),
        "ece_em15": ece_equal_mass(conf, correct),
        "ece_sweep": ece_sweep(conf, correct),
        "option_ece": option_ece(probs_list, [r.label_index for r in records]),
        "mean_tvd": (sum(tvds) / len(tvds)) if len(tvds) == n else None,
        "reliability": reliability(conf, correct),
    }
