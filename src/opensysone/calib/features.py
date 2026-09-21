"""Cheap per-decision features the adaptive head conditions on."""
from __future__ import annotations

import math
from collections.abc import Sequence

KIND_ORDER = ("noul", "choice", "score", "pairwise")
FEATURE_NAMES = (
    "log_k", "top", "margin", "norm_entropy", "log_prompt_tokens",
    "kind_noul", "kind_choice", "kind_score", "kind_pairwise",
)


def decision_features(kind: str, probs: Sequence[float], prompt_tokens: int) -> list[float]:
    k = len(probs)
    ranked = sorted((float(p) for p in probs), reverse=True)
    top = ranked[0]
    second = ranked[1] if k > 1 else 0.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    norm_entropy = h / math.log(k) if k > 1 else 0.0
    return [
        math.log(k),
        top,
        top - second,
        norm_entropy,
        math.log1p(max(int(prompt_tokens), 0)),
        *[1.0 if kind == name else 0.0 for name in KIND_ORDER],
    ]
