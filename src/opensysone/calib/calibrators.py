"""Calibrators that rescale confidence without changing the winning option.

identity:    probabilities unchanged
temperature: one fitted T per question type, p_i ** (1/T) renormalized
adaptive:    a small head predicts T per decision from decision features
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .features import decision_features
from .metrics import EPS, soft_nll

T_MIN = 0.05
T_MAX = 20.0
METHODS = ("identity", "temperature", "adaptive")
CALIBRATION_FILE = "calibration.json"
HEAD_FILE = "head.safetensors"
FORMAT = 1


def temperature_scale(probs: Sequence[float], t: float) -> list[float]:
    logits = [math.log(max(float(p), EPS)) / t for p in probs]
    top = max(logits)
    weights = [math.exp(x - top) for x in logits]
    z = sum(weights)
    return [w / z for w in weights]


def _mean_nll(samples: Sequence[tuple[Sequence[float], Sequence[float]]], t: float) -> float:
    return sum(soft_nll(temperature_scale(p, t), y) for p, y in samples) / len(samples)


def fit_temperature(samples: Sequence[tuple[Sequence[float], Sequence[float]]], iterations: int = 80) -> float:
    """Golden-section search for T in [T_MIN, T_MAX] (log space) minimizing mean soft NLL."""
    if not samples:
        return 1.0
    ratio = (math.sqrt(5) - 1) / 2
    lo, hi = math.log(T_MIN), math.log(T_MAX)
    a = hi - ratio * (hi - lo)
    b = lo + ratio * (hi - lo)
    fa, fb = _mean_nll(samples, math.exp(a)), _mean_nll(samples, math.exp(b))
    for _ in range(iterations):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = _mean_nll(samples, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = _mean_nll(samples, math.exp(b))
    return round(math.exp((lo + hi) / 2), 6)


@dataclass(frozen=True)
class KindCalibration:
    method: str
    temperature: float = 1.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown calibration method {self.method!r}; expected one of {METHODS}")


class Calibrator:
    def __init__(self, kinds: dict[str, KindCalibration], head: Any = None, meta: dict[str, Any] | None = None):
        self.kinds = dict(kinds)
        self.head = head
        self.meta = dict(meta or {})
        self._sha: str | None = None
        if any(k.method == "adaptive" for k in self.kinds.values()) and head is None:
            raise ValueError("an adaptive kind needs a trained head")

    @property
    def sha(self) -> str | None:
        return self._sha

    def apply(self, kind: str, probs: Sequence[float], prompt_tokens: int) -> list[float]:
        kc = self.kinds.get(kind)
        if kc is None or kc.method == "identity":
            return [float(p) for p in probs]
        if kc.method == "temperature":
            return temperature_scale(probs, kc.temperature)
        t = self.head.temperature(decision_features(kind, probs, prompt_tokens))
        return temperature_scale(probs, t)

    def _body(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kinds": {k: {"method": v.method, "temperature": v.temperature, "metrics": v.metrics}
                      for k, v in sorted(self.kinds.items())},
            "head": HEAD_FILE if self.head is not None else None,
            "meta": self.meta,
        }

    @staticmethod
    def _sha_of(directory: Path) -> str:
        h = hashlib.sha256((directory / CALIBRATION_FILE).read_bytes())
        head = directory / HEAD_FILE
        if head.exists():
            h.update(head.read_bytes())
        return h.hexdigest()[:16]

    def save(self, directory: str | Path) -> str:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CALIBRATION_FILE).write_text(json.dumps(self._body(), indent=2, sort_keys=True) + "\n")
        if self.head is not None:
            self.head.save_weights(str(directory / HEAD_FILE))
        self._sha = self._sha_of(directory)
        return self._sha

    @classmethod
    def load(cls, directory: str | Path) -> "Calibrator":
        directory = Path(directory)
        body = json.loads((directory / CALIBRATION_FILE).read_text())
        if body.get("format") != FORMAT:
            raise ValueError(f"unsupported calibration format {body.get('format')!r}")
        head = None
        if body.get("head"):
            from .head import AdaptiveHead

            head = AdaptiveHead()
            head.load_weights(str(directory / body["head"]))
        kinds = {k: KindCalibration(v["method"], float(v.get("temperature", 1.0)), v.get("metrics", {}))
                 for k, v in body["kinds"].items()}
        cal = cls(kinds, head=head, meta=body.get("meta"))
        cal._sha = cls._sha_of(directory)
        return cal
