"""Dirichlet calibration (Kull et al. 2019): q = softmax(W log p + b), fit to soft labels.

Unlike one temperature, W and b give every option its own scale and offset, so a model that
over-predicts one option can be corrected. That can change the winner, so the fit selection
checks accuracy before using it. ODIR keeps W near diagonal and b near zero on small data.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .metrics import EPS

Sample = tuple[Sequence[float], Sequence[float]]  # probs, target distribution
LAMBDAS = (1e-3, 1e-2, 1e-1)
STEPS = 400
LR = 0.05


@dataclass(frozen=True)
class DirichletMap:
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    @property
    def k(self) -> int:
        return len(self.bias)

    @classmethod
    def identity(cls, k: int) -> "DirichletMap":
        return cls(tuple(tuple(1.0 if i == j else 0.0 for j in range(k)) for i in range(k)), (0.0,) * k)

    def apply(self, probs: Sequence[float]) -> list[float]:
        logp = [math.log(max(float(p), EPS)) for p in probs]
        z = [sum(w * x for w, x in zip(row, logp)) + b for row, b in zip(self.weights, self.bias)]
        top = max(z)
        e = [math.exp(v - top) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def to_dict(self) -> dict[str, list]:
        return {"weights": [list(r) for r in self.weights], "bias": list(self.bias)}

    @classmethod
    def from_dict(cls, d: dict[str, list]) -> "DirichletMap":
        return cls(tuple(tuple(float(x) for x in r) for r in d["weights"]), tuple(float(x) for x in d["bias"]))


def _train(samples: Sequence[Sample], lam: float):
    import mlx.core as mx
    import mlx.optimizers as optim

    k = len(samples[0][0])
    logp = mx.array([[math.log(max(float(p), EPS)) for p in s[0]] for s in samples], dtype=mx.float32)
    target = mx.array([list(s[1]) for s in samples], dtype=mx.float32)
    off = 1.0 - mx.eye(k)
    params = {"w": mx.eye(k), "b": mx.zeros((k,))}

    def loss(p):
        z = logp @ p["w"].T + p["b"]
        ce = -(target * (z - mx.logsumexp(z, axis=-1, keepdims=True))).sum(axis=-1).mean()
        odir = ((p["w"] * off) ** 2).sum() / max(1, k * (k - 1)) + (p["b"] ** 2).sum() / k
        return ce + lam * odir

    opt = optim.Adam(learning_rate=LR)
    grad = mx.value_and_grad(loss)
    for _ in range(STEPS):
        _, g = grad(params)
        params = opt.apply_gradients(g, params)
        mx.eval(params)
    w, b = params["w"].tolist(), params["b"].tolist()
    return DirichletMap(tuple(tuple(r) for r in w), tuple(b))


def _nll(m: DirichletMap, samples: Sequence[Sample]) -> float:
    return sum(-sum(t * math.log(max(q, EPS)) for q, t in zip(m.apply(p), y) if t > 0)
               for p, y in samples) / len(samples)


def fit_dirichlet(train: Sequence[Sample], val: Sequence[Sample]) -> DirichletMap:
    """Fit one map per regularization strength on `train` and keep the best on `val` (soft NLL)."""
    k = len(train[0][0])
    if any(len(p) != k for p, _ in [*train, *val]):
        raise ValueError("every sample in a Dirichlet group needs the same number of options")
    best, best_nll = DirichletMap.identity(k), _nll(DirichletMap.identity(k), val) if val else math.inf
    for lam in LAMBDAS:
        m = _train(train, lam)
        v = _nll(m, val) if val else 0.0
        if v < best_nll:
            best, best_nll = m, v
    return best
