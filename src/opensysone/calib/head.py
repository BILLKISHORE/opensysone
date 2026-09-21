"""Adaptive temperature: a small MLX head predicts T per decision.

The head sees only decision features (option count, top probability, margin,
entropy, prompt length, question type), so scaling by its T keeps the winner.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .features import FEATURE_NAMES
from .metrics import EPS

T_FLOOR = 0.05
PAD_LOGIT = -1e9

Sample = tuple[Sequence[float], Sequence[float], Sequence[float]]  # features, probs, target


class AdaptiveHead(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 16):
        super().__init__()
        self.l1 = nn.Linear(n_features, hidden)
        self.l2 = nn.Linear(hidden, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.softplus(self.l2(nn.relu(self.l1(x)))).squeeze(-1) + T_FLOOR

    def temperature(self, features: Sequence[float]) -> float:
        return float(self(mx.array([list(features)], dtype=mx.float32))[0].item())


def _batch(samples: Sequence[Sample]) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    width = max(len(p) for _, p, _ in samples)
    feats, logp, mask, target = [], [], [], []
    for f, p, y in samples:
        pad = width - len(p)
        feats.append(list(f))
        logp.append([math.log(max(float(x), EPS)) for x in p] + [0.0] * pad)
        mask.append([True] * len(p) + [False] * pad)
        target.append(list(y) + [0.0] * pad)
    return (mx.array(feats, dtype=mx.float32), mx.array(logp, dtype=mx.float32),
            mx.array(mask), mx.array(target, dtype=mx.float32))


def _loss(head: AdaptiveHead, x: mx.array, logp: mx.array, mask: mx.array, target: mx.array) -> mx.array:
    t = head(x)[:, None]
    logits = mx.where(mask, logp / t, PAD_LOGIT)
    log_q = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -(target * mx.where(mask, log_q, 0.0)).sum(axis=-1).mean()


def fit_adaptive(
    train: Sequence[Sample],
    val: Sequence[Sample],
    *,
    epochs: int = 400,
    lr: float = 0.01,
    patience: int = 40,
    seed: int = 0,
) -> tuple[AdaptiveHead, float]:
    mx.random.seed(seed)
    head = AdaptiveHead()
    mx.eval(head.parameters())
    opt = optim.Adam(learning_rate=lr)
    train_b, val_b = _batch(train), _batch(val)
    step = nn.value_and_grad(head, _loss)
    best_val = float(_loss(head, *val_b).item())
    best_params = dict(tree_flatten(head.parameters()))
    stale = 0
    for _ in range(epochs):
        _, grads = step(head, *train_b)
        opt.update(head, grads)
        mx.eval(head.parameters(), opt.state)
        v = float(_loss(head, *val_b).item())
        if v < best_val - 1e-6:
            best_val, best_params, stale = v, dict(tree_flatten(head.parameters())), 0
        else:
            stale += 1
            if stale >= patience:
                break
    head.update(tree_unflatten(list(best_params.items())))
    mx.eval(head.parameters())
    return head, best_val
