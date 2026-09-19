"""Turn probability distributions into Jev answers.

All functions are pure. The engine collects one distribution per question
(possibly averaged over samples) and calls shape_answer.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from .schema import QuestionPlan


def confidence(probs: Sequence[float]) -> float:
    """1 - H(p) / ln K. 1 when certain, 0 when uniform. Same as openjev."""
    k = len(probs)
    if k < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def _normalize(probs: Sequence[float]) -> list[float]:
    z = float(sum(probs))
    if z <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [float(p) / z for p in probs]


def distribution_from_telemetry(plan: QuestionPlan, telemetry: dict[str, Any]) -> list[float]:
    """Full distribution over plan.options from one jevmlx field_telemetry entry."""
    if plan.kind == "multi":
        per = telemetry.get("per_option") or {}
        return [float(per.get(o, 0.0)) for o in plan.options]
    if plan.kind == "noul":
        p = float(telemetry["probability"])
        yes = p if telemetry["value"] is True else 1.0 - p
        return [yes, 1.0 - yes]
    log_scores = telemetry.get("log_scores") or {}
    if log_scores:
        top = max(log_scores.values())
        weights = {k: math.exp(v - top) for k, v in log_scores.items()}
        z = sum(weights.values()) or 1.0
        return [weights.get(o, 0.0) / z for o in plan.options]
    return [1.0 if o == telemetry.get("value") else 0.0 for o in plan.options]


def _action(top: float, plan: QuestionPlan) -> str:
    if top >= plan.policy.act:
        return "act"
    if top >= plan.policy.review:
        return "review"
    return "escalate"


def shape_answer(plan: QuestionPlan, probs: Sequence[float]) -> dict[str, Any]:
    if plan.kind == "noul":
        p = _normalize(probs)
        ans: dict[str, Any] = {"type": "noul", "noul": p[0]}
        top = max(p)
    elif plan.kind in ("choice", "pairwise"):
        p = _normalize(probs)
        order = sorted(range(len(p)), key=lambda i: -p[i])
        winner = plan.options[order[0]]
        ans = {
            "type": "choice",
            "choice": winner,
            "probabilities": {o: p[i] for i, o in enumerate(plan.options)},
            "confidence": confidence(p),
            "ranked": [plan.options[i] for i in order],
        }
        if plan.abstain_option is not None:
            ans["abstained"] = winner == plan.abstain_option
            if ans["abstained"]:
                ans["choice"] = None
        top = p[order[0]]
    elif plan.kind == "score":
        p = _normalize(probs)
        ev = sum(i * x for i, x in enumerate(p))
        var = sum(x * (i - ev) ** 2 for i, x in enumerate(p))
        legend = plan.legend or tuple(plan.options)
        ans = {
            "type": "score",
            "score": ev,
            "legend": {str(i): legend[i] for i in range(len(p))},
            "probabilities": {str(i): p[i] for i in range(len(p))},
            "confidence": confidence(p),
            "spread": math.sqrt(var),
        }
        top = max(p)
    elif plan.kind == "multi":
        p = [float(x) for x in probs]
        ans = {
            "type": "multi",
            "selected": [o for o, x in zip(plan.options, p) if x >= 0.5],
            "probabilities": dict(zip(plan.options, p)),
        }
        top = min((x if x >= 0.5 else 1.0 - x) for x in p) if p else 1.0
    else:
        raise ValueError(f"unknown plan kind {plan.kind!r}")
    if plan.policy is not None:
        ans["action"] = _action(top, plan)
    return ans


def average_distributions(dists: Sequence[Sequence[float]]) -> list[float]:
    n = len(dists)
    width = len(dists[0])
    return [sum(d[i] for d in dists) / n for i in range(width)]


def permute_choices(spec: dict[str, Any], seed: int, sample_index: int) -> dict[str, Any]:
    """Seeded shuffle of an enum or multi field's prompt order. Sample 0 keeps the caller's order."""
    if sample_index == 0 or spec.get("type") == "boolean":
        return spec
    rng = random.Random(f"{seed}:{spec.get('description', '')}:{sample_index}")
    choices = list(spec["choices"])
    rng.shuffle(choices)
    out = dict(spec)
    out["choices"] = choices
    return out
