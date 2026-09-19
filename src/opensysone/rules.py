"""Rules-first: keyword or regex rules that answer a question before the model runs."""
from __future__ import annotations

import re
from typing import Any

from .schema import QuestionPlan, SchemaError, render_state
from .wire import SystemOneRequest

_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def _compile(pattern: str, flags: str, loc: list[Any]) -> re.Pattern[str]:
    bits = 0
    for ch in flags:
        if ch not in _FLAGS:
            raise SchemaError(loc, f"unknown regex flag {ch!r}; use any of i, m, s")
        bits |= _FLAGS[ch]
    try:
        return re.compile(pattern, bits)
    except re.error as e:
        raise SchemaError(loc, f"invalid regex: {e}") from e


def rule_distribution(plan: QuestionPlan, answer: Any) -> list[float]:
    """One-hot (per-option for multi) distribution over plan.options for a rule's answer."""
    if plan.kind == "noul":
        if not isinstance(answer, bool):
            raise ValueError("noul rule answer must be true or false")
        return [1.0, 0.0] if answer else [0.0, 1.0]
    if plan.kind == "multi":
        if not isinstance(answer, list):
            raise ValueError("multi rule answer must be a list of option names")
        unknown = [a for a in answer if a not in plan.options]
        if unknown:
            raise ValueError(f"unknown options {unknown}")
        return [1.0 if o in answer else 0.0 for o in plan.options]
    if plan.kind == "score":
        if isinstance(answer, bool) or not isinstance(answer, (int, str)):
            raise ValueError("score rule answer must be a level index or level text")
        if isinstance(answer, str):
            if plan.legend is None or answer not in plan.legend:
                raise ValueError(f"unknown level {answer!r}")
            idx = plan.legend.index(answer)
        else:
            idx = answer
        if not 0 <= idx < len(plan.options):
            raise ValueError(f"level index {idx} out of range")
        return [1.0 if i == idx else 0.0 for i in range(len(plan.options))]
    if answer not in plan.options:
        raise ValueError(f"unknown option {answer!r}")
    return [1.0 if o == answer else 0.0 for o in plan.options]


def apply_rules(
    req: SystemOneRequest, plans: dict[str, QuestionPlan]
) -> dict[str, tuple[list[float], str]]:
    """qid -> (distribution, pattern) for every question decided by a rule.

    The first matching rule for a question wins. Rules are checked in order
    against the rendered state.
    """
    if not req.rules:
        return {}
    text = render_state(req.state)
    out: dict[str, tuple[list[float], str]] = {}
    for index, rule in enumerate(req.rules):
        loc = ["body", "rules", index]
        plan = plans.get(rule.question)
        if plan is None:
            raise SchemaError(loc, f"rule targets unknown question {rule.question!r}")
        try:
            dist = rule_distribution(plan, rule.answer)
        except ValueError as e:
            raise SchemaError(loc, str(e)) from e
        if rule.question in out:
            continue
        if _compile(rule.pattern, rule.flags, loc).search(text):
            out[rule.question] = (dist, rule.pattern)
    return out
