"""Schema lint: catch questions the model will answer badly before calling it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import CompiledRequest

MANY_OPTIONS = 128
NEAR_DUPLICATE_MIN_LEN = 4  # "a" vs "b" is a label choice, not a typo


@dataclass(frozen=True)
class LintFinding:
    qid: str
    code: str
    message: str


def _edit_distance_is_one(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1 :]


def _jevmlx_findings(compiled: CompiledRequest, tokenizer: Any) -> list[Any]:
    from jevmlx.lint import lint_schema
    from jevmlx.schema import StructuredSchema

    return list(lint_schema(StructuredSchema(compiled.schema_dict), tokenizer))


def lint_compiled(compiled: CompiledRequest, tokenizer: Any = None) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for qid, plan in compiled.plans.items():
        spec = compiled.schema_dict[qid]
        fallback = qid.replace("_", " ")
        desc = spec.get("description", "")
        if desc in ("", fallback, fallback + " Answer with the level number."):
            findings.append(LintFinding(
                qid, "empty_instructions", "no instructions given; the model only sees the question id"))
        if plan.kind not in ("choice", "multi"):
            continue
        by_lower: dict[str, list[str]] = {}
        for o in plan.options:
            by_lower.setdefault(o.lower(), []).append(o)
        for names in by_lower.values():
            if len(names) > 1:
                findings.append(LintFinding(qid, "duplicate_option", f"options differ only by case: {names}"))
        opts = list(plan.options)
        for i in range(len(opts)):
            for j in range(i + 1, len(opts)):
                if min(len(opts[i]), len(opts[j])) < NEAR_DUPLICATE_MIN_LEN:
                    continue
                if _edit_distance_is_one(opts[i].lower(), opts[j].lower()):
                    findings.append(LintFinding(
                        qid, "near_duplicate", f"{opts[i]!r} and {opts[j]!r} differ by one character"))
        if len(opts) > MANY_OPTIONS:
            findings.append(LintFinding(
                qid, "many_options", f"{len(opts)} options; accuracy and latency degrade above {MANY_OPTIONS}"))
    if tokenizer is not None:
        for f in _jevmlx_findings(compiled, tokenizer):
            message = f.message
            if getattr(f, "suggestion", None):
                message += f" Suggestion: {f.suggestion}"
            findings.append(LintFinding(f.field, f"token_{f.kind}", message))
    return findings
