"""Compile a wire request into a jevmlx schema dict and per-question plans.

The plan is what maps a probability distribution over jevmlx choices back
into a Jev answer (see answers.py). jevmlx field names are the question ids.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .wire import (
    ChoiceQuestion,
    Described,
    JSONContent,
    MultiQuestion,
    NoulQuestion,
    PairwiseQuestion,
    Policy,
    ScoreQuestion,
    SystemOneRequest,
)

NONE_OPTION = "none_of_these"
NONE_DESCRIPTION = "none of the options apply"
PAIRWISE_PREFIX = "Compare candidate a and candidate b and pick the better one."


class SchemaError(ValueError):
    """A request that parsed but cannot be compiled. Carries a 422 location."""

    def __init__(self, loc: list[Any], msg: str):
        super().__init__(msg)
        self.loc = loc
        self.msg = msg


@dataclass(frozen=True)
class QuestionPlan:
    qid: str
    kind: str  # noul | choice | score | multi | pairwise
    options: tuple[str, ...]  # jevmlx choice names in prompt order
    legend: tuple[str, ...] | None  # score only: level texts by index
    abstain_option: str | None
    policy: Policy | None


@dataclass(frozen=True)
class CompiledRequest:
    context: str
    schema_dict: dict[str, dict[str, Any]]
    plans: dict[str, QuestionPlan]
    constraints: list[dict[str, Any]]


def text_of(content: Described) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def render_state(state: JSONContent) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)


def _instructions(qid: str, q) -> str:
    return text_of(q.instructions) or qid.replace("_", " ")


def _glosses(qid: str, criteria: dict[str, Described]) -> dict[str, str]:
    glosses: dict[str, str] = {}
    for name, desc in criteria.items():
        if name == "":
            raise SchemaError(["body", "questions", qid, "criteria"], "option names must not be empty")
        text = text_of(desc)
        if text:
            glosses[name] = text
    return glosses


def _noul(qid: str, q: NoulQuestion) -> tuple[dict[str, Any], QuestionPlan]:
    desc = _instructions(qid, q)
    if q.criteria is not None:
        t, f = text_of(q.criteria.true), text_of(q.criteria.false)
        if t:
            desc += f" True means: {t}."
        if f:
            desc += f" False means: {f}."
    return {"type": "boolean", "description": desc}, QuestionPlan(
        qid, "noul", ("true", "false"), None, None, q.policy
    )


def _choice(qid: str, q: ChoiceQuestion) -> tuple[dict[str, Any], QuestionPlan]:
    names = list(q.criteria)
    glosses = _glosses(qid, q.criteria)
    abstain = None
    if q.abstain:
        if NONE_OPTION in names:
            raise SchemaError(
                ["body", "questions", qid, "criteria"],
                f"option name {NONE_OPTION!r} is reserved when abstain is set",
            )
        names.append(NONE_OPTION)
        glosses[NONE_OPTION] = NONE_DESCRIPTION
        abstain = NONE_OPTION
    spec = {
        "type": "enum",
        "description": _instructions(qid, q),
        "choices": names,
        "choice_descriptions": glosses,
    }
    return spec, QuestionPlan(qid, "choice", tuple(names), None, abstain, q.policy)


def _score(qid: str, q: ScoreQuestion) -> tuple[dict[str, Any], QuestionPlan]:
    legend = tuple(text_of(level) for level in q.criteria)
    names = [str(i) for i in range(len(legend))]
    spec = {
        "type": "enum",
        "description": _instructions(qid, q) + " Answer with the level number.",
        "choices": names,
        "choice_descriptions": {n: legend[i] for i, n in enumerate(names)},
    }
    return spec, QuestionPlan(qid, "score", tuple(names), legend, None, q.policy)


def _multi(qid: str, q: MultiQuestion) -> tuple[dict[str, Any], QuestionPlan]:
    names = list(q.criteria)
    spec: dict[str, Any] = {
        "type": "multi",
        "description": _instructions(qid, q),
        "choices": names,
        "choice_descriptions": _glosses(qid, q.criteria),
    }
    if q.constraints:
        spec["set_constraints"] = [dict(c) for c in q.constraints]
    return spec, QuestionPlan(qid, "multi", tuple(names), None, None, q.policy)


def _pairwise(qid: str, q: PairwiseQuestion) -> tuple[dict[str, Any], QuestionPlan]:
    desc = PAIRWISE_PREFIX
    if q.instructions:
        desc += " " + text_of(q.instructions)
    spec = {
        "type": "enum",
        "description": desc,
        "choices": ["a", "b"],
        "choice_descriptions": {"a": text_of(q.a), "b": text_of(q.b)},
    }
    return spec, QuestionPlan(qid, "pairwise", ("a", "b"), None, None, q.policy)


_COMPILERS: dict[type, Any] = {
    NoulQuestion: _noul,
    ChoiceQuestion: _choice,
    ScoreQuestion: _score,
    MultiQuestion: _multi,
    PairwiseQuestion: _pairwise,
}


def _validated_constraints(constraints: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not constraints:
        return []
    from jevmlx.constraints import ConstraintError, validate_constraints

    try:
        return [dict(c) for c in validate_constraints(list(constraints))]
    except (ConstraintError, ValueError) as e:
        raise SchemaError(["body", "constraints"], str(e)) from e


def _check_compiles(schema_dict: dict[str, dict[str, Any]]) -> None:
    """jevmlx validates set constraints and field shapes at construction."""
    from jevmlx.schema import SchemaCompileError, StructuredSchema

    try:
        StructuredSchema(schema_dict)
    except SchemaCompileError as e:
        field = getattr(e, "field", None)
        loc = ["body", "questions", field] if field else ["body", "questions"]
        raise SchemaError(loc, str(e)) from e


def compile_request(req: SystemOneRequest) -> CompiledRequest:
    schema_dict: dict[str, dict[str, Any]] = {}
    plans: dict[str, QuestionPlan] = {}
    for qid, q in req.questions.items():
        compiler = _COMPILERS.get(type(q))
        if compiler is None:
            raise SchemaError(["body", "questions", qid, "type"], f"unsupported question type {q.type!r}")
        spec, plan = compiler(qid, q)
        schema_dict[qid] = spec
        plans[qid] = plan
    _check_compiles(schema_dict)
    constraints = _validated_constraints(req.constraints)
    return CompiledRequest(render_state(req.state), schema_dict, plans, constraints)
