"""Jev wire format (TypeSafe OpenAPI 0.2.0 shapes) plus opensysone extensions.

Question models ignore unknown fields so requests from TypeSafe's SDKs keep
working if they add fields. Our own models (Policy, Rule) forbid extras.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

JSONContent = Union[str, dict[str, Any], list[Any]]
Described = Union[str, dict[str, Any], list[Any], None]

QID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_QUESTIONS = 128
MAX_OPTIONS = 255  # jevmlx and Jev both cap choice fields here
MAX_MULTI_OPTIONS = 64  # jevmlx cap for multi fields
MAX_SCORE_LEVELS = 10
MAX_SAMPLES = 16
MAX_BULK_ITEMS = 1000


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    act: float = Field(ge=0.0, le=1.0)
    review: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "Policy":
        if self.review > self.act:
            raise ValueError("policy.review must be <= policy.act")
        return self


class NoulCriteria(BaseModel):
    true: Described = None
    false: Described = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Described = None
    criteria: NoulCriteria | None = None
    policy: Policy | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Described = None
    criteria: dict[str, Described] = Field(min_length=2, max_length=MAX_OPTIONS)
    abstain: bool = False
    policy: Policy | None = None


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Described = None
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_SCORE_LEVELS)
    policy: Policy | None = None


class MultiQuestion(BaseModel):
    type: Literal["multi"]
    instructions: Described = None
    criteria: dict[str, Described] = Field(min_length=2, max_length=MAX_MULTI_OPTIONS)
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    policy: Policy | None = None


class PairwiseQuestion(BaseModel):
    type: Literal["pairwise"]
    instructions: Described = None
    a: JSONContent
    b: JSONContent
    policy: Policy | None = None


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion, MultiQuestion, PairwiseQuestion],
    Field(discriminator="type"),
]


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    pattern: str
    answer: bool | int | str | list[str]
    flags: str = "i"


class SystemOneRequest(BaseModel):
    model: str = "opensysone-latest"
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)
    constraints: list[dict[str, Any]] | None = None
    samples: int = Field(default=1, ge=1, le=MAX_SAMPLES)
    prior_correction: bool = False
    rules: list[Rule] | None = None
    strict: bool = False
    cache: bool = True

    @model_validator(mode="after")
    def _qids(self) -> "SystemOneRequest":
        for qid in self.questions:
            if not QID_RE.match(qid):
                raise ValueError(f"question id {qid!r} must match {QID_RE.pattern}")
        return self


class BulkRequest(BaseModel):
    items: list[SystemOneRequest] = Field(min_length=1, max_length=MAX_BULK_ITEMS)


def error_response(
    status: int, error_type: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Jev-shaped error: {"detail": {"error_type", "message"}}."""
    return JSONResponse(
        {"detail": {"error_type": error_type, "message": message}},
        status_code=status,
        headers=headers,
    )


def validation_error(loc: list[Any], msg: str, value: Any = None) -> JSONResponse:
    """FastAPI-shaped 422 list, the shape Jev returns for bad questions."""
    return JSONResponse(
        {"detail": [{"type": "value_error", "loc": loc, "msg": msg, "input": value}]},
        status_code=422,
    )
