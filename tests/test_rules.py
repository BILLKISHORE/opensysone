import pytest

from opensysone.rules import apply_rules, rule_distribution
from opensysone.schema import SchemaError, compile_request
from opensysone.wire import SystemOneRequest

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Refund?"},
    "team": {"type": "choice", "criteria": {"billing": "b", "ops": "o"}},
    "tone": {"type": "score", "criteria": ["calm", "furious"]},
    "tags": {"type": "multi", "criteria": {"x": "", "y": ""}},
}


def _req(rules, state="Please REFUND me now"):
    return SystemOneRequest.model_validate({"state": state, "questions": QUESTIONS, "rules": rules})


def test_rule_distributions_per_kind():
    plans = compile_request(_req([])).plans
    assert rule_distribution(plans["refund"], True) == [1.0, 0.0]
    assert rule_distribution(plans["team"], "ops") == [0.0, 1.0]
    assert rule_distribution(plans["tone"], 1) == [0.0, 1.0]
    assert rule_distribution(plans["tone"], "furious") == [0.0, 1.0]
    assert rule_distribution(plans["tags"], ["y"]) == [0.0, 1.0]


def test_first_matching_rule_wins_and_is_case_insensitive_by_default():
    req = _req([
        {"question": "refund", "pattern": r"\brefund\b", "answer": True},
        {"question": "refund", "pattern": r"now", "answer": False},
        {"question": "team", "pattern": r"charge", "answer": "billing"},
    ])
    out = apply_rules(req, compile_request(req).plans)
    assert out["refund"] == ([1.0, 0.0], r"\brefund\b")
    assert "team" not in out


def test_flags_are_honoured():
    req = _req([{"question": "refund", "pattern": r"refund", "answer": True, "flags": ""}],
               state="no lowercase here")
    assert apply_rules(req, compile_request(req).plans) == {}


def test_unknown_question_and_bad_answer_and_bad_regex_are_schema_errors():
    plans = compile_request(_req([])).plans
    with pytest.raises(SchemaError) as e:
        apply_rules(_req([{"question": "nope", "pattern": "x", "answer": True}]), plans)
    assert e.value.loc == ["body", "rules", 0]
    with pytest.raises(SchemaError):
        apply_rules(_req([{"question": "team", "pattern": "refund", "answer": "sales"}]), plans)
    with pytest.raises(SchemaError):
        apply_rules(_req([{"question": "refund", "pattern": "(", "answer": True}]), plans)
