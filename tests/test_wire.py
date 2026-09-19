import pytest
from pydantic import ValidationError

from opensysone.wire import (
    BulkRequest,
    ChoiceQuestion,
    Policy,
    Rule,
    SystemOneRequest,
    error_response,
    validation_error,
)


def _req(**kw):
    base = {
        "state": "I was charged twice.",
        "questions": {"billing": {"type": "noul", "instructions": "Is this a billing issue?"}},
    }
    base.update(kw)
    return SystemOneRequest.model_validate(base)


def test_minimal_jev_request_parses_with_defaults():
    req = _req()
    assert req.model == "opensysone-latest"
    assert req.samples == 1
    assert req.prior_correction is False
    assert req.cache is True
    assert req.questions["billing"].type == "noul"


def test_choice_needs_two_options():
    with pytest.raises(ValidationError):
        ChoiceQuestion.model_validate({"type": "choice", "criteria": {"only": "one"}})


def test_score_levels_bounded():
    with pytest.raises(ValidationError):
        _req(questions={"tone": {"type": "score", "criteria": ["a"]}})
    with pytest.raises(ValidationError):
        _req(questions={"tone": {"type": "score", "criteria": [str(i) for i in range(11)]}})


def test_discriminator_rejects_unknown_type():
    with pytest.raises(ValidationError):
        _req(questions={"x": {"type": "essay", "instructions": "write"}})


def test_question_id_pattern():
    with pytest.raises(ValidationError):
        _req(questions={"bad id!": {"type": "noul"}})


def test_samples_range():
    assert _req(samples=16).samples == 16
    with pytest.raises(ValidationError):
        _req(samples=17)


def test_policy_ordering():
    assert Policy(act=0.9, review=0.6).act == 0.9
    with pytest.raises(ValidationError):
        Policy(act=0.5, review=0.6)


def test_rule_answer_types():
    assert Rule(question="q", pattern="refund", answer=True).answer is True
    assert Rule(question="q", pattern="refund", answer=2).answer == 2
    assert Rule(question="q", pattern="refund", answer=["a", "b"]).answer == ["a", "b"]


def test_extra_jev_fields_are_ignored_not_rejected():
    req = _req(questions={"q": {"type": "noul", "instructions": "x", "future_field": 1}})
    assert req.questions["q"].instructions == "x"


def test_bulk_bounds():
    one = {"state": "s", "questions": {"q": {"type": "noul"}}}
    assert len(BulkRequest(items=[one]).items) == 1
    with pytest.raises(ValidationError):
        BulkRequest(items=[])


def test_error_helpers_shape():
    body = error_response(401, "authentication_error", "no key").body
    assert b'"error_type":"authentication_error"' in body or b'"error_type": "authentication_error"' in body
    v = validation_error(["body", "questions"], "bad")
    assert v.status_code == 422


def test_option_caps_match_jevmlx():
    too_many = {f"o{i}": "" for i in range(256)}
    with pytest.raises(ValidationError):
        _req(questions={"q": {"type": "choice", "criteria": too_many}})
    with pytest.raises(ValidationError):
        _req(questions={"q": {"type": "multi", "criteria": {f"o{i}": "" for i in range(65)}}})
    assert _req(questions={"q": {"type": "choice", "criteria": {f"o{i}": "" for i in range(255)}}})
