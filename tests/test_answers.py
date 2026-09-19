import math

import pytest

from opensysone.answers import (
    average_distributions,
    confidence,
    distribution_from_telemetry,
    permute_choices,
    shape_answer,
)
from opensysone.schema import NONE_OPTION, QuestionPlan
from opensysone.wire import Policy


def _plan(kind, options, legend=None, abstain=None, policy=None):
    return QuestionPlan("q", kind, tuple(options), legend, abstain, policy)


def test_confidence_is_one_when_certain_and_zero_when_uniform():
    assert confidence([1.0, 0.0]) == 1.0
    assert confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert confidence([0.25] * 4) == pytest.approx(0.0)
    assert 0.0 < confidence([0.9, 0.1]) < 1.0


def test_noul_distribution_from_winner_probability():
    plan = _plan("noul", ["true", "false"])
    assert distribution_from_telemetry(plan, {"value": True, "probability": 0.8}) == [0.8, pytest.approx(0.2)]
    assert distribution_from_telemetry(plan, {"value": False, "probability": 0.7}) == [pytest.approx(0.3), 0.7]


def test_choice_distribution_from_log_scores_in_plan_order():
    plan = _plan("choice", ["a", "b", "c"])
    tele = {"value": "b", "probability": 0.6,
            "log_scores": {"b": math.log(0.6), "a": math.log(0.3), "c": math.log(0.1)}}
    d = distribution_from_telemetry(plan, tele)
    assert d == [pytest.approx(0.3), pytest.approx(0.6), pytest.approx(0.1)]


def test_multi_distribution_is_per_option():
    plan = _plan("multi", ["x", "y"])
    tele = {"value": ["x"], "probability": None, "per_option": {"x": 0.9, "y": 0.2}}
    assert distribution_from_telemetry(plan, tele) == [0.9, 0.2]


def test_noul_answer_shape():
    ans = shape_answer(_plan("noul", ["true", "false"]), [0.97, 0.03])
    assert ans == {"type": "noul", "noul": pytest.approx(0.97)}


def test_choice_answer_has_ranked_and_confidence():
    ans = shape_answer(_plan("choice", ["outage", "billing", "feature"]), [0.94, 0.05, 0.01])
    assert ans["type"] == "choice"
    assert ans["choice"] == "outage"
    assert ans["ranked"] == ["outage", "billing", "feature"]
    assert ans["probabilities"]["billing"] == pytest.approx(0.05)
    assert 0.0 < ans["confidence"] <= 1.0


def test_choice_abstain_maps_none_option_to_null():
    plan = _plan("choice", ["a", "b", NONE_OPTION], abstain=NONE_OPTION)
    ans = shape_answer(plan, [0.1, 0.2, 0.7])
    assert ans["choice"] is None
    assert ans["abstained"] is True
    assert ans["probabilities"][NONE_OPTION] == pytest.approx(0.7)


def test_score_answer_expected_value_legend_and_spread():
    plan = _plan("score", ["0", "1", "2"], legend=("calm", "annoyed", "furious"))
    ans = shape_answer(plan, [0.0, 0.5, 0.5])
    assert ans["type"] == "score"
    assert ans["score"] == pytest.approx(1.5)
    assert ans["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert ans["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5}
    assert ans["spread"] == pytest.approx(0.5)


def test_multi_answer_selects_above_half():
    ans = shape_answer(_plan("multi", ["x", "y", "z"]), [0.9, 0.5, 0.1])
    assert ans["type"] == "multi"
    assert ans["selected"] == ["x", "y"]
    assert ans["probabilities"]["z"] == 0.1


def test_policy_bands():
    pol = Policy(act=0.9, review=0.6)
    assert shape_answer(_plan("noul", ["true", "false"], policy=pol), [0.95, 0.05])["action"] == "act"
    assert shape_answer(_plan("noul", ["true", "false"], policy=pol), [0.3, 0.7])["action"] == "review"
    assert shape_answer(_plan("choice", ["a", "b"], policy=pol), [0.55, 0.45])["action"] == "escalate"


def test_average_distributions():
    assert average_distributions([[1.0, 0.0], [0.0, 1.0]]) == [0.5, 0.5]


def test_permute_choices_is_seeded_and_identity_for_sample_zero():
    spec = {"type": "enum", "choices": [str(i) for i in range(20)], "choice_descriptions": {}}
    assert permute_choices(spec, 7, 0) is spec
    p1 = permute_choices(spec, 7, 1)
    assert sorted(p1["choices"]) == sorted(spec["choices"])
    assert p1["choices"] != spec["choices"]
    assert permute_choices(spec, 7, 1)["choices"] == p1["choices"]
    assert permute_choices({"type": "boolean", "description": "d"}, 7, 3)["type"] == "boolean"
