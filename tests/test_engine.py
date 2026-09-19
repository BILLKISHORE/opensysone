import math

import pytest

from opensysone import engine as engine_mod
from opensysone.cache import DecisionCache
from opensysone.schema import SchemaError
from opensysone.wire import SystemOneRequest

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"outage": "down", "billing": "money"}},
    "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
}


def _req(**kw):
    return SystemOneRequest.model_validate({"state": "Everything is down.", "questions": QUESTIONS, **kw})


def _choices_of(schema, name):
    field = schema[name]
    return list(field.choices) if hasattr(field, "choices") else list(field["choices"])


def _type_of(schema, name):
    field = schema[name]
    return field.field_type if hasattr(field, "field_type") else field["type"]


class FakeRuns:
    """Records calls to the jevmlx entry point and returns canned telemetry."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, tokenizer, context, schema, **kw):
        self.calls.append({"context": context, "schema": schema, **kw})
        names = list(schema.fields) if hasattr(schema, "fields") else list(schema)
        tele = {}
        for name in names:
            if _type_of(schema, name) == "boolean":
                tele[name] = {"value": True, "probability": 0.9}
            else:
                choices = _choices_of(schema, name)
                n = len(choices)
                probs = [0.7] + [0.3 / (n - 1)] * (n - 1)  # first in prompt order wins
                tele[name] = {"value": choices[0], "probability": 0.7,
                              "log_scores": {c: math.log(p) for c, p in zip(choices, probs)}}
        return {"field_telemetry": tele, "prompt_tokens": 42, "prefill_ms": 1.0,
                "suffix_eval_ms": 2.0, "elapsed_ms": 3.0, "passes": 1}


@pytest.fixture
def engine(monkeypatch):
    runs = FakeRuns()
    monkeypatch.setattr(engine_mod, "_load", lambda model_id: (object(), None))
    monkeypatch.setattr(engine_mod, "_parity", lambda model, tok: {"passed": True, "max_abs_drift_nats": 0.01})
    monkeypatch.setattr(engine_mod, "_run_generation", runs)
    monkeypatch.setattr(engine_mod, "build_stamp",
                        lambda m, c: {"backbone": m, "calibration": c, "opensysone": "test"})
    e = engine_mod.Engine("fake-model", cache=DecisionCache(max_entries=10))
    e.load()
    e._runs = runs
    return e


def test_load_sets_ready_and_stamp(engine):
    assert engine.ready is True
    assert engine.stamp["backbone"] == "fake-model"
    assert engine.max_rows is None


def test_parity_failure_forces_one_row_per_pass(monkeypatch):
    monkeypatch.setattr(engine_mod, "_load", lambda model_id: (object(), None))
    monkeypatch.setattr(engine_mod, "_parity", lambda model, tok: {"passed": False, "max_abs_drift_nats": 0.23})
    monkeypatch.setattr(engine_mod, "build_stamp", lambda m, c: {})
    e = engine_mod.Engine("fake-model")
    e.load()
    assert e.max_rows == 1
    assert e.parity["passed"] is False


def test_decide_returns_jev_shape_with_extensions(engine):
    out = engine.decide(_req())
    assert out["model"] == engine_mod.MODEL_VERSION
    assert out["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert out["cached"] is False
    assert out["answers"]["urgent"] == {"type": "noul", "noul": pytest.approx(0.9)}
    team = out["answers"]["team"]
    assert team["choice"] == "outage" and team["ranked"][0] == "outage"
    tone = out["answers"]["tone"]
    assert tone["legend"]["2"] == "furious" and "spread" in tone
    assert out["stamp"]["backbone"] == "fake-model"
    assert out["timing"]["samples"] == 1


def test_cache_hit_on_second_call(engine):
    first = engine.decide(_req())
    second = engine.decide(_req())
    assert second["cached"] is True
    assert second["answers"] == first["answers"]
    assert len(engine._runs.calls) == 1


def test_cache_can_be_bypassed(engine):
    engine.decide(_req())
    engine.decide(_req(cache=False))
    assert len(engine._runs.calls) == 2


def test_samples_permute_and_average(engine):
    out = engine.decide(_req(samples=4))
    assert len(engine._runs.calls) == 4
    orders = [_choices_of(c["schema"], "team") for c in engine._runs.calls]
    assert orders[0] == ["outage", "billing"]
    assert any(o != orders[0] for o in orders[1:])
    p = out["answers"]["team"]["probabilities"]
    assert 0.3 < p["outage"] < 0.7  # averaged across orders, no longer 0.7 for the first slot


def test_rules_skip_the_model_when_everything_is_ruled(engine):
    req = SystemOneRequest.model_validate({
        "state": "REFUND please", "questions": {"refund": {"type": "noul", "instructions": "refund?"}},
        "rules": [{"question": "refund", "pattern": "refund", "answer": True}]})
    out = engine.decide(req)
    assert engine._runs.calls == []
    assert out["answers"]["refund"] == {"type": "noul", "noul": 1.0, "source": "rule", "rule": "refund"}
    assert out["usage"]["input_tokens"] == 0


def test_prior_correction_and_constraints_pass_through(engine):
    engine.decide(_req(prior_correction=True, constraints=[
        {"type": "implies", "parent": "urgent", "child": "team", "mapping": {"true": "outage"}}]))
    call = engine._runs.calls[0]
    assert call["prior_correction"] is True
    assert call["constraints"][0]["type"] == "implies"


def test_strict_turns_lint_into_schema_error(engine):
    bad = SystemOneRequest.model_validate({"state": "s", "strict": True,
                                           "questions": {"q": {"type": "choice", "criteria": {"a": "", "A": ""}}}})
    with pytest.raises(SchemaError):
        engine.decide(bad)


def test_lint_warnings_are_returned_when_not_strict(engine):
    out = engine.decide(SystemOneRequest.model_validate({"state": "s", "questions": {"q": {"type": "noul"}}}))
    assert any("empty_instructions" in w for w in out["warnings"])


def test_bulk_runs_each_item(engine):
    outs = engine.decide_bulk([_req(), _req(state="Other state")])
    assert len(outs) == 2 and outs[0]["cached"] is False and outs[1]["cached"] is False
