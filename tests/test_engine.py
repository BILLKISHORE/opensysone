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
            elif _type_of(schema, name) == "multi":
                tele[name] = {"value": [], "probability": None,
                              "per_option": {c: 0.4 for c in _choices_of(schema, name)}}
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


class Sharpen:
    sha = "cafe"

    def apply(self, kind, probs, prompt_tokens, signature=None):
        self.signatures = getattr(self, "signatures", []) + [signature]
        sq = [p * p for p in probs]
        return [x / sum(sq) for x in sq]


def test_raw_distributions_are_exposed_before_calibration(engine):
    out = engine.decide(_req())
    raw = out[engine_mod.RAW_KEY]
    assert raw["prompt_tokens"] == 42
    team = raw["questions"]["team"]
    assert team["kind"] == "choice" and team["options"] == ["outage", "billing"]
    assert team["probs"][0] == pytest.approx(0.7) and team["source"] == "model"


def test_calibrator_changes_confidence_not_winner_and_stamp(engine):
    before = engine.decide(_req(cache=False))["answers"]["team"]["probabilities"]["outage"]
    engine.set_calibrator(Sharpen())
    out = engine.decide(_req())
    assert out["stamp"]["calibration"] == "cafe"
    assert out["answers"]["team"]["choice"] == "outage"
    assert out["answers"]["team"]["probabilities"]["outage"] > before
    assert out[engine_mod.RAW_KEY]["questions"]["team"]["probs"][0] == pytest.approx(0.7)


def test_rules_and_multi_are_not_calibrated(engine):
    engine.set_calibrator(Sharpen())
    req = SystemOneRequest.model_validate({
        "state": "REFUND now", "cache": False,
        "questions": {"refund": {"type": "noul", "instructions": "refund?"},
                      "tags": {"type": "multi", "instructions": "tags?", "criteria": {"x": "", "y": ""}}},
        "rules": [{"question": "refund", "pattern": "refund", "answer": True}]})
    out = engine.decide(req)
    assert out["answers"]["refund"]["noul"] == 1.0
    assert out[engine_mod.RAW_KEY]["questions"]["refund"]["source"] == "rule"


class TinyLM:
    """A stand-in with an untied bf16 head, like Qwen3.8's lm_head."""

    def __init__(self):
        import mlx.core as mx
        import mlx.nn as nn

        self.lm_head = nn.Linear(4, 3, bias=False)
        self.lm_head.weight = (mx.arange(12).reshape(3, 4) / 7 + 16).astype(mx.bfloat16)


def test_float32_head_stops_bf16_rounding_of_logits():
    import mlx.core as mx

    model = TinyLM()
    x = (mx.arange(4).reshape(1, 4) / 3 + 1).astype(mx.bfloat16)
    exact = x.astype(mx.float32) @ model.lm_head.weight.astype(mx.float32).T
    assert model.lm_head(x).dtype == mx.bfloat16

    assert engine_mod.float32_head(model) == "float32"
    out = model.lm_head(x)
    assert out.dtype == mx.float32
    assert mx.allclose(out, exact, atol=1e-4).item()
    assert engine_mod.float32_head(model) == "float32"  # idempotent: no second wrap
    assert model.lm_head(x).dtype == mx.float32


def test_float32_head_leaves_models_without_a_head_alone():
    assert engine_mod.float32_head(object()) == "unchanged"


def test_load_records_head_precision_in_stamp(engine):
    assert engine.stamp["head"] == "unchanged"


def test_calibrator_gets_the_same_signature_the_training_records_have(engine):
    from opensysone.calib.records import decision_line, parse_decision

    question = {"type": "choice", "instructions": "Which team?", "criteria": {"outage": "", "billing": ""}}
    req = SystemOneRequest.model_validate({"state": "x", "cache": False, "questions": {"team": question}})
    cal = Sharpen()
    engine.set_calibrator(cal)
    out = engine.decide(req)
    raw = out[engine_mod.RAW_KEY]["questions"]["team"]
    rec = parse_decision(decision_line(case_id="c", group_id="c", qid="team", question=question, kind=raw["kind"],
                                       options=raw["options"], probs=raw["probs"], prompt_tokens=1, label="outage",
                                       label_distribution=None, source="t", stamp={}))
    assert cal.signatures == [rec.signature] and rec.signature
