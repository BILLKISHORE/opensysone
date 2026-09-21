import random

from opensysone.calib.fit import MIN_ADAPTIVE, fit_calibrator
from opensysone.calib.records import DecisionRecord


def _records(n, kind="choice", top=0.95, acc=0.6, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        label = "a" if rng.random() < acc else "b"
        out.append(DecisionRecord(id=f"{kind}-{i}", group_id=f"{kind}-{i}", kind=kind, options=("a", "b", "c"),
                                  probs=(top, (1 - top) * 0.7, (1 - top) * 0.3), prompt_tokens=100 + i % 50,
                                  label=label, label_distribution=None))
    return out


def test_overconfident_kind_gets_calibrated_and_ece_drops():
    result = fit_calibrator(_records(600))
    s = result.summary["choice"]
    assert s["chosen"] in ("temperature", "adaptive")
    assert s["after"]["ece"] < s["before"]["ece"]
    assert s["after"]["accuracy"] == s["before"]["accuracy"]  # winner never changes
    assert s["n_train"] + s["n_heldout"] == 600


def test_well_calibrated_kind_keeps_identity_or_near_one():
    recs = _records(600, kind="noul", top=0.6, acc=0.6)
    recs = [DecisionRecord(r.id, r.group_id, "noul", ("a", "b"), (0.6, 0.4), r.prompt_tokens, r.label, None)
            for r in recs if r.label in ("a", "b")]
    s = fit_calibrator(recs).summary["noul"]
    assert s["chosen"] == "identity" or abs(s["temperature"] - 1.0) < 0.3


def test_small_kind_skips_adaptive():
    s = fit_calibrator(_records(MIN_ADAPTIVE - 10, kind="score")).summary["score"]
    assert "adaptive" not in s["heldout_nll"]


def test_calibrator_applies_chosen_method():
    result = fit_calibrator(_records(600))
    cal = result.calibrator
    out = cal.apply("choice", [0.95, 0.035, 0.015], 120)
    assert out[0] < 0.95 and out.index(max(out)) == 0
