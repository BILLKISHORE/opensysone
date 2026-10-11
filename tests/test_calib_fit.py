import random

from opensysone.calib.fit import MIN_ADAPTIVE, choose_method, fit_calibrator
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
    assert s["chosen"] != "identity"
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


def test_already_calibrated_kind_keeps_raw_probabilities():
    """A backbone that is already calibrated must not be 'fixed': identity wins."""
    rng = random.Random(7)
    recs = []
    for i in range(600):
        top = rng.choice([0.55, 0.7, 0.9])
        label = "a" if rng.random() < top else "b"  # accuracy matches confidence
        recs.append(DecisionRecord(f"cal-{i}", f"cal-{i}", "choice", ("a", "b"), (top, 1 - top),
                                   120, label, None))
    s = fit_calibrator(recs).summary["choice"]
    assert s["chosen"] == "identity"
    assert s["after"]["ece"] == s["before"]["ece"]


def test_selection_never_raises_heldout_ece():
    for seed, top, acc in ((1, 0.95, 0.6), (2, 0.6, 0.85), (3, 0.8, 0.8)):
        s = fit_calibrator(_records(600, top=top, acc=acc, seed=seed)).summary["choice"]
        assert s["after"]["ece"] <= s["before"]["ece"] + 1e-9, (seed, s["chosen"])


def _m(nll, ece, acc):
    return {"nll": nll, "ece": ece, "accuracy": acc}


def test_choose_lowest_nll_that_passes_the_guards():
    metrics = {"identity": _m(0.90, 0.06, 0.70), "temperature": _m(0.85, 0.04, 0.70),
               "dirichlet": _m(0.80, 0.03, 0.72)}
    assert choose_method(metrics) == "dirichlet"


def test_choose_rejects_methods_that_raise_ece_or_drop_accuracy():
    metrics = {"identity": _m(0.90, 0.05, 0.70), "temperature": _m(0.80, 0.07, 0.70),
               "dirichlet": _m(0.75, 0.03, 0.68), "ets": _m(0.88, 0.05, 0.70)}
    assert choose_method(metrics) == "ets"


def test_choose_keeps_identity_without_a_real_gain():
    metrics = {"identity": _m(0.90, 0.05, 0.70), "temperature": _m(0.8995, 0.05, 0.70)}
    assert choose_method(metrics) == "identity"


def _biased_by_question(n, seed=0):
    """Two recurring questions; the model over-predicts option a on q1 only."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        q = "q1" if i % 2 else "q2"
        if q == "q1":
            label = "b" if rng.random() < 0.7 else "a"
        else:
            label = "a" if rng.random() < 0.7 else "b"
        out.append(DecisionRecord(f"{q}-{i}", f"g-{i}", "choice", ("a", "b"), (0.65, 0.35), 100, label, None,
                                  signature=q))
    return out


def test_dirichlet_fixes_a_per_question_bias_and_can_raise_accuracy():
    result = fit_calibrator(_biased_by_question(1200))
    s = result.summary["choice"]
    assert s["chosen"] == "dirichlet"
    assert s["after"]["accuracy"] > s["before"]["accuracy"]
    cal = result.calibrator
    assert cal.apply("choice", [0.65, 0.35], 100, signature="q1")[1] > 0.5
    assert cal.apply("choice", [0.65, 0.35], 100, signature="q2")[0] > 0.5
