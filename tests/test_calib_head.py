import random

import pytest

from opensysone.calib.calibrators import Calibrator, KindCalibration, temperature_scale
from opensysone.calib.features import decision_features
from opensysone.calib.head import AdaptiveHead, fit_adaptive
from opensysone.calib.metrics import soft_nll


def _synthetic(n, seed):
    """Confident decisions (top 0.95) are right 60% of the time; unsure ones (0.55) 55%."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        top = 0.95 if i % 2 == 0 else 0.55
        acc = 0.6 if top == 0.95 else 0.55
        probs = [top, 1.0 - top]
        target = [1.0, 0.0] if rng.random() < acc else [0.0, 1.0]
        out.append((decision_features("noul", probs, 200), probs, target))
    return out


def _nll(samples, head=None):
    total = 0.0
    for f, p, y in samples:
        q = temperature_scale(p, head.temperature(f)) if head else p
        total += soft_nll(q, y)
    return total / len(samples)


def test_head_outputs_positive_temperatures():
    head = AdaptiveHead()
    t = head.temperature(decision_features("choice", [0.5, 0.3, 0.2], 50))
    assert t > 0.05


def test_fit_adaptive_beats_identity_on_heldout():
    train, val, test = _synthetic(800, 1), _synthetic(200, 2), _synthetic(400, 3)
    head, best_val = fit_adaptive(train, val, epochs=300, seed=0)
    assert best_val < _nll(val)
    assert _nll(test, head) < _nll(test) - 0.05
    confident_t = head.temperature(decision_features("noul", [0.95, 0.05], 200))
    assert confident_t > 1.5


def test_fit_is_deterministic():
    train, val = _synthetic(200, 4), _synthetic(60, 5)
    h1, v1 = fit_adaptive(train, val, epochs=50, seed=0)
    h2, v2 = fit_adaptive(train, val, epochs=50, seed=0)
    f = decision_features("noul", [0.9, 0.1], 100)
    assert v1 == pytest.approx(v2)
    assert h1.temperature(f) == pytest.approx(h2.temperature(f))


def test_calibrator_with_head_roundtrip(tmp_path):
    head, _ = fit_adaptive(_synthetic(200, 6), _synthetic(60, 7), epochs=50, seed=0)
    cal = Calibrator({"noul": KindCalibration("adaptive")}, head=head)
    sha = cal.save(tmp_path)
    loaded = Calibrator.load(tmp_path)
    assert loaded.sha == sha
    assert loaded.apply("noul", [0.95, 0.05], 200) == pytest.approx(cal.apply("noul", [0.95, 0.05], 200), abs=1e-5)


def test_adaptive_without_head_rejected():
    with pytest.raises(ValueError, match="head"):
        Calibrator({"noul": KindCalibration("adaptive")})
