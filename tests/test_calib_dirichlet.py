import random

import pytest

from opensysone.calib.dirichlet import DirichletMap, fit_dirichlet
from opensysone.calib.records import question_signature


def _biased(n, seed=0):
    """Option 0 is over-predicted: the model says 0.6 for it, but option 1 is right 60% of the time."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = [0.6, 0.3, 0.1]
        y = 1 if rng.random() < 0.6 else 0
        out.append((p, [1.0 if k == y else 0.0 for k in range(3)]))
    return out


def test_identity_map_returns_input():
    m = DirichletMap.identity(3)
    assert m.apply([0.2, 0.3, 0.5]) == pytest.approx([0.2, 0.3, 0.5])


def test_fit_learns_class_bias_temperature_cannot():
    train, val = _biased(600, 1), _biased(300, 2)
    m = fit_dirichlet(train, val)
    q = m.apply([0.6, 0.3, 0.1])
    assert q.index(max(q)) == 1  # the bias is fixed, so the winner can change
    assert sum(q) == pytest.approx(1.0)


def test_fit_on_soft_targets_matches_vote_shares():
    rng = random.Random(3)
    data = [([0.5, 0.5], [0.8, 0.2]) for _ in range(rng.randint(400, 400))]
    m = fit_dirichlet(data[:300], data[300:])
    assert m.apply([0.5, 0.5])[0] == pytest.approx(0.8, abs=0.03)


def test_roundtrip_through_dict():
    m = fit_dirichlet(_biased(200, 4), _biased(100, 5))
    again = DirichletMap.from_dict(m.to_dict())
    assert again.apply([0.6, 0.3, 0.1]) == pytest.approx(m.apply([0.6, 0.3, 0.1]))


def test_signature_depends_on_kind_text_and_options():
    a = question_signature("choice", "Which team?", ("billing", "tech"))
    assert a == question_signature("choice", "Which team?", ("billing", "tech"))
    assert a != question_signature("choice", "Which team?", ("tech", "billing"))
    assert a != question_signature("score", "Which team?", ("billing", "tech"))
    assert question_signature("noul", {"b": 1, "a": 2}, ("false", "true")) == \
        question_signature("noul", {"a": 2, "b": 1}, ("false", "true"))
    assert all(c in "0123456789abcdef" for c in a) and len(a) == 12
