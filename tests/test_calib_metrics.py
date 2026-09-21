import math

import pytest

from opensysone.calib.metrics import brier, ece, reliability, soft_nll, summarize, tvd
from opensysone.calib.records import DecisionRecord


def _rec(probs, label_index, dist=None, kind="choice"):
    opts = tuple(str(i) for i in range(len(probs)))
    return DecisionRecord(id="x", group_id="x", kind=kind, options=opts, probs=tuple(probs), prompt_tokens=10,
                          label=opts[label_index], label_distribution=dist)


def test_soft_nll_and_brier():
    assert soft_nll([0.5, 0.5], [1.0, 0.0]) == pytest.approx(math.log(2))
    assert soft_nll([1.0, 0.0], [0.0, 1.0]) == pytest.approx(-math.log(1e-12))
    assert brier([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert brier([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.5)


def test_tvd():
    assert tvd([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert tvd([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_ece_perfect_and_overconfident():
    conf = [0.8] * 10
    assert ece(conf, [True] * 8 + [False] * 2) == pytest.approx(0.0)
    assert ece([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)


def test_reliability_bins_skip_empty_and_clamp_one():
    bins = reliability([0.05, 0.95, 1.0], [False, True, True])
    assert [(b["lo"], b["n"]) for b in bins] == [(0.0, 1), (0.9, 2)]
    assert bins[1]["accuracy"] == 1.0


def test_summarize_uses_top_prob_and_optional_tvd():
    recs = [_rec([0.9, 0.1], 0), _rec([0.6, 0.4], 1)]
    s = summarize(recs, [r.probs for r in recs])
    assert s["n"] == 2 and s["accuracy"] == 0.5
    assert s["mean_tvd"] is None
    soft = [_rec([0.9, 0.1], 0, dist=(0.8, 0.2))]
    assert summarize(soft, [soft[0].probs])["mean_tvd"] == pytest.approx(0.1)


def test_summarize_empty():
    assert summarize([], [])["n"] == 0
