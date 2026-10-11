import random

from opensysone.calib.crossfit import crossfit, fold_of
from opensysone.calib.records import DecisionRecord


def _overconfident(n, seed=0):
    rng = random.Random(seed)
    return [DecisionRecord(f"c{i}/q", f"c{i}", "choice", ("a", "b"), (0.95, 0.05), 50,
                           "a" if rng.random() < 0.6 else "b", None, signature="q") for i in range(n)]


def test_every_record_gets_out_of_fold_probabilities():
    recs = _overconfident(500)
    probs, folds = crossfit(recs, folds=5)
    assert len(probs) == len(recs) and all(abs(sum(p) - 1) < 1e-6 for p in probs)
    assert sorted(set(folds)) == [0, 1, 2, 3, 4]
    assert sum(p[0] for p in probs) / len(probs) < 0.8  # overconfidence corrected out of fold


def test_folds_follow_groups():
    assert fold_of("g1", 5) == fold_of("g1", 5)
    recs = _overconfident(50)
    _, folds = crossfit(recs, folds=5)
    by_group = {}
    for r, f in zip(recs, folds):
        by_group.setdefault(r.group_id, set()).add(f)
    assert all(len(v) == 1 for v in by_group.values())
