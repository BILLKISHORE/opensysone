from opensysone.calib.records import is_heldout
from opensysone.train.examples import CaseExample, split_train_val


def _ex(i, group):
    return CaseExample(case_id=f"c{i}", group_id=group, prompt=(1,), fields=())


def _pool(n=600, per_group=3):
    return [_ex(i, f"g{i // per_group}") for i in range(n)]


def test_split_keeps_every_non_heldout_case_and_skips_outer_heldout():
    examples = _pool()
    train, val = split_train_val(examples, val_cases=60)
    kept = [e for e in examples if not is_heldout(e.group_id)]
    assert sorted(e.case_id for e in [*train, *val]) == sorted(e.case_id for e in kept)


def test_split_never_puts_one_group_in_both_sides():
    train, val = split_train_val(_pool(), val_cases=60)
    assert not {e.group_id for e in train} & {e.group_id for e in val}


def test_split_takes_whole_groups_until_the_val_size_is_reached():
    train, val = split_train_val(_pool(per_group=3), val_cases=60)
    assert 60 <= len(val) < 63
    assert all(is_heldout(e.group_id, salt="inner") for e in val)
