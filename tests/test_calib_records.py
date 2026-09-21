import pytest

from opensysone.calib.records import (
    CALIBRATED_KINDS,
    append_jsonl,
    decision_line,
    is_heldout,
    load_decisions,
    normalize_label,
    parse_decision,
    read_jsonl,
)


def _line(**kw):
    base = dict(case_id="c1", group_id="c1", qid="team", question={"type": "choice"}, kind="choice",
                options=["a", "b", "c"], probs=[0.7, 0.2, 0.1], prompt_tokens=120, label="b",
                label_distribution={"a": 0.25, "b": 0.5, "c": 0.25}, source="teacher", stamp={"backbone": "m"})
    base.update(kw)
    return decision_line(**base)


def test_normalize_label():
    assert normalize_label(True) == "true"
    assert normalize_label(False) == "false"
    assert normalize_label(2) == "2"
    assert normalize_label("billing") == "billing"


def test_decision_line_shape():
    line = _line()
    assert line["id"] == "c1/team"
    assert line["answer"] == {"kind": "choice", "options": ["a", "b", "c"], "probs": [0.7, 0.2, 0.1],
                              "prompt_tokens": 120}
    assert line["label"] == "b" and line["source"] == "teacher" and line["ts"].endswith("Z")


def test_parse_decision_aligns_soft_labels_to_options():
    rec = parse_decision(_line())
    assert rec.kind == "choice" and rec.label_index == 1
    assert rec.target() == [0.25, 0.5, 0.25]
    hard = parse_decision(_line(label_distribution=None))
    assert hard.target() == [0.0, 1.0, 0.0]


def test_parse_decision_rejects_unknown_label_and_length_mismatch():
    with pytest.raises(ValueError, match="not in options"):
        parse_decision(_line(label="z"))
    with pytest.raises(ValueError, match="probs"):
        parse_decision(_line(probs=[0.5, 0.5]))


def test_bool_label_matches_noul_options():
    rec = parse_decision(_line(kind="noul", options=["true", "false"], probs=[0.8, 0.2], label=False,
                               label_distribution=None))
    assert rec.label == "false" and rec.label_index == 1


def test_jsonl_roundtrip_and_kind_filter(tmp_path):
    p = tmp_path / "d.jsonl"
    assert append_jsonl(p, [_line(), _line(qid="tags", kind="multi", options=["x", "y"], probs=[0.9, 0.1],
                                         label="x", label_distribution=None)]) == 2
    assert len(list(read_jsonl(p))) == 2
    recs = load_decisions([p])
    assert [r.kind for r in recs] == ["choice"]
    assert "multi" not in CALIBRATED_KINDS


def test_read_jsonl_reports_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ok": 1}\n{broken\n')
    with pytest.raises(ValueError, match=":2:"):
        list(read_jsonl(p))


def test_heldout_is_deterministic_and_about_one_in_five():
    ids = [f"case_{i}" for i in range(2000)]
    held = [i for i in ids if is_heldout(i)]
    assert held == [i for i in ids if is_heldout(i)]
    assert 300 < len(held) < 500
    assert [i for i in ids if is_heldout(i, salt="inner")] != held
