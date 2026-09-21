import pytest

from opensysone.calib.records import read_jsonl
from opensysone.calib.store import OutcomeStore, UnknownDecision, served_distribution
from opensysone.wire import SystemOneRequest

REQ = SystemOneRequest.model_validate({
    "state": "Card charged twice, want it fixed.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "b", "ops": "o"}},
        "urgent": {"type": "noul", "instructions": "Urgent?"},
        "tags": {"type": "multi", "instructions": "Tags?", "criteria": {"x": "", "y": ""}},
    },
})


def _response(team_p=0.9, urgent_p=0.55):
    return {
        "answers": {
            "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": team_p, "ops": 1 - team_p}},
            "urgent": {"type": "noul", "noul": urgent_p},
            "tags": {"type": "multi", "selected": [], "probabilities": {"x": 0.1, "y": 0.2}},
        },
        "stamp": {"backbone": "m", "calibration": "abc"},
        "_raw": {"prompt_tokens": 77, "questions": {
            "team": {"kind": "choice", "options": ["billing", "ops"], "probs": [0.97, 0.03], "source": "model"},
            "urgent": {"kind": "noul", "options": ["true", "false"], "probs": [0.6, 0.4], "source": "model"},
            "tags": {"kind": "multi", "options": ["x", "y"], "probs": [0.1, 0.2], "source": "model"},
        }},
    }


def test_served_distribution():
    assert served_distribution({"type": "noul", "noul": 0.8}, ["true", "false"]) == pytest.approx([0.8, 0.2])
    assert served_distribution({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}}, ["a", "b"]) == [0.3, 0.7]


def test_record_skips_multi_and_rules(tmp_path):
    store = OutcomeStore(str(tmp_path / "o.sqlite"))
    assert store.record("r1", REQ, _response()) == 2


def test_feedback_writes_raw_probs_line_and_reports_correctness(tmp_path):
    fb = tmp_path / "feedback.jsonl"
    store = OutcomeStore(str(tmp_path / "o.sqlite"), feedback_out=str(fb))
    store.record("r1", REQ, _response())
    assert store.add_outcome("r1", "team", "ops") == {"correct": False, "label": "ops"}
    assert store.add_outcome("r1", "urgent", True) == {"correct": True, "label": "true"}
    lines = list(read_jsonl(fb))
    assert [line["id"] for line in lines] == ["r1/team", "r1/urgent"]
    assert lines[0]["answer"]["probs"] == [0.97, 0.03] and lines[0]["source"] == "outcome"
    assert lines[0]["answer"]["prompt_tokens"] == 77


def test_feedback_errors(tmp_path):
    store = OutcomeStore(str(tmp_path / "o.sqlite"))
    store.record("r1", REQ, _response())
    with pytest.raises(UnknownDecision):
        store.add_outcome("nope", "team", "ops")
    with pytest.raises(UnknownDecision):
        store.add_outcome("r1", "tags", "x")
    with pytest.raises(ValueError, match="not an option"):
        store.add_outcome("r1", "team", "sales")


def test_audit_uses_served_confidence(tmp_path):
    store = OutcomeStore(str(tmp_path / "o.sqlite"))
    for i in range(10):
        store.record(f"r{i}", REQ, _response(team_p=0.9))
        store.add_outcome(f"r{i}", "team", "billing" if i < 5 else "ops")
    a = store.audit(window=1000)
    assert a["n"] == 10
    team = a["by_kind"]["choice"]
    assert team["accuracy"] == 0.5 and team["ece"] == pytest.approx(0.4)
    assert a["overall"]["n"] == 10


def test_queue_lists_unlabeled_low_confidence_newest_first(tmp_path):
    store = OutcomeStore(str(tmp_path / "o.sqlite"))
    store.record("old", REQ, _response(urgent_p=0.55))
    store.record("new", REQ, _response(urgent_p=0.52))
    store.add_outcome("old", "urgent", True)
    items = store.queue(threshold=0.6)
    assert [(i["request_id"], i["qid"]) for i in items] == [("new", "urgent")]
    assert items[0]["state"] == REQ.state and items[0]["options"] == ["true", "false"]
    assert store.queue(threshold=0.6, offset=1) == []
