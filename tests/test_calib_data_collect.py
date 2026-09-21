import json
from pathlib import Path

from opensysone.calib.collect import collect
from opensysone.calib.data import case_from_typed_decisions_row
from opensysone.calib.records import load_decisions, read_jsonl
from opensysone.wire import SystemOneRequest

ROW = json.loads((Path(__file__).parent / "fixtures" / "typed_decisions_row.json").read_text())


def test_row_becomes_a_valid_case():
    case = case_from_typed_decisions_row(ROW)
    assert case["id"] == "customer_service_000000" and case["group_id"] == case["id"]
    assert set(case["questions"]) == {"action", "category", "churn_risk", "needs_human", "urgency"}
    assert case["labels"]["needs_human"] == "false" and case["labels"]["urgency"] == "1"
    assert abs(sum(case["label_distributions"]["action"].values()) - 1.0) < 1e-3
    assert isinstance(case["state"], dict) and case["source"] == "teacher"
    SystemOneRequest.model_validate({"state": case["state"], "questions": case["questions"]})


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def decide(self, req):
        self.calls += 1
        qs = {}
        for qid, q in req.questions.items():
            if q.type == "noul":
                qs[qid] = {"kind": "noul", "options": ["true", "false"], "probs": [0.3, 0.7], "source": "model"}
            elif q.type == "score":
                n = len(q.criteria)
                qs[qid] = {"kind": "score", "options": [str(i) for i in range(n)], "probs": [1.0 / n] * n,
                           "source": "model"}
            else:
                names = list(q.criteria)
                qs[qid] = {"kind": "choice", "options": names, "probs": [1.0 / len(names)] * len(names),
                           "source": "model"}
        return {"answers": {}, "stamp": {"backbone": "fake"}, "_raw": {"prompt_tokens": 99, "questions": qs}}


def test_collect_writes_one_line_per_labeled_question_and_resumes(tmp_path):
    out = tmp_path / "decisions.jsonl"
    cases = [case_from_typed_decisions_row(ROW)]
    engine = FakeEngine()
    stats = collect(engine, cases, out)
    assert stats == {"cases": 1, "skipped": 0, "decisions": 5}
    lines = list(read_jsonl(out))
    assert {line["qid"] for line in lines} == {"action", "category", "churn_risk", "needs_human", "urgency"}
    assert all(line["answer"]["prompt_tokens"] == 99 and line["source"] == "teacher" for line in lines)
    recs = load_decisions([out])
    assert len(recs) == 5 and all(r.label_distribution is not None for r in recs)
    again = collect(engine, cases, out)
    assert again == {"cases": 0, "skipped": 1, "decisions": 0}
    assert engine.calls == 1
