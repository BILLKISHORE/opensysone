import json

from opensysone import cli
from opensysone.calib.records import append_jsonl, decision_line


def _write_decisions(path, n=300, top=0.95, acc=0.6):
    rows = []
    for i in range(n):
        label = "a" if (i * 7919) % 100 < acc * 100 else "b"
        rows.append(decision_line(case_id=f"c{i}", group_id=f"c{i}", qid="q", question={"type": "choice"},
                                  kind="choice", options=["a", "b"], probs=[top, 1 - top], prompt_tokens=80,
                                  label=label, label_distribution=None, source="teacher", stamp={}))
    append_jsonl(path, rows)


def test_fit_report_and_test_pipeline(tmp_path, capsys):
    d = tmp_path / "dec.jsonl"
    _write_decisions(d)
    assert cli.main(["fit", str(d), "--out", str(tmp_path / "cal")]) == 0
    assert (tmp_path / "cal" / "calibration.json").exists()
    summary = json.loads((tmp_path / "cal" / "fit_summary.json").read_text())
    assert summary["choice"]["after"]["ece"] < summary["choice"]["before"]["ece"]
    assert cli.main(["report", str(d), "--out", str(tmp_path / "rep"), "--calibration", str(tmp_path / "cal")]) == 0
    assert (tmp_path / "rep" / "report.md").exists() and (tmp_path / "rep" / "reliability-choice.svg").exists()
    metrics = json.loads((tmp_path / "rep" / "metrics.json").read_text())
    assert metrics["all"]["calibrated"]["ece"] < metrics["all"]["raw"]["ece"]
    capsys.readouterr()
    assert cli.main(["test", str(d), "--calibration", str(tmp_path / "cal"), "--min-accuracy", "0.5",
                     "--max-ece", "0.2"]) == 0
    assert cli.main(["test", str(d), "--min-accuracy", "0.5", "--max-ece", "0.05"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_test_reads_gate_file(tmp_path):
    d = tmp_path / "dec.jsonl"
    _write_decisions(d)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"min_accuracy": 0.99, "max_ece": 1.0}))
    assert cli.main(["test", str(d), "--gate", str(gate)]) == 1


def test_collect_uses_engine_and_limit(tmp_path, monkeypatch):
    calls = {}

    class E:
        def __init__(self, model_id, **kw):
            calls["model"] = model_id

        def load(self):
            calls["loaded"] = True

    monkeypatch.setattr(cli, "Engine", E)
    monkeypatch.setattr(cli, "run_collect", lambda engine, cases, out, samples, log: calls.update(
        {"n": len(list(cases)), "out": str(out), "samples": samples}) or {"cases": 1, "skipped": 0, "decisions": 5})
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps({"id": f"x{i}", "state": "s", "questions": {}, "labels": {}})
                               for i in range(3)) + "\n")
    rc = cli.main(["collect", str(cases), "--out", str(tmp_path / "d.jsonl"), "--model", "m", "--limit", "2",
                   "--samples", "2"])
    assert rc == 0 and calls["model"] == "m" and calls["n"] == 2 and calls["samples"] == 2


def test_data_command_writes_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "typed_decisions_cases",
                        lambda split, workflow, revision, limit: [{"id": "a"}, {"id": "b"}][: limit or 2])
    out = tmp_path / "cases.jsonl"
    assert cli.main(["data", "typed-decisions", "--split", "test", "--out", str(out), "--limit", "1"]) == 0
    assert out.read_text().count("\n") == 1
