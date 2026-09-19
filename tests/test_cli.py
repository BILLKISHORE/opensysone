import json

from opensysone import cli


def test_lint_without_model_reports_findings(tmp_path, capsys):
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"team": {"type": "choice", "criteria": {"a": "", "A": ""}}}))
    rc = cli.main(["lint", "--questions", str(q)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "duplicate_option" in out


def test_lint_clean_returns_zero(tmp_path, capsys):
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"team": {"type": "choice", "instructions": "which", "criteria": {"a": "x", "b": "y"}}}))
    assert cli.main(["lint", "--questions", str(q)]) == 0
    assert "no findings" in capsys.readouterr().out


def test_decide_uses_engine_and_prints_json(tmp_path, capsys, monkeypatch):
    class E:
        def __init__(self, *a, **k):
            self.loaded = False

        def load(self):
            self.loaded = True

        def decide(self, req):
            return {"answers": {q: {"type": "noul", "noul": 1.0} for q in req.questions}}

    monkeypatch.setattr(cli, "Engine", E)
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"ok": {"type": "noul", "instructions": "fine?"}}))
    rc = cli.main(["decide", "--model", "m", "--questions", str(q), "--state", "all good"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["answers"]["ok"]["noul"] == 1.0


def test_serve_builds_engine_and_runs_uvicorn(monkeypatch):
    calls = {}

    class E:
        def __init__(self, model_id, **kw):
            calls["engine"] = (model_id, kw)

        def load(self):
            calls["loaded"] = True

    monkeypatch.setattr(cli, "Engine", E)
    monkeypatch.setattr(cli, "create_app", lambda engine, **kw: ("app", kw))
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kw: calls.setdefault("run", (app, kw)))
    assert cli.main(["serve", "--model", "m", "--port", "9000", "--api-key", "k", "--no-parity-check"]) == 0
    assert calls["engine"][0] == "m" and calls["engine"][1]["parity_check"] is False
    assert calls["loaded"] is True
    assert calls["run"][1]["port"] == 9000
