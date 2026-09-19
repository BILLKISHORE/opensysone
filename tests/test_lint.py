from dataclasses import dataclass

from opensysone.lint import LintFinding, lint_compiled
from opensysone.schema import compile_request
from opensysone.wire import SystemOneRequest


def _lint(questions):
    return lint_compiled(compile_request(SystemOneRequest.model_validate({"state": "s", "questions": questions})))


def _codes(findings):
    return sorted(f.code for f in findings)


def test_clean_request_has_no_findings():
    assert _lint({"team": {"type": "choice", "instructions": "Which team?",
                           "criteria": {"outage": "down", "billing": "money"}}}) == []


def test_duplicate_and_near_duplicate_options():
    f = _lint({"team": {"type": "choice", "instructions": "x",
                        "criteria": {"Billing": "a", "billing": "b", "billings": "c"}}})
    assert "duplicate_option" in _codes(f)
    assert "near_duplicate" in _codes(f)
    assert all(isinstance(x, LintFinding) and x.qid == "team" for x in f)


def test_empty_instructions_flagged_when_only_qid_fallback():
    assert _codes(_lint({"q": {"type": "noul"}})) == ["empty_instructions"]


def test_many_options_warning():
    crit = {f"opt{i}": "" for i in range(200)}
    assert "many_options" in _codes(_lint({"big": {"type": "choice", "instructions": "x", "criteria": crit}}))


def test_tokenizer_findings_are_prefixed(monkeypatch):
    import opensysone.lint as lint_mod

    @dataclass(frozen=True)
    class F:
        field: str
        kind: str
        message: str
        suggestion: str | None = None

    monkeypatch.setattr(lint_mod, "_jevmlx_findings",
                        lambda compiled, tokenizer: [F("team", "collision", "a and b share a token", "rename b")])
    req = SystemOneRequest.model_validate(
        {"state": "s", "questions": {"team": {"type": "choice", "instructions": "x", "criteria": {"a": "", "bb": ""}}}})
    f = lint_compiled(compile_request(req), tokenizer=object())
    assert [x.code for x in f] == ["token_collision"]
    assert "rename b" in f[0].message
