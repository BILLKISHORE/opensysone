"""Outcome store: served decisions, feedback labels, audit and the labeling queue."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

from ..engine import RAW_KEY
from ..schema import render_state
from ..wire import SystemOneRequest
from .metrics import ece, reliability
from .records import CALIBRATED_KINDS, append_jsonl, decision_line, normalize_label

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, state TEXT NOT NULL, stamp TEXT NOT NULL,
  ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (
  request_id TEXT NOT NULL, qid TEXT NOT NULL, kind TEXT NOT NULL, question TEXT NOT NULL,
  options TEXT NOT NULL, raw_probs TEXT NOT NULL, served_probs TEXT NOT NULL,
  confidence REAL NOT NULL, prompt_tokens INTEGER NOT NULL, ts REAL NOT NULL,
  PRIMARY KEY (request_id, qid));
CREATE TABLE IF NOT EXISTS outcomes (request_id TEXT NOT NULL, qid TEXT NOT NULL, label TEXT NOT NULL,
  ts REAL NOT NULL, PRIMARY KEY (request_id, qid));
CREATE INDEX IF NOT EXISTS decisions_conf ON decisions (confidence, ts);
"""


class UnknownDecision(KeyError):
    pass


def served_distribution(answer: dict[str, Any], options: list[str]) -> list[float]:
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return [p, 1.0 - p]
    probs = answer["probabilities"]
    return [float(probs[o]) for o in options]


def _block(rows: list[tuple[list[float], int]]) -> dict[str, Any]:
    conf = [max(p) for p, _ in rows]
    correct = [max(range(len(p)), key=lambda i: p[i]) == li for p, li in rows]
    n = len(rows)
    return {"n": n, "accuracy": (sum(correct) / n) if n else None,
            "ece": ece(conf, correct) if n else None, "reliability": reliability(conf, correct)}


class OutcomeStore:
    def __init__(self, path: str, *, feedback_out: str | None = None):
        self.path = path
        self.feedback_out = feedback_out
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()

    def record(self, request_id: str, req: SystemOneRequest, response: dict[str, Any]) -> int:
        raw = response.get(RAW_KEY) or {}
        now = time.time()
        rows = []
        for qid, q in (raw.get("questions") or {}).items():
            if q["source"] != "model" or q["kind"] not in CALIBRATED_KINDS:
                continue
            served = served_distribution(response["answers"][qid], q["options"])
            rows.append((request_id, qid, q["kind"], json.dumps(req.questions[qid].model_dump()),
                         json.dumps(q["options"]), json.dumps(q["probs"]), json.dumps(served),
                         max(served), int(raw.get("prompt_tokens", 0)), now))
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO requests VALUES (?, ?, ?, ?)",
                             (request_id, render_state(req.state), json.dumps(response.get("stamp", {})), now))
            self._db.executemany("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            self._db.commit()
        return len(rows)

    def add_outcome(self, request_id: str, qid: str, label: Any) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT d.kind, d.question, d.options, d.raw_probs, d.served_probs, d.prompt_tokens, r.stamp "
                "FROM decisions d JOIN requests r USING (request_id) WHERE d.request_id = ? AND d.qid = ?",
                (request_id, qid)).fetchone()
            if row is None:
                raise UnknownDecision(f"no recorded decision for request {request_id!r} question {qid!r}")
            kind, question, options_s, raw_s, served_s, prompt_tokens, stamp_s = row
            options = json.loads(options_s)
            label_s = normalize_label(label)
            if label_s not in options:
                raise ValueError(f"label {label_s!r} is not an option: {options}")
            self._db.execute("INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?)",
                             (request_id, qid, label_s, time.time()))
            self._db.commit()
        served = json.loads(served_s)
        correct = max(range(len(served)), key=lambda i: served[i]) == options.index(label_s)
        if self.feedback_out:
            append_jsonl(self.feedback_out, [decision_line(
                case_id=request_id, group_id=request_id, qid=qid, question=json.loads(question), kind=kind,
                options=options, probs=json.loads(raw_s), prompt_tokens=prompt_tokens, label=label_s,
                label_distribution=None, source="outcome", stamp=json.loads(stamp_s))])
        return {"correct": correct, "label": label_s}

    def audit(self, window: int = 1000) -> dict[str, Any]:
        with self._lock:
            rows = self._db.execute(
                "SELECT d.kind, d.options, d.served_probs, o.label FROM outcomes o "
                "JOIN decisions d USING (request_id, qid) ORDER BY o.ts DESC LIMIT ?", (window,)).fetchall()
        by_kind: dict[str, list[tuple[list[float], int]]] = {}
        everything: list[tuple[list[float], int]] = []
        for kind, options_s, served_s, label in rows:
            item = (json.loads(served_s), json.loads(options_s).index(label))
            by_kind.setdefault(kind, []).append(item)
            everything.append(item)
        return {"n": len(everything), "window": window, "overall": _block(everything),
                "by_kind": {k: _block(v) for k, v in sorted(by_kind.items())}}

    def queue(self, threshold: float = 0.6, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT d.request_id, d.qid, d.kind, d.question, d.options, d.served_probs, d.confidence, "
                "d.ts, r.state FROM decisions d JOIN requests r USING (request_id) "
                "LEFT JOIN outcomes o ON o.request_id = d.request_id AND o.qid = d.qid "
                "WHERE o.label IS NULL AND d.confidence < ? ORDER BY d.ts DESC, d.rowid DESC LIMIT ? OFFSET ?",
                (threshold, limit, offset)).fetchall()
        return [{"request_id": r[0], "qid": r[1], "kind": r[2], "question": json.loads(r[3]),
                 "options": json.loads(r[4]), "probabilities": json.loads(r[5]), "confidence": r[6],
                 "ts": r[7], "state": r[8]} for r in rows]
