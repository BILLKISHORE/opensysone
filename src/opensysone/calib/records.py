"""The one JSONL format every learning feature reads and writes.

A decision line is one question of one case: the question, the backbone's raw
(pre-calibration) distribution over its options, and the label.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CALIBRATED_KINDS = frozenset({"noul", "choice", "score", "pairwise"})
HELDOUT_MODULUS = 5  # 1 group in 5 is held out


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    group_id: str
    kind: str
    options: tuple[str, ...]
    probs: tuple[float, ...]
    prompt_tokens: int
    label: str
    label_distribution: tuple[float, ...] | None
    signature: str = ""  # question_signature(): the same question asked again gets the same id

    @property
    def label_index(self) -> int:
        return self.options.index(self.label)

    def target(self) -> list[float]:
        if self.label_distribution is not None:
            return list(self.label_distribution)
        return [1.0 if i == self.label_index else 0.0 for i in range(len(self.options))]


def question_signature(kind: str, instructions: Any, options: Iterable[str]) -> str:
    """A short stable id for one question: its type, its instructions and its options in prompt order.

    Option and level descriptions are not part of it, so two questions that differ only in those
    descriptions share one per-question calibration map.
    """
    text = instructions if isinstance(instructions, str) or instructions is None else \
        json.dumps(instructions, ensure_ascii=False, sort_keys=True)
    body = json.dumps([kind, text or "", list(options)], ensure_ascii=False)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e.msg}") from e


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        f.flush()
    return count


def is_heldout(group_id: str, salt: str = "") -> bool:
    digest = hashlib.sha1(f"{salt}{group_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % HELDOUT_MODULUS == 0


def decision_line(
    *,
    case_id: str,
    group_id: str,
    qid: str,
    question: dict[str, Any],
    kind: str,
    options: Iterable[str],
    probs: Iterable[float],
    prompt_tokens: int,
    label: Any,
    label_distribution: dict[str, float] | None,
    source: str,
    stamp: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{case_id}/{qid}",
        "group_id": group_id,
        "case_id": case_id,
        "qid": qid,
        "question": question,
        "answer": {"kind": kind, "options": list(options), "probs": [float(p) for p in probs],
                   "prompt_tokens": int(prompt_tokens)},
        "label": normalize_label(label),
        "label_distribution": label_distribution,
        "source": source,
        "stamp": stamp,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def parse_decision(line: dict[str, Any]) -> DecisionRecord:
    answer = line["answer"]
    options = tuple(str(o) for o in answer["options"])
    probs = tuple(float(p) for p in answer["probs"])
    if len(probs) != len(options):
        raise ValueError(f"{line.get('id')}: {len(probs)} probs for {len(options)} options")
    label = normalize_label(line["label"])
    if label not in options:
        raise ValueError(f"{line.get('id')}: label {label!r} not in options {list(options)}")
    aligned = None
    dist = line.get("label_distribution")
    if dist:
        raw = [float(dist.get(o, 0.0)) for o in options]
        z = sum(raw)
        aligned = tuple(x / z for x in raw) if z > 0 else None
    return DecisionRecord(
        id=str(line["id"]),
        group_id=str(line.get("group_id") or line["id"]),
        kind=str(answer["kind"]),
        options=options,
        probs=probs,
        prompt_tokens=int(answer.get("prompt_tokens", 0)),
        label=label,
        label_distribution=aligned,
        signature=question_signature(str(answer["kind"]), (line.get("question") or {}).get("instructions"), options),
    )


def load_decisions(
    paths: Iterable[str | Path], kinds: frozenset[str] = CALIBRATED_KINDS
) -> list[DecisionRecord]:
    out: list[DecisionRecord] = []
    for path in paths:
        for line in read_jsonl(path):
            if line["answer"]["kind"] in kinds:
                out.append(parse_decision(line))
    return out
