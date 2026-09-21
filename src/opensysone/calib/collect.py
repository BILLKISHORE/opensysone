"""Run the backbone once over labeled cases and store raw answers as decision lines."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..engine import RAW_KEY
from ..wire import SystemOneRequest
from .records import CALIBRATED_KINDS, append_jsonl, decision_line, read_jsonl


def _done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return {line["case_id"] for line in read_jsonl(out_path)}


def collect(
    engine: Any,
    cases: Iterable[dict[str, Any]],
    out_path: str | Path,
    *,
    samples: int = 1,
    log: Callable[[str], None] | None = None,
) -> dict[str, int]:
    out_path = Path(out_path)
    done = _done(out_path)
    stats = {"cases": 0, "skipped": 0, "decisions": 0}
    for case in cases:
        if case["id"] in done:
            stats["skipped"] += 1
            continue
        t0 = time.perf_counter()
        req = SystemOneRequest.model_validate(
            {"state": case["state"], "questions": case["questions"], "samples": samples, "cache": False})
        out = engine.decide(req)
        raw = out[RAW_KEY]
        lines = []
        for qid, label in case["labels"].items():
            q = raw["questions"].get(qid)
            if q is None or q["source"] != "model" or q["kind"] not in CALIBRATED_KINDS:
                continue
            lines.append(decision_line(
                case_id=case["id"], group_id=case.get("group_id", case["id"]), qid=qid,
                question=case["questions"][qid], kind=q["kind"], options=q["options"], probs=q["probs"],
                prompt_tokens=raw["prompt_tokens"], label=label,
                label_distribution=(case.get("label_distributions") or {}).get(qid),
                source=case.get("source", "human"), stamp=out["stamp"]))
        stats["decisions"] += append_jsonl(out_path, lines)
        stats["cases"] += 1
        if log is not None:
            log(f"{case['id']}: {len(lines)} decisions in {time.perf_counter() - t0:.1f}s "
                f"({stats['cases']} done, {stats['skipped']} skipped)")
    return stats
