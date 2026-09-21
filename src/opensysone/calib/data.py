"""Labeled cases from LocalLLaMA/typed-decisions (TypeSafe's task, Apache-2.0).

Labels are frontier-model consensus, not human gold: a calibrated probability
fitted on them is the chance the consensus agrees.
"""
from __future__ import annotations

import json
from typing import Any

TD_REPO = "LocalLLaMA/typed-decisions"
TD_REVISION = "ea9306458d6e9563628369a3d1e72e362fb381d2"


def _obj(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def case_from_typed_decisions_row(row: dict[str, Any]) -> dict[str, Any]:
    gold = _obj(row["gold"])
    return {
        "id": row["id"],
        "group_id": row["id"],
        "workflow": row["workflow"],
        "split": row["split"],
        "state": _obj(row["state"]),
        "questions": _obj(row["questions"]),
        "labels": {qid: str(g["label"]) for qid, g in gold.items()},
        "label_distributions": {qid: {str(k): float(v) for k, v in g["probabilities"].items()}
                                for qid, g in gold.items()},
        "source": "teacher",
    }


def typed_decisions_cases(
    split: str, *, workflow: str = "all", revision: str = TD_REVISION, limit: int | None = None
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(TD_REPO, f"{workflow}/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=revision)
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda r: r["id"])
    if limit is not None:
        rows = rows[:limit]
    return [case_from_typed_decisions_row(r) for r in rows]
