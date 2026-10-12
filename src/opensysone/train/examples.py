"""Labeled cases as the exact token rows the engine scores.

The engine prefills one prompt per case, then reads each question's option-code logits at one row
position, log-softmaxed over the allowed codes (jevmlx slot scoring). Training must see the same
prompt, the same rows and the same positions, or the adapter learns a different task than the one
it is served on. So this builds everything with jevmlx's own functions.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..calib.records import CALIBRATED_KINDS, is_heldout
from ..schema import compile_request
from ..wire import SystemOneRequest

SCORING = "slots"


@dataclass(frozen=True)
class FieldExample:
    qid: str
    row: tuple[int, ...]  # tokens scored after the shared prompt
    position: int  # index in `row` whose logits decide the question
    allowed: tuple[int, ...]  # option-code tokens read at that position, in read order
    target: tuple[float, ...]  # label mass on each allowed token
    options: tuple[str, ...] = ()  # the answer option each allowed token stands for


@dataclass(frozen=True)
class CaseExample:
    case_id: str
    group_id: str
    prompt: tuple[int, ...]
    fields: tuple[FieldExample, ...]


def _option_id(choice: Any) -> str:
    if isinstance(choice, bool):
        return "true" if choice else "false"
    return str(choice)


def _target(dist: dict[str, float] | None, label: Any, labels: str = "hard") -> dict[str, float]:
    if dist and labels == "soft":
        z = sum(dist.values())
        return {str(k): float(v) / z for k, v in dist.items()}
    return {_option_id(label): 1.0}


def build_case(case: dict[str, Any], tokenizer, stats: Counter | None = None,
               cache: dict | None = None, labels: str = "hard") -> CaseExample | None:
    """`cache` keeps the compiled rows per question set and the chat profile: both depend only on the
    questions and the tokenizer, and compiling them is most of the cost."""
    from jevmlx.engine import PROMPT_V2_SYSTEM, _build_schema_rows, _chat_ids, _resolve_profile, _user_content
    from jevmlx.schema import StructuredSchema

    stats = stats if stats is not None else Counter()
    cache = cache if cache is not None else {}
    req = SystemOneRequest.model_validate({"state": case["state"], "questions": case["questions"]})
    compiled = compile_request(req)
    key = json.dumps(compiled.schema_dict, sort_keys=True, default=str)
    if key not in cache:
        schema = StructuredSchema(compiled.schema_dict)
        cache[key] = (schema, _build_schema_rows(schema, tokenizer, SCORING))
    schema, built = cache[key]
    if "profile" not in cache:
        cache["profile"] = _resolve_profile(tokenizer)
    prompt = _chat_ids(tokenizer, _user_content(compiled.context, schema, tokenizer, SCORING), PROMPT_V2_SYSTEM,
                       cache["profile"])
    fields = []
    for qid, plan in compiled.plans.items():
        if plan.kind not in CALIBRATED_KINDS or qid not in case.get("labels", {}):
            stats["skipped_unlabeled_or_kind"] += 1
            continue
        rows = [i for i, f in enumerate(built["row_field"]) if f == qid]
        if len(rows) != 1:  # multi-token codes need several branch rows; not trained here
            stats["skipped_multi_row"] += 1
            continue
        ridx = rows[0]
        position, allowed = built["row_decision"][ridx]
        node = built["tries"][qid][built["row_branch"][ridx]]
        choices = [_option_id(c) for c in built["field_plans"][qid]["choices"]]
        target = _target((case.get("label_distributions") or {}).get(qid), case["labels"][qid], labels)
        mass = [sum(target.get(choices[ci], 0.0) for ci in node["children"][tok]) for tok in allowed]
        if abs(sum(mass) - 1.0) > 1e-6:
            raise ValueError(f"{case['id']}/{qid}: label options {sorted(target)} do not match {choices}")
        names = tuple("+".join(choices[ci] for ci in node["children"][tok]) for tok in allowed)
        fields.append(FieldExample(qid, tuple(built["rows"][ridx]), int(position), tuple(allowed), tuple(mass), names))
        stats["fields"] += 1
    if not fields:
        return None
    stats["cases"] += 1
    return CaseExample(case["id"], case.get("group_id", case["id"]), tuple(prompt), tuple(fields))


def build_examples(cases: Iterable[dict[str, Any]], tokenizer, labels: str = "hard") -> tuple[list[CaseExample], Counter]:
    """`labels="hard"` trains toward the winning label (what accuracy-based scores reward); `"soft"` toward
    the vote shares, which leaves the model less confident than its accuracy."""
    if labels not in ("hard", "soft"):
        raise ValueError(f"labels must be hard or soft, got {labels!r}")
    stats: Counter = Counter()
    cache: dict = {}
    out = [ex for c in cases if (ex := build_case(c, tokenizer, stats, cache, labels)) is not None]
    return out, stats


def split_train_val(examples: list[CaseExample], val_cases: int) -> tuple[list[CaseExample], list[CaseExample]]:
    """Train and validation cases from everything outside the outer held-out groups (the calibration test set).

    Validation takes whole inner held-out groups until it has `val_cases` cases; every other case trains,
    so no case is dropped and no group sits on both sides.
    """
    pool = [e for e in examples if not is_heldout(e.group_id)]
    train = [e for e in pool if not is_heldout(e.group_id, salt="inner")]
    val: list[CaseExample] = []
    val_groups: set[str] = set()
    for e in pool:
        if not is_heldout(e.group_id, salt="inner"):
            continue
        if e.group_id in val_groups or len(val) < val_cases:
            val_groups.add(e.group_id)
            val.append(e)
        else:
            train.append(e)
    return train, val
