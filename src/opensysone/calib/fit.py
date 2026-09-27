"""Pick the best calibration method per question type on held-out data."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .calibrators import Calibrator, KindCalibration, fit_temperature, temperature_scale
from .features import decision_features
from .metrics import soft_nll, summarize
from .records import DecisionRecord, is_heldout

MIN_ADAPTIVE = 50
PREFERENCE = ("identity", "temperature", "adaptive")  # simpler wins a tie
MIN_ECE_GAIN = 0.005  # a method must beat raw probabilities by this much to be used


@dataclass(frozen=True)
class FitResult:
    calibrator: Calibrator
    summary: dict[str, Any]


def _mean_nll(records: list[DecisionRecord], probs_list: list[list[float]]) -> float:
    return sum(soft_nll(p, r.target()) for r, p in zip(records, probs_list)) / len(records)


def _samples(records: list[DecisionRecord]) -> list[tuple[list[float], list[float], list[float]]]:
    return [(decision_features(r.kind, r.probs, r.prompt_tokens), list(r.probs), r.target()) for r in records]


def fit_calibrator(records: list[DecisionRecord], *, meta: dict[str, Any] | None = None, seed: int = 0) -> FitResult:
    by_kind: dict[str, list[DecisionRecord]] = defaultdict(list)
    for r in records:
        by_kind[r.kind].append(r)
    train_all = [r for r in records if not is_heldout(r.group_id)]
    head = None
    fit_train = [r for r in train_all if not is_heldout(r.group_id, salt="inner")]
    fit_val = [r for r in train_all if is_heldout(r.group_id, salt="inner")]
    if len(fit_train) >= MIN_ADAPTIVE and fit_val:
        from .head import fit_adaptive

        head, _ = fit_adaptive(_samples(fit_train), _samples(fit_val), seed=seed)

    kinds: dict[str, KindCalibration] = {}
    summary: dict[str, Any] = {}
    for kind, recs in sorted(by_kind.items()):
        train = [r for r in recs if not is_heldout(r.group_id)]
        held = [r for r in recs if is_heldout(r.group_id)]
        raw = [list(r.probs) for r in held]
        if not train or not held:
            kinds[kind] = KindCalibration("identity", metrics={"note": "not enough data to hold out"})
            summary[kind] = {"n_train": len(train), "n_heldout": len(held), "heldout_nll": {},
                             "chosen": "identity", "temperature": 1.0,
                             "before": summarize(held, raw), "after": summarize(held, raw)}
            continue
        t = fit_temperature([(list(r.probs), r.target()) for r in train])
        candidates: dict[str, list[list[float]]] = {
            "identity": raw,
            "temperature": [temperature_scale(r.probs, t) for r in held],
        }
        if head is not None and len(train) >= MIN_ADAPTIVE:
            candidates["adaptive"] = [
                temperature_scale(r.probs, head.temperature(decision_features(r.kind, r.probs, r.prompt_tokens)))
                for r in held
            ]
        nlls = {m: _mean_nll(held, p) for m, p in candidates.items()}
        # Selection is on held-out calibration error, the number the report
        # leads with. A candidate must beat raw probabilities by MIN_ECE_GAIN
        # or the kind keeps them: on a backbone that is already calibrated,
        # the honest answer is to change nothing.
        eces = {m: summarize(held, p)["ece"] for m, p in candidates.items()}
        ranked = min(eces, key=lambda m: (round(eces[m], 4), PREFERENCE.index(m)))
        chosen = ranked if eces[ranked] <= eces["identity"] - MIN_ECE_GAIN else "identity"
        before, after = summarize(held, raw), summarize(held, candidates[chosen])
        kinds[kind] = KindCalibration(chosen, t if chosen == "temperature" else 1.0,
                                      {"heldout_nll": nlls, "heldout_ece": eces,
                                       "ece_before": before["ece"], "ece_after": after["ece"]})
        summary[kind] = {"n_train": len(train), "n_heldout": len(held), "heldout_nll": nlls,
                         "heldout_ece": eces, "chosen": chosen, "temperature": t,
                         "before": before, "after": after}
    uses_head = any(k.method == "adaptive" for k in kinds.values())
    return FitResult(Calibrator(kinds, head=head if uses_head else None, meta=meta), summary)
