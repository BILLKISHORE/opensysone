"""Pick the best calibration method per question type on held-out data."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .calibrators import (Calibrator, KindCalibration, ets_scale, fit_ets, fit_temperature,
                          temperature_scale)
from .dirichlet import DirichletMap, fit_dirichlet
from .features import decision_features
from .metrics import summarize
from .records import DecisionRecord, is_heldout

MIN_ADAPTIVE = 50
MIN_DIRICHLET = 50  # decisions in a (type, option count) group before it gets its own map
MIN_SIGNATURE = 100  # decisions of one recurring question before it gets its own map
PREFERENCE = ("identity", "temperature", "ets", "adaptive", "dirichlet")  # simpler wins a tie
MIN_NLL_GAIN = 0.002  # a method must lower held-out NLL by this much to replace raw probabilities
ECE_TOLERANCE = 0.0  # ...and must not raise held-out calibration error
ACCURACY_TOLERANCE = 0.0  # ...or lower held-out accuracy (dirichlet can change the winner)


@dataclass(frozen=True)
class FitResult:
    calibrator: Calibrator
    summary: dict[str, Any]


def choose_method(metrics: dict[str, dict[str, float]]) -> str:
    """Lowest held-out NLL among methods that keep calibration error and accuracy at least as good as raw.

    NLL rewards being both honest and decisive; selecting on calibration error alone would reward a
    model that always says the base rate. The guards stop a lower NLL from buying worse calibration
    or a lost answer.
    """
    raw = metrics["identity"]
    ok = [m for m, v in metrics.items()
          if v["ece"] <= raw["ece"] + ECE_TOLERANCE and v["accuracy"] >= raw["accuracy"] - ACCURACY_TOLERANCE]
    best = min(ok, key=lambda m: (round(metrics[m]["nll"], 6), PREFERENCE.index(m)))
    return best if metrics[best]["nll"] <= raw["nll"] - MIN_NLL_GAIN else "identity"


def _samples(records: list[DecisionRecord]) -> list[tuple[list[float], list[float], list[float]]]:
    return [(decision_features(r.kind, r.probs, r.prompt_tokens), list(r.probs), r.target()) for r in records]


def _fit_maps(train: list[DecisionRecord]) -> dict[str, DirichletMap]:
    """One Dirichlet map per recurring question and per (type, option count), regularization picked on an
    inner split of the training cases."""
    groups: dict[str, list[DecisionRecord]] = defaultdict(list)
    for r in train:
        groups[f"kind:{r.kind}:{len(r.options)}"].append(r)
        if r.signature:
            groups[f"sig:{r.signature}"].append(r)
    maps = {}
    for key, recs in sorted(groups.items()):
        if len(recs) < (MIN_SIGNATURE if key.startswith("sig:") else MIN_DIRICHLET):
            continue
        inner_train = [(r.probs, r.target()) for r in recs if not is_heldout(r.group_id, salt="inner")]
        inner_val = [(r.probs, r.target()) for r in recs if is_heldout(r.group_id, salt="inner")]
        if inner_train and inner_val:
            maps[key] = fit_dirichlet(inner_train, inner_val)
    return maps


def _candidates(held: list[DecisionRecord], t: float, ets_w: tuple[float, float, float], head: Any,
                maps: dict[str, DirichletMap], n_train: int) -> dict[str, list[list[float]]]:
    probe = Calibrator({}, maps=maps)
    out: dict[str, list[list[float]]] = {
        "identity": [list(r.probs) for r in held],
        "temperature": [temperature_scale(r.probs, t) for r in held],
        "ets": [ets_scale(r.probs, t, ets_w) for r in held],
    }
    if head is not None and n_train >= MIN_ADAPTIVE:
        out["adaptive"] = [
            temperature_scale(r.probs, head.temperature(decision_features(r.kind, r.probs, r.prompt_tokens)))
            for r in held
        ]
    if maps:
        dirichlet = []
        for r in held:
            m = probe.dirichlet_map(r.kind, len(r.probs), r.signature)
            dirichlet.append(m.apply(r.probs) if m is not None else temperature_scale(r.probs, t))
        out["dirichlet"] = dirichlet
    return out


def _used_maps(maps: dict[str, DirichletMap], kinds: Iterable[str], records: list[DecisionRecord]):
    sigs = {f"sig:{r.signature}" for r in records if r.kind in kinds and r.signature}
    return {k: m for k, m in maps.items()
            if k in sigs or (k.startswith("kind:") and k.split(":")[1] in kinds)}


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
    maps = _fit_maps(train_all)

    kinds: dict[str, KindCalibration] = {}
    summary: dict[str, Any] = {}
    for kind, recs in sorted(by_kind.items()):
        train = [r for r in recs if not is_heldout(r.group_id)]
        held = [r for r in recs if is_heldout(r.group_id)]
        raw = [list(r.probs) for r in held]
        if not train or not held:
            kinds[kind] = KindCalibration("identity", metrics={"note": "not enough data to hold out"})
            summary[kind] = {"n_train": len(train), "n_heldout": len(held), "heldout": {},
                             "heldout_nll": {}, "chosen": "identity", "temperature": 1.0,
                             "before": summarize(held, raw), "after": summarize(held, raw)}
            continue
        target = [(list(r.probs), r.target()) for r in train]
        t = fit_temperature(target)
        ets_w = fit_ets(target, t)
        candidates = _candidates(held, t, ets_w, head, maps, len(train))
        scored = {m: summarize(held, p) for m, p in candidates.items()}
        heldout = {m: {k: s[k] for k in ("nll", "ece", "ece_em15", "accuracy", "brier")} for m, s in scored.items()}
        chosen = choose_method(heldout)
        before, after = scored["identity"], scored[chosen]
        kinds[kind] = KindCalibration(chosen, t if chosen in ("temperature", "ets", "dirichlet") else 1.0,
                                      {"heldout": heldout, "ece_before": before["ece"], "ece_after": after["ece"]},
                                      ets_w if chosen == "ets" else None)
        summary[kind] = {"n_train": len(train), "n_heldout": len(held), "heldout": heldout,
                         "heldout_nll": {m: v["nll"] for m, v in heldout.items()},
                         "chosen": chosen, "temperature": t, "ets": ets_w,
                         "before": before, "after": after}
    uses_head = any(k.method == "adaptive" for k in kinds.values())
    dirichlet_kinds = {k for k, v in kinds.items() if v.method == "dirichlet"}
    used = _used_maps(maps, dirichlet_kinds, records) if dirichlet_kinds else {}
    return FitResult(Calibrator(kinds, head=head if uses_head else None, meta=meta, maps=used), summary)
