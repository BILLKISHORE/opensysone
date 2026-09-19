"""One loaded backbone, one decision at a time, Jev-shaped responses."""
from __future__ import annotations

import logging
import time
from typing import Any

from . import __version__
from .answers import average_distributions, distribution_from_telemetry, permute_choices, shape_answer
from .cache import DecisionCache, cache_key
from .lint import LintFinding, lint_compiled
from .rules import apply_rules
from .schema import SchemaError, compile_request
from .stamp import build_stamp
from .wire import SystemOneRequest

MODEL_VERSION = f"opensysone-{__version__}"
MODEL_ALIASES = frozenset({"opensysone-latest", MODEL_VERSION, "jev-latest", "jev-preview"})
_TIMING_KEYS = ("prefill_ms", "suffix_eval_ms", "elapsed_ms", "passes")

logger = logging.getLogger(__name__)


def _load(model_id: str):
    from jevmlx.engine import load_engine

    return load_engine(model_id)


def _parity(model, tokenizer) -> dict[str, Any]:
    from jevmlx.parity import check_scoring_parity

    return check_scoring_parity(model, tokenizer)


def _run_generation(model, tokenizer, context: str, schema, **kw) -> dict[str, Any]:
    from jevmlx.engine import run_parallel_generation

    return run_parallel_generation(model, tokenizer, context, schema, **kw)


def _structured(schema_dict: dict[str, dict[str, Any]]):
    from jevmlx.schema import StructuredSchema

    return StructuredSchema(schema_dict)


class Engine:
    def __init__(
        self,
        model_id: str,
        *,
        scoring: str = "slots",
        parity_check: bool = True,
        max_rows: int | None = None,
        cache: DecisionCache | None = None,
    ):
        self.model_id = model_id
        self.scoring = scoring
        self.parity_check = parity_check
        self.max_rows = max_rows
        self.cache = cache
        self.parity: dict[str, Any] | None = None
        self.stamp: dict[str, Any] = {}
        self.ready = False
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        self._model, self._tokenizer = _load(self.model_id)
        self.stamp = build_stamp(self.model_id, None)
        if self.parity_check and self.max_rows is None:
            self.parity = _parity(self._model, self._tokenizer)
            if not self.parity.get("passed", False):
                self.max_rows = 1
                logger.warning(
                    "scoring parity failed (drift %.3f nats): batched scoring disabled, one row per pass",
                    self.parity.get("max_abs_drift_nats", float("nan")),
                )
        self.ready = True

    def lint(self, req: SystemOneRequest) -> list[LintFinding]:
        return lint_compiled(compile_request(req), self._tokenizer)

    def decide(self, req: SystemOneRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        compiled = compile_request(req)
        findings = lint_compiled(compiled, self._tokenizer)
        warnings = [f"{f.qid}: {f.code}: {f.message}" for f in findings]
        if req.strict and warnings:
            raise SchemaError(["body", "questions"], "; ".join(warnings))
        ruled = apply_rules(req, compiled.plans)
        options = {
            "samples": req.samples,
            "prior_correction": req.prior_correction,
            "scoring": self.scoring,
            "max_rows": self.max_rows,
        }
        key = cache_key(
            compiled.context, compiled.schema_dict, compiled.constraints, options,
            self.stamp, {q: pattern for q, (_, pattern) in ruled.items()},
        )
        if req.cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                out = dict(hit)
                out["cached"] = True
                return out
        dists: dict[str, list[float]] = {q: d for q, (d, _) in ruled.items()}
        # Constraints may reference ruled questions, so keep every field in the
        # model schema when constraints are present; rule answers still win.
        pending = {q: s for q, s in compiled.schema_dict.items() if compiled.constraints or q not in ruled}
        input_tokens = 0
        timing: dict[str, Any] = {}
        if pending:
            seed = int(key[:8], 16)
            per_sample: list[dict[str, list[float]]] = []
            for s in range(req.samples):
                permuted = {q: permute_choices(spec, seed, s) for q, spec in pending.items()}
                result = _run_generation(
                    self._model, self._tokenizer, compiled.context, _structured(permuted),
                    max_rows=self.max_rows, scoring=self.scoring,
                    prior_correction=req.prior_correction,
                    constraints=compiled.constraints or None,
                )
                input_tokens = int(result.get("prompt_tokens", 0))
                timing = {k: result.get(k) for k in _TIMING_KEYS}
                per_sample.append({
                    q: distribution_from_telemetry(compiled.plans[q], result["field_telemetry"][q])
                    for q in pending if q not in ruled
                })
            for q in pending:
                if q not in ruled:
                    dists[q] = average_distributions([ps[q] for ps in per_sample])
        answers: dict[str, Any] = {}
        for qid, plan in compiled.plans.items():
            ans = shape_answer(plan, dists[qid])
            if qid in ruled:
                ans["source"] = "rule"
                ans["rule"] = ruled[qid][1]
            answers[qid] = ans
        response = {
            "model": MODEL_VERSION,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "stamp": self.stamp,
            "cached": False,
            "warnings": warnings,
            "timing": {
                **timing,
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
                "samples": req.samples,
                "max_rows": self.max_rows,
            },
        }
        if req.cache and self.cache is not None:
            self.cache.put(key, response)
        return response

    def decide_bulk(self, reqs: list[SystemOneRequest]) -> list[dict[str, Any]]:
        return [self.decide(r) for r in reqs]
