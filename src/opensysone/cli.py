"""opensysone command line: serve, decide, lint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from .cache import DecisionCache
from .calib.calibrators import Calibrator
from .calib.collect import collect as run_collect
from .calib.data import TD_REVISION, typed_decisions_cases
from .calib.fit import fit_calibrator
from .calib.records import load_decisions, read_jsonl
from .calib.report import evaluate, render_report
from .engine import Engine
from .lint import lint_compiled
from .schema import SchemaError, compile_request
from .server import create_app
from .wire import SystemOneRequest

DEFAULT_MODEL = "mlx-community/Qwen3.8-27B-4bit"
CONSENSUS_NOTE = ("Labels in typed-decisions are frontier-model consensus, not human gold. "
                  "A calibrated probability here is the chance the consensus agrees.")


def _questions(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _state(args) -> str:
    if args.state_file:
        return Path(args.state_file).read_text()
    return args.state


def _calibrator(path: str | None):
    return Calibrator.load(path) if path else None


def _write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cmd_serve(args) -> int:
    cache = DecisionCache(path=args.cache_path) if args.cache_path else DecisionCache()
    engine = Engine(args.model, parity_check=not args.no_parity_check, cache=cache,
                    calibrator=_calibrator(args.calibration))
    engine.load()
    app = create_app(engine, api_key=args.api_key, max_queue=args.max_queue)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_decide(args) -> int:
    req = SystemOneRequest.model_validate({
        "state": _state(args),
        "questions": _questions(args.questions),
        "samples": args.samples,
        "prior_correction": args.prior_correction,
        "cache": not args.no_cache,
    })
    engine = Engine(args.model, parity_check=not args.no_parity_check, calibrator=_calibrator(args.calibration))
    engine.load()
    try:
        out = engine.decide(req)
    except SchemaError as e:
        print(json.dumps({"error": e.msg, "loc": e.loc}), file=sys.stderr)
        return 2
    out = {k: v for k, v in out.items() if k != "_raw"}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_lint(args) -> int:
    req = SystemOneRequest.model_validate({"state": "", "questions": _questions(args.questions)})
    try:
        compiled = compile_request(req)
    except SchemaError as e:
        print(f"schema error at {e.loc}: {e.msg}")
        return 2
    tokenizer = None
    if args.model:
        engine = Engine(args.model, parity_check=False)
        engine.load()
        tokenizer = engine._tokenizer
    findings = lint_compiled(compiled, tokenizer)
    if not findings:
        print("no findings")
        return 0
    for f in findings:
        print(f"{f.qid}: {f.code}: {f.message}")
    return 1


def cmd_data(args) -> int:
    cases = typed_decisions_cases(args.split, workflow=args.workflow, revision=args.revision, limit=args.limit)
    _write_jsonl(args.out, cases)
    print(f"wrote {len(cases)} cases to {args.out}")
    return 0


def cmd_collect(args) -> int:
    cases = list(read_jsonl(args.cases))
    if args.limit is not None:
        cases = cases[: args.limit]
    engine = Engine(args.model, parity_check=not args.no_parity_check)
    engine.load()
    stats = run_collect(engine, cases, args.out, samples=args.samples, log=lambda m: print(m, flush=True))
    print(json.dumps(stats))
    return 0


def cmd_fit(args) -> int:
    records = load_decisions(args.decisions)
    result = fit_calibrator(records, meta={"decisions": [str(p) for p in args.decisions]})
    sha = result.calibrator.save(args.out)
    Path(args.out, "fit_summary.json").write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n")
    print(f"calibration {sha} written to {args.out}")
    for kind, s in result.summary.items():
        print(f"{kind:9s} n={s['n_train']}+{s['n_heldout']} chosen={s['chosen']:11s} "
              f"ece {s['before']['ece']:.3f} -> {s['after']['ece']:.3f}")
    return 0


def cmd_report(args) -> int:
    ev = evaluate(load_decisions(args.decisions), _calibrator(args.calibration))
    md, files = render_report(ev, title=args.title, note=CONSENSUS_NOTE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md)
    for name, svg in files.items():
        (out / name).write_text(svg)
    (out / "metrics.json").write_text(json.dumps(ev, indent=2, sort_keys=True) + "\n")
    print(f"report written to {out / 'report.md'}")
    return 0


def cmd_test(args) -> int:
    min_acc, max_ece = args.min_accuracy, args.max_ece
    if args.gate:
        gate = json.loads(Path(args.gate).read_text())
        min_acc = gate.get("min_accuracy", min_acc)
        max_ece = gate.get("max_ece", max_ece)
    ev = evaluate(load_decisions(args.decisions), _calibrator(args.calibration))["all"]["calibrated"]
    failures = []
    if min_acc is not None and ev["accuracy"] < min_acc:
        failures.append(f"accuracy {ev['accuracy']:.3f} < {min_acc}")
    if max_ece is not None and ev["ece"] > max_ece:
        failures.append(f"ece {ev['ece']:.3f} > {max_ece}")
    print(f"n={ev['n']} accuracy={ev['accuracy']:.3f} ece={ev['ece']:.3f} nll={ev['nll']:.3f}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opensysone")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the Jev-compatible server")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--api-key", default=None)
    s.add_argument("--cache-path", default=None)
    s.add_argument("--max-queue", type=int, default=64)
    s.add_argument("--no-parity-check", action="store_true")
    s.add_argument("--calibration", default=None, help="calibration directory from `opensysone fit`")
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("decide", help="decide one state from the terminal")
    d.add_argument("--model", default=DEFAULT_MODEL)
    d.add_argument("--questions", required=True, help="JSON file: {qid: question}")
    g = d.add_mutually_exclusive_group(required=True)
    g.add_argument("--state")
    g.add_argument("--state-file")
    d.add_argument("--samples", type=int, default=1)
    d.add_argument("--prior-correction", action="store_true")
    d.add_argument("--no-cache", action="store_true")
    d.add_argument("--no-parity-check", action="store_true")
    d.add_argument("--calibration", default=None, help="calibration directory from `opensysone fit`")
    d.set_defaults(func=cmd_decide)

    l = sub.add_parser("lint", help="check a questions file without calling the model")
    l.add_argument("--questions", required=True)
    l.add_argument("--model", default=None, help="also run token-level lint with this model's tokenizer")
    l.set_defaults(func=cmd_lint)

    da = sub.add_parser("data", help="build labeled cases")
    da_sub = da.add_subparsers(dest="dataset", required=True)
    td = da_sub.add_parser("typed-decisions", help="LocalLLaMA/typed-decisions (TypeSafe task)")
    td.add_argument("--split", choices=["train", "test"], required=True)
    td.add_argument("--out", required=True)
    td.add_argument("--workflow", default="all")
    td.add_argument("--limit", type=int, default=None)
    td.add_argument("--revision", default=TD_REVISION)
    td.set_defaults(func=cmd_data)

    c = sub.add_parser("collect", help="run the backbone over labeled cases (resumable)")
    c.add_argument("cases")
    c.add_argument("--out", required=True)
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--samples", type=int, default=1)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--no-parity-check", action="store_true")
    c.set_defaults(func=cmd_collect)

    f = sub.add_parser("fit", help="fit calibration on decision records")
    f.add_argument("decisions", nargs="+")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("report", help="calibration report with reliability diagrams")
    r.add_argument("decisions", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--calibration", default=None)
    r.add_argument("--title", default="Calibration report")
    r.set_defaults(func=cmd_report)

    t = sub.add_parser("test", help="fail when accuracy or calibration regresses")
    t.add_argument("decisions", nargs="+")
    t.add_argument("--calibration", default=None)
    t.add_argument("--min-accuracy", type=float, default=None)
    t.add_argument("--max-ece", type=float, default=None)
    t.add_argument("--gate", default=None, help='JSON file: {"min_accuracy": x, "max_ece": y}')
    t.set_defaults(func=cmd_test)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
