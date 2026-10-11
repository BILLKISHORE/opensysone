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
from .calib.records import is_heldout, load_decisions, read_jsonl
from .calib.store import OutcomeStore
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
                    calibrator=_calibrator(args.calibration), adapter=args.adapter)
    engine.load()
    store = OutcomeStore(args.outcomes, feedback_out=args.feedback_out) if args.outcomes else None
    app = create_app(engine, api_key=args.api_key, max_queue=args.max_queue,
                     store=store, calibration_dir=args.calibration)
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
    engine = Engine(args.model, parity_check=not args.no_parity_check, calibrator=_calibrator(args.calibration),
                    adapter=args.adapter)
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
    engine = Engine(args.model, parity_check=not args.no_parity_check, adapter=args.adapter)
    engine.load()
    stats = run_collect(engine, cases, args.out, samples=args.samples, log=lambda m: print(m, flush=True))
    print(json.dumps(stats))
    return 0


def build_examples(cases, tokenizer, labels="hard"):
    from .train.examples import build_examples as build

    return build(cases, tokenizer, labels)


def cmd_lora(args) -> int:
    """Train a LoRA adapter on labeled cases; validation is the held-out share of the training cases."""
    from .train.lora import LoraConfig, attach_lora, save_adapter, train

    cases = list(read_jsonl(args.cases))
    if args.limit is not None:
        cases = cases[: args.limit]
    engine = Engine(args.model, parity_check=False)
    engine.load()
    examples, stats = build_examples(cases, engine._tokenizer, args.labels)
    # The outer held-out groups are the calibration fit's test set; checkpoint selection must not see them.
    pool = [e for e in examples if not is_heldout(e.group_id)]
    train_ex = [e for e in pool if not is_heldout(e.group_id, salt="inner")]
    val_ex = [e for e in pool if is_heldout(e.group_id, salt="inner")][: args.val_cases]
    cfg = LoraConfig(layers=args.layers, rank=args.rank, scale=args.scale, lr=args.lr, epochs=args.epochs,
                     warmup=args.warmup, eval_every=args.eval_every, seed=args.seed)
    print(f"examples {dict(stats)}; train {len(train_ex)} cases, val {len(val_ex)} cases", flush=True)
    attach_lora(engine._model, cfg)
    result = train(engine._model, train_ex, val_ex, cfg, log=lambda m: print(m, flush=True))
    sha = save_adapter(engine._model, args.out, cfg, meta={
        "backbone": args.model, "cases": str(args.cases), "n_train_cases": len(train_ex),
        "n_val_cases": len(val_ex), **result})
    print(f"adapter {sha} written to {args.out} (best val loss {result['best_val_loss']:.4f})")
    return 0


def cmd_fit(args) -> int:
    records = load_decisions(args.decisions)
    result = fit_calibrator(records, meta={"decisions": [str(p) for p in args.decisions]}, target=args.target,
                            split_salt=args.split_salt)
    sha = result.calibrator.save(args.out)
    Path(args.out, "fit_summary.json").write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n")
    print(f"calibration {sha} written to {args.out}")
    for kind, s in result.summary.items():
        b, a = s["before"], s["after"]
        print(f"{kind:9s} n={s['n_train']}+{s['n_heldout']} chosen={s['chosen']:11s} "
              f"ece {b['ece']:.3f} -> {a['ece']:.3f}  nll {b['nll']:.3f} -> {a['nll']:.3f}  "
              f"acc {b['accuracy']:.3f} -> {a['accuracy']:.3f}")
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


def cmd_crossfit(args) -> int:
    from .calib.crossfit import crossfit

    records = load_decisions(args.decisions)
    probs, _ = crossfit(records, folds=args.folds, target=args.target)
    ev = evaluate(records, probs=probs)
    note = (f"{CONSENSUS_NOTE} Calibrated columns are cross-fitted: {args.folds} folds by case, each decision "
            f"calibrated by a fit that never saw its case (target {args.target}).")
    md, files = render_report(ev, title=args.title, note=note)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md)
    for name, svg in files.items():
        (out / name).write_text(svg)
    (out / "metrics.json").write_text(json.dumps(ev, indent=2, sort_keys=True) + "\n")
    a = ev["all"]
    for side in ("raw", "calibrated"):
        m, ci = a[side], a[f"ci_{side}"]
        print(f"{side:10s} acc {m['accuracy']:.4f} ece {m['ece']:.4f} [{ci['ece'][0]:.4f}-{ci['ece'][1]:.4f}] "
              f"nll {m['nll']:.4f}")
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
    s.add_argument("--outcomes", default=None, help="SQLite file for decisions and feedback (enables audit/queue)")
    s.add_argument("--feedback-out", default=None, help="append labeled feedback as decision JSONL for `fit`")
    s.add_argument("--adapter", default=None, help="LoRA adapter directory from `opensysone lora`")
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
    d.add_argument("--adapter", default=None, help="LoRA adapter directory from `opensysone lora`")
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
    c.add_argument("--adapter", default=None, help="LoRA adapter directory from `opensysone lora`")
    c.set_defaults(func=cmd_collect)

    lo = sub.add_parser("lora", help="fine-tune a LoRA adapter on labeled cases (soft labels)")
    lo.add_argument("cases")
    lo.add_argument("--out", required=True)
    lo.add_argument("--model", default=DEFAULT_MODEL)
    lo.add_argument("--layers", type=int, default=16, help="adapt the last N transformer blocks")
    lo.add_argument("--rank", type=int, default=8)
    lo.add_argument("--scale", type=float, default=20.0)
    lo.add_argument("--lr", type=float, default=1e-4)
    lo.add_argument("--epochs", type=int, default=1)
    lo.add_argument("--warmup", type=int, default=20)
    lo.add_argument("--eval-every", type=int, default=100)
    lo.add_argument("--val-cases", type=int, default=60)
    lo.add_argument("--limit", type=int, default=None)
    lo.add_argument("--seed", type=int, default=0)
    lo.add_argument("--labels", choices=["hard", "soft"], default="hard",
                    help="train toward the winning label (default) or the vote shares")
    lo.set_defaults(func=cmd_lora)

    f = sub.add_parser("fit", help="fit calibration on decision records")
    f.add_argument("decisions", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--target", choices=["hard", "soft"], default="hard",
                   help="fit to the winning label (default) or to the label vote shares")
    f.add_argument("--split-salt", default="",
                   help="salt for the fit/held-out case split; use one (e.g. calib) when all records are held out "
                        "from LoRA training")
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("report", help="calibration report with reliability diagrams")
    r.add_argument("decisions", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--calibration", default=None)
    r.add_argument("--title", default="Calibration report")
    r.set_defaults(func=cmd_report)

    x = sub.add_parser("crossfit", help="cross-fitted calibration report: no separate training set needed")
    x.add_argument("decisions", nargs="+")
    x.add_argument("--out", required=True)
    x.add_argument("--folds", type=int, default=5)
    x.add_argument("--target", choices=["hard", "soft"], default="hard")
    x.add_argument("--title", default="Cross-fitted calibration report")
    x.set_defaults(func=cmd_crossfit)

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
