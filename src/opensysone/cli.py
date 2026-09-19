"""opensysone command line: serve, decide, lint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from .cache import DecisionCache
from .engine import Engine
from .lint import lint_compiled
from .schema import SchemaError, compile_request
from .server import create_app
from .wire import SystemOneRequest

DEFAULT_MODEL = "mlx-community/Qwen3.8-27B-4bit"


def _questions(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _state(args) -> str:
    if args.state_file:
        return Path(args.state_file).read_text()
    return args.state


def cmd_serve(args) -> int:
    cache = DecisionCache(path=args.cache_path) if args.cache_path else DecisionCache()
    engine = Engine(args.model, parity_check=not args.no_parity_check, cache=cache)
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
    engine = Engine(args.model, parity_check=not args.no_parity_check)
    engine.load()
    try:
        out = engine.decide(req)
    except SchemaError as e:
        print(json.dumps({"error": e.msg, "loc": e.loc}), file=sys.stderr)
        return 2
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
    d.set_defaults(func=cmd_decide)

    l = sub.add_parser("lint", help="check a questions file without calling the model")
    l.add_argument("--questions", required=True)
    l.add_argument("--model", default=None, help="also run token-level lint with this model's tokenizer")
    l.set_defaults(func=cmd_lint)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
