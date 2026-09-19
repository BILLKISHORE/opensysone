"""Latency of one choice question as the option count grows. Needs the model."""
import json
import sys
import time

import mlx.core as mx

from opensysone.engine import Engine
from opensysone.wire import SystemOneRequest

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.8-27B-4bit"
OUT = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/results/options-qwen3.8-27b.json"
STATE = "Customer writes: my card was charged twice for order 8841 and support has not replied."


def main() -> None:
    engine = Engine(MODEL)
    engine.load()
    rows = []
    for n in (16, 64, 128, 255):
        criteria = {f"queue_{i:04d}": f"queue number {i}" for i in range(n - 1)}
        criteria["billing"] = "charges, refunds, double charges"
        req = SystemOneRequest.model_validate({
            "state": STATE, "cache": False,
            "questions": {"queue": {"type": "choice", "instructions": "Which queue?", "criteria": criteria}}})
        engine.decide(req)  # warm shapes
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = engine.decide(req)
        wall = (time.perf_counter() - t0) * 1000
        rows.append({
            "options": n,
            "wall_ms": round(wall, 1),
            "chose_billing": out["answers"]["queue"]["choice"] == "billing",
            "p_billing": round(out["answers"]["queue"]["probabilities"]["billing"], 4),
            "input_tokens": out["usage"]["input_tokens"],
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
            "max_rows": engine.max_rows,
        })
        print(rows[-1], flush=True)
    with open(OUT, "w") as f:
        json.dump({"model": MODEL, "stamp": engine.stamp, "parity": engine.parity, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
