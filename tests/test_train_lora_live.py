"""Live checks for LoRA training on the real backbone: run with `pytest -m live -s tests/test_train_lora_live.py`."""
import json
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.live
MODEL = "mlx-community/Qwen3.8-27B-4bit"
DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def engine():
    from opensysone.engine import Engine

    e = Engine(MODEL, parity_check=False, max_rows=1)
    e.load()
    return e


def _cases(split, picks):
    rows = [json.loads(line) for line in open(DATA / f"td-{split}.cases.jsonl")]
    return [rows[i] for i in picks]


def test_training_rows_read_exactly_the_served_probabilities(engine):
    import mlx.core as mx
    from jevmlx.engine import _broadcast_cache
    from mlx_lm.models.cache import make_prompt_cache

    from opensysone.train.examples import build_examples
    from opensysone.wire import SystemOneRequest

    cases = _cases("test", [0, 100, 200, 300])  # one case per workflow
    examples, _ = build_examples(cases, engine._tokenizer)
    worst = 0.0
    for case, ex in zip(cases, examples):
        req = SystemOneRequest.model_validate({"state": case["state"], "questions": case["questions"], "cache": False})
        served = engine.decide(req)["_raw"]["questions"]
        cache = make_prompt_cache(engine._model)
        engine._model(mx.array(ex.prompt)[None], cache=cache)
        for f in ex.fields:
            logits = engine._model(mx.array(f.row)[None], cache=_broadcast_cache(cache, 1))[0, f.position]
            mine = dict(zip(f.options, mx.softmax(logits[mx.array(f.allowed)].astype(mx.float32)).tolist()))
            theirs = dict(zip(served[f.qid]["options"], served[f.qid]["probs"]))
            assert set(mine) == set(theirs), (f.qid, mine, theirs)
            worst = max(worst, max(abs(mine[o] - theirs[o]) for o in mine))
    print(f"\nworst probability difference, training rows vs served: {worst:.2e}")
    assert worst < 1e-3


def test_lora_step_time_and_memory(engine):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from opensysone.train.examples import build_examples
    from opensysone.train.lora import LoraConfig, attach_lora, prefill, rows_loss

    examples, _ = build_examples(_cases("train", range(6)), engine._tokenizer)
    attach_lora(engine._model, LoraConfig(layers=16, rank=8))
    opt = optim.Adam(learning_rate=1e-5)
    step = nn.value_and_grad(engine._model, rows_loss)
    mx.reset_peak_memory()
    times = []
    for ex in examples:
        t = time.perf_counter()
        cache = prefill(engine._model, ex)
        engine._model.train()  # rows use the ops path for linear attention: the fused kernel has no gradient
        loss, grads = step(engine._model, ex, cache)
        opt.update(engine._model, grads)
        mx.eval(engine._model.trainable_parameters(), opt.state, loss)
        times.append(time.perf_counter() - t)
    engine._model.eval()
    peak = mx.get_peak_memory() / 2**30
    report = {"seconds_per_case": times, "median": sorted(times)[len(times) // 2], "peak_gb": peak,
              "prompt_tokens": [len(e.prompt) for e in examples]}
    (DATA / "lora-live.json").write_text(json.dumps(report, indent=2))
    print(f"\nLoRA step: median {report['median']:.1f}s per case, peak memory {peak:.1f} GB, times {times}")
    assert peak < 44


def test_multi_token_rows_give_the_served_probability_of_the_label(engine):
    # 77- and 151-option intent questions need two-token codes. With a hard label the training loss is
    # -log P(label), and P(label) must be what the engine serves: root probability times branch probability.
    import math

    from opensysone.train.examples import build_examples
    from opensysone.train.lora import case_loss
    from opensysone.wire import SystemOneRequest

    cases = []
    for name in ("banking77-test", "clinc150-test"):
        rows = [json.loads(line) for line in open(DATA / "human-gold" / f"{name}-s500.cases.jsonl")]
        cases += rows[:3]
    examples, _ = build_examples(cases, engine._tokenizer)
    assert any(len(ex.fields) > 1 for ex in examples)
    worst = 0.0
    for case, ex in zip(cases, examples):
        req = SystemOneRequest.model_validate({"state": case["state"], "questions": case["questions"], "cache": False})
        served = engine.decide(req)["_raw"]["questions"]["answer"]
        theirs = dict(zip(served["options"], served["probs"]))[case["labels"]["answer"]]
        mine = math.exp(-float(case_loss(engine._model, ex).item()))
        worst = max(worst, abs(mine - theirs))
    print(f"\nworst label probability difference, training rows vs served: {worst:.2e}")
    assert worst < 1e-3
