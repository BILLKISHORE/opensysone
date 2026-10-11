"""LoRA training on the engine's own scoring rows, with soft-label cross-entropy.

One step = one case: prefill the shared prompt once, then score each question's row on a copy of
that cache, exactly as the engine does at max_rows=1. Gradients flow through the cache, so the
prompt is read once per case instead of once per question.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .examples import CaseExample

ADAPTER_FILE = "adapters.safetensors"
CONFIG_FILE = "adapter_config.json"


@dataclass(frozen=True)
class LoraConfig:
    layers: int = 16  # adapt the last N transformer blocks
    rank: int = 8
    scale: float = 20.0
    dropout: float = 0.0
    lr: float = 1e-4
    epochs: int = 1
    warmup: int = 20
    eval_every: int = 100
    seed: int = 0


def attach_lora(model, cfg: LoraConfig) -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if any("lora_" in k for k, _ in tree_flatten(model.parameters())):
        raise RuntimeError("this model already has LoRA layers; load a fresh model before attaching another adapter")
    model.freeze()
    linear_to_lora_layers(model, cfg.layers, {"rank": cfg.rank, "scale": cfg.scale, "dropout": cfg.dropout})


def case_loss(model, ex: CaseExample) -> mx.array:
    """Mean soft cross-entropy over the case's questions, read where the engine reads them."""
    from jevmlx.engine import _broadcast_cache
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    model(mx.array(ex.prompt)[None], cache=cache)
    total = mx.array(0.0)
    for f in ex.fields:
        logits = model(mx.array(f.row)[None], cache=_broadcast_cache(cache, 1))[0, f.position]
        z = logits[mx.array(f.allowed)].astype(mx.float32)
        total = total - (mx.array(f.target) * (z - mx.logsumexp(z))).sum()
    return total / len(ex.fields)


def mean_loss(model, examples: Sequence[CaseExample]) -> float:
    losses = []
    for ex in examples:
        loss = case_loss(model, ex)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else math.nan


def _lr(step: int, total: int, cfg: LoraConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup) / max(1, total - cfg.warmup)))


def train(model, train_ex: Sequence[CaseExample], val_ex: Sequence[CaseExample], cfg: LoraConfig,
          log: Callable[[str], None] = print) -> dict:
    """Train attached LoRA weights; keep the weights with the lowest validation loss."""
    rng = random.Random(cfg.seed)
    total = cfg.epochs * len(train_ex)
    opt = optim.Adam(learning_rate=cfg.lr)
    step_fn = nn.value_and_grad(model, case_loss)
    best_val = mean_loss(model, val_ex)
    best = dict(tree_flatten(model.trainable_parameters()))
    history = [{"step": 0, "val_loss": best_val}]
    log(f"step 0/{total} val_loss {best_val:.4f}")
    step, t0, running = 0, time.perf_counter(), []
    for _ in range(cfg.epochs):
        order = list(train_ex)
        rng.shuffle(order)
        for ex in order:
            opt.learning_rate = _lr(step, total, cfg)
            loss, grads = step_fn(model, ex)
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state, loss)
            running.append(float(loss.item()))
            step += 1
            if step % cfg.eval_every == 0 or step == total:
                val = mean_loss(model, val_ex)
                rate = (time.perf_counter() - t0) / step
                history.append({"step": step, "train_loss": sum(running) / len(running), "val_loss": val})
                log(f"step {step}/{total} train_loss {sum(running) / len(running):.4f} val_loss {val:.4f} "
                    f"{rate:.1f}s/step eta {(total - step) * rate / 60:.0f}m")
                running = []
                if val < best_val:
                    best_val, best = val, dict(tree_flatten(model.trainable_parameters()))
    model.update(tree_unflatten(list(best.items())))
    mx.eval(model.trainable_parameters())
    return {"best_val_loss": best_val, "history": history}


def save_adapter(model, directory: str | Path, cfg: LoraConfig, meta: dict | None = None) -> str:
    """mlx-lm adapter layout (adapters.safetensors + adapter_config.json); returns a short sha."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(directory / ADAPTER_FILE), dict(tree_flatten(model.trainable_parameters())))
    config = {"fine_tune_type": "lora", "num_layers": cfg.layers,
              "lora_parameters": {"rank": cfg.rank, "scale": cfg.scale, "dropout": cfg.dropout},
              "opensysone": {"config": asdict(cfg), **(meta or {})}}
    (directory / CONFIG_FILE).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return adapter_sha(directory)


def adapter_sha(directory: str | Path) -> str:
    return hashlib.sha256((Path(directory) / ADAPTER_FILE).read_bytes()).hexdigest()[:16]


def load_adapter(model, directory: str | Path) -> str:
    """Attach LoRA layers as saved and load their weights; returns the adapter sha."""
    directory = Path(directory)
    config = json.loads((directory / CONFIG_FILE).read_text())
    lp = config["lora_parameters"]
    attach_lora(model, LoraConfig(layers=config["num_layers"], rank=lp["rank"], scale=lp["scale"],
                                  dropout=lp.get("dropout", 0.0)))
    weights = mx.load(str(directory / ADAPTER_FILE))
    expected = {k: v.shape for k, v in tree_flatten(model.trainable_parameters())}
    got = {k: v.shape for k, v in weights.items()}
    if got != expected:
        missing, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
        shapes = sorted(k for k in set(got) & set(expected) if got[k] != expected[k])
        raise ValueError(f"adapter does not fit this model: missing {missing[:3]}, unexpected {extra[:3]}, "
                         f"wrong shape {shapes[:3]}")
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return adapter_sha(directory)
