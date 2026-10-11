import mlx.core as mx
import pytest

from opensysone.train.examples import CaseExample, FieldExample
from opensysone.train.lora import LoraConfig, attach_lora, case_loss, load_adapter, mean_loss, save_adapter, train


def _tiny():
    from mlx_lm.models import qwen3_5 as q

    args = q.TextModelArgs(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, vocab_size=97, linear_num_value_heads=4,
                           linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16, head_dim=16,
                           full_attention_interval=2,
                           rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25, "type": "default"})
    mx.random.seed(0)
    model = q.TextModel(args)
    mx.eval(model.parameters())
    return model


def _case(i, target_first):
    t = (0.9, 0.1) if target_first else (0.1, 0.9)
    return CaseExample(f"c{i}", f"g{i}", (5, 6, 7, 8 + i % 5),
                       (FieldExample("a", (20, 21, 22), 2, (30, 31), t),
                        FieldExample("b", (40, 41), 1, (50, 51, 52), (0.2, 0.3, 0.5))))


def test_case_loss_reads_the_same_logits_as_one_full_sequence():
    model = _tiny()
    ex = _case(0, True)
    expected = 0.0
    for f in ex.fields:
        logits = model(mx.array(ex.prompt + f.row)[None])[0, len(ex.prompt) + f.position]
        z = logits[mx.array(f.allowed)]
        expected += float(-(mx.array(f.target) * (z - mx.logsumexp(z))).sum().item())
    assert float(case_loss(model, ex).item()) == pytest.approx(expected / len(ex.fields), rel=1e-4)


def test_training_lowers_loss_and_only_touches_lora():
    model = _tiny()
    cfg = LoraConfig(layers=2, rank=4, scale=10.0, lr=5e-3, epochs=8, warmup=2, eval_every=10)
    attach_lora(model, cfg)
    names = [k for k, _ in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
        model.trainable_parameters())]
    assert names and all("lora" in k for k in names)
    data = [_case(i, True) for i in range(6)]
    before = mean_loss(model, data)
    result = train(model, data, data[:2], cfg, log=lambda s: None)
    assert mean_loss(model, data) < before
    assert result["best_val_loss"] <= result["history"][0]["val_loss"]


def test_adapter_roundtrip_gives_identical_scores(tmp_path):
    model = _tiny()
    cfg = LoraConfig(layers=2, rank=4, scale=10.0, lr=5e-3, epochs=2, warmup=1, eval_every=5)
    attach_lora(model, cfg)
    data = [_case(i, False) for i in range(4)]
    train(model, data, data[:1], cfg, log=lambda s: None)
    sha = save_adapter(model, tmp_path, cfg)
    fresh = _tiny()
    assert load_adapter(fresh, tmp_path) == sha
    assert float(case_loss(fresh, data[0]).item()) == pytest.approx(float(case_loss(model, data[0]).item()), rel=1e-5)
