import mlx.core as mx
import pytest

from opensysone.train.examples import CaseExample, FieldExample
from opensysone.train.lora import LoraConfig, attach_lora, case_loss, load_adapter, mean_loss, save_adapter, train


@pytest.fixture(autouse=True)
def _cpu():
    """The tiny model's head sizes have no fused Metal kernel; on CPU every mode uses the plain ops path."""
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


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


def test_engine_loads_an_adapter_and_stamps_its_sha(tmp_path, monkeypatch):
    from opensysone import engine as engine_mod

    trained = _tiny()
    cfg = LoraConfig(layers=2, rank=4, scale=10.0, lr=5e-3, epochs=1, warmup=1, eval_every=5)
    attach_lora(trained, cfg)
    train(trained, [_case(0, True), _case(1, True)], [_case(2, True)], cfg, log=lambda s: None)
    sha = save_adapter(trained, tmp_path, cfg)

    served = _tiny()
    monkeypatch.setattr(engine_mod, "_load", lambda model_id: (served, None))
    monkeypatch.setattr(engine_mod, "build_stamp", lambda m, c: {"backbone": m})
    e = engine_mod.Engine("tiny", parity_check=False, adapter=str(tmp_path))
    e.load()
    assert e.stamp["adapter"] == sha
    assert float(case_loss(e._model, _case(0, True)).item()) == pytest.approx(
        float(case_loss(trained, _case(0, True)).item()), rel=1e-5)


def test_engine_without_adapter_stamps_none(monkeypatch):
    from opensysone import engine as engine_mod

    monkeypatch.setattr(engine_mod, "_load", lambda model_id: (object(), None))
    monkeypatch.setattr(engine_mod, "build_stamp", lambda m, c: {})
    e = engine_mod.Engine("x", parity_check=False)
    e.load()
    assert e.stamp["adapter"] is None


def test_cli_lora_trains_and_writes_an_adapter(tmp_path, monkeypatch, capsys):
    import json as _json
    from collections import Counter

    from opensysone import cli
    from opensysone import engine as engine_mod

    model = _tiny()
    monkeypatch.setattr(engine_mod, "_load", lambda model_id: (model, None))
    monkeypatch.setattr(engine_mod, "build_stamp", lambda m, c: {})
    examples = [_case(i, True) for i in range(12)]
    monkeypatch.setattr(cli, "build_examples", lambda cases, tok, labels="hard": (examples, Counter(cases=12)))
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(_json.dumps({"id": f"c{i}"}) for i in range(12)) + "\n")
    out = tmp_path / "adapter"
    assert cli.main(["lora", str(cases), "--out", str(out), "--layers", "2", "--rank", "4", "--lr", "5e-3",
                     "--warmup", "1", "--eval-every", "5"]) == 0
    assert (out / "adapters.safetensors").exists()
    meta = _json.loads((out / "adapter_config.json").read_text())["opensysone"]
    assert meta["n_train_cases"] + meta["n_val_cases"] <= 12 and "best_val_loss" in meta
    assert "adapter" in capsys.readouterr().out


def test_attaching_twice_is_refused():
    model = _tiny()
    attach_lora(model, LoraConfig(layers=2, rank=4))
    with pytest.raises(RuntimeError):
        attach_lora(model, LoraConfig(layers=2, rank=4))


def test_adapter_for_a_different_shape_is_refused(tmp_path):
    model = _tiny()
    attach_lora(model, LoraConfig(layers=2, rank=4))
    save_adapter(model, tmp_path, LoraConfig(layers=2, rank=4))
    import json as _json
    cfg = _json.loads((tmp_path / "adapter_config.json").read_text())
    cfg["lora_parameters"]["rank"] = 8
    (tmp_path / "adapter_config.json").write_text(_json.dumps(cfg))
    with pytest.raises(ValueError, match="does not fit"):
        load_adapter(_tiny(), tmp_path)


def test_cli_lora_never_validates_on_outer_heldout_groups(tmp_path, monkeypatch):
    from collections import Counter

    from opensysone import cli
    from opensysone import engine as engine_mod
    from opensysone.calib.records import is_heldout

    seen = {}
    monkeypatch.setattr(engine_mod, "_load", lambda model_id: (_tiny(), None))
    monkeypatch.setattr(engine_mod, "build_stamp", lambda m, c: {})
    examples = [_case(i, True) for i in range(60)]
    monkeypatch.setattr(cli, "build_examples", lambda cases, tok, labels="hard": (examples, Counter()))

    def fake_train(model, train_ex, val_ex, cfg, log):
        seen["groups"] = [e.group_id for e in [*train_ex, *val_ex]]
        return {"best_val_loss": 0.0, "history": []}

    monkeypatch.setattr("opensysone.train.lora.train", fake_train)
    cases = tmp_path / "c.jsonl"
    cases.write_text('{"id": "x"}\n')
    assert cli.main(["lora", str(cases), "--out", str(tmp_path / "a"), "--layers", "2", "--rank", "4"]) == 0
    assert seen["groups"] and not any(is_heldout(g) for g in seen["groups"])


def test_hard_and_soft_targets():
    from opensysone.train.examples import _target

    dist = {"a": 0.6, "b": 0.4}
    assert _target(dist, "a", "hard") == {"a": 1.0}
    assert _target(dist, "a", "soft") == {"a": 0.6, "b": 0.4}
    assert _target(None, True, "soft") == {"true": 1.0}


def test_train_steps_run_in_training_mode_and_leave_eval_mode(monkeypatch):
    import opensysone.train.lora as lora_mod

    model = _tiny()
    cfg = LoraConfig(layers=2, rank=4, lr=1e-3, epochs=1, warmup=1, eval_every=2)
    attach_lora(model, cfg)
    modes = []
    real = lora_mod.rows_loss

    def spy(m, ex, cache):
        modes.append(m.training)
        return real(m, ex, cache)

    monkeypatch.setattr(lora_mod, "rows_loss", spy)
    train(model, [_case(i, True) for i in range(4)], [_case(9, True)], cfg, log=lambda s: None)
    assert True in modes and False in modes  # training steps in train mode, validation in eval mode
    assert model.training is False
