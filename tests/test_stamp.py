import opensysone
from opensysone import stamp as stamp_mod


def test_build_stamp_collects_provenance(monkeypatch):
    monkeypatch.setattr(stamp_mod, "_metadata", lambda model_id: {
        "model_id": model_id, "revision": "abc123", "mlx_version": "0.32.2",
        "mlx_lm_version": "0.31.3", "quantization": {"bits": 4, "group_size": 64}})
    monkeypatch.setattr(stamp_mod, "_prompt_version", lambda: "jevmlx-parallel-v8")
    s = stamp_mod.build_stamp("mlx-community/Qwen3.8-27B-4bit", None)
    assert s == {
        "backbone": "mlx-community/Qwen3.8-27B-4bit",
        "backbone_revision": "abc123",
        "quantization": {"bits": 4, "group_size": 64},
        "calibration": None,
        "prompt_version": "jevmlx-parallel-v8",
        "mlx": "0.32.2",
        "mlx_lm": "0.31.3",
        "opensysone": opensysone.__version__,
    }
