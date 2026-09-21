import json
import math
import random

import pytest

from opensysone.calib.calibrators import (
    CALIBRATION_FILE,
    Calibrator,
    KindCalibration,
    fit_temperature,
    temperature_scale,
)
from opensysone.calib.features import FEATURE_NAMES, decision_features


def test_features_shape_and_values():
    f = decision_features("choice", [0.7, 0.2, 0.1], 100)
    assert len(f) == len(FEATURE_NAMES) == 9
    named = dict(zip(FEATURE_NAMES, f))
    assert named["log_k"] == pytest.approx(math.log(3))
    assert named["top"] == pytest.approx(0.7)
    assert named["margin"] == pytest.approx(0.5)
    assert 0.0 < named["norm_entropy"] < 1.0
    assert named["log_prompt_tokens"] == pytest.approx(math.log1p(100))
    assert (named["kind_noul"], named["kind_choice"], named["kind_score"], named["kind_pairwise"]) == (0, 1, 0, 0)


def test_temperature_scale_keeps_argmax_and_softens_or_sharpens():
    p = [0.7, 0.2, 0.1]
    assert temperature_scale(p, 1.0) == pytest.approx(p)
    soft = temperature_scale(p, 3.0)
    sharp = temperature_scale(p, 0.5)
    assert soft[0] < 0.7 < sharp[0]
    assert soft.index(max(soft)) == sharp.index(max(sharp)) == 0
    assert sum(soft) == pytest.approx(1.0)


def test_fit_temperature_recovers_overconfidence():
    rng = random.Random(0)
    samples = []
    for _ in range(500):
        correct = rng.random() < 0.6
        target = [1.0, 0.0] if correct else [0.0, 1.0]
        samples.append(([0.95, 0.05], target))
    t = fit_temperature(samples)
    assert t > 1.5
    calibrated = temperature_scale([0.95, 0.05], t)[0]
    assert calibrated == pytest.approx(0.6, abs=0.03)


def test_calibrator_identity_and_temperature_apply():
    c = Calibrator({"choice": KindCalibration("temperature", 2.0), "noul": KindCalibration("identity")})
    assert c.apply("noul", [0.9, 0.1], 10) == [0.9, 0.1]
    assert c.apply("score", [0.9, 0.1], 10) == [0.9, 0.1]  # unknown kind: unchanged
    assert c.apply("choice", [0.9, 0.1], 10)[0] < 0.9


def test_save_load_roundtrip_and_sha(tmp_path):
    c = Calibrator({"choice": KindCalibration("temperature", 1.7, {"heldout_nll": 0.4})}, meta={"backbone": "m"})
    sha = c.save(tmp_path)
    assert len(sha) == 16 and (tmp_path / CALIBRATION_FILE).exists()
    body = json.loads((tmp_path / CALIBRATION_FILE).read_text())
    assert body["format"] == 1 and body["head"] is None
    loaded = Calibrator.load(tmp_path)
    assert loaded.sha == sha
    assert loaded.kinds["choice"].temperature == pytest.approx(1.7)
    assert loaded.apply("choice", [0.9, 0.1], 5) == pytest.approx(c.apply("choice", [0.9, 0.1], 5))


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method"):
        KindCalibration("magic")
