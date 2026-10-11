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
    assert body["format"] == 2 and body["head"] is None
    loaded = Calibrator.load(tmp_path)
    assert loaded.sha == sha
    assert loaded.kinds["choice"].temperature == pytest.approx(1.7)
    assert loaded.apply("choice", [0.9, 0.1], 5) == pytest.approx(c.apply("choice", [0.9, 0.1], 5))


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method"):
        KindCalibration("magic")


def test_ets_keeps_argmax_and_mixes_toward_uniform():
    from opensysone.calib.calibrators import ets_scale

    p = [0.7, 0.2, 0.1]
    q = ets_scale(p, 2.0, (0.5, 0.3, 0.2))
    assert q.index(max(q)) == 0 and sum(q) == pytest.approx(1.0)
    assert ets_scale(p, 1.0, (0.0, 1.0, 0.0)) == pytest.approx(p)
    assert ets_scale(p, 1.0, (0.0, 0.0, 1.0)) == pytest.approx([1 / 3] * 3)


def test_fit_ets_prefers_uniform_when_model_is_noise():
    from opensysone.calib.calibrators import fit_ets

    rng = random.Random(0)
    samples = []
    for _ in range(300):
        y = rng.randrange(2)
        samples.append(([0.95, 0.05], [1.0 if k == y else 0.0 for k in range(2)]))
    w = fit_ets(samples, 1.0)
    assert w[2] > 0.5


def test_dirichlet_kind_uses_signature_map_then_kind_fallback(tmp_path):
    from opensysone.calib.dirichlet import DirichletMap

    flip = DirichletMap(((0.0, 1.0), (1.0, 0.0)), (0.0, 0.0))
    maps = {"sig:abc": flip, "kind:noul:2": DirichletMap.identity(2)}
    cal = Calibrator({"noul": KindCalibration("dirichlet", 1.0)}, maps=maps)
    assert cal.apply("noul", [0.8, 0.2], 10, signature="abc") == pytest.approx([0.2, 0.8])
    assert cal.apply("noul", [0.8, 0.2], 10, signature="other") == pytest.approx([0.8, 0.2])
    assert cal.apply("noul", [0.8, 0.1, 0.1], 10, signature="abc")[0] == pytest.approx(0.8)  # no map for k=3
    cal.save(tmp_path)
    again = Calibrator.load(tmp_path)
    assert again.apply("noul", [0.8, 0.2], 10, signature="abc") == pytest.approx([0.2, 0.8])
    assert again.sha == cal.sha


def test_ets_kind_roundtrips(tmp_path):
    cal = Calibrator({"choice": KindCalibration("ets", 1.5, ets=(0.6, 0.3, 0.1))})
    cal.save(tmp_path)
    again = Calibrator.load(tmp_path)
    assert again.apply("choice", [0.7, 0.3], 10) == pytest.approx(cal.apply("choice", [0.7, 0.3], 10))


def test_format_one_files_still_load(tmp_path):
    (tmp_path / CALIBRATION_FILE).write_text(json.dumps(
        {"format": 1, "kinds": {"choice": {"method": "temperature", "temperature": 2.0, "metrics": {}}},
         "head": None, "meta": {}}))
    cal = Calibrator.load(tmp_path)
    assert cal.apply("choice", [0.9, 0.1], 5) == pytest.approx(temperature_scale([0.9, 0.1], 2.0))
