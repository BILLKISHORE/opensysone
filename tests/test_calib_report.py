import xml.etree.ElementTree as ET

from opensysone.calib.calibrators import Calibrator, KindCalibration
from opensysone.calib.records import DecisionRecord
from opensysone.calib.report import evaluate, reliability_svg, render_report


def _recs():
    out = []
    for i in range(40):
        out.append(DecisionRecord(f"c{i}", f"c{i}", "choice", ("a", "b"), (0.9, 0.1), 50,
                                  "a" if i % 2 else "b", None))
    for i in range(20):
        out.append(DecisionRecord(f"n{i}", f"n{i}", "noul", ("true", "false"), (0.8, 0.2), 50, "true", None))
    return out


def test_evaluate_has_kinds_and_all_and_calibration_lowers_ece():
    cal = Calibrator({"choice": KindCalibration("temperature", 5.0)})
    ev = evaluate(_recs(), cal)
    assert set(ev) == {"choice", "noul", "all"}
    assert ev["all"]["raw"]["n"] == 60
    assert ev["choice"]["calibrated"]["ece"] < ev["choice"]["raw"]["ece"]
    assert ev["noul"]["calibrated"] == ev["noul"]["raw"]


def test_svg_is_valid_xml_with_two_series_and_tooltips():
    ev = evaluate(_recs(), Calibrator({"choice": KindCalibration("temperature", 5.0)}))
    svg = reliability_svg("choice", ev["choice"]["raw"]["reliability"], ev["choice"]["calibrated"]["reliability"])
    root = ET.fromstring(svg)  # our own generated SVG, not outside input
    ns = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{ns}svg"
    assert len(root.findall(f".//{ns}polyline")) == 2
    titles = [t.text for t in root.iter(f"{ns}title")]
    assert any(t and t.startswith("raw") for t in titles) and any(t and t.startswith("calibrated") for t in titles)
    assert "prefers-color-scheme: dark" in svg


def test_render_report_markdown_and_files():
    ev = evaluate(_recs(), Calibrator({"choice": KindCalibration("temperature", 5.0)}))
    md, files = render_report(ev, title="Calibration on test", note="Labels are consensus.")
    assert md.startswith("# Calibration on test")
    assert "| choice |" in md and "Labels are consensus." in md
    assert "![choice reliability](reliability-choice.svg)" in md
    assert set(files) == {"reliability-choice.svg", "reliability-noul.svg", "reliability-all.svg"}
