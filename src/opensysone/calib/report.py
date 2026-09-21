"""Calibration report: metrics per question type, reliability table and SVG diagram."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any
from xml.sax.saxutils import escape

from .metrics import summarize
from .records import DecisionRecord

W, H = 480, 360
LEFT, RIGHT, TOP, BOTTOM = 56, 88, 56, 48
LABEL_GAP = 14  # minimum vertical distance between the two end labels
PW, PH = W - LEFT - RIGHT, H - TOP - BOTTOM

STYLE = """
.viz{--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme: dark){.viz{--surface:#1a1a19;--text:#ffffff;--text2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926}}
.viz text{font-family:system-ui,-apple-system,sans-serif}
"""


def evaluate(records: Sequence[DecisionRecord], calibrator: Any = None) -> dict[str, dict[str, Any]]:
    by_kind: dict[str, list[DecisionRecord]] = defaultdict(list)
    for r in records:
        by_kind[r.kind].append(r)
    out: dict[str, dict[str, Any]] = {}
    all_raw, all_cal = [], []
    for kind, recs in sorted(by_kind.items()):
        raw = [list(r.probs) for r in recs]
        cal = [calibrator.apply(kind, r.probs, r.prompt_tokens) if calibrator else list(r.probs) for r in recs]
        out[kind] = {"raw": summarize(recs, raw), "calibrated": summarize(recs, cal)}
        all_raw += raw
        all_cal += cal
    ordered = [r for kind in sorted(by_kind) for r in by_kind[kind]]
    out["all"] = {"raw": summarize(ordered, all_raw), "calibrated": summarize(ordered, all_cal)}
    return out


def _x(v: float) -> float:
    return LEFT + v * PW


def _y(v: float) -> float:
    return TOP + (1.0 - v) * PH


def _series(name: str, var: str, bins: list[dict[str, Any]]) -> str:
    if not bins:
        return ""
    pts = " ".join(f"{_x(b['confidence']):.1f},{_y(b['accuracy']):.1f}" for b in bins)
    parts = [f'<polyline points="{pts}" fill="none" stroke="var({var})" stroke-width="2" '
             f'stroke-linejoin="round" stroke-linecap="round"/>']
    for b in bins:
        tip = (f"{name}, confidence {b['lo']:.1f} to {b['hi']:.1f}: n={b['n']}, "
               f"mean confidence {b['confidence']:.2f}, accuracy {b['accuracy']:.2f}")
        parts.append(f'<circle cx="{_x(b["confidence"]):.1f}" cy="{_y(b["accuracy"]):.1f}" r="4" '
                     f'fill="var({var})" stroke="var(--surface)" stroke-width="2"><title>{escape(tip)}</title></circle>')
    return "\n".join(parts)


def _end_labels(series: list[tuple[str, list[dict[str, Any]]]]) -> str:
    """Direct labels in the right margin, level with each line's end, pushed apart when close."""
    ends = [(name, _x(1.0) + 8, _y(b[-1]["accuracy"]) + 4) for name, b in series if b]
    if len(ends) == 2 and abs(ends[0][2] - ends[1][2]) < LABEL_GAP:
        (n0, x0, y0), (n1, x1, y1) = ends
        mid = (y0 + y1) / 2
        top, bottom = mid - LABEL_GAP / 2, mid + LABEL_GAP / 2
        ends = [(n0, x0, top if y0 <= y1 else bottom), (n1, x1, bottom if y0 <= y1 else top)]
    return "\n".join(f'<text x="{x:.1f}" y="{y:.1f}" font-size="12" fill="var(--text2)">{escape(name)}</text>'
                     for name, x, y in ends)


def reliability_svg(title: str, raw_bins: list[dict[str, Any]], cal_bins: list[dict[str, Any]]) -> str:
    grid = []
    for i in range(6):
        v = i / 5
        grid.append(f'<line x1="{_x(0):.1f}" y1="{_y(v):.1f}" x2="{_x(1):.1f}" y2="{_y(v):.1f}" '
                    f'stroke="var(--grid)" stroke-width="1"/>')
        grid.append(f'<text x="{LEFT - 8}" y="{_y(v) + 4:.1f}" font-size="11" fill="var(--muted)" '
                    f'text-anchor="end">{v:.1f}</text>')
        grid.append(f'<text x="{_x(v):.1f}" y="{H - BOTTOM + 18}" font-size="11" fill="var(--muted)" '
                    f'text-anchor="middle">{v:.1f}</text>')
    legend = (f'<rect x="{LEFT}" y="{TOP - 24}" width="10" height="10" rx="2" fill="var(--s1)"/>'
              f'<text x="{LEFT + 16}" y="{TOP - 15}" font-size="12" fill="var(--text2)">raw</text>'
              f'<rect x="{LEFT + 60}" y="{TOP - 24}" width="10" height="10" rx="2" fill="var(--s2)"/>'
              f'<text x="{LEFT + 76}" y="{TOP - 15}" font-size="12" fill="var(--text2)">calibrated</text>')
    label = escape(f"Reliability diagram, {title}")
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" class="viz" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="{label}">',
        f"<title>{label}</title>",
        f"<style>{STYLE}</style>",
        f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
        f'<text x="{LEFT}" y="20" font-size="14" font-weight="600" fill="var(--text)">{escape(title)}</text>',
        *grid,
        f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(1):.1f}" y2="{_y(0):.1f}" stroke="var(--axis)" '
        f'stroke-width="1"/>',
        f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(1):.1f}" y2="{_y(1):.1f}" stroke="var(--muted)" '
        f'stroke-width="1" stroke-dasharray="4 4"/>',
        f'<text x="{_x(0.5):.1f}" y="{H - 8}" font-size="12" fill="var(--muted)" '
        f'text-anchor="middle">mean confidence</text>',
        f'<text x="14" y="{_y(0.5):.1f}" font-size="12" fill="var(--muted)" text-anchor="middle" '
        f'transform="rotate(-90 14 {_y(0.5):.1f})">accuracy</text>',
        legend,
        _series("raw", "--s1", raw_bins),
        _series("calibrated", "--s2", cal_bins),
        _end_labels([("raw", raw_bins), ("calibrated", cal_bins)]),
        "</svg>",
    ])


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def render_report(evaluation: dict[str, dict[str, Any]], *, title: str, note: str) -> tuple[str, dict[str, str]]:
    lines = [f"# {title}", "", note, "",
             "| Question type | n | Accuracy | ECE raw | ECE calibrated | NLL raw | NLL calibrated "
             "| Brier raw | Brier calibrated |",
             "|---|---|---|---|---|---|---|---|---|"]
    for kind, ev in evaluation.items():
        r, c = ev["raw"], ev["calibrated"]
        lines.append(f"| {kind} | {r['n']} | {_fmt(r['accuracy'])} | {_fmt(r['ece'])} | {_fmt(c['ece'])} | "
                     f"{_fmt(r['nll'])} | {_fmt(c['nll'])} | {_fmt(r['brier'])} | {_fmt(c['brier'])} |")
    files: dict[str, str] = {}
    for kind, ev in evaluation.items():
        name = f"reliability-{kind}.svg"
        files[name] = reliability_svg(kind, ev["raw"]["reliability"], ev["calibrated"]["reliability"])
        lines += ["", f"## {kind}", "", f"![{kind} reliability]({name})", "",
                  "| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated "
                  "| confidence calibrated | accuracy calibrated |",
                  "|---|---|---|---|---|---|---|"]
        raw_bins = {b["lo"]: b for b in ev["raw"]["reliability"]}
        cal_bins = {b["lo"]: b for b in ev["calibrated"]["reliability"]}
        for lo in sorted(set(raw_bins) | set(cal_bins)):
            rb, cb = raw_bins.get(lo), cal_bins.get(lo)
            lines.append(f"| {lo:.1f} to {lo + 0.1:.1f} | {rb['n'] if rb else 0} | {_fmt(rb and rb['confidence'])} | "
                         f"{_fmt(rb and rb['accuracy'])} | {cb['n'] if cb else 0} | {_fmt(cb and cb['confidence'])} | "
                         f"{_fmt(cb and cb['accuracy'])} |")
    return "\n".join(lines) + "\n", files
