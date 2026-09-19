"""Version stamp carried on every response so a decision can be reproduced."""
from __future__ import annotations

from typing import Any

from . import __version__


def _metadata(model_id: str) -> dict[str, Any]:
    from jevmlx.engine import engine_metadata

    return engine_metadata(model_id)


def _prompt_version() -> str:
    from jevmlx.engine import PROMPT_VERSION

    return PROMPT_VERSION


def build_stamp(model_id: str, calibration_sha: str | None) -> dict[str, Any]:
    meta = _metadata(model_id)
    return {
        "backbone": model_id,
        "backbone_revision": meta.get("revision"),
        "quantization": meta.get("quantization"),
        "calibration": calibration_sha,
        "prompt_version": _prompt_version(),
        "mlx": meta.get("mlx_version"),
        "mlx_lm": meta.get("mlx_lm_version"),
        "opensysone": __version__,
    }
