"""Lightweight sampling helpers for Monte Carlo.

We avoid heavy dependencies and keep specs JSON-serializable so they can be
constructed directly from Streamlit widgets.

Distribution spec format
------------------------
A *spec* is a dict with at minimum a ``dist`` key:

- Fixed:
    {"dist": "fixed", "value": 0.72}

- Uniform:
    {"dist": "uniform", "min": 0.68, "max": 0.76}

- Normal (clipped/truncated by clipping):
    {"dist": "normal", "mean": 0.72, "sd": 0.01, "min": 0.20, "max": 0.95}

All numbers are interpreted as floats in *native units* (e.g. turnout is 0..1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _get(spec: Dict, key: str, default):
    v = spec.get(key, default)
    return default if v is None else v


def sample_float(
    rng: np.random.Generator,
    spec: Optional[Dict[str, float]],
    default: float,
    *,
    hard_min: Optional[float] = None,
    hard_max: Optional[float] = None,
) -> float:
    """Sample a float from spec, falling back to default."""

    if spec is None:
        x = float(default)
        if hard_min is not None:
            x = max(hard_min, x)
        if hard_max is not None:
            x = min(hard_max, x)
        return x

    dist = str(spec.get("dist", "fixed")).lower().strip()

    if dist in {"fixed", "const", "constant"}:
        x = float(_get(spec, "value", default))

    elif dist in {"uniform", "range"}:
        lo = float(_get(spec, "min", default))
        hi = float(_get(spec, "max", default))
        if hi < lo:
            lo, hi = hi, lo
        x = float(rng.uniform(lo, hi))

    elif dist in {"normal", "gaussian"}:
        mu = float(_get(spec, "mean", default))
        sd = float(_get(spec, "sd", 0.0))
        x = float(rng.normal(mu, sd))
        lo = spec.get("min", None)
        hi = spec.get("max", None)
        if lo is not None:
            x = max(float(lo), x)
        if hi is not None:
            x = min(float(hi), x)

    else:
        # Unknown spec -> treat as fixed.
        x = float(_get(spec, "value", default))

    if hard_min is not None:
        x = max(hard_min, x)
    if hard_max is not None:
        x = min(hard_max, x)
    return x


def sample_int(
    rng: np.random.Generator,
    spec: Optional[Dict[str, float]],
    default: int,
    *,
    hard_min: Optional[int] = None,
    hard_max: Optional[int] = None,
) -> int:
    x = sample_float(rng, spec, float(default))
    xi = int(round(x))
    if hard_min is not None:
        xi = max(int(hard_min), xi)
    if hard_max is not None:
        xi = min(int(hard_max), xi)
    return xi


def normalize_weights(keys: Sequence[str], raw: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    raw = np.clip(raw, 0.0, None)
    s = float(raw.sum())
    if not np.isfinite(s) or s <= 0:
        fb = np.asarray(fallback, dtype=float)
        fb = np.clip(fb, 0.0, None)
        s2 = float(fb.sum())
        return fb / s2 if s2 > 0 else np.ones(len(keys), dtype=float) / max(1, len(keys))
    return raw / s


def sample_weights_minmax(
    rng: np.random.Generator,
    base_weights: Dict[str, float],
    minmax: Optional[Dict[str, List[float]]],
) -> Tuple[List[str], np.ndarray]:
    """Sample weights within per-key [min,max] intervals, then renormalize."""

    keys = list(base_weights.keys())
    base = np.array([float(base_weights[k]) for k in keys], dtype=float)

    if not minmax:
        return keys, normalize_weights(keys, base, base)

    draws = []
    for k, b in zip(keys, base):
        mm = minmax.get(k, None)
        if mm is None or len(mm) != 2:
            draws.append(float(b))
            continue
        lo, hi = float(mm[0]), float(mm[1])
        if hi < lo:
            lo, hi = hi, lo
        draws.append(float(rng.uniform(lo, hi)))

    draws_arr = np.array(draws, dtype=float)
    return keys, normalize_weights(keys, draws_arr, base)
