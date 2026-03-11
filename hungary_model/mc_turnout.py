from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .mc_distributions import sample_float
from .turnout import predict_station_votes_2026
from .aggregation import aggregate_station_to_evk
from .types import ModelData, ScenarioConfig, MonteCarloConfig


def sample_turnout_target(mc: MonteCarloConfig, base_target: float, rng: np.random.Generator) -> float:
    """Sample turnout target (fraction 0..1)."""

    if mc.turnout_target_spec is not None:
        return sample_float(rng, mc.turnout_target_spec, base_target, hard_min=0.20, hard_max=0.95)

    # Backward compatible: normal around base_target, in percentage points
    sd = float(mc.turnout_sigma_pp) / 100.0
    t = float(rng.normal(base_target, sd))
    return float(np.clip(t, 0.20, 0.95))


def choose_turnout_model_and_granularity(
    mc: MonteCarloConfig, base_cfg: ScenarioConfig, rng: np.random.Generator
) -> Tuple[str, str]:
    """Randomly pick turnout model/granularity from MC lists, else use scenario."""

    models = [m for m in (mc.turnout_models or []) if m]
    grans = [g for g in (mc.turnout_granularities or []) if g]

    model = str(rng.choice(models)) if models else base_cfg.turnout_model
    gran = str(rng.choice(grans)) if grans else base_cfg.turnout_granularity

    return model, gran


def make_turnout_grid_targets(mc: MonteCarloConfig, base_target: float) -> List[float]:
    """Pick a small set of turnout targets to precompute row vote shapes.

    This is used to allow turnout geography to depend (approximately) on the
    target turnout level without recomputing turnout maps inside every draw.
    """

    spec = mc.turnout_target_spec

    if spec is not None:
        dist = str(spec.get("dist", "fixed")).lower().strip()
        if dist in {"fixed", "const", "constant"}:
            v = float(spec.get("value", base_target))
            return [float(np.clip(v, 0.20, 0.95))]

        if dist in {"uniform", "range"}:
            lo = float(spec.get("min", base_target))
            hi = float(spec.get("max", base_target))
            if hi < lo:
                lo, hi = hi, lo
            pts = [lo, 0.5 * (lo + hi), hi]
            return sorted({float(np.clip(p, 0.20, 0.95)) for p in pts})

        if dist in {"normal", "gaussian"}:
            mu = float(spec.get("mean", base_target))
            sd = float(spec.get("sd", 0.0))
            pts = [mu - 2 * sd, mu, mu + 2 * sd]
            return sorted({float(np.clip(p, 0.20, 0.95)) for p in pts})

    # default: use legacy sigma
    sd = float(mc.turnout_sigma_pp) / 100.0
    pts = [base_target - 2 * sd, base_target, base_target + 2 * sd]
    return sorted({float(np.clip(p, 0.20, 0.95)) for p in pts})


def build_turnout_row_votes_cache(
    data: ModelData,
    base_cfg: ScenarioConfig,
    *,
    unit: str,
    unit_index: pd.Index,
    turnout_models: List[str],
    turnout_granularities: List[str],
    grid_targets: List[float],
) -> Dict[Tuple[str, str], np.ndarray]:
    """Precompute row-vote vectors for (turnout_model, turnout_granularity) x grid_targets.

    Returns
    -------
    dict
        (model, granularity) -> row_votes_grid, shape (len(grid_targets), n_units)
    """

    cache: Dict[Tuple[str, str], np.ndarray] = {}

    # Ensure deterministic order for stable caching / debugging
    turnout_models = list(dict.fromkeys(turnout_models))
    turnout_granularities = list(dict.fromkeys(turnout_granularities))

    for m in turnout_models:
        for g in turnout_granularities:
            rows = []
            for t in grid_targets:
                cfg_t = replace(base_cfg, turnout_model=m, turnout_granularity=g, turnout_target=float(t))
                votes_station, _ = predict_station_votes_2026(data, cfg_t)

                if unit == "station":
                    rv = votes_station.reindex(unit_index).fillna(0.0).to_numpy(dtype=float)

                elif unit == "evk":
                    votes_evk = aggregate_station_to_evk(votes_station, data.station_meta)
                    rv = votes_evk.reindex(unit_index).fillna(0.0).to_numpy(dtype=float)

                else:
                    raise ValueError(f"Unsupported unit for turnout cache: {unit}")

                rows.append(rv)

            cache[(m, g)] = np.vstack(rows)

    return cache


def interpolate_row_votes(
    grid_targets: List[float],
    row_votes_grid: np.ndarray,
    target: float,
) -> np.ndarray:
    """Linear interpolation over precomputed grid targets."""

    if row_votes_grid.shape[0] == 1:
        return row_votes_grid[0].copy()

    gt = np.asarray(grid_targets, dtype=float)
    t = float(target)

    # Clip to grid range
    if t <= gt.min():
        return row_votes_grid[0].copy()
    if t >= gt.max():
        return row_votes_grid[-1].copy()

    # Find bracketing indices
    hi = int(np.searchsorted(gt, t, side="right"))
    lo = hi - 1

    t0, t1 = float(gt[lo]), float(gt[hi])
    if t1 <= t0:
        return row_votes_grid[lo].copy()

    w = (t - t0) / (t1 - t0)
    return (1.0 - w) * row_votes_grid[lo] + w * row_votes_grid[hi]
