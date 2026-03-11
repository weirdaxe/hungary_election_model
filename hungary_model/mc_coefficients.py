from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import MODEL_PARTIES
from .mc_distributions import sample_weights_minmax
from .types import ModelData, MonteCarloConfig, ScenarioConfig


GEO_LEVEL_ORDER = ["station", "location", "settlement", "evk"]


def apply_coef_noise(coefs: np.ndarray, mc: MonteCarloConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply log-normal multiplicative noise to coefficients."""

    sigma = float(mc.coef_log_sigma)
    if sigma <= 0:
        return coefs

    noise = rng.normal(0.0, sigma, size=coefs.shape)
    out = coefs * np.exp(noise)
    return np.clip(out, 1e-6, 1e6)


def _blocks_to_parties_df(df_blocks: pd.DataFrame) -> pd.DataFrame:
    """Convert a BLOCK-bucket coefficient frame to MODEL_PARTIES columns."""

    # Expected block columns: FIDESZ, OPP, DK?, MH, MKKP, OTHER.
    # In coefficient frames, DK may not be present; OPP exists.
    out = pd.DataFrame(index=df_blocks.index)

    if "FIDESZ" in df_blocks.columns:
        out["FIDESZ"] = df_blocks["FIDESZ"]
    else:
        out["FIDESZ"] = 1.0

    # OPP coefficients apply to both TISZA and DK in party-space mapping
    opp = df_blocks["OPP"] if "OPP" in df_blocks.columns else 1.0
    out["TISZA"] = opp
    out["DK"] = opp

    for k in ["MH", "MKKP", "OTHER"]:
        out[k] = df_blocks[k] if k in df_blocks.columns else 1.0

    # Ensure column order
    out = out.reindex(columns=MODEL_PARTIES)
    return out


def _evk_agg_logmean(
    station_values: pd.DataFrame,
    station_meta: pd.DataFrame,
    station_weights: pd.Series,
    evk_index: pd.Index,
) -> pd.DataFrame:
    """Aggregate station-level values to EVK by weighted mean in log-space."""

    # IMPORTANT:
    # station_values can include station_ids that are not present in the 2026 station universe
    # (station_meta). Using `.loc[...]` with a strict indexer will crash with KeyError.
    # We use `.reindex(...)` so missing stations become NaN and are ignored by groupby.

    w = station_weights.reindex(station_values.index).fillna(0.0)

    # Avoid log(0)
    safe = station_values.clip(lower=1e-6)
    logv = np.log(safe)

    # Map station rows -> EVK ids (NaN for missing station_ids / missing mapping)
    evk = station_meta["evk_id"].reindex(logv.index)

    logv_w = logv.mul(w, axis=0)
    denom = w.groupby(evk).sum().replace(0.0, np.nan)

    grouped = logv_w.groupby(evk).sum().div(denom, axis=0)
    grouped = grouped.reindex(evk_index).fillna(0.0)

    return grouped


def build_log_coef_components_evk(
    data: ModelData,
    cfg: ScenarioConfig,
    mc: MonteCarloConfig,
    *,
    evk_index: pd.Index,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Precompute EVK-level log(coef) components for each election and geo level.

    Returns
    -------
    components : np.ndarray
        Shape (n_elections, n_levels, n_evks, n_parties)
    elections : list[str]
    levels : list[str]
    """

    # Choose coefficient source based on scope
    coef_by_election = data.coef_by_election_no_budapest if cfg.exclude_budapest else data.coef_by_election

    elections = list(cfg.election_weights.keys())
    levels = list(cfg.geo_weights.keys()) if cfg.geo_weights else GEO_LEVEL_ORDER

    # Ensure standard order where possible (for UI consistency)
    levels = [l for l in GEO_LEVEL_ORDER if l in levels] + [l for l in levels if l not in GEO_LEVEL_ORDER]

    n_e = len(elections)
    n_l = len(levels)
    n_u = len(evk_index)
    n_p = len(MODEL_PARTIES)

    comp = np.zeros((n_e, n_l, n_u, n_p), dtype=float)

    station_meta = data.station_meta
    station_weights = data.registered_2026

    for ei, ekey in enumerate(elections):
        if ekey not in coef_by_election:
            # missing election -> leave zeros (log 1)
            continue

        c_e = coef_by_election[ekey]

        for li, lvl in enumerate(levels):
            if lvl not in c_e:
                continue

            df_lvl_blocks = c_e[lvl]
            df_lvl_party = _blocks_to_parties_df(df_lvl_blocks).fillna(1.0)

            if lvl == "evk":
                log_evk = np.log(df_lvl_party.reindex(evk_index).fillna(1.0).clip(lower=1e-6))

            elif lvl == "station":
                # station_id index
                # Restrict to the 2026 station universe so that historical station IDs that
                # disappeared by 2026 cannot leak into EVK aggregation.
                station_df = df_lvl_party.reindex(station_meta.index).fillna(1.0)
                log_evk = _evk_agg_logmean(station_df, station_meta, station_weights, evk_index)

            elif lvl == "location":
                # station_name_id index -> map to station_id then agg
                mapped = df_lvl_party.reindex(station_meta["station_name_id"]).set_index(station_meta.index)
                mapped = mapped.fillna(1.0)
                log_evk = _evk_agg_logmean(mapped, station_meta, station_weights, evk_index)

            elif lvl == "settlement":
                mapped = df_lvl_party.reindex(station_meta["settlement_id"]).set_index(station_meta.index)
                mapped = mapped.fillna(1.0)
                log_evk = _evk_agg_logmean(mapped, station_meta, station_weights, evk_index)

            else:
                # Unknown level: try mapping via station_meta if column exists
                if lvl in station_meta.columns:
                    mapped = df_lvl_party.reindex(station_meta[lvl]).set_index(station_meta.index)
                    mapped = mapped.fillna(1.0)
                    log_evk = _evk_agg_logmean(mapped, station_meta, station_weights, evk_index)
                else:
                    log_evk = pd.DataFrame(index=evk_index, columns=MODEL_PARTIES, data=0.0)

            comp[ei, li, :, :] = log_evk.reindex(columns=MODEL_PARTIES).to_numpy(dtype=float)

    return comp, elections, levels


def sample_election_weights(mc: MonteCarloConfig, cfg: ScenarioConfig, rng: np.random.Generator) -> Tuple[List[str], np.ndarray]:
    """Sample election weights (renormalized)."""

    base = {k: float(v) for k, v in cfg.election_weights.items()}
    keys, w = sample_weights_minmax(rng, base, mc.election_weight_minmax)
    return keys, w


def sample_geo_weights(mc: MonteCarloConfig, cfg: ScenarioConfig, rng: np.random.Generator) -> Tuple[List[str], np.ndarray]:
    """Sample geography weights (renormalized) in a stable order."""

    ordered_keys = [k for k in GEO_LEVEL_ORDER if k in cfg.geo_weights] + [k for k in cfg.geo_weights.keys() if k not in GEO_LEVEL_ORDER]
    base = {k: float(cfg.geo_weights.get(k, 0.0)) for k in ordered_keys}
    keys, w = sample_weights_minmax(rng, base, mc.geo_weight_minmax)
    return keys, w
