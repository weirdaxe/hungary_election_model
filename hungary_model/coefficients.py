from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import BLOCK_BUCKETS, MODEL_PARTIES


def compute_group_coefficients(stations: pd.DataFrame, group_col: str, buckets: List[str]) -> pd.DataFrame:
    """Compute multiplicative geography coefficients: (group share) / (national share)."""
    df = stations.copy()
    g_votes = df.groupby(group_col)[buckets].sum()
    g_total = g_votes.sum(axis=1)
    g_shares = g_votes.div(g_total.replace(0, np.nan), axis=0)

    nat_votes = df[buckets].sum()
    nat_shares = (nat_votes / nat_votes.sum()).replace(0, np.nan)

    coef = g_shares.div(nat_shares, axis=1).replace([np.inf, -np.inf], np.nan)
    return coef


def clip_and_fill_coef(coef: pd.DataFrame, lo: float = 0.05, hi: float = 20.0) -> pd.DataFrame:
    """Clip extreme coefficients and fill missing with 1.0."""
    return coef.clip(lower=lo, upper=hi).fillna(1.0)


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Normalize non-negative weights to sum to 1."""
    w2 = {k: float(v) for k, v in (w or {}).items() if float(v) >= 0}
    s = float(sum(w2.values()))
    if s <= 0:
        return {k: 0.0 for k in (w or {})}
    return {k: v / s for k, v in w2.items()}


def _weighted_group_log_mean(
    log_df: pd.DataFrame,
    groups: pd.Series,
    weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute group-wise weighted mean of a log-coefficient DataFrame.

    If weights is None, uses simple mean.
    If a group has zero total weight, falls back to simple mean.

    Returns: DataFrame indexed by group with same columns as log_df.
    """
    groups = groups.reindex(log_df.index)

    if weights is None:
        return log_df.groupby(groups).mean().fillna(0.0)

    w = weights.reindex(log_df.index).fillna(0.0).astype(float)

    num = log_df.mul(w, axis=0).groupby(groups).sum()
    den = w.groupby(groups).sum()

    out = num.div(den.replace(0.0, np.nan), axis=0)

    # Fallback to unweighted mean if a group has no weight.
    if bool((den <= 0).any()):
        mean = log_df.groupby(groups).mean()
        out = out.fillna(mean)

    return out.fillna(0.0)


def blend_geo_levels_for_election_blocks(
    coef_by_election: Dict[str, Dict[str, pd.DataFrame]],
    station_meta: pd.DataFrame,
    election_key: str,
    geo_weights: Dict[str, float],
    blocks: List[str],
    station_index: pd.Index,
    unit: str,
    station_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Blend coefficients within a single election in log-space.

    - unit='station': returns a station-indexed coefficient matrix.
    - unit='evk': returns an EVK-indexed coefficient matrix.

    Fix: When unit='evk', geo_weights should still matter.
    We compute blended coefficients at station level over the 2026 station universe,
    then aggregate to EVK via a (weighted) geometric mean.
    """
    w = normalize_weights(geo_weights)
    eps = 1e-9

    if election_key not in coef_by_election:
        raise KeyError(f"Missing coefficients for election_key={election_key}")

    c = coef_by_election[election_key]

    def _blend_station_level(station_ids: pd.Index) -> pd.DataFrame:
        station = c["station"].reindex(station_ids).fillna(1.0)

        loc_id = station_meta["station_name_id"].reindex(station_ids)
        settle_id = station_meta["settlement_id"].reindex(station_ids)
        evk_id = station_meta["evk_id"].reindex(station_ids)

        location = c["location"].reindex(loc_id.values).set_index(station_ids)
        settlement = c["settlement"].reindex(settle_id.values).set_index(station_ids)
        evk = c["evk"].reindex(evk_id.values).set_index(station_ids)

        for df in (station, location, settlement, evk):
            for b in blocks:
                if b not in df.columns:
                    df[b] = 1.0

        station = station[blocks]
        location = location[blocks].fillna(1.0)
        settlement = settlement[blocks].fillna(1.0)
        evk = evk[blocks].fillna(1.0)

        log_coef = (
            w.get("station", 0.0) * np.log(station.clip(eps))
            + w.get("location", 0.0) * np.log(location.clip(eps))
            + w.get("settlement", 0.0) * np.log(settlement.clip(eps))
            + w.get("evk", 0.0) * np.log(evk.clip(eps))
        )
        return np.exp(log_coef)

    unit = str(unit).strip().lower()

    if unit == "station":
        return _blend_station_level(station_index)

    if unit == "evk":
        # Aggregate station-level blended coefficients to EVK.
        station_ids = station_meta.index
        st_coef = _blend_station_level(station_ids)

        evk_map = station_meta["evk_id"].reindex(station_ids)
        logc = np.log(st_coef[blocks].clip(eps))

        log_mean = _weighted_group_log_mean(logc, groups=evk_map, weights=station_weights)
        out = np.exp(log_mean)

        out = out.reindex(station_index).fillna(1.0)
        for b in blocks:
            if b not in out.columns:
                out[b] = 1.0
        return out[blocks]

    raise ValueError(f"Unknown unit={unit}")


def build_coefs_2026(
    coef_by_election: Dict[str, Dict[str, pd.DataFrame]],
    station_meta: pd.DataFrame,
    election_weights: Dict[str, float],
    geo_weights: Dict[str, float],
    index: pd.Index,
    unit: str,
    station_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Blend coefficients across elections in BLOCK space, then map to 2026 party space.

    Mapping assumption:
      - TISZA and DK inherit the historical OPP geography.

    station_weights is used only when unit='evk' (for station→EVK aggregation).
    """
    if not election_weights:
        raise ValueError("election_weights missing")
    if not geo_weights:
        raise ValueError("geo_weights missing")

    W = normalize_weights(election_weights)

    blocks = list(BLOCK_BUCKETS)
    eps = 1e-9
    log_sum: Optional[pd.DataFrame] = None

    for e, w_e in W.items():
        coef_e = blend_geo_levels_for_election_blocks(
            coef_by_election=coef_by_election,
            station_meta=station_meta,
            election_key=e,
            geo_weights=geo_weights,
            blocks=blocks,
            station_index=index,
            unit=unit,
            station_weights=station_weights,
        ).clip(eps, 20.0)
        log_e = np.log(coef_e)
        log_sum = log_e * float(w_e) if log_sum is None else log_sum + log_e * float(w_e)

    coef_blocks = np.exp(log_sum) if log_sum is not None else pd.DataFrame(index=index, columns=blocks).fillna(1.0)

    out = pd.DataFrame(index=coef_blocks.index)
    out["FIDESZ"] = coef_blocks["FIDESZ"]
    out["TISZA"] = coef_blocks["OPP"]
    out["DK"] = coef_blocks["OPP"]
    out["MH"] = coef_blocks["MH"]
    out["MKKP"] = coef_blocks["MKKP"]
    out["OTHER"] = coef_blocks["OTHER"]

    for p in MODEL_PARTIES:
        if p not in out.columns:
            out[p] = 1.0

    return out[MODEL_PARTIES].clip(lower=0.05, upper=20.0).fillna(1.0)
