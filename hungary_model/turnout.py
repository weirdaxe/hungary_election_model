
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .constants import TURNOUT_GRANULARITIES, TURNOUT_MODELS
from .math_utils import logit, sigmoid
from .types import ModelData, ScenarioConfig


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    den = float(w.sum())
    if den <= 0:
        return float("nan")
    return float((x * w).sum() / den)


def national_turnout_rate(stations: pd.DataFrame) -> float:
    num = stations["voters_appeared_calc"].sum()
    den = stations["registered_voters_imp"].sum()
    return float(num / den) if den > 0 else float("nan")


def build_group_turnout_panel(panel: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Wide group turnout (sum votes / sum registered) by election."""
    tmp = panel.copy()
    g_votes = tmp.groupby([group_col, "election"])["voters_appeared_calc"].sum()
    g_reg = tmp.groupby([group_col, "election"])["registered_voters_imp"].sum().replace(0, np.nan)
    g_turnout = (g_votes / g_reg).unstack("election")
    return g_turnout


def estimate_turnout_model_logit_slope(
    panel: pd.DataFrame,
    nat_logit: pd.Series,
    group_col: str,
    include_ep_dummy: bool = False,
    k_shrink: float = 4.0,
) -> pd.DataFrame:
    """
    Estimate turnout model at group level in logit space.

    If include_ep_dummy=True:
      y = alpha + beta * nat_logit + gamma * I(EP)

    beta is shrinked toward 1.0 by weight n/(n+k_shrink).
    """
    wide = build_group_turnout_panel(panel, group_col=group_col)
    elections = [e for e in nat_logit.index if e in wide.columns]
    wide = wide[elections]

    wide_logit = wide.apply(lambda col: pd.Series(logit(col.values), index=col.index), axis=0)

    x = nat_logit.reindex(elections).values.astype(float)
    is_ep = np.array([1.0 if str(e).startswith("ep_") else 0.0 for e in elections], dtype=float)

    rows = []
    x_mean = float(np.mean(x))

    for gid, row in wide_logit.iterrows():
        y = row.values.astype(float)
        mask = ~np.isnan(y) & ~np.isnan(x)
        n = int(mask.sum())
        if n < 2:
            rows.append((gid, np.nan, np.nan, np.nan, n))
            continue

        xm = x[mask]
        ym = y[mask]

        if not include_ep_dummy:
            beta = float(np.cov(xm, ym, ddof=0)[0, 1] / np.var(xm, ddof=0))
            gamma = 0.0
            y_mean = float(np.mean(ym))
            alpha = y_mean - beta * x_mean
            rows.append((gid, alpha, beta, gamma, n))
        else:
            epm = is_ep[mask]
            X = np.column_stack([np.ones_like(xm), xm, epm])
            coef, _, _, _ = np.linalg.lstsq(X, ym, rcond=None)
            alpha, beta, gamma = [float(c) for c in coef.tolist()]
            rows.append((gid, alpha, beta, gamma, n))

    out = pd.DataFrame(rows, columns=[group_col, "alpha_raw", "beta_raw", "gamma_raw", "n_obs"]).set_index(group_col)

    beta_prior = 1.0
    w = out["n_obs"] / (out["n_obs"] + float(k_shrink))
    out["beta"] = w * out["beta_raw"].fillna(beta_prior) + (1 - w) * beta_prior

    # Recompute alpha to preserve group mean turnout (in logit space)
    y_mean = wide_logit.mean(axis=1, skipna=True)
    out["alpha"] = y_mean - out["beta"] * x_mean

    out["gamma"] = out["gamma_raw"].fillna(0.0)
    out["beta"] = out["beta"].clip(lower=0.0, upper=3.0)
    return out[["alpha", "beta", "gamma", "n_obs"]]



def build_reference_turnout_map(
    *,
    data: ModelData,
    election_key: str,
    station_index: pd.Index,
    fallback_granularity: str = "settlement",
    turnout_clip: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[pd.Series, float]:
    """
    Build a station-level turnout-rate map aligned to the 2026 station universe.

    Notes
    -----
    2026 has fewer (and also some different) polling-station IDs than previous elections.
    For turnout modelling we therefore:
      1) use station-level turnout where the station exists in the reference election;
      2) otherwise fall back to a higher-level geography (default: settlement) computed from the full
         reference-election dataset; and
      3) finally fall back to national turnout for any remaining missing values.
    """
    key_to_df = {
        "parl_2018_list": data.parl18_list_i,
        "parl_2022_list": data.parl22_list_i,
        "ep_2019_list": data.ep19_i,
        "ep_2024_list": data.ep24_i,
    }
    if election_key not in key_to_df:
        raise KeyError(f"Unsupported turnout reference election: {election_key}")

    df = key_to_df[election_key]
    if df is None or len(df) == 0:
        raise ValueError(f"Reference election dataframe for {election_key} is empty")

    nat_ref = float(data.nat_turnout.get(election_key, np.nan))
    if not np.isfinite(nat_ref):
        nat_ref = float(df["voters_appeared_calc"].sum() / max(df["registered_voters_imp"].sum(), 1.0))

    # Station-level turnout where available
    t_station = df.set_index("station_id")["turnout_rate"]
    t_ref = t_station.reindex(station_index)

    # Group-level fallback (settlement / location / evk) using full reference-election data
    if fallback_granularity in TURNOUT_GRANULARITIES:
        gcol = TURNOUT_GRANULARITIES[fallback_granularity]
        if gcol in df.columns and gcol in data.station_meta.columns:
            votes_g = df.groupby(gcol)["voters_appeared_calc"].sum()
            reg_g = df.groupby(gcol)["registered_voters_imp"].sum()
            t_g = (votes_g / reg_g).replace([np.inf, -np.inf], np.nan)

            gkey = data.station_meta.reindex(station_index)[gcol]
            t_ref = t_ref.combine_first(gkey.map(t_g))

    t_ref = t_ref.fillna(nat_ref).clip(turnout_clip[0], turnout_clip[1]).astype(float)
    return t_ref, nat_ref

def compute_relative_offset_parl(panel: pd.DataFrame, nat_logit: pd.Series, group_col: str) -> pd.Series:
    """
    Parl-only offset:
      offset_g = mean_parl( logit(turnout_g) - logit(turnout_nat) )
    """
    wide = build_group_turnout_panel(panel, group_col=group_col)
    elections = [e for e in nat_logit.index if e in wide.columns and str(e).startswith("parl_")]
    if not elections:
        return pd.Series(0.0, index=wide.index)

    y = wide[elections].apply(lambda col: pd.Series(logit(col.values), index=col.index), axis=0)
    x = nat_logit.reindex(elections)
    off = (y.sub(x, axis=1)).mean(axis=1, skipna=True).fillna(0.0)
    off.name = "offset_parl"
    return off


def compute_marginal_propensity(panel: pd.DataFrame, group_col: str) -> pd.Series:
    """
    Proxy for "marginal voter" density:
      mp_g = max(0, mean_parl(turnout_g) - mean_ep(turnout_g))
    """
    wide = build_group_turnout_panel(panel, group_col=group_col)
    parl_cols = [c for c in wide.columns if str(c).startswith("parl_")]
    ep_cols = [c for c in wide.columns if str(c).startswith("ep_")]

    parl_mean = wide[parl_cols].mean(axis=1, skipna=True) if parl_cols else pd.Series(np.nan, index=wide.index)
    ep_mean = wide[ep_cols].mean(axis=1, skipna=True) if ep_cols else pd.Series(np.nan, index=wide.index)

    mp = (parl_mean - ep_mean).fillna(0.0)
    mp = mp.clip(lower=0.0)
    mp.name = "marginal_propensity"
    return mp


def enforce_turnout_caps_and_total(
    votes: pd.Series,
    reg: pd.Series,
    total_target: float,
    clip: Tuple[float, float],
    max_iter: int = 20,
) -> pd.Series:
    """
    Iteratively:
      - clip turnout rates to [clip_lo, clip_hi]
      - rescale all votes to match total_target
    """
    v = votes.copy().astype(float)
    for _ in range(int(max_iter)):
        t = (v / reg.replace(0, np.nan)).fillna(0.0)
        t_clipped = t.clip(lower=float(clip[0]), upper=float(clip[1]))
        v = (t_clipped * reg).fillna(0.0)

        s = float(v.sum())
        if s <= 0:
            break

        f = float(total_target) / s
        v2 = v * f

        if float(np.max(np.abs(v2 - v))) / max(1.0, float(v.sum())) < 1e-9:
            v = v2
            break
        v = v2
    return v


def distribute_group_votes_to_stations(
    group_votes: pd.Series,
    group_key: pd.Series,
    reg: pd.Series,
    baseline_turnout_station: pd.Series,
) -> pd.Series:
    """Allocate group total votes to stations proportionally to baseline station votes within group."""
    base_t = baseline_turnout_station.reindex(reg.index)
    if base_t.isna().any():
        base_t = base_t.fillna(_weighted_mean(baseline_turnout_station.dropna(), reg.reindex(baseline_turnout_station.index).fillna(0)))

    w = (reg * base_t).fillna(0.0)

    df = pd.DataFrame({"group": group_key.values, "w": w.values, "reg": reg.values}, index=reg.index)
    wsum = df.groupby("group")["w"].transform("sum").replace(0, np.nan)
    share = (df["w"] / wsum).fillna(df["reg"] / df.groupby("group")["reg"].transform("sum").replace(0, np.nan)).fillna(0.0)

    gv = group_votes.reindex(df["group"]).fillna(0.0).values
    station_votes = gv * share.values
    return pd.Series(station_votes, index=reg.index, name="votes_station")


def get_group_key(granularity: str, station_meta: pd.DataFrame, station_index: pd.Index) -> pd.Series:
    if granularity not in TURNOUT_GRANULARITIES:
        raise ValueError(f"Unknown turnout_granularity: {granularity}")

    col = TURNOUT_GRANULARITIES[granularity]
    if col == "station_id":
        return pd.Series(station_index, index=station_index, name="group")

    key = station_meta.reindex(station_index)[col].copy()
    # If some stations are missing the chosen geography, fall back to station_id so that every station
    # participates in group-based allocations (important when 2026 station IDs differ from historical ones).
    if key.isna().any():
        key = key.astype(object)
        mask = key.isna()
        key.loc[mask] = station_index[mask]
    key.name = "group"
    return key


def predict_station_votes_2026(data: ModelData, cfg: ScenarioConfig) -> Tuple[pd.Series, pd.Series]:
    """
    Predict station-level vote totals (voters appeared) for 2026.

    Returns:
      votes_station (Series indexed by station_id)
      turnout_rate_station (Series indexed by station_id)

    Notes:
    - This module models the *geographic distribution of turnout* (how many people vote in each area).
    - It is separate from the vote-allocation model, which determines *who* those voters vote for.
    
    Turnout models implemented
    -------------------------
    All models ultimately produce station vote totals `votes_i = turnout_i × registered_i` and then
    `enforce_turnout_caps_and_total()` is applied to guarantee:

    - `turnout_i ∈ [turnout_clip_low, turnout_clip_high]`
    - `Σ votes_i = turnout_target × Σ registered_i`

    The pre-enforcement logic differs by model:

    - `uniform`: `turnout_i = turnout_target`
    - `scaled_reference`: `votes_i ∝ (reference_turnout_i × registered_i)` scaled to national target
    - `logit_slope_original`: station-level logit-slope elasticity: `logit(t_i) = α_i + β_i logit(T)`
    - `logit_slope_ep_offset`: grouped logit-slope with EP dummy; parliamentary prediction uses EP=0
    - `relative_offset_parl`: grouped parliamentary-only logit offset added to `logit(T)`
    - `logit_slope_reserve_adjusted`: reweights the change vs the reference map toward groups with more headroom
    - `baseline_plus_marginal`: baseline from reference map + allocate Δvotes by a marginal-propensity proxy
    - `baseline_ref_plus_elastic_delta`: anchor a baseline turnout level B with the reference map,
      and allocate the surplus/deficit using elasticity-implied deltas between B and T
    """
    if cfg.turnout_model not in TURNOUT_MODELS:
        raise ValueError(f"Unknown turnout_model: {cfg.turnout_model}")

    station_index = data.registered_2026.index
    reg = data.registered_2026.reindex(station_index).fillna(0.0)
    total_reg = float(reg.sum())
    target_total_votes = float(cfg.turnout_target) * total_reg

    group_key = get_group_key(str(cfg.turnout_granularity), data.station_meta, station_index)

    # Baseline reference map (aligned to 2026 stations)
    clip_lo, clip_hi = float(cfg.turnout_clip[0]), float(cfg.turnout_clip[1])

    # Reference turnout map on the 2026 station universe. We use station-level turnout where the station exists
    # in the reference election; otherwise fall back to settlement-level turnout from the full reference dataset.
    t_ref, nat_ref = build_reference_turnout_map(
        data=data,
        election_key=cfg.turnout_reference_election,
        station_index=station_index,
        fallback_granularity="settlement",
        turnout_clip=(clip_lo, clip_hi),
    )

    direction = float(np.sign(float(cfg.turnout_target) - nat_ref))

    if cfg.turnout_model == "uniform":
        votes = pd.Series(float(cfg.turnout_target), index=station_index) * reg

    elif cfg.turnout_model == "scaled_reference":
        base_votes = (t_ref * reg).fillna(0.0)
        base_total = float(base_votes.sum())
        if base_total <= 0:
            votes = pd.Series(float(cfg.turnout_target), index=station_index) * reg
        else:
            votes = base_votes * (target_total_votes / base_total)

    elif cfg.turnout_model == "logit_slope_original":
        el = data.elasticity_station_original.reindex(station_index).fillna({"alpha": 0.0, "beta": 1.0, "gamma": 0.0})
        x = float(logit(np.array([cfg.turnout_target]))[0])
        t_hat = sigmoid((el["alpha"].values + el["beta"].values * x))
        t_hat = np.clip(t_hat, clip_lo, clip_hi)
        votes = pd.Series(t_hat, index=station_index) * reg

    elif cfg.turnout_model == "logit_slope_ep_offset":
        el = data.elasticity_ep_offset[str(cfg.turnout_granularity)]
        x = float(logit(np.array([cfg.turnout_target]))[0])
        t_g = sigmoid((el["alpha"] + el["beta"] * x).values)  # EP dummy=0 for 2026
        t_g = np.clip(t_g, clip_lo, clip_hi)
        t_g = pd.Series(t_g, index=el.index)

        reg_g = reg.groupby(group_key).sum()
        votes_g = t_g.reindex(reg_g.index).fillna(float(cfg.turnout_target)) * reg_g
        votes = distribute_group_votes_to_stations(votes_g, group_key, reg, t_ref)

    elif cfg.turnout_model == "relative_offset_parl":
        off = data.offset_parl[str(cfg.turnout_granularity)]
        x = float(logit(np.array([cfg.turnout_target]))[0])
        y_g = x + off.reindex(off.index).fillna(0.0).values
        t_g = sigmoid(y_g)
        t_g = np.clip(t_g, clip_lo, clip_hi)
        t_g = pd.Series(t_g, index=off.index)

        reg_g = reg.groupby(group_key).sum()
        votes_g = t_g.reindex(reg_g.index).fillna(float(cfg.turnout_target)) * reg_g
        votes = distribute_group_votes_to_stations(votes_g, group_key, reg, t_ref)

    elif cfg.turnout_model == "logit_slope_reserve_adjusted":
        el = data.elasticity_ep_offset[str(cfg.turnout_granularity)]
        x = float(logit(np.array([cfg.turnout_target]))[0])
        t_g_raw = sigmoid((el["alpha"] + el["beta"] * x).values)
        t_g_raw = np.clip(t_g_raw, clip_lo, clip_hi)
        t_g_raw = pd.Series(t_g_raw, index=el.index)

        # baseline group turnout from reference election map
        ref_votes = (t_ref * reg).fillna(0.0)
        ref_reg_g = reg.groupby(group_key).sum()
        ref_votes_g = ref_votes.groupby(group_key).sum()
        t_g_base = (ref_votes_g / ref_reg_g.replace(0, np.nan)).fillna(float(cfg.turnout_target))

        delta = (t_g_raw.reindex(ref_reg_g.index).fillna(float(cfg.turnout_target)) - t_g_base)

        if direction >= 0:
            reserve = (clip_hi - t_g_base).clip(lower=0.0)
            scale = (reserve / max(1e-9, float(reserve.mean()))) ** float(cfg.reserve_adjust_power)
        else:
            reserve = (t_g_base - clip_lo).clip(lower=0.0)
            scale = (reserve / max(1e-9, float(reserve.mean()))) ** float(cfg.reserve_adjust_power)

        t_g_adj = (t_g_base + delta * scale).clip(lower=clip_lo, upper=clip_hi)
        votes_g = t_g_adj * ref_reg_g
        votes = distribute_group_votes_to_stations(votes_g, group_key, reg, t_ref)

    elif cfg.turnout_model == "baseline_plus_marginal":
        base_votes = (t_ref * reg).fillna(0.0)
        base_total = float(base_votes.sum())
        delta_total = target_total_votes - base_total

        if abs(delta_total) < 1e-9:
            votes = base_votes
        else:
            mp = data.marginal_propensity[str(cfg.turnout_granularity)]
            mp_station = mp.reindex(group_key.values).fillna(0.0).values
            mp_station = np.maximum(mp_station, 0.0)

            if delta_total > 0:
                cap = (clip_hi - t_ref).clip(lower=0.0).values
            else:
                cap = (t_ref - clip_lo).clip(lower=0.0).values

            w = (mp_station + 1e-9) ** float(cfg.marginal_concentration)
            w = w * (cap + 1e-9)
            w_sum = float(np.sum(w))
            if w_sum <= 0:
                w = np.ones_like(w)
                w_sum = float(np.sum(w))

            add = delta_total * (w / w_sum)
            votes = base_votes + pd.Series(add, index=station_index)

    elif cfg.turnout_model == "baseline_ref_plus_elastic_delta":
        # 1) Anchor a baseline distribution at a fixed turnout level (e.g., 70%) using the chosen reference election.
        # 2) Allocate the surplus/deficit vs that baseline using the turnout-elasticity model.
        baseline_level = float(cfg.turnout_baseline_level)
        baseline_level = float(np.clip(baseline_level, clip_lo, clip_hi))

        baseline_total_votes = baseline_level * total_reg
        base_votes_raw = (t_ref * reg).fillna(0.0)
        raw_sum = float(base_votes_raw.sum())
        if raw_sum <= 0:
            base_votes = pd.Series(baseline_level, index=station_index) * reg
        else:
            base_votes = base_votes_raw * (baseline_total_votes / raw_sum)

        delta_total = target_total_votes - baseline_total_votes
        if abs(delta_total) <= 1e-9:
            votes = base_votes
        else:
            # Elasticity-implied delta pattern across groups (from baseline_level to turnout_target)
            el = data.elasticity_ep_offset[str(cfg.turnout_granularity)]
            reg_g = reg.groupby(group_key).sum()

            alpha = el["alpha"].reindex(reg_g.index).fillna(0.0)
            beta = el["beta"].reindex(reg_g.index).fillna(1.0)

            x_base = logit(baseline_level)
            x_target = logit(float(cfg.turnout_target))

            t_g_base = sigmoid(alpha + beta * x_base).clip(lower=clip_lo, upper=clip_hi)
            t_g_target = sigmoid(alpha + beta * x_target).clip(lower=clip_lo, upper=clip_hi)

            delta_g = (t_g_target - t_g_base) * reg_g
            sum_delta_g = float(delta_g.sum())

            if abs(sum_delta_g) <= 1e-12:
                # fallback: spread delta proportional to registered voters
                delta_g = (reg_g / max(float(reg_g.sum()), 1.0)) * delta_total
            else:
                delta_g = delta_g * (delta_total / sum_delta_g)

            station_delta = distribute_group_votes_to_stations(delta_g, group_key, reg, t_ref)
            votes = base_votes + station_delta

    else:
        raise ValueError(f"Unknown turnout_model: {cfg.turnout_model}")

    # Enforce caps and exact national turnout total
    votes = enforce_turnout_caps_and_total(votes, reg, total_target=target_total_votes, clip=(clip_lo, clip_hi))
    turnout_rate = (votes / reg.replace(0, np.nan)).fillna(0.0).clip(lower=clip_lo, upper=clip_hi)

    return votes.rename("votes_station_2026"), turnout_rate.rename("turnout_rate_2026")
