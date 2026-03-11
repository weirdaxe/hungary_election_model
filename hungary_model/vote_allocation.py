
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .constants import MODEL_PARTIES
from .math_utils import logit, sigmoid
from .types import ScenarioConfig


def build_initial_matrix(
    index: pd.Index,
    row_votes: pd.Series,
    coefs: pd.DataFrame,
    national_shares: pd.Series,
    cfg: ScenarioConfig,
    elasticity_beta: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build initial unit×party vote matrix:
      P_ij ∝ coef_ij * national_share_j
      then scaled to row totals.

    Optional: tilt Fidesz vs Tisza along elasticity beta (legacy option).
    """
    parties = list(MODEL_PARTIES)
    row_votes = row_votes.reindex(index).fillna(0.0)
    coefs = coefs.reindex(index).fillna(1.0)
    nat = national_shares.reindex(parties).fillna(0.0)

    P = coefs[parties].values * nat.values[None, :]

    if cfg.undecided_elasticity_link_strength and float(cfg.undecided_elasticity_link_strength) != 0.0 and elasticity_beta is not None:
        beta = elasticity_beta.reindex(index).fillna(1.0).values
        z = (beta - np.mean(beta)) / (np.std(beta) + 1e-9)
        strength = float(cfg.undecided_elasticity_link_strength)
        # sign uses the user's undecided split: if Fidesz gets >=50%, tilt toward high-beta areas for Fidesz.
        sign = 1.0 if float(_normalize_undecided_split(cfg)[0]) >= 0.5 else -1.0
        tilt = np.exp(strength * sign * z)

        j_f = parties.index("FIDESZ")
        j_t = parties.index("TISZA")
        P[:, j_f] *= tilt
        P[:, j_t] /= tilt

    P_sum = P.sum(axis=1, keepdims=True)
    shares = np.where(P_sum > 0, P / P_sum, 1.0 / len(parties))
    M0 = shares * row_votes.values[:, None]
    return pd.DataFrame(M0, index=index, columns=parties)


def ipf_rake(M: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray, max_iter: int = 2000, tol: float = 1e-6) -> np.ndarray:
    """Iterative proportional fitting (raking) to match both row and column totals."""
    M = M.copy().astype(float)
    total_r = float(row_targets.sum())
    total_c = float(col_targets.sum())
    if abs(total_r - total_c) / max(total_c, 1e-9) > 1e-6:
        col_targets = col_targets * (total_r / max(total_c, 1e-9))

    eps = 1e-12
    for it in range(int(max_iter)):
        row_sums = M.sum(axis=1)
        row_f = np.where(row_sums > 0, row_targets / np.maximum(row_sums, eps), 0.0)
        M *= row_f[:, None]

        col_sums = M.sum(axis=0)
        col_f = np.where(col_sums > 0, col_targets / np.maximum(col_sums, eps), 0.0)
        M *= col_f[None, :]

        if it % 25 == 0 or it == max_iter - 1:
            max_row_err = float(np.max(np.abs(M.sum(axis=1) - row_targets)))
            max_col_err = float(np.max(np.abs(M.sum(axis=0) - col_targets)))
            if max(max_row_err, max_col_err) < tol:
                break
    return M


def aggregate_station_to_evk(unit_votes: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    """Aggregate station-level votes to EVK level using the station_meta mapping."""
    df = unit_votes.copy()
    df["evk_id"] = station_meta["evk_id"].reindex(df.index).values
    out = df.groupby("evk_id")[MODEL_PARTIES].sum()
    out.index.name = "evk_id"
    return out


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    den = float(w.sum())
    if den <= 0:
        return float("nan")
    return float((x * w).sum() / den)


def _normalize_undecided_split(cfg: ScenarioConfig) -> Tuple[float, float]:
    uF = float(cfg.undecided_to_fidesz)
    uT = float(cfg.undecided_to_tisza)
    s = uF + uT
    if s <= 0:
        return 0.5, 0.5
    if s > 1.0:
        return uF / s, uT / s
    return uF, uT


def build_undecided_geo_factor(
    station_index: pd.Index,
    reg: pd.Series,
    cfg: ScenarioConfig,
    elasticity_station: pd.DataFrame,
    baseline_turnout_station: pd.Series,
) -> pd.Series:
    """
    Factor used to distribute UNDECIDED_TRUE geographically in baseline+marginal vote model.

    Returned factor has weighted mean 1 (weights=reg).

    Interpretation:
      - uniform: undecideds spread proportionally to registered voters
      - elasticity: undecideds concentrated where turnout is more sensitive to national swings
      - low_turnout: undecideds concentrated where baseline turnout is low
    """
    model = str(cfg.undecided_geo_model).strip().lower()
    if model == "uniform":
        f = pd.Series(1.0, index=station_index)
    elif model == "elasticity":
        beta = elasticity_station.reindex(station_index)["beta"].fillna(1.0)
        z = (beta - beta.mean()) / (beta.std() + 1e-9)
        strength = float(cfg.undecided_elasticity_link_strength or 0.0)
        f = pd.Series(np.exp(strength * z), index=station_index)
    elif model == "low_turnout":
        t0 = baseline_turnout_station.reindex(station_index).fillna(baseline_turnout_station.median())
        f = (1 - t0).clip(lower=1e-3)
    else:
        raise ValueError(f"Unknown undecided_geo_model: {cfg.undecided_geo_model}")

    w = reg.reindex(station_index).fillna(0.0)
    mean_w = _weighted_mean(f, w)
    if mean_w <= 0:
        return pd.Series(1.0, index=station_index)
    return f / mean_w


def _mobilization_vectors(cfg: ScenarioConfig) -> Tuple[dict, dict]:
    """
    Build (mobilization_rate, reserve_strength) dicts per party, based on cfg.

    - If cfg.use_mobilization_model is False: all parties fully mobilized, no reserves.
    - If cfg.mobilization_all_parties is False: defaults match legacy behavior (only FIDESZ/TISZA can have rates < 1).
    - If True: allows user to set rates for all parties.
    """
    parties = list(MODEL_PARTIES)

    if not bool(getattr(cfg, "use_mobilization_model", True)):
        mob = {p: 1.0 for p in parties}
        res = {p: 0.0 for p in parties}
        return mob, res

    mob_in = cfg.mobilization_rates or {}
    res_in = cfg.reserve_strength or {}

    mob = {}
    res = {}
    allow_all = bool(getattr(cfg, "mobilization_all_parties", False))

    for p in parties:
        if (not allow_all) and (p not in ["FIDESZ", "TISZA"]):
            # Legacy behavior: other parties fully mobilized
            mob[p] = 1.0
            res[p] = 0.0
            continue

        # Defaults: keep original priors for the big two; others default to 1.0.
        if p == "FIDESZ":
            mob_default = 0.70
        elif p == "TISZA":
            mob_default = 0.90
        else:
            mob_default = 1.0

        mob[p] = float(np.clip(mob_in.get(p, mob_default), 0.0, 1.0))
        # Reserve strength: if mob==1, reserves are zero anyway, so default=1 is harmless.
        res[p] = float(max(0.0, res_in.get(p, 1.0)))

    return mob, res


def allocate_station_votes_baseline_marginal(
    station_index: pd.Index,
    votes_station: pd.Series,
    reg: pd.Series,
    coefs: pd.DataFrame,
    pop_shares: pd.Series,
    cfg: ScenarioConfig,
    elasticity_station: pd.DataFrame,
    baseline_turnout_station: pd.Series,
) -> pd.DataFrame:
    """
    Endogenous national totals model:

    1) Build a station×(parties+UNDECIDED_TRUE) *population* matrix (row sums = reg, col sums = pop shares)
       using coefficients + undecided geography factor, then IPF-rake it.

    2) Apply mobilization rates per party to get baseline voters (this is the key place where
       you can choose to model turnout differentially by party via cfg.mobilization_all_parties).

    3) Fill remaining turnout (station shortfall) with:
         - party reserves (unmobilized supporters), weighted by reserve_strength
         - then undecideds allocated to parties via cfg undecided split and optional local lean
    """
    parties = list(MODEL_PARTIES)
    reg = reg.reindex(station_index).fillna(0.0)
    votes_station = votes_station.reindex(station_index).fillna(0.0)
    coefs = coefs.reindex(station_index).fillna(1.0)

    pop = pop_shares.reindex(parties + ["UNDECIDED_TRUE"]).fillna(0.0).clip(lower=0.0)
    pop = pop / float(pop.sum()) if float(pop.sum()) > 0 else pop * 0.0

    f_u = build_undecided_geo_factor(station_index, reg, cfg, elasticity_station, baseline_turnout_station)

    # Initial population matrix (counts)
    P = np.zeros((len(station_index), len(parties) + 1), dtype=float)
    for j, p in enumerate(parties):
        P[:, j] = coefs[p].values * float(pop[p])
    P[:, -1] = f_u.values * float(pop["UNDECIDED_TRUE"])

    # row-normalize then scale to reg
    P_sum = P.sum(axis=1, keepdims=True)
    shares = np.where(P_sum > 0, P / P_sum, 1.0 / P.shape[1])
    M0 = shares * reg.values[:, None]

    # IPF so col totals match pop shares exactly
    row_targets = reg.values
    col_targets = pop.values * float(reg.sum())
    Mr = ipf_rake(M0, row_targets=row_targets, col_targets=col_targets, max_iter=2000, tol=1e-5)

    supporters = pd.DataFrame(Mr[:, : len(parties)], index=station_index, columns=parties)
    undec = pd.Series(Mr[:, -1], index=station_index, name="UNDECIDED_TRUE")

    # Mobilization vectors
    mob, res_w = _mobilization_vectors(cfg)

    base_votes = supporters.copy()
    for p in parties:
        base_votes[p] = base_votes[p] * mob[p]

    base_sum = base_votes.sum(axis=1)
    shortfall = (votes_station - base_sum).clip(lower=0.0)

    reserves = supporters.copy()
    for p in parties:
        reserves[p] = reserves[p] * (1 - mob[p]) * res_w[p]

    res_sum = reserves.sum(axis=1).replace(0, np.nan)
    take_res = pd.concat([shortfall, reserves.sum(axis=1)], axis=1).min(axis=1)

    add_from_reserves = reserves.div(res_sum, axis=0).fillna(0.0).mul(take_res, axis=0)
    votes_after_res = base_votes + add_from_reserves

    shortfall2 = (votes_station - votes_after_res.sum(axis=1)).clip(lower=0.0)

    # Undecided voting: allocate remaining shortfall to undecideds
    take_u = pd.concat([shortfall2, undec], axis=1).min(axis=1)

    uF, uT = _normalize_undecided_split(cfg)
    u_rem = max(0.0, 1.0 - (uF + uT))

    base_logit = float(logit(np.array([uF]))[0])

    # Local lean: log(coef_F / coef_T)
    lean = np.log((coefs["FIDESZ"].values + 1e-9) / (coefs["TISZA"].values + 1e-9))
    pf = sigmoid(base_logit + float(cfg.undecided_local_lean_strength or 0.0) * lean)
    pf = np.clip(pf, 0.0, 1.0)
    pt = 1.0 - pf

    votes_after_res["FIDESZ"] += take_u.values * pf
    votes_after_res["TISZA"] += take_u.values * pt

    # Allocate remainder (if any) to other parties pro-rata of their mobilized baseline
    other_parties = [p for p in parties if p not in ["FIDESZ", "TISZA"]]
    other_base = votes_after_res[other_parties].sum(axis=1).replace(0, np.nan)
    for p in other_parties:
        votes_after_res[p] += (take_u.values * u_rem) * (votes_after_res[p] / other_base).fillna(0.0).values

    # If there is still shortfall (rare), distribute proportionally to existing votes
    rem = (votes_station - votes_after_res.sum(axis=1)).clip(lower=0.0)
    if float(rem.sum()) > 1e-6:
        w_row = votes_after_res.sum(axis=1).replace(0, np.nan)
        add = votes_after_res.div(w_row, axis=0).fillna(0.0).mul(rem, axis=0)
        votes_after_res = votes_after_res + add

    # Final adjustment: scale rows to exactly match votes_station (numerical)
    row_sum = votes_after_res.sum(axis=1).replace(0, np.nan)
    votes_final = votes_after_res.div(row_sum, axis=0).fillna(0.0).mul(votes_station, axis=0)
    return votes_final.reindex(columns=parties).fillna(0.0)
