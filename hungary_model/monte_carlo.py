from __future__ import annotations

import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import multiprocessing as mp

import numpy as np
import pandas as pd

from .constants import MODEL_PARTIES
from .mc_coefficients import apply_coef_noise, build_log_coef_components_evk
from .mc_diaspora import sample_diaspora_votes_with_meta
from .mc_distributions import sample_float, sample_weights_minmax
from .mc_national_shares import sample_share_vector
from .mc_turnout import (
    build_turnout_row_votes_cache,
    choose_turnout_model_and_granularity,
    interpolate_row_votes,
    make_turnout_grid_targets,
    sample_turnout_target,
)
from .scenario import _is_budapest_evk
from .types import ModelData, MonteCarloConfig, MonteCarloResults, ScenarioResults


def _default_workers() -> int:
    try:
        return max(1, (os.cpu_count() or 1) - 1)
    except Exception:
        return 1


def _build_initial_matrix(coefs: np.ndarray, nat: np.ndarray, row_votes: np.ndarray) -> np.ndarray:
    """Initial positive matrix for IPF: coefs * nat, then row-scaled to row_votes."""

    mat = coefs * nat[None, :]
    mat = np.clip(mat, 1e-12, None)
    rs = mat.sum(axis=1)
    rs = np.where(rs <= 0, 1.0, rs)
    mat = mat / rs[:, None]
    mat = mat * row_votes[:, None]
    return mat


def ipf_rake_numpy(
    mat: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    max_iter: int = 250,
    tol: float = 1e-4,
    check_every: int = 10,
) -> np.ndarray:
    """Iterative proportional fitting (raking) for a dense matrix.

    mat must be strictly positive.
    """

    X = np.asarray(mat, dtype=float)
    r = np.asarray(row_targets, dtype=float)
    c = np.asarray(col_targets, dtype=float)

    r = np.clip(r, 0.0, None)
    c = np.clip(c, 0.0, None)

    # Avoid division by zero in targets
    r_tot = float(r.sum())
    c_tot = float(c.sum())
    if r_tot <= 0 or c_tot <= 0:
        return np.zeros_like(X)

    # Force consistent totals
    if abs(r_tot - c_tot) > 1e-6:
        c = c * (r_tot / c_tot)

    for it in range(max_iter):
        # Row scaling
        rs = X.sum(axis=1)
        rs = np.where(rs == 0, 1.0, rs)
        X *= (r / rs)[:, None]

        # Column scaling
        cs = X.sum(axis=0)
        cs = np.where(cs == 0, 1.0, cs)
        X *= (c / cs)[None, :]

        if (it + 1) % check_every == 0:
            max_row_err = float(np.max(np.abs(X.sum(axis=1) - r)))
            max_col_err = float(np.max(np.abs(X.sum(axis=0) - c)))
            if max(max_row_err, max_col_err) < tol:
                break

    return X


def _dhondt_alloc_fast(votes: np.ndarray, seats: int) -> np.ndarray:
    """Fast D'Hondt allocation for small party counts.

    votes: shape (n_parties,)
    returns: seats per party
    """

    v = np.asarray(votes, dtype=float)
    v = np.clip(v, 0.0, None)
    n_p = v.shape[0]
    if seats <= 0 or v.sum() <= 0:
        return np.zeros(n_p, dtype=int)

    div = np.arange(1, seats + 1, dtype=float)
    quot = (v[:, None] / div[None, :]).ravel()

    # top-k indices
    k = int(seats)
    idx = np.argpartition(quot, -k)[-k:]
    party_idx = idx // seats
    out = np.bincount(party_idx, minlength=n_p)
    return out.astype(int)


def _top2_indices_values(votes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return winner_idx, runner_idx, winner_val, runner_val for each row."""

    # votes shape: (n_rows, n_cols)
    n_rows, _ = votes.shape

    top2 = np.argpartition(votes, -2, axis=1)[:, -2:]
    vals = votes[np.arange(n_rows)[:, None], top2]
    order = np.argsort(vals, axis=1)

    runner_idx = top2[np.arange(n_rows), order[:, 0]]
    winner_idx = top2[np.arange(n_rows), order[:, 1]]

    runner_val = vals[np.arange(n_rows), order[:, 0]]
    winner_val = vals[np.arange(n_rows), order[:, 1]]

    return winner_idx, runner_idx, winner_val, runner_val


def _poll_to_population_shares(
    poll_probs: np.ndarray,
    *,
    nonresponse_nonvoter: float,
) -> Tuple[np.ndarray, float]:
    """Convert raw poll shares (including UNDECIDED) to population shares.

    poll_probs: length n_parties+1 where last entry is UNDECIDED.

    nonresponse_nonvoter: additional share (0..1) to add to UNDECIDED_TRUE.

    Returns (party_pop_probs, undecided_true_prob).
    """

    p = np.asarray(poll_probs, dtype=float)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0:
        n = len(p)
        p = np.ones(n, dtype=float) / n
    else:
        p = p / s

    parties = p[:-1]
    undec = float(p[-1])

    nv = float(np.clip(nonresponse_nonvoter, 0.0, 0.95))
    undec_true = float(np.clip(undec + nv, 0.0, 0.999))

    decided_poll = max(0.0, 1.0 - undec)
    decided_true = max(0.0, 1.0 - undec_true)

    if decided_poll <= 1e-12:
        parties_true = np.zeros_like(parties)
    else:
        scale = decided_true / decided_poll
        parties_true = parties * scale

    # Numerical normalize (parties_true sum might be slightly off)
    parties_sum = float(parties_true.sum())
    if parties_sum > 0:
        parties_true = parties_true * (decided_true / parties_sum)

    return parties_true, undec_true


def _poll_population_to_voter_shares(
    parties_pop: np.ndarray,
    undec_true: float,
    *,
    turnout_target: float,
    u_fidesz: float,
    u_others: float,
    use_mobilization: bool,
    mob_f: float,
    mob_t: float,
    res_f: float,
    res_t: float,
    j_f: int,
    j_t: int,
) -> np.ndarray:
    """Convert population shares to voter shares using the marginal-turnout model."""

    turnout = float(np.clip(turnout_target, 0.20, 0.95))

    votes = np.asarray(parties_pop, dtype=float).copy()
    votes = np.clip(votes, 0.0, None)

    decided_sum = float(votes.sum())
    if decided_sum <= 1e-12:
        # No decided mass -> fall back to two-party split
        out = np.zeros_like(votes)
        out[j_f] = float(np.clip(u_fidesz, 0.0, 1.0))
        out[j_t] = 1.0 - out[j_f]
        return out

    if turnout <= decided_sum + 1e-12:
        # Turnout is fully within the decided mass -> composition unchanged
        out = votes / decided_sum
        return out

    extra = min(turnout - decided_sum, float(max(0.0, undec_true)))

    uO = float(np.clip(u_others, 0.0, 1.0))
    remain = max(0.0, 1.0 - uO)

    uF = float(np.clip(u_fidesz, 0.0, remain))
    uT = float(max(0.0, remain - uF))

    # Mobilization tilt: reweight uF/uT by reserve size
    if use_mobilization:
        mob_f = float(np.clip(mob_f, 0.0, 1.0))
        mob_t = float(np.clip(mob_t, 0.0, 1.0))
        res_f = float(np.clip(res_f, 0.0, 1.0))
        res_t = float(np.clip(res_t, 0.0, 1.0))

        reserveF = (1.0 - mob_f) * res_f + 1e-9
        reserveT = (1.0 - mob_t) * res_t + 1e-9

        uF_adj = uF * reserveF
        uT_adj = uT * reserveT
        s_adj = uF_adj + uT_adj
        if s_adj > 0:
            uF = uF_adj / s_adj
            uT = uT_adj / s_adj

    # Allocate marginal voters
    votes[j_f] += extra * uF
    votes[j_t] += extra * uT

    if uO > 0:
        other_mask = np.ones(len(votes), dtype=bool)
        other_mask[[j_f, j_t]] = False
        other_base = votes * other_mask
        other_sum = float(other_base.sum())
        if other_sum > 0:
            votes += (extra * uO) * (other_base / other_sum)
        else:
            # If no other parties, dump the remainder into the two-party split.
            votes[j_f] += extra * uO * uF
            votes[j_t] += extra * uO * uT

    # Return shares among voters
    s = float(votes.sum())
    if s <= 0:
        return np.ones_like(votes) / len(votes)
    return votes / s


def _estimate_budapest_shares_from_block(
    bp_block_shares: np.ndarray,
    *,
    tisza_dk_ratio: float,
    j_f: int,
    j_t: int,
    j_dk: int,
) -> np.ndarray:
    """Budapest party shares given Budapest BLOCK shares and a TISZA:DK ratio."""

    # bp_block_shares order: [FIDESZ, OPP, MH, MKKP, OTHER]
    f, opp, mh, mkkp, other = [float(x) for x in bp_block_shares]

    ratio = float(np.clip(tisza_dk_ratio, 0.0, 1.0))

    out = np.zeros(len(MODEL_PARTIES), dtype=float)
    out[j_f] = f
    out[j_t] = opp * ratio
    out[j_dk] = opp * (1.0 - ratio)
    out[MODEL_PARTIES.index("MH")] = mh
    out[MODEL_PARTIES.index("MKKP")] = mkkp
    out[MODEL_PARTIES.index("OTHER")] = other

    # Enforce TISZA > FIDESZ by shifting DK -> TISZA if necessary
    if out[j_t] < out[j_f]:
        needed = out[j_f] - out[j_t] + 1e-6
        take = min(out[j_dk], needed)
        out[j_dk] -= take
        out[j_t] += take

    s = float(out.sum())
    if s > 0:
        out /= s
    return out


def _simulate_chunk(payload: Dict[str, Any], n_sims: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))

    mc: MonteCarloConfig = payload["mc_cfg"]

    n_units = int(payload["n_units"])
    n_parties = int(payload["n_parties"])

    # Indices
    j_f = int(payload["j_f"])
    j_t = int(payload["j_t"])
    j_dk = int(payload["j_dk"])

    # Precomputed components
    coef_components = payload["coef_components"]  # (E,L,U,P) log-space
    base_election_weights = payload["base_election_weights"]
    base_geo_weights = payload["base_geo_weights"]

    # Turnout cache
    grid_targets = payload["turnout_grid_targets"]
    turnout_cache = payload["turnout_row_votes_cache"]
    turnout_models = payload["turnout_models"]
    turnout_grans = payload["turnout_granularities"]
    base_turnout_target = float(payload["base_turnout_target"])
    total_registered = float(payload["total_registered"])

    # Poll base
    base_nat_probs = payload["base_nat_probs"]  # (P,)
    base_poll_probs_raw = payload["base_poll_probs_raw"]  # (P+1,)
    base_poll_type = payload["base_poll_type"]

    base_nonresponse = float(payload["base_nonresponse_nonvoter"])
    base_und_to_f = float(payload["base_undecided_to_fidesz"])
    base_und_to_others = float(payload["base_undecided_to_others"])

    base_use_mobilization = bool(payload["base_use_mobilization"])
    base_mob_f = float(payload["base_mob_f"])
    base_mob_t = float(payload["base_mob_t"])
    base_res_f = float(payload["base_res_f"])
    base_res_t = float(payload["base_res_t"])

    # Seats config
    smc_plus1_minus1 = bool(payload["smc_plus1_minus1"])
    winner_two_party_only = bool(payload["winner_two_party_only"])
    threshold = float(payload["threshold"])
    list_seats_n = int(payload["list_seats_n"])
    nationality_seats = payload["nationality_seats"].astype(int)

    # Budapest handling
    exclude_budapest = bool(payload["exclude_budapest"])
    mask_bp = payload["mask_bp"]
    mask_nb = payload["mask_nb"]
    bp_block_shares = payload.get("bp_block_shares", None)

    base_diaspora = payload["base_diaspora"]

    seats_draws = np.zeros((n_sims, n_parties), dtype=np.int16)
    nat_draws = np.zeros((n_sims, n_parties), dtype=np.float32)
    turnout_draws = np.zeros(n_sims, dtype=np.float32)

    # --- Per-draw input capture (for diagnostics) ---
    parties: List[str] = payload.get("parties", []) or []
    coef_elections: List[str] = payload.get("coef_elections", list(base_election_weights.keys()))
    coef_levels: List[str] = payload.get("coef_levels", list(base_geo_weights.keys()))

    turnout_model_draws = np.empty(n_sims, dtype=object)
    turnout_granularity_draws = np.empty(n_sims, dtype=object)
    poll_type_draws = np.empty(n_sims, dtype=object)

    poll_undecided_share_draws = np.full(n_sims, np.nan, dtype=np.float32)
    undecided_true_share_draws = np.full(n_sims, np.nan, dtype=np.float32)
    nonresponse_nonvoter_pct_draws = np.full(n_sims, np.nan, dtype=np.float32)
    undecided_to_fidesz_draws = np.full(n_sims, np.nan, dtype=np.float32)

    use_mobilization_draws = np.empty(n_sims, dtype=object)
    use_mobilization_draws[:] = None

    mob_f_draws = np.full(n_sims, np.nan, dtype=np.float32)
    mob_t_draws = np.full(n_sims, np.nan, dtype=np.float32)
    res_f_draws = np.full(n_sims, np.nan, dtype=np.float32)
    res_t_draws = np.full(n_sims, np.nan, dtype=np.float32)

    diaspora_total_draws = np.zeros(n_sims, dtype=np.int32)
    diaspora_fidesz_share_draws = np.full(n_sims, np.nan, dtype=np.float32)

    nat_target_draws = np.zeros((n_sims, n_parties), dtype=np.float32)

    w_e_draws = np.zeros((n_sims, len(coef_elections)), dtype=np.float32)
    w_g_draws = np.zeros((n_sims, len(coef_levels)), dtype=np.float32)

    # Precompute party indices for two-party restriction
    if winner_two_party_only:
        two_party_idx = np.array([j_f, j_t], dtype=int)

    for i in range(n_sims):
        # --- Turnout ---
        t_target = sample_turnout_target(mc, base_turnout_target, rng)
        turnout_draws[i] = float(t_target)

        # Randomly choose turnout model & granularity if requested
        # (Done here instead of via mc_turnout helper to avoid pickling ScenarioConfig)
        t_model = str(rng.choice(turnout_models)) if turnout_models else payload["base_turnout_model"]
        t_gran = str(rng.choice(turnout_grans)) if turnout_grans else payload["base_turnout_granularity"]

        turnout_model_draws[i] = t_model
        turnout_granularity_draws[i] = t_gran

        rv_grid = turnout_cache[(t_model, t_gran)]
        row_votes = interpolate_row_votes(grid_targets, rv_grid, t_target)

        # Rescale to exact target total votes
        tgt_total = t_target * total_registered
        cur_total = float(row_votes.sum())
        if cur_total > 0:
            row_votes = row_votes * (tgt_total / cur_total)

        # --- Poll / national shares ---
        poll_mode = mc.poll_mode
        if poll_mode == "scenario":
            poll_mode = "raw" if str(base_poll_type).lower().startswith("raw") else "decided"

        poll_type_draws[i] = poll_mode

        if poll_mode == "decided":
            nat_probs = sample_share_vector(base_nat_probs, mc, rng)

        else:
            poll_probs = sample_share_vector(base_poll_probs_raw, mc, rng)
            poll_undecided_share_draws[i] = float(poll_probs[-1])

            nv = sample_float(rng, mc.nonresponse_nonvoter_spec, base_nonresponse, hard_min=0.0, hard_max=0.50)
            nonresponse_nonvoter_pct_draws[i] = float(nv)

            parties_pop, undec_true = _poll_to_population_shares(poll_probs, nonresponse_nonvoter=float(nv))
            undecided_true_share_draws[i] = float(undec_true)

            uF = sample_float(rng, mc.undecided_to_fidesz_spec, base_und_to_f, hard_min=0.0, hard_max=1.0)
            undecided_to_fidesz_draws[i] = float(uF)

            use_mob = base_use_mobilization
            if mc.use_mobilization_choices:
                use_mob = bool(rng.choice(mc.use_mobilization_choices))

            use_mobilization_draws[i] = bool(use_mob)

            mob_f = sample_float(rng, mc.mobilization_rate_fidesz_spec, base_mob_f, hard_min=0.0, hard_max=1.0)
            mob_t = sample_float(rng, mc.mobilization_rate_tisza_spec, base_mob_t, hard_min=0.0, hard_max=1.0)

            res_f = sample_float(rng, mc.reserve_strength_fidesz_spec, base_res_f, hard_min=0.0, hard_max=1.0)
            res_t = sample_float(rng, mc.reserve_strength_tisza_spec, base_res_t, hard_min=0.0, hard_max=1.0)

            mob_f_draws[i] = float(mob_f)
            mob_t_draws[i] = float(mob_t)
            res_f_draws[i] = float(res_f)
            res_t_draws[i] = float(res_t)

            nat_probs = _poll_population_to_voter_shares(
                parties_pop,
                undec_true,
                turnout_target=t_target,
                u_fidesz=uF,
                u_others=base_und_to_others,
                use_mobilization=use_mob,
                mob_f=mob_f,
                mob_t=mob_t,
                res_f=res_f,
                res_t=res_t,
                j_f=j_f,
                j_t=j_t,
            )

        nat_target_draws[i, :] = nat_probs.astype(np.float32)

        # --- Coefficients (sample weights + noise) ---
        _, w_e = sample_weights_minmax(rng, base_election_weights, mc.election_weight_minmax)
        _, w_g = sample_weights_minmax(rng, base_geo_weights, mc.geo_weight_minmax)

        if w_e_draws.shape[1] == len(w_e):
            w_e_draws[i, :] = w_e.astype(np.float32)
        else:
            n = min(w_e_draws.shape[1], len(w_e))
            w_e_draws[i, :n] = w_e[:n].astype(np.float32)

        if w_g_draws.shape[1] == len(w_g):
            w_g_draws[i, :] = w_g.astype(np.float32)
        else:
            n = min(w_g_draws.shape[1], len(w_g))
            w_g_draws[i, :n] = w_g[:n].astype(np.float32)

        # log_coef = sum_e sum_l w_e[e]*w_g[l]*coef_components[e,l]
        tmp = np.tensordot(w_e, coef_components, axes=(0, 0))  # (L,U,P)
        log_coef = np.tensordot(w_g, tmp, axes=(0, 0))  # (U,P)
        coefs = np.exp(log_coef)
        coefs = apply_coef_noise(coefs, mc, rng)

        # --- Diaspora votes ---
        diaspora, diaspora_total, diaspora_f_share = sample_diaspora_votes_with_meta(base_diaspora, mc, rng)
        diaspora_total_draws[i] = int(diaspora_total)
        diaspora_fidesz_share_draws[i] = float(diaspora_f_share)

        # --- Build unit vote matrix via IPF ---
        if exclude_budapest:
            # Non-Budapest IPF
            row_nb = row_votes[mask_nb]
            coefs_nb = coefs[mask_nb]

            total_nb = float(row_nb.sum())
            nat_nb = nat_probs
            col_nb = nat_nb * total_nb

            init_nb = _build_initial_matrix(coefs_nb, nat_nb, row_nb)
            mat_nb = ipf_rake_numpy(init_nb, row_nb, col_nb, max_iter=mc.ipf_max_iter, tol=mc.ipf_tol)

            # Budapest fixed shares
            row_bp = row_votes[mask_bp]

            # ratio for splitting OPP between TISZA/DK
            denom = float(nat_nb[j_t] + nat_nb[j_dk])
            ratio = float(nat_nb[j_t] / denom) if denom > 0 else 1.0

            bp_shares = _estimate_budapest_shares_from_block(bp_block_shares, tisza_dk_ratio=ratio, j_f=j_f, j_t=j_t, j_dk=j_dk)
            mat_bp = row_bp[:, None] * bp_shares[None, :]

            # Combine
            mat = np.zeros((n_units, n_parties), dtype=float)
            mat[mask_nb, :] = mat_nb
            mat[mask_bp, :] = mat_bp

        else:
            total_votes = float(row_votes.sum())
            col_targets = nat_probs * total_votes
            init = _build_initial_matrix(coefs, nat_probs, row_votes)
            mat = ipf_rake_numpy(init, row_votes, col_targets, max_iter=mc.ipf_max_iter, tol=mc.ipf_tol)

        # --- Realized national shares (domestic) ---
        tot_dom = float(mat.sum())
        if tot_dom > 0:
            nat_draws[i, :] = (mat.sum(axis=0) / tot_dom).astype(np.float32)
        else:
            nat_draws[i, :] = (np.ones(n_parties) / n_parties).astype(np.float32)

        # --- Seat simulation ---
        smc_votes = mat

        # Apply +1pp/-1pp SMC tweak (shares domain)
        if smc_plus1_minus1:
            row_sums = smc_votes.sum(axis=1)
            row_sums_safe = np.where(row_sums <= 0, 1.0, row_sums)
            shares = smc_votes / row_sums_safe[:, None]
            shares[:, j_f] += 0.01
            shares[:, j_t] = np.clip(shares[:, j_t] - 0.01, 0.0, 1.0)
            shares_sum = shares.sum(axis=1)
            shares_sum = np.where(shares_sum <= 0, 1.0, shares_sum)
            shares = shares / shares_sum[:, None]
            smc_votes = shares * row_sums_safe[:, None]

        # Winners & runners
        if winner_two_party_only:
            two_votes = smc_votes[:, two_party_idx]
            win_in_two = np.argmax(two_votes, axis=1)
            winner_idx = two_party_idx[win_in_two]
            runner_idx = two_party_idx[1 - win_in_two]
            winner_val = two_votes[np.arange(two_votes.shape[0]), win_in_two]
            runner_val = two_votes[np.arange(two_votes.shape[0]), 1 - win_in_two]
        else:
            winner_idx, runner_idx, winner_val, runner_val = _top2_indices_values(smc_votes)

        # Force Budapest EVKs to TISZA in exclude mode
        if exclude_budapest:
            winner_idx = winner_idx.copy()
            runner_idx = runner_idx.copy()
            winner_val = winner_val.copy()
            runner_val = runner_val.copy()

            bp_rows = mask_bp
            winner_idx[bp_rows] = j_t
            if winner_two_party_only:
                runner_idx[bp_rows] = j_f
            else:
                # runner is best among remaining parties
                v = smc_votes[bp_rows].copy()
                v[:, j_t] = -1.0
                runner_idx[bp_rows] = np.argmax(v, axis=1)

            winner_val[bp_rows] = smc_votes[bp_rows, winner_idx[bp_rows]]
            runner_val[bp_rows] = smc_votes[bp_rows, runner_idx[bp_rows]]

        # SMC seats
        smc_seats = np.bincount(winner_idx, minlength=n_parties).astype(int)

        # Compensation votes
        loser_votes = smc_votes.copy()
        loser_votes[np.arange(smc_votes.shape[0]), winner_idx] = 0.0
        loser_comp = loser_votes.sum(axis=0)

        margins = np.clip(winner_val - runner_val, 0.0, None)
        winner_comp = np.bincount(winner_idx, weights=margins, minlength=n_parties)

        # List totals
        domestic_list_total = mat.sum(axis=0)
        list_total = domestic_list_total + loser_comp + winner_comp + diaspora

        # Threshold eligibility
        lt_sum = float(list_total.sum())
        if lt_sum > 0:
            eligible = (list_total / lt_sum) >= threshold
        else:
            eligible = np.ones(n_parties, dtype=bool)

        votes_eligible = list_total.copy()
        votes_eligible[~eligible] = 0.0

        list_seats = _dhondt_alloc_fast(votes_eligible, list_seats_n)

        seats = smc_seats + list_seats + nationality_seats
        seats_draws[i, :] = seats.astype(np.int16)

    # Build per-draw inputs table (enables diagnostics / sensitivity analysis)
    input_dict: Dict[str, Any] = {
        "turnout_target": turnout_draws.astype(np.float32),
        "turnout_model": turnout_model_draws,
        "turnout_granularity": turnout_granularity_draws,
        "poll_type": poll_type_draws,
        "poll_undecided_share": poll_undecided_share_draws,
        "undecided_true_share": undecided_true_share_draws,
        "nonresponse_nonvoter_pct": nonresponse_nonvoter_pct_draws,
        "undecided_to_fidesz": undecided_to_fidesz_draws,
        "use_mobilization": use_mobilization_draws,
        "mobilization_rate_fidesz": mob_f_draws,
        "mobilization_rate_tisza": mob_t_draws,
        "reserve_strength_fidesz": res_f_draws,
        "reserve_strength_tisza": res_t_draws,
        "diaspora_total": diaspora_total_draws,
        "diaspora_fidesz_share": diaspora_fidesz_share_draws,
    }

    # Drawn national targets among voters
    if parties:
        for pi, p in enumerate(parties):
            input_dict[f"nat_target_{p}"] = nat_target_draws[:, pi]

    # Drawn coefficient weights
    for ei, e in enumerate(coef_elections):
        input_dict[f"w_e_{e}"] = w_e_draws[:, ei]
    for gi, g in enumerate(coef_levels):
        input_dict[f"w_g_{g}"] = w_g_draws[:, gi]

    input_draws = pd.DataFrame(input_dict)

    return seats_draws, nat_draws, turnout_draws, input_draws


def run_monte_carlo(
    data: ModelData,
    base: ScenarioResults,
    mc: MonteCarloConfig,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> MonteCarloResults:
    """Run Monte Carlo simulations around a scenario.

    This MC is optimized for EVK-level IPF (106 units), which enables 100k+ draws.

    It supports:
      - turnout uncertainty (including sampling turnout model / granularity)
      - poll uncertainty (gaussian / multinomial / dirichlet-multinomial)
      - undecided/nonresponse conversion in raw poll mode
      - coefficient uncertainty including sampling election & geography weights
      - diaspora/mail-in vote uncertainty

    Results include seat distributions, turnout draws and realized national shares.
    """

    n_total = int(mc.n_sims)
    parties = list(MODEL_PARTIES)
    n_parties = len(parties)

    if n_total <= 0:
        empty_seats = pd.DataFrame(columns=parties, dtype=int)
        summary = pd.DataFrame(index=parties, columns=["mean", "median", "p05", "p95"]).fillna(0.0)
        prob_majority = pd.Series(0.0, index=parties)
        prob_winner = pd.DataFrame(0.0, index=parties, columns=parties)
        nat_draws = pd.DataFrame(columns=parties, dtype=float)
        turnout_draws = pd.Series(dtype=float)
        input_draws = pd.DataFrame()
        return MonteCarloResults(
            cfg=mc,
            seat_draws=empty_seats,
            seat_summary=summary,
            prob_majority=prob_majority,
            prob_winner=prob_winner,
            nat_share_draws=nat_draws,
            turnout_draws=turnout_draws,
            input_draws=input_draws,
            evk_winner_draws=None,
            doom_prob_fidesz_majority=0.0,
            doom_prob_fidesz_mh_majority=0.0,
            doom_prob_any=0.0,
        )

    # MC is run on EVK universe for speed
    evk_index = base.evk_list_votes.index

    # party indices
    j_f = parties.index("FIDESZ")
    j_t = parties.index("TISZA")
    j_dk = parties.index("DK")

    # --- Build coefficient components (log-space) ---
    coef_components, elections, levels = build_log_coef_components_evk(data, base.cfg, mc, evk_index=evk_index)

    # Stable base weights aligned to component order
    base_election_weights = {k: float(base.cfg.election_weights.get(k, 0.0)) for k in elections}
    base_geo_weights = {k: float(base.cfg.geo_weights.get(k, 0.0)) for k in levels}

    # --- Turnout cache ---
    base_turnout_target = float(base.cfg.turnout_target)

    turnout_models = mc.turnout_models or [base.cfg.turnout_model]
    turnout_grans = mc.turnout_granularities or [base.cfg.turnout_granularity]

    grid_targets = make_turnout_grid_targets(mc, base_turnout_target)

    row_votes_cache = build_turnout_row_votes_cache(
        data,
        base.cfg,
        unit="evk",
        unit_index=evk_index,
        turnout_models=turnout_models,
        turnout_granularities=turnout_grans,
        grid_targets=grid_targets,
    )

    total_registered = float(data.registered_2026.sum())

    # --- Poll bases ---
    base_nat = base.national_shares_2026.reindex(parties).fillna(0.0).to_numpy(dtype=float)
    base_nat = np.clip(base_nat, 1e-12, None)
    base_nat = base_nat / base_nat.sum()

    # Raw poll base: MODEL_PARTIES + UNDECIDED
    poll_cols = parties + ["UNDECIDED"]
    base_poll_pp = base.avg_poll_2026.reindex(poll_cols).fillna(0.0).to_numpy(dtype=float)
    base_poll_pp = np.clip(base_poll_pp, 0.0, None)
    if base_poll_pp.sum() <= 0:
        base_poll_probs_raw = np.ones(len(base_poll_pp), dtype=float) / len(base_poll_pp)
    else:
        base_poll_probs_raw = base_poll_pp / base_poll_pp.sum()

    # --- Diaspora base ---
    base_diaspora = base.diaspora_votes.reindex(parties).fillna(0.0).to_numpy(dtype=float)

    # --- Seat constants ---
    nat_seats = np.zeros(n_parties, dtype=int)
    n_nat = 0
    if bool(base.cfg.nationality_seat_to_fidesz):
        nat_seats[j_f] = 1
        n_nat = 1
    list_seats_n = 93 - n_nat

    # --- Budapest precompute for exclude mode ---
    exclude_budapest = bool(base.cfg.exclude_budapest)
    is_bp = _is_budapest_evk(evk_index)
    mask_bp = is_bp.to_numpy(dtype=bool)
    mask_nb = ~mask_bp

    bp_block_shares = None
    if exclude_budapest:
        # Use turnout reference election as the basis, like scenario._estimate_budapest_voter_shares
        ref = str(base.cfg.turnout_reference_election)
        if ref == "parl_2018_list":
            df = data.parl18_list_i
        elif ref == "ep_2019":
            df = data.ep19_i
        elif ref == "parl_2022_list":
            df = data.parl22_list_i
        elif ref == "ep_2024":
            df = data.ep24_i
        else:
            df = data.parl22_list_i

        bp = df[df["county_id"] == "01"]
        if bp.empty:
            # fallback: assume Budapest shares match national
            # block order: FIDESZ, OPP, MH, MKKP, OTHER
            bp_block_shares = np.array([0.30, 0.45, 0.10, 0.05, 0.10], dtype=float)
        else:
            # Blocks are in df columns
            cols = ["FIDESZ", "OPP", "MH", "MKKP", "OTHER"]
            sums = bp[cols].sum(axis=0).to_numpy(dtype=float)
            sums = np.clip(sums, 0.0, None)
            s = float(sums.sum())
            if s > 0:
                bp_block_shares = sums / s
            else:
                bp_block_shares = np.array([0.30, 0.45, 0.10, 0.05, 0.10], dtype=float)

    # Mobilization defaults come from ScenarioConfig dicts
    mob_rates0 = getattr(base.cfg, "mobilization_rates", None) or {}
    res_strength0 = getattr(base.cfg, "reserve_strength", None) or {}

    base_mob_f = float(mob_rates0.get("FIDESZ", 0.70))
    base_mob_t = float(mob_rates0.get("TISZA", 0.90))
    base_res_f = float(res_strength0.get("FIDESZ", 1.0))
    base_res_t = float(res_strength0.get("TISZA", 1.0))

    payload: Dict[str, Any] = {
        "mc_cfg": mc,
        "n_units": int(len(evk_index)),
        "n_parties": int(n_parties),
        "parties": parties,
        "coef_elections": elections,
        "coef_levels": levels,
        "j_f": int(j_f),
        "j_t": int(j_t),
        "j_dk": int(j_dk),
        "coef_components": coef_components,
        "base_election_weights": base_election_weights,
        "base_geo_weights": base_geo_weights,
        "turnout_grid_targets": grid_targets,
        "turnout_row_votes_cache": row_votes_cache,
        "turnout_models": turnout_models,
        "turnout_granularities": turnout_grans,
        "base_turnout_model": base.cfg.turnout_model,
        "base_turnout_granularity": base.cfg.turnout_granularity,
        "base_turnout_target": base_turnout_target,
        "total_registered": total_registered,
        "base_nat_probs": base_nat,
        "base_poll_probs_raw": base_poll_probs_raw,
        "base_poll_type": base.cfg.poll_type,
        "base_nonresponse_nonvoter": float(base.cfg.nonresponse_nonvoter_pct),
        "base_undecided_to_fidesz": float(base.cfg.undecided_to_fidesz),
        # MC currently models a two-party undecided split (FIDESZ vs TISZA). Others get 0 from marginal undecideds.
        "base_undecided_to_others": 0.0,
        "base_use_mobilization": bool(base.cfg.use_mobilization_model),
        "base_mob_f": float(base_mob_f),
        "base_mob_t": float(base_mob_t),
        "base_res_f": float(base_res_f),
        "base_res_t": float(base_res_t),
        "smc_plus1_minus1": bool(base.cfg.smc_plus1_minus1),
        "winner_two_party_only": bool(base.cfg.winner_two_party_only),
        "threshold": float(base.cfg.threshold),
        "list_seats_n": int(list_seats_n),
        "nationality_seats": nat_seats,
        "exclude_budapest": exclude_budapest,
        "mask_bp": mask_bp,
        "mask_nb": mask_nb,
        "bp_block_shares": bp_block_shares,
        "base_diaspora": base_diaspora,
    }

    # Chunk plan
    chunk = int(mc.chunk_size or 500)
    chunk = max(200, chunk)

    chunks: List[int] = []
    left = n_total
    while left > 0:
        k = min(chunk, left)
        chunks.append(k)
        left -= k

    ss = np.random.SeedSequence(int(mc.seed))
    child = ss.spawn(len(chunks))
    seeds = [int(s.generate_state(1)[0]) for s in child]

    n_workers = int(mc.n_workers or 0)
    if n_workers <= 0:
        n_workers = _default_workers()

    # Progress helper
    done = 0

    def _report(done_now: int) -> None:
        if progress_cb:
            progress_cb(done_now, n_total)

    # Single worker path
    if n_workers == 1:
        seats_all: List[np.ndarray] = []
        nat_all: List[np.ndarray] = []
        turnout_all: List[np.ndarray] = []
        input_all: List[pd.DataFrame] = []

        for k, s in zip(chunks, seeds):
            seats_c, nat_c, t_c, inp_c = _simulate_chunk(payload, int(k), int(s))
            seats_all.append(seats_c)
            nat_all.append(nat_c)
            turnout_all.append(t_c)
            input_all.append(inp_c)
            done += int(k)
            _report(done)

    else:
        # Parallel backend
        backend = str(getattr(mc, "backend", "auto")).lower().strip()
        if backend == "auto":
            # In Streamlit on Windows, process-based parallelism is more fragile
            # (large payload pickling, spawn semantics). Default to threads.
            backend = "threads" if os.name == "nt" else "processes"
        if backend not in {"processes", "threads"}:
            backend = "processes"

        Executor = ProcessPoolExecutor if backend == "processes" else ThreadPoolExecutor
        ex_kwargs = {"max_workers": n_workers}
        if Executor is ProcessPoolExecutor and os.name == "nt":
            ex_kwargs["mp_context"] = mp.get_context("spawn")

        seats_all = []
        nat_all = []
        turnout_all = []
        input_all = []

        with Executor(**ex_kwargs) as ex:
            futs: List[Future] = []
            for k, s in zip(chunks, seeds):
                futs.append(ex.submit(_simulate_chunk, payload, int(k), int(s)))

            for fut in as_completed(futs):
                seats_c, nat_c, t_c, inp_c = fut.result()
                seats_all.append(seats_c)
                nat_all.append(nat_c)
                turnout_all.append(t_c)
                input_all.append(inp_c)

                done += int(seats_c.shape[0])
                _report(done)

    seats_mat = np.vstack(seats_all)[:n_total]
    nat_mat = np.vstack(nat_all)[:n_total]
    turnout_vec = np.concatenate(turnout_all)[:n_total]
    input_draws = pd.concat(input_all, ignore_index=True).iloc[:n_total].reset_index(drop=True)

    seat_draws = pd.DataFrame(seats_mat, columns=parties)
    nat_share_draws = pd.DataFrame(nat_mat, columns=parties)
    turnout_draws = pd.Series(turnout_vec, name="turnout")

    summary = pd.DataFrame(
        {
            "mean": seat_draws.mean(),
            "median": seat_draws.median(),
            "p05": seat_draws.quantile(0.05),
            "p95": seat_draws.quantile(0.95),
        }
    ).round(2)

    prob_majority = (seat_draws >= 100).mean()

    # Pairwise winner probabilities
    vals = seat_draws.to_numpy(dtype=float)
    prob = np.zeros((n_parties, n_parties), dtype=float)
    for a in range(n_parties):
        for b in range(n_parties):
            if a == b:
                prob[a, b] = 0.5
            else:
                prob[a, b] = float(np.mean(vals[:, a] > vals[:, b]))
    prob_winner = pd.DataFrame(prob, index=parties, columns=parties)

    # Doom scenario probabilities (convenience fields for diagnostics)
    seats_f = seat_draws["FIDESZ"] if "FIDESZ" in seat_draws.columns else pd.Series([], dtype=float)
    seats_mh = seat_draws["MH"] if "MH" in seat_draws.columns else 0
    doom_prob_fidesz_majority = float((seats_f >= 100).mean()) if len(seat_draws) else 0.0
    doom_prob_fidesz_mh_majority = float(((seats_f + seats_mh) >= 100).mean()) if len(seat_draws) else 0.0
    doom_prob_any = float(((seats_f >= 100) | ((seats_f + seats_mh) >= 100)).mean()) if len(seat_draws) else 0.0

    return MonteCarloResults(
        cfg=mc,
        seat_draws=seat_draws,
        seat_summary=summary,
        prob_majority=prob_majority,
        prob_winner=prob_winner,
        nat_share_draws=nat_share_draws,
        turnout_draws=turnout_draws,
        input_draws=input_draws,
        evk_winner_draws=None,
        doom_prob_fidesz_majority=doom_prob_fidesz_majority,
        doom_prob_fidesz_mh_majority=doom_prob_fidesz_mh_majority,
        doom_prob_any=doom_prob_any,
    )
