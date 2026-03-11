
from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import numpy as np

from .constants import BLOCK_BUCKETS, MODEL_PARTIES
from .coefficients import build_coefs_2026
from .polls import get_poll_average_2026, poll_to_population_shares, poll_to_voter_shares
from .seats import determine_winner_row, dhondt_alloc
from .turnout import predict_station_votes_2026
from .types import ModelData, ScenarioConfig, ScenarioResults
from .vote_allocation import (
    aggregate_station_to_evk,
    allocate_station_votes_baseline_marginal,
    build_initial_matrix,
    ipf_rake,
)


def default_config() -> ScenarioConfig:
    """Project default scenario settings."""
    return ScenarioConfig(
        turnout_target=0.72,
        turnout_model="logit_slope_ep_offset",
        turnout_granularity="location",
        turnout_reference_election="parl_2022_list",
        turnout_baseline_level=0.70,
        turnout_clip=(0.20, 0.95),
        reserve_adjust_power=1.0,
        marginal_concentration=1.0,

        poll_type="raw",
        last_n_days=60,
        pollster_filter=None,
        poll_decay_half_life_days=0.0,
        manual_poll_override=None,

        nonresponse_nonvoter_pct=0.10,
        undecided_to_fidesz=0.40,
        undecided_to_tisza=0.60,

        election_weights={"ep_2024": 0.55, "parl_2022_list": 0.30, "ep_2019": 0.10, "parl_2018_list": 0.05},
        geo_weights={"station": 0.10, "location": 0.55, "settlement": 0.25, "evk": 0.10},

        undecided_elasticity_link_strength=0.0,

        use_mobilization_model=True,
        mobilization_all_parties=False,
        mobilization_rates={"FIDESZ": 0.70, "TISZA": 0.90},
        reserve_strength={"FIDESZ": 1.0, "TISZA": 1.0},

        vote_allocation_model="ipf",
        undecided_geo_model="uniform",
        undecided_local_lean_strength=0.0,

        smc_plus1_minus1=True,
        winner_two_party_only=True,

        diaspora_votes={"FIDESZ": 250_000, "TISZA": 50_000, "MH": 5_000, "DK": 1_000, "MKKP": 1_000, "OTHER": 0},
        nationality_seat_to_fidesz=True,
        threshold=0.05,
        modelling_unit="station",
    )


def _attach_evk_names(df_by_evk: pd.DataFrame, data: ModelData) -> pd.DataFrame:
    if df_by_evk is None or df_by_evk.empty:
        return df_by_evk
    if data.evk_meta is None or data.evk_meta.empty:
        return df_by_evk
    if "evk_name" in df_by_evk.columns:
        return df_by_evk
    out = df_by_evk.copy()
    out["evk_name"] = data.evk_meta.reindex(out.index)["evk_name"].values
    return out


def _is_budapest_station(data: ModelData, station_index: pd.Index) -> pd.Series:
    """Boolean mask for stations that are in Budapest (county_id == '01')."""
    if data.station_meta is None or data.station_meta.empty:
        return pd.Series(False, index=station_index)
    county = data.station_meta.reindex(station_index)["county_id"].astype(str)
    return county.eq("01").fillna(False)


def _is_budapest_evk(evk_index: pd.Index) -> pd.Series:
    """Boolean mask for EVKs that are in Budapest (evk_id startswith '01-')."""
    s = pd.Index(evk_index).astype(str)
    return pd.Series(s.str.startswith("01-"), index=evk_index)


def _estimate_budapest_voter_shares(data: ModelData, cfg: ScenarioConfig, tisza_dk_ratio: float) -> pd.Series:
    """Estimate Budapest party shares among voters.

    Uses Budapest BLOCK shares from the selected turnout reference election and maps
    OPP -> (TISZA, DK) using the provided TISZA/(TISZA+DK) ratio.
    """
    ref = str(cfg.turnout_reference_election)
    if ref == "parl_2022_list":
        df = data.parl22_list_i
    elif ref == "parl_2018_list":
        df = data.parl18_list_i
    else:
        df = data.parl22_list_i

    if df is None or df.empty or "county_id" not in df.columns:
        s = pd.Series({"FIDESZ": 0.30, "TISZA": 0.45, "DK": 0.10, "MH": 0.05, "MKKP": 0.05, "OTHER": 0.05})
        return (s / s.sum()).reindex(MODEL_PARTIES).fillna(0.0)

    df_bp = df.loc[df["county_id"].astype(str) == "01"]
    if df_bp.empty:
        s = pd.Series({"FIDESZ": 0.30, "TISZA": 0.45, "DK": 0.10, "MH": 0.05, "MKKP": 0.05, "OTHER": 0.05})
        return (s / s.sum()).reindex(MODEL_PARTIES).fillna(0.0)

    v = df_bp.reindex(columns=list(BLOCK_BUCKETS)).fillna(0.0).sum(axis=0)
    tot = float(v.sum())
    if tot <= 0:
        s = pd.Series({"FIDESZ": 0.30, "TISZA": 0.45, "DK": 0.10, "MH": 0.05, "MKKP": 0.05, "OTHER": 0.05})
        return (s / s.sum()).reindex(MODEL_PARTIES).fillna(0.0)

    sh = (v / tot).astype(float)
    r = float(np.clip(tisza_dk_ratio, 0.0, 1.0))

    out = pd.Series(
        {
            "FIDESZ": float(sh.get("FIDESZ", 0.0)),
            "TISZA": float(sh.get("OPP", 0.0)) * r,
            "DK": float(sh.get("OPP", 0.0)) * (1.0 - r),
            "MH": float(sh.get("MH", 0.0)),
            "MKKP": float(sh.get("MKKP", 0.0)),
            "OTHER": float(sh.get("OTHER", 0.0)),
        }
    ).reindex(MODEL_PARTIES).fillna(0.0)

    # Enforce the user's Budapest assumption by ensuring TISZA >= FIDESZ,
    # shifting within the OPP bucket (DK -> TISZA) when needed.
    if float(out.get("TISZA", 0.0)) <= float(out.get("FIDESZ", 0.0)):
        needed = float(out["FIDESZ"] - out["TISZA"] + 1e-3)
        shift = min(needed, float(out.get("DK", 0.0)))
        out["TISZA"] += shift
        out["DK"] -= shift

    out = out.clip(lower=0.0)
    s = float(out.sum())
    return (out / s) if s > 0 else out


def run_scenario(data: ModelData, cfg: ScenarioConfig) -> ScenarioResults:
    """Run a single deterministic scenario."""
    # --- polls -> shares
    avg_poll_2026 = get_poll_average_2026(data.polls_all, cfg)
    pop_shares_2026 = poll_to_population_shares(avg_poll_2026, cfg)
    # Interpreted as a national voter-share input. In "Exclude Budapest" mode, this is treated
    # as Hungary ex Budapest and Budapest is added back later via a historical estimate.
    national_shares_2026 = poll_to_voter_shares(avg_poll_2026, cfg).reindex(MODEL_PARTIES).fillna(0.0)

    # --- station votes / turnout rates (always station-level first)
    votes_station, turnout_rate_station = predict_station_votes_2026(data, cfg)

    # --- choose vote allocation model
    vote_model = str(cfg.vote_allocation_model).strip().lower()
    unit = str(cfg.modelling_unit).strip().lower()

    exclude_bp = bool(getattr(cfg, "exclude_budapest", False))

    # Choose coefficient source depending on scope.
    coef_source = data.coef_by_election_no_budapest if exclude_bp else data.coef_by_election

    if vote_model not in ["ipf", "baseline_marginal"]:
        raise ValueError("vote_allocation_model must be 'ipf' or 'baseline_marginal'")

    if vote_model == "baseline_marginal":
        if exclude_bp:
            raise ValueError("exclude_budapest is currently supported only for vote_allocation_model='ipf'.")
        if str(cfg.poll_type).strip().lower() != "raw":
            raise ValueError("baseline_marginal requires poll_type='raw' (needs undecided pool)")
        if unit != "station":
            raise ValueError("baseline_marginal requires modelling_unit='station'")

        station_index = data.registered_2026.index
        reg = data.registered_2026.reindex(station_index).fillna(0.0)

        coefs_2026 = build_coefs_2026(
            coef_by_election=coef_source,
            station_meta=data.station_meta,
            election_weights=cfg.election_weights or {},
            geo_weights=cfg.geo_weights or {},
            index=station_index,
            unit="station",
        )

        unit_list_votes = allocate_station_votes_baseline_marginal(
            station_index=station_index,
            votes_station=votes_station,
            reg=reg,
            coefs=coefs_2026,
            pop_shares=pop_shares_2026,
            cfg=cfg,
            elasticity_station=data.elasticity_station_original,
            baseline_turnout_station=data.baseline_turnout_station,
        )

        # realized national shares among voters
        total_votes = float(unit_list_votes.sum().sum())
        if total_votes > 0:
            national_shares_2026 = (unit_list_votes.sum(axis=0) / total_votes).reindex(MODEL_PARTIES).fillna(0.0)
        else:
            national_shares_2026 = pd.Series(0.0, index=MODEL_PARTIES)

        evk_list_votes = aggregate_station_to_evk(unit_list_votes, data.station_meta)

    else:
        # IPF mode (fixed national totals)
        if unit not in ["station", "evk"]:
            raise ValueError("modelling_unit must be 'station' or 'evk'")

        if unit == "station":
            station_index = data.registered_2026.index
            row_votes = votes_station.reindex(station_index).fillna(0.0)

            coefs_2026 = build_coefs_2026(
                coef_by_election=coef_source,
                station_meta=data.station_meta,
                election_weights=cfg.election_weights or {},
                geo_weights=cfg.geo_weights or {},
                index=station_index,
                unit="station",
            )

            beta_series = data.elasticity_station_original["beta"]

            if not exclude_bp:
                M0 = build_initial_matrix(
                    index=station_index,
                    row_votes=row_votes,
                    coefs=coefs_2026,
                    national_shares=national_shares_2026,
                    cfg=cfg,
                    elasticity_beta=beta_series,
                )
                row_targets = row_votes.values
                col_targets = national_shares_2026.values * float(row_targets.sum())

                Mr = ipf_rake(M0.values, row_targets=row_targets, col_targets=col_targets, max_iter=2000, tol=1e-5)
                unit_list_votes = pd.DataFrame(Mr, index=station_index, columns=MODEL_PARTIES)
            else:
                # Split into Budapest vs rest-of-country. Poll-derived national_shares_2026 are
                # treated as "Hungary ex Budapest" targets. Budapest is added back separately.
                is_bp = _is_budapest_station(data, station_index)
                idx_nb = station_index[~is_bp.values]
                idx_bp = station_index[is_bp.values]

                # Non-Budapest via IPF
                rv_nb = row_votes.reindex(idx_nb).fillna(0.0)
                cf_nb = coefs_2026.reindex(idx_nb).fillna(1.0)
                beta_nb = beta_series.reindex(idx_nb).fillna(1.0)
                M0_nb = build_initial_matrix(
                    index=idx_nb,
                    row_votes=rv_nb,
                    coefs=cf_nb,
                    national_shares=national_shares_2026,
                    cfg=cfg,
                    elasticity_beta=beta_nb,
                )
                row_targets_nb = rv_nb.values
                col_targets_nb = national_shares_2026.values * float(row_targets_nb.sum())
                Mr_nb = ipf_rake(M0_nb.values, row_targets=row_targets_nb, col_targets=col_targets_nb, max_iter=2000, tol=1e-5)
                votes_nb = pd.DataFrame(Mr_nb, index=idx_nb, columns=MODEL_PARTIES)

                # Budapest: constant shares based on historical Budapest BLOCK shares
                denom = float(national_shares_2026.get("TISZA", 0.0) + national_shares_2026.get("DK", 0.0))
                ratio = float(national_shares_2026.get("TISZA", 0.0) / denom) if denom > 0 else 0.85
                bp_shares = _estimate_budapest_voter_shares(data, cfg, tisza_dk_ratio=ratio)

                rv_bp = row_votes.reindex(idx_bp).fillna(0.0)
                votes_bp = pd.DataFrame(0.0, index=idx_bp, columns=MODEL_PARTIES)
                for p in MODEL_PARTIES:
                    votes_bp[p] = rv_bp.values * float(bp_shares.get(p, 0.0))

                unit_list_votes = pd.concat([votes_nb, votes_bp], axis=0).reindex(station_index).fillna(0.0)

                # Realized national shares
                tot = float(unit_list_votes.sum().sum())
                if tot > 0:
                    national_shares_2026 = (unit_list_votes.sum(axis=0) / tot).reindex(MODEL_PARTIES).fillna(0.0)

            evk_list_votes = aggregate_station_to_evk(unit_list_votes, data.station_meta)

        else:
            # EVK mode: aggregate station turnout to EVK and rake at EVK level
            station_index = data.registered_2026.index
            evk_map = data.station_meta["evk_id"].reindex(station_index)
            votes_evk = votes_station.reindex(station_index).groupby(evk_map).sum()
            votes_evk = votes_evk[votes_evk.index.notna()].copy()
            votes_evk.index.name = "evk_id"

            unit_index = votes_evk.index
            row_votes = votes_evk

            coefs_2026 = build_coefs_2026(
                coef_by_election=coef_source,
                station_meta=data.station_meta,
                election_weights=cfg.election_weights or {},
                geo_weights=cfg.geo_weights or {},
                index=unit_index,
                unit="evk",
                station_weights=data.registered_2026,
            )

            # approximate EVK beta as registered-weighted mean of station betas
            beta_station = data.elasticity_station_original["beta"].reindex(station_index).fillna(1.0)
            w_station = data.registered_2026.reindex(station_index).fillna(0.0)

            def _wmean(s: pd.Series) -> float:
                ww = w_station.loc[s.index]
                den = float(ww.sum())
                return float((s * ww).sum() / den) if den > 0 else 1.0

            beta_evk = beta_station.groupby(evk_map).apply(_wmean)
            beta_evk.index.name = "evk_id"

            if not exclude_bp:
                M0 = build_initial_matrix(
                    index=unit_index,
                    row_votes=row_votes,
                    coefs=coefs_2026,
                    national_shares=national_shares_2026,
                    cfg=cfg,
                    elasticity_beta=beta_evk,
                )
                row_targets = row_votes.values
                col_targets = national_shares_2026.values * float(row_targets.sum())

                Mr = ipf_rake(M0.values, row_targets=row_targets, col_targets=col_targets, max_iter=2000, tol=1e-5)
                unit_list_votes = pd.DataFrame(Mr, index=unit_index, columns=MODEL_PARTIES)
            else:
                # Split EVKs into Budapest vs rest-of-country
                is_bp_evk = _is_budapest_evk(unit_index)
                idx_nb = unit_index[~is_bp_evk.values]
                idx_bp = unit_index[is_bp_evk.values]

                # Non-Budapest via IPF
                rv_nb = row_votes.reindex(idx_nb).fillna(0.0)
                cf_nb = coefs_2026.reindex(idx_nb).fillna(1.0)
                beta_nb = beta_evk.reindex(idx_nb).fillna(1.0)
                M0_nb = build_initial_matrix(
                    index=idx_nb,
                    row_votes=rv_nb,
                    coefs=cf_nb,
                    national_shares=national_shares_2026,
                    cfg=cfg,
                    elasticity_beta=beta_nb,
                )
                row_targets_nb = rv_nb.values
                col_targets_nb = national_shares_2026.values * float(row_targets_nb.sum())
                Mr_nb = ipf_rake(M0_nb.values, row_targets=row_targets_nb, col_targets=col_targets_nb, max_iter=2000, tol=1e-5)
                votes_nb = pd.DataFrame(Mr_nb, index=idx_nb, columns=MODEL_PARTIES)

                # Budapest: constant shares
                denom = float(national_shares_2026.get("TISZA", 0.0) + national_shares_2026.get("DK", 0.0))
                ratio = float(national_shares_2026.get("TISZA", 0.0) / denom) if denom > 0 else 0.85
                bp_shares = _estimate_budapest_voter_shares(data, cfg, tisza_dk_ratio=ratio)

                rv_bp = row_votes.reindex(idx_bp).fillna(0.0)
                votes_bp = pd.DataFrame(0.0, index=idx_bp, columns=MODEL_PARTIES)
                for p in MODEL_PARTIES:
                    votes_bp[p] = rv_bp.values * float(bp_shares.get(p, 0.0))

                unit_list_votes = pd.concat([votes_nb, votes_bp], axis=0).reindex(unit_index).fillna(0.0)

                # Realized national shares
                tot = float(unit_list_votes.sum().sum())
                if tot > 0:
                    national_shares_2026 = (unit_list_votes.sum(axis=0) / tot).reindex(MODEL_PARTIES).fillna(0.0)

            evk_list_votes = unit_list_votes.copy()

    # SMC votes start equal to list votes
    evk_smc_votes = evk_list_votes.copy()

    # +/- 1pp correction for SMC
    if cfg.smc_plus1_minus1:
        total = evk_smc_votes.sum(axis=1)
        shares = evk_smc_votes.div(total.replace(0, np.nan), axis=0).fillna(0.0)
        shares["FIDESZ"] = shares["FIDESZ"] + 0.01
        shares["TISZA"] = shares["TISZA"] - 0.01
        shares["TISZA"] = shares["TISZA"].clip(lower=0.0)
        shares = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        evk_smc_votes = shares.mul(total, axis=0)

    # Winners + SMC seats
    winners_rows = []
    for evk, row in evk_smc_votes.iterrows():
        w, r = determine_winner_row(row, two_party_only=cfg.winner_two_party_only)
        margin = float(row[w] - row[r])
        winners_rows.append((evk, w, r, margin, abs(margin)))

    winners = pd.DataFrame(winners_rows, columns=["evk_id", "winner", "runner_up", "margin_votes", "abs_margin_votes"]).set_index("evk_id")
    winners = _attach_evk_names(winners, data)

    # "Exclude Budapest" scope: force Budapest EVKs to be won by TISZA.
    if exclude_bp and not winners.empty:
        bp_mask = _is_budapest_evk(winners.index)
        if bool(bp_mask.any()):
            for evk_id in winners.index[bp_mask.values]:
                row = evk_smc_votes.loc[evk_id]
                winner = "TISZA"
                if bool(cfg.winner_two_party_only):
                    runner = "FIDESZ"
                else:
                    runner = row.drop(index=[winner], errors="ignore").idxmax()
                margin = float(row.get(winner, 0.0) - row.get(runner, 0.0))
                winners.loc[evk_id, "winner"] = winner
                winners.loc[evk_id, "runner_up"] = runner
                winners.loc[evk_id, "margin_votes"] = margin
                winners.loc[evk_id, "abs_margin_votes"] = abs(margin)

    smc_seats = winners["winner"].value_counts().reindex(MODEL_PARTIES).fillna(0).astype(int)

    # Compensation votes (loser + winner)
    loser_comp = pd.Series(0.0, index=MODEL_PARTIES)
    winner_comp = pd.Series(0.0, index=MODEL_PARTIES)

    for evk, row in evk_smc_votes.iterrows():
        w = winners.loc[evk, "winner"]
        r = winners.loc[evk, "runner_up"]
        v_w = float(row[w])
        v_r = float(row[r])

        for p in MODEL_PARTIES:
            if p != w:
                loser_comp[p] += float(row[p])

        winner_comp[w] += max(0.0, v_w - v_r - 1.0)

    total_comp = loser_comp + winner_comp

    # List votes totals
    domestic_list_votes = unit_list_votes.sum(axis=0)
    list_plus_comp = domestic_list_votes + total_comp

    diaspora_votes = pd.Series(cfg.diaspora_votes or {}).reindex(MODEL_PARTIES).fillna(0.0)
    list_total_votes = list_plus_comp + diaspora_votes

    # Nationality seat
    nationality_seats = pd.Series(0, index=MODEL_PARTIES, dtype=int)
    n_nat = 0
    if cfg.nationality_seat_to_fidesz:
        nationality_seats["FIDESZ"] = 1
        n_nat = 1

    # Threshold
    threshold = float(cfg.threshold)
    list_shares = list_total_votes / list_total_votes.sum() if float(list_total_votes.sum()) > 0 else list_total_votes * 0.0
    eligible_parties = list_shares[list_shares >= threshold].index.tolist()

    # D'Hondt list seats
    list_seats_n = 93 - n_nat
    list_seats, dhondt_table = dhondt_alloc(list_total_votes.reindex(eligible_parties), seats=list_seats_n)
    list_seats = list_seats.reindex(MODEL_PARTIES).fillna(0).astype(int)

    total_seats = smc_seats.reindex(MODEL_PARTIES).fillna(0).astype(int) + list_seats + nationality_seats

    seats_table = pd.DataFrame(
        {
            "SMC_seats": smc_seats.reindex(MODEL_PARTIES).fillna(0).astype(int),
            "List_seats": list_seats,
            "Nationality": nationality_seats,
            "Total": total_seats,
            "List_share_%": (list_shares.reindex(MODEL_PARTIES) * 100).round(2),
        }
    )

    return ScenarioResults(
        cfg=cfg,
        avg_poll_2026=avg_poll_2026,
        population_shares_2026=pop_shares_2026,
        national_shares_2026=national_shares_2026,
        turnout_rate_station=turnout_rate_station,
        votes_station=votes_station,
        coefs_2026=coefs_2026,
        unit_list_votes=unit_list_votes,
        evk_list_votes=evk_list_votes,
        evk_smc_votes=evk_smc_votes,
        winners=winners,
        loser_comp=loser_comp,
        winner_comp=winner_comp,
        total_comp=total_comp,
        domestic_list_votes=domestic_list_votes,
        diaspora_votes=diaspora_votes,
        list_total_votes=list_total_votes,
        list_shares=list_shares,
        eligible_parties=eligible_parties,
        smc_seats=smc_seats,
        list_seats=list_seats,
        nationality_seats=nationality_seats,
        seats_table=seats_table,
        dhondt_table=dhondt_table,
    )
