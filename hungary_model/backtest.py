from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .coefficients import build_coefs_2026
from .constants import BLOCK_BUCKETS, MODEL_PARTIES
from .seats import determine_winner_row, dhondt_alloc
from .types import ModelData, ScenarioConfig
from .vote_allocation import build_initial_matrix, ipf_rake


@dataclass
class Backtest2022Results:
    """Dirty 2022 backtest results (EVK-level).

    The goal is to test the *national vote -> seats* mapping sensitivity to the
    coefficient settings (election weights + geo weights), using 2022 national
    totals and 2022 EVK turnout distribution.

    Caveats:
      - 2022 stations are mapped into 2026 EVKs (station_id -> 2026 maz-evk).
      - Party-space approximation: OPP -> TISZA, DK=0.
    """

    table: pd.DataFrame
    mae_fidesz_pp: float
    mae_tisza_pp: float
    winner_accuracy_ft: float
    n_evks: int

    seats_actual: pd.DataFrame
    seats_pred: pd.DataFrame
    seats_diff: pd.DataFrame


def _seat_projection_from_evk_list_votes(
    evk_list_votes: pd.DataFrame,
    cfg: ScenarioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the model's seat rules on an EVK×party list-vote table.

    Returns:
      - evk_smc_votes
      - winners (index evk_id)
      - seats_table (index party)

    Notes:
      - The model uses evk_list_votes as the base for SMC votes, optionally applying
        the +/-1pp correction.
      - Uses cfg.diaspora_votes / threshold / nationality seat toggle.
    """
    evk_smc_votes = evk_list_votes.copy().reindex(columns=MODEL_PARTIES).fillna(0.0)

    # +/- 1pp correction for SMC
    if bool(cfg.smc_plus1_minus1):
        total = evk_smc_votes.sum(axis=1)
        shares = evk_smc_votes.div(total.replace(0, np.nan), axis=0).fillna(0.0)
        shares["FIDESZ"] = shares["FIDESZ"] + 0.01
        shares["TISZA"] = (shares["TISZA"] - 0.01).clip(lower=0.0)
        shares = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        evk_smc_votes = shares.mul(total, axis=0)

    # Winners + SMC seats
    winners_rows = []
    for evk, row in evk_smc_votes.iterrows():
        w, r = determine_winner_row(row, two_party_only=bool(cfg.winner_two_party_only))
        margin = float(row[w] - row[r])
        winners_rows.append((evk, w, r, margin, abs(margin)))

    winners = (
        pd.DataFrame(winners_rows, columns=["evk_id", "winner", "runner_up", "margin_votes", "abs_margin_votes"])
        .set_index("evk_id")
        .sort_index()
    )

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
    domestic_list_votes = evk_list_votes.reindex(columns=MODEL_PARTIES).fillna(0.0).sum(axis=0)
    list_plus_comp = domestic_list_votes + total_comp

    diaspora_votes = pd.Series(cfg.diaspora_votes or {}).reindex(MODEL_PARTIES).fillna(0.0)
    list_total_votes = list_plus_comp + diaspora_votes

    # Nationality seat
    nationality_seats = pd.Series(0, index=MODEL_PARTIES, dtype=int)
    n_nat = 0
    if bool(cfg.nationality_seat_to_fidesz):
        nationality_seats["FIDESZ"] = 1
        n_nat = 1

    # Threshold
    threshold = float(cfg.threshold)
    total_list = float(list_total_votes.sum())
    list_shares = (list_total_votes / total_list) if total_list > 0 else list_total_votes * 0.0
    eligible_parties = list_shares[list_shares >= threshold].index.tolist()

    # D'Hondt list seats
    list_seats_n = 93 - n_nat
    list_seats, _dh = dhondt_alloc(list_total_votes.reindex(eligible_parties), seats=list_seats_n)
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

    return evk_smc_votes, winners, seats_table


def backtest_2022_dirty(data: ModelData, cfg: ScenarioConfig) -> Backtest2022Results:
    """Evaluate 2022 national vote -> EVK geography -> seats mapping under current parameters."""
    df = data.parl22_list_i.copy()
    if df.empty or "evk_id" not in df.columns:
        empty_seats = pd.DataFrame(index=MODEL_PARTIES)
        return Backtest2022Results(
            table=pd.DataFrame(),
            mae_fidesz_pp=float("nan"),
            mae_tisza_pp=float("nan"),
            winner_accuracy_ft=float("nan"),
            n_evks=0,
            seats_actual=empty_seats,
            seats_pred=empty_seats,
            seats_diff=empty_seats,
        )

    # Aggregate 2022 list votes to EVKs in BLOCK space.
    v_evk = df.groupby("evk_id")[list(BLOCK_BUCKETS)].sum().copy()
    v_evk = v_evk.loc[v_evk.index.notna()]
    v_evk.index = v_evk.index.astype(str)

    # Exclude Budapest EVKs if requested.
    if bool(getattr(cfg, "exclude_budapest", False)):
        v_evk = v_evk.loc[~v_evk.index.str.startswith("01-")].copy()

    # Convert to MODEL_PARTIES votes.
    actual_votes = pd.DataFrame(index=v_evk.index, columns=MODEL_PARTIES, dtype=float)
    actual_votes["FIDESZ"] = v_evk.get("FIDESZ", 0.0)
    actual_votes["TISZA"] = v_evk.get("OPP", 0.0)
    actual_votes["DK"] = 0.0
    actual_votes["MH"] = v_evk.get("MH", 0.0)
    actual_votes["MKKP"] = v_evk.get("MKKP", 0.0)
    actual_votes["OTHER"] = v_evk.get("OTHER", 0.0)
    actual_votes = actual_votes.fillna(0.0).clip(lower=0.0)

    row_votes = actual_votes.sum(axis=1)
    total_votes = float(row_votes.sum())
    if total_votes <= 0:
        empty_seats = pd.DataFrame(index=MODEL_PARTIES)
        return Backtest2022Results(
            table=pd.DataFrame(),
            mae_fidesz_pp=float("nan"),
            mae_tisza_pp=float("nan"),
            winner_accuracy_ft=float("nan"),
            n_evks=int(len(actual_votes)),
            seats_actual=empty_seats,
            seats_pred=empty_seats,
            seats_diff=empty_seats,
        )

    nat_shares = (actual_votes.sum(axis=0) / total_votes).reindex(MODEL_PARTIES).fillna(0.0)

    # Coefficients for the chosen settings (using the scenario's election_weights + geo_weights).
    coef_source = data.coef_by_election_no_budapest if bool(getattr(cfg, "exclude_budapest", False)) else data.coef_by_election
    coefs = build_coefs_2026(
        coef_by_election=coef_source,
        station_meta=data.station_meta,
        election_weights=cfg.election_weights or {},
        geo_weights=cfg.geo_weights or {},
        index=actual_votes.index,
        unit="evk",
        station_weights=data.registered_2026,
    )

    # IPF to reproduce EVK allocations given national totals.
    M0 = build_initial_matrix(
        index=actual_votes.index,
        row_votes=row_votes,
        coefs=coefs,
        national_shares=nat_shares,
        cfg=cfg,
        elasticity_beta=None,
    )
    Mr = ipf_rake(
        M0.values,
        row_targets=row_votes.values,
        col_targets=nat_shares.values * total_votes,
        max_iter=2000,
        tol=1e-5,
    )
    pred_votes = pd.DataFrame(Mr, index=actual_votes.index, columns=MODEL_PARTIES).fillna(0.0).clip(lower=0.0)

    # Shares and errors (EVK-level)
    actual_sh = actual_votes.div(row_votes.replace(0, np.nan), axis=0).fillna(0.0)
    pred_row_votes = pred_votes.sum(axis=1)
    pred_sh = pred_votes.div(pred_row_votes.replace(0, np.nan), axis=0).fillna(0.0)

    err_f = (pred_sh["FIDESZ"] - actual_sh["FIDESZ"]) * 100.0
    err_t = (pred_sh["TISZA"] - actual_sh["TISZA"]) * 100.0

    mae_f = float(np.mean(np.abs(err_f.values)))
    mae_t = float(np.mean(np.abs(err_t.values)))

    # Winner accuracy (Fidesz vs Tisza)
    actual_w = (actual_votes[["FIDESZ", "TISZA"]].idxmax(axis=1)).astype(str)
    pred_w = (pred_votes[["FIDESZ", "TISZA"]].idxmax(axis=1)).astype(str)
    acc = float((actual_w == pred_w).mean())

    table = pd.DataFrame(
        {
            "FIDESZ_actual_%": (actual_sh["FIDESZ"] * 100.0).round(2),
            "TISZA_actual_%": (actual_sh["TISZA"] * 100.0).round(2),
            "FIDESZ_pred_%": (pred_sh["FIDESZ"] * 100.0).round(2),
            "TISZA_pred_%": (pred_sh["TISZA"] * 100.0).round(2),
            "err_F_pp": err_f.round(2),
            "err_T_pp": err_t.round(2),
            "winner_actual": actual_w,
            "winner_pred": pred_w,
        }
    )

    # Seats mapping under current model rules.
    _, _, seats_actual = _seat_projection_from_evk_list_votes(actual_votes, cfg)
    _, _, seats_pred = _seat_projection_from_evk_list_votes(pred_votes, cfg)

    seats_diff = seats_pred.reindex(seats_actual.index).fillna(0) - seats_actual.fillna(0)

    return Backtest2022Results(
        table=table.reset_index().rename(columns={"index": "evk_id"}),
        mae_fidesz_pp=mae_f,
        mae_tisza_pp=mae_t,
        winner_accuracy_ft=acc,
        n_evks=int(len(table)),
        seats_actual=seats_actual,
        seats_pred=seats_pred,
        seats_diff=seats_diff,
    )
