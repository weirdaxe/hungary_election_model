
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import MODEL_PARTIES
from .types import ModelPaths, ScenarioConfig


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read CSV with a small encoding fall-back list."""
    encodings = ["utf-8", "utf-8-sig", "cp1250", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to read CSV (unknown error)")


def parse_poll_date(x) -> pd.Timestamp:
    """Parse either ISO-ish strings or Excel date serial numbers."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float)) and x > 20000:
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(int(x), unit="D")
    return pd.to_datetime(x, errors="coerce")


_POLL_COL_ALIASES = {
    "FIDESZ": ["FIDESZ", "Fidesz"],
    "TISZA": ["TISZA", "Tisza", "TISZA PÁRT", "TISZA PART"],
    "DK": ["DK", "Democratic Coalition", "Demokratikus Koalíció", "Demokratikus Koalicio"],
    "MH": ["MH", "Mi Hazánk", "Mi Hazank", "Our Homeland"],
    "MKKP": ["MKKP", "Two Tailed Dog", "Two-Tailed Dog", "Kétfarkú", "Ketfarku"],
    "OTHER": ["Other", "OTHER", "Others"],
    "UNDECIDED": ["Don't Know", "Dont Know", "Undecided", "UNDECIDED"],
    "CHALLENGER": ["Challenger", "CHALLENGER", "Opposition", "Opp."],
}


def _standardize_poll_party_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = {c: str(c).strip() for c in out.columns}
    out = out.rename(columns=cols)

    alias_to_can: Dict[str, str] = {}
    for can, aliases in _POLL_COL_ALIASES.items():
        for a in aliases:
            alias_to_can[str(a).strip().lower()] = can

    rename = {}
    for c in out.columns:
        key = str(c).strip().lower()
        if key in alias_to_can:
            rename[c] = alias_to_can[key]
    out = out.rename(columns=rename)

    for c in ["FIDESZ", "TISZA", "DK", "MH", "MKKP", "OTHER", "UNDECIDED", "CHALLENGER"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def load_polls(paths: ModelPaths, year: str) -> pd.DataFrame:
    """Load a single poll CSV (one year) with canonical columns."""
    df = read_csv_auto(paths.poll_files()[year]).copy()

    # common meta columns
    if "Date" not in df.columns and "date" in df.columns:
        df["Date"] = df["date"]
    if "Poll Type" not in df.columns and "poll_type" in df.columns:
        df["Poll Type"] = df["poll_type"]
    if "Pollster" not in df.columns and "pollster" in df.columns:
        df["Pollster"] = df["pollster"]

    df["year"] = int(year)
    df["date"] = df["Date"].apply(parse_poll_date)
    df["poll_type"] = df["Poll Type"].astype(str).str.strip().str.lower()
    df["pollster"] = df["Pollster"].astype(str).str.strip()

    df = _standardize_poll_party_columns(df)
    return df


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    mask = (~series.isna()) & (~weights.isna()) & (weights > 0)
    if not mask.any():
        return float("nan")
    x = series[mask].astype(float)
    w = weights[mask].astype(float)
    return float((x * w).sum() / w.sum())


def get_poll_average_2026(polls_all: pd.DataFrame, cfg: ScenarioConfig) -> pd.Series:
    """Average 2026 polls with optional pollster filter and exponential time decay."""
    df = polls_all[(polls_all["year"] == 2026) & (polls_all["poll_type"] == str(cfg.poll_type).strip().lower())].copy()
    if cfg.pollster_filter:
        df = df[df["pollster"].isin(cfg.pollster_filter)]

    if df.empty:
        raise ValueError(f"No polls for 2026 poll_type={cfg.poll_type} and pollster_filter={cfg.pollster_filter}")

    max_date = df["date"].max()
    start = max_date - pd.Timedelta(days=int(cfg.last_n_days))
    df = df[df["date"] >= start]
    if df.empty:
        raise ValueError(f"No polls in last_n_days={cfg.last_n_days} for poll_type={cfg.poll_type}")

    cols = ["FIDESZ", "TISZA", "DK", "MH", "MKKP", "OTHER", "UNDECIDED"]

    # Optional exponential decay (half-life in days)
    if cfg.poll_decay_half_life_days and cfg.poll_decay_half_life_days > 0:
        age_days = (max_date - df["date"]).dt.total_seconds() / 86400.0
        lam = math.log(2.0) / float(cfg.poll_decay_half_life_days)
        w = np.exp(-lam * age_days.clip(lower=0.0))
        w = pd.Series(w.values, index=df.index, name="weight")
    else:
        w = pd.Series(1.0, index=df.index, name="weight")

    out = {c: weighted_mean(df[c], w) for c in cols}
    s = pd.Series(out)

    # Manual override (percent points)
    if cfg.manual_poll_override:
        for k, v in cfg.manual_poll_override.items():
            if k in s.index:
                s[k] = float(v)
    return s


def poll_to_population_shares(avg_poll: pd.Series, cfg: ScenarioConfig) -> pd.Series:
    """
    Convert poll to population shares (sum to 1) over:
      MODEL_PARTIES + UNDECIDED_TRUE

    For 'raw': uses UNDECIDED + nonresponse_nonvoter_pct.
    For 'decided': treats undecided as 0.
    """
    poll_type = str(cfg.poll_type).strip().lower()

    f = float(avg_poll.get("FIDESZ", 0.0) or 0.0)
    t = float(avg_poll.get("TISZA", 0.0) or 0.0)
    dk = float(avg_poll.get("DK", 0.0) or 0.0)
    mh = float(avg_poll.get("MH", 0.0) or 0.0)
    mk = float(avg_poll.get("MKKP", 0.0) or 0.0)
    oth = float(avg_poll.get("OTHER", 0.0) or 0.0)
    u = float(avg_poll.get("UNDECIDED", 0.0) or 0.0)

    if poll_type in ["decided", "pártválasztók", "partvalasztok"]:
        vec = pd.Series({"FIDESZ": f, "TISZA": t, "DK": dk, "MH": mh, "MKKP": mk, "OTHER": oth, "UNDECIDED_TRUE": 0.0})
        s = float(vec.sum())
        return vec / s if s > 0 else vec * 0.0

    if poll_type == "raw":
        nv = 100.0 * float(cfg.nonresponse_nonvoter_pct)
        u_true = u + nv
        decided_sum_poll = max(0.0, 100.0 - u)
        decided_sum_true = max(0.0, 100.0 - u_true)
        if decided_sum_poll <= 0:
            raise ValueError("Poll has 100% undecided; cannot proceed")

        # scale parties down so that parties + u_true = 100
        scale = decided_sum_true / decided_sum_poll
        f_pop, t_pop, dk_pop, mh_pop, mk_pop, oth_pop = [x * scale for x in [f, t, dk, mh, mk, oth]]

        vec = pd.Series(
            {
                "FIDESZ": f_pop,
                "TISZA": t_pop,
                "DK": dk_pop,
                "MH": mh_pop,
                "MKKP": mk_pop,
                "OTHER": oth_pop,
                "UNDECIDED_TRUE": u_true,
            }
        )
        s = float(vec.sum())
        return vec / s if s > 0 else vec * 0.0

    raise ValueError(f"Unknown poll_type: {cfg.poll_type}")


def _normalize_undecided_split(cfg: ScenarioConfig) -> Tuple[float, float]:
    uF = float(cfg.undecided_to_fidesz)
    uT = float(cfg.undecided_to_tisza)
    s = uF + uT
    if s <= 0:
        return 0.5, 0.5
    if s > 1.0:
        return uF / s, uT / s
    return uF, uT


def poll_to_voter_shares_simple(avg_poll: pd.Series, cfg: ScenarioConfig) -> pd.Series:
    """
    Simple conversion to voter shares (sum=1) over MODEL_PARTIES.
    - decided polls: undecided allocated by cfg split (Fidesz/Tisza) and remainder pro-rata to others
    - raw polls: treats poll numbers as population shares; turnout_target determines how many undecideds vote.
      If turnout_target is below the decided population, all parties are proportionally scaled (shares unchanged).
    """
    poll_type = str(cfg.poll_type).strip().lower()

    f = float(avg_poll.get("FIDESZ", 0.0) or 0.0)
    t = float(avg_poll.get("TISZA", 0.0) or 0.0)
    dk = float(avg_poll.get("DK", 0.0) or 0.0)
    mh = float(avg_poll.get("MH", 0.0) or 0.0)
    mk = float(avg_poll.get("MKKP", 0.0) or 0.0)
    oth = float(avg_poll.get("OTHER", 0.0) or 0.0)
    u = float(avg_poll.get("UNDECIDED", 0.0) or 0.0)

    uF, uT = _normalize_undecided_split(cfg)
    u_rem = max(0.0, 1.0 - (uF + uT))

    if poll_type in ["decided", "pártválasztók", "partvalasztok"]:
        base = pd.Series({"DK": dk, "MH": mh, "MKKP": mk, "OTHER": oth}).clip(lower=0.0)
        base_sum = float(base.sum())
        base_w = (base / base_sum) if base_sum > 0 else base * 0.0

        f2 = f + uF * u
        t2 = t + uT * u
        others_add = u_rem * u
        vec = pd.Series(
            {
                "FIDESZ": f2,
                "TISZA": t2,
                "DK": dk + others_add * float(base_w.get("DK", 0.0)),
                "MH": mh + others_add * float(base_w.get("MH", 0.0)),
                "MKKP": mk + others_add * float(base_w.get("MKKP", 0.0)),
                "OTHER": oth + others_add * float(base_w.get("OTHER", 0.0)),
            }
        )
        vec = vec.clip(lower=0.0)
        return vec / vec.sum() if float(vec.sum()) > 0 else vec * 0.0

    if poll_type == "raw":
        nv = 100.0 * float(cfg.nonresponse_nonvoter_pct)
        u_true = u + nv
        decided_sum_poll = max(0.0, 100.0 - u)
        decided_sum_true = max(0.0, 100.0 - u_true)
        if decided_sum_poll <= 0:
            raise ValueError("Poll has 100% undecided; cannot proceed")

        # population shares
        scale = decided_sum_true / decided_sum_poll
        f_pop, t_pop, dk_pop, mh_pop, mk_pop, oth_pop = [x * scale for x in [f, t, dk, mh, mk, oth]]
        decided_pop = decided_sum_true

        T = 100.0 * float(cfg.turnout_target)  # turnout among registered, interpreted in same % base

        base = pd.Series({"DK": dk_pop, "MH": mh_pop, "MKKP": mk_pop, "OTHER": oth_pop}).clip(lower=0.0)
        base_sum = float(base.sum())
        base_w = (base / base_sum) if base_sum > 0 else base * 0.0

        if T >= decided_pop:
            extra = T - decided_pop
            f_votes = f_pop + uF * extra
            t_votes = t_pop + uT * extra
            others_add = u_rem * extra
            dk_votes = dk_pop + others_add * float(base_w.get("DK", 0.0))
            mh_votes = mh_pop + others_add * float(base_w.get("MH", 0.0))
            mk_votes = mk_pop + others_add * float(base_w.get("MKKP", 0.0))
            oth_votes = oth_pop + others_add * float(base_w.get("OTHER", 0.0))
        else:
            shrink = T / max(1e-9, decided_pop)
            f_votes, t_votes, dk_votes, mh_votes, mk_votes, oth_votes = [x * shrink for x in [f_pop, t_pop, dk_pop, mh_pop, mk_pop, oth_pop]]

        vec_votes = pd.Series({"FIDESZ": f_votes, "TISZA": t_votes, "DK": dk_votes, "MH": mh_votes, "MKKP": mk_votes, "OTHER": oth_votes}).clip(lower=0.0)
        return vec_votes / vec_votes.sum() if float(vec_votes.sum()) > 0 else vec_votes * 0.0

    raise ValueError(f"Unknown poll_type: {cfg.poll_type}")



def poll_to_voter_shares_mobilization(avg_poll: pd.Series, cfg: ScenarioConfig) -> pd.Series:
    """
    Mobilization-aware conversion for raw polls.

    Two modes:

    1) cfg.mobilization_all_parties == False (default):
       - preserves baseline decided split when UNDECIDED=0
       - mobilization inputs only *tilt the marginal undecided allocation* (Fidesz vs Tisza)

    2) cfg.mobilization_all_parties == True:
       - treats mobilization_rates as turnout propensities for *all* parties (if provided)
       - allocates additional turnout first from party "reserves" (unmobilized decided supporters),
         then from UNDECIDED_TRUE, with the undecided split and optional Fidesz/Tisza reserve tilt

    Output: voter shares over MODEL_PARTIES, sum=1.
    """
    pop = poll_to_population_shares(avg_poll, cfg)

    # For non-raw poll types, return decided shares directly.
    if str(cfg.poll_type).strip().lower() != "raw":
        decided = pop.reindex(MODEL_PARTIES).fillna(0.0).clip(lower=0.0)
        s = float(decided.sum())
        return decided / s if s > 0 else decided * 0.0

    decided_pop = pop.reindex(MODEL_PARTIES).fillna(0.0).clip(lower=0.0)
    undec_pop = float(pop.get("UNDECIDED_TRUE", 0.0) or 0.0)
    turnout = float(np.clip(float(cfg.turnout_target), 0.0, 1.0))

    # --- Default behaviour: undecided-only marginal logic (backwards compatible)
    if not bool(getattr(cfg, "mobilization_all_parties", False)):
        decided_sum = float(decided_pop.sum())

        if turnout <= decided_sum + 1e-12:
            s = float(decided_pop.sum())
            return decided_pop / s if s > 0 else decided_pop * 0.0

        extra = turnout - decided_sum
        extra = min(extra, undec_pop)

        uF, uT = _normalize_undecided_split(cfg)
        u_rem = max(0.0, 1.0 - (uF + uT))

        # Reserve tilt for Fidesz vs Tisza
        mob = cfg.mobilization_rates or {}
        res_w = cfg.reserve_strength or {}
        mobF = float(np.clip(mob.get("FIDESZ", 0.80), 0.0, 1.0))
        mobT = float(np.clip(mob.get("TISZA", 0.80), 0.0, 1.0))
        rF = float(max(0.0, res_w.get("FIDESZ", 1.0)))
        rT = float(max(0.0, res_w.get("TISZA", 1.0)))

        reserveF = (1.0 - mobF) * rF + 1e-9
        reserveT = (1.0 - mobT) * rT + 1e-9

        uF_adj = uF * reserveF
        uT_adj = uT * reserveT
        s_adj = uF_adj + uT_adj
        if s_adj > 0:
            uF_adj /= s_adj
            uT_adj /= s_adj
        else:
            uF_adj, uT_adj = uF, uT

        other_base = decided_pop.drop(index=["FIDESZ", "TISZA"], errors="ignore")
        other_sum = float(other_base.sum())
        other_w = (other_base / other_sum) if other_sum > 0 else other_base * 0.0

        votes = decided_pop.copy()
        votes["FIDESZ"] += extra * uF_adj
        votes["TISZA"] += extra * uT_adj

        other_add = extra * u_rem
        for p in other_w.index:
            votes[p] += other_add * float(other_w[p])

        s = float(votes.sum())
        return votes / s if s > 0 else votes * 0.0

    # --- All-party turnout composition mode
    mob_in = cfg.mobilization_rates or {}
    res_in = cfg.reserve_strength or {}

    mob = {}
    res_w = {}
    for p in MODEL_PARTIES:
        if p == "FIDESZ":
            mob[p] = float(np.clip(mob_in.get("FIDESZ", 0.70), 0.0, 1.0))
            res_w[p] = float(max(0.0, res_in.get("FIDESZ", 1.0)))
        elif p == "TISZA":
            mob[p] = float(np.clip(mob_in.get("TISZA", 0.90), 0.0, 1.0))
            res_w[p] = float(max(0.0, res_in.get("TISZA", 1.0)))
        else:
            mob[p] = float(np.clip(mob_in.get(p, 1.0), 0.0, 1.0))
            # If user supplies reserve strengths for smaller parties, honor them; else default 1.
            res_w[p] = float(max(0.0, res_in.get(p, 1.0)))

    base_votes = decided_pop.copy()
    for p in MODEL_PARTIES:
        base_votes[p] = base_votes[p] * mob[p]

    base_sum = float(base_votes.sum())

    if base_sum <= 1e-12:
        # fallback
        return pd.Series(1.0 / len(MODEL_PARTIES), index=MODEL_PARTIES)

    if turnout <= base_sum + 1e-12:
        # Scale down the "mobilized" decided voters to hit turnout.
        votes = base_votes * (turnout / base_sum)
        s = float(votes.sum())
        return votes / s if s > 0 else votes * 0.0

    # Remaining turnout after mobilized decided voters
    remaining = turnout - base_sum

    votes = base_votes.copy()

    # Step A: take from decided reserves (unmobilized supporters)
    reserves = decided_pop.copy()
    for p in MODEL_PARTIES:
        reserves[p] = reserves[p] * (1.0 - mob[p]) * res_w[p]

    res_sum = float(reserves.sum())
    take_res = min(remaining, res_sum)
    if take_res > 0 and res_sum > 0:
        votes += reserves * (take_res / res_sum)
        remaining -= take_res

    # Step B: take from undecided pool
    take_u = min(remaining, undec_pop)
    if take_u > 0:
        uF, uT = _normalize_undecided_split(cfg)
        u_rem = max(0.0, 1.0 - (uF + uT))

        # Optional Fidesz-vs-Tisza reserve tilt (uses their mobilization/reserve params)
        reserveF = (1.0 - mob["FIDESZ"]) * res_w["FIDESZ"] + 1e-9
        reserveT = (1.0 - mob["TISZA"]) * res_w["TISZA"] + 1e-9

        uF_adj = uF * reserveF
        uT_adj = uT * reserveT
        s_adj = uF_adj + uT_adj
        if s_adj > 0:
            uF_adj /= s_adj
            uT_adj /= s_adj
        else:
            uF_adj, uT_adj = uF, uT

        votes["FIDESZ"] += take_u * uF_adj
        votes["TISZA"] += take_u * uT_adj

        other_add = take_u * u_rem
        other_base = decided_pop.drop(index=["FIDESZ", "TISZA"], errors="ignore")
        other_sum = float(other_base.sum())
        if other_sum > 0:
            for p in other_base.index:
                votes[p] += other_add * float(other_base[p] / other_sum)

        remaining -= take_u

    # Step C: if turnout still not met (rare), distribute proportionally
    if remaining > 1e-9:
        w = float(votes.sum())
        if w > 0:
            votes += votes * (remaining / w)

    s = float(votes.sum())
    return votes / s if s > 0 else votes * 0.0

def poll_to_voter_shares(avg_poll: pd.Series, cfg: ScenarioConfig) -> pd.Series:
    """Wrapper selecting between simple and mobilization-aware conversion."""
    if str(cfg.poll_type).strip().lower() == "raw" and bool(cfg.use_mobilization_model):
        return poll_to_voter_shares_mobilization(avg_poll, cfg)
    return poll_to_voter_shares_simple(avg_poll, cfg)