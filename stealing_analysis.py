
"""
Station-level anomaly / "stealing" diagnostics for Hungary elections.

This module is designed to be embedded into the main Streamlit app (app.py)
as an additional tab.

It expects a polling-station results CSV with columns similar to:
  - votes_individual_party_*
  - votes_list_party_* / votes_list_comp_* / votes_list_minority_*
plus metadata such as maz, constituency_code, polling_station_address, etc.
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Constants / helpers
# -----------------------------
VOTE_PREFIXES = (
    "votes_individual_party_",
    "votes_list_party_",
    "votes_list_comp_",
    "votes_list_minority_",
)

TOTAL_COL_BY_CONTEST = {
    "Individual (single-member district)": "valid_votes_individual",
    "List (party lists)": "valid_votes_list",
    "List (party + minority lists)": "valid_votes_list",
}

# Expected poll parties (from your xlsx / inline fallback)
POLL_PARTIES = ["Fidesz", "United for Hungary", "MKKP", "Our Homeland", "Others"]


def humanize_vote_col(col: str) -> str:
    for p in VOTE_PREFIXES:
        if col.startswith(p):
            return col[len(p):].replace("_", " ")
    return col


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


# -----------------------------
# Data loading / preparation
# -----------------------------
@st.cache_data(show_spinner=False)
def load_raw_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    return _prepare_raw(df)


@st.cache_data(show_spinner=False)
def load_raw_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return _prepare_raw(df)


def _prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create stable station id
    required = {"maz", "taz", "sorsz"}
    if required.issubset(df.columns):
        df["polling_station_id"] = (
            df["maz"].astype(str) + "-" + df["taz"].astype(str) + "-" + df["sorsz"].astype(str)
        )
        df = df.set_index("polling_station_id")
    elif "polling_station_id" in df.columns:
        df = df.set_index("polling_station_id")

    return df


def contest_vote_cols(df_raw: pd.DataFrame, contest: str) -> List[str]:
    if contest == "Individual (single-member district)":
        return [c for c in df_raw.columns if c.startswith("votes_individual_party_")]

    if contest == "List (party lists)":
        return [c for c in df_raw.columns if c.startswith("votes_list_comp_") or c.startswith("votes_list_party_")]

    if contest == "List (party + minority lists)":
        return [
            c
            for c in df_raw.columns
            if c.startswith("votes_list_comp_")
            or c.startswith("votes_list_party_")
            or c.startswith("votes_list_minority_")
        ]

    return []


def build_votes_df(
    df_raw: pd.DataFrame, vote_cols: List[str], total_col: Optional[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not vote_cols:
        st.error("No vote columns found for the selected contest.")
        st.stop()

    # votes-only DF
    df_votes = df_raw[vote_cols].copy()
    for c in df_votes.columns:
        df_votes[c] = safe_numeric(df_votes[c])

    # totals
    if total_col and total_col in df_raw.columns:
        total_votes = safe_numeric(df_raw[total_col])
    else:
        total_votes = df_votes.sum(axis=1)

    # drop 0-vote stations
    mask = total_votes > 0
    df_votes = df_votes.loc[mask].copy()
    total_votes = total_votes.loc[mask].copy()

    # extra safety: drop rows where all vote cols are 0
    row_sums = df_votes.sum(axis=1)
    df_votes = df_votes.loc[row_sums > 0].copy()
    total_votes = total_votes.loc[df_votes.index].copy()

    # metadata used downstream
    meta_cols = [
        c
        for c in [
            "oevk_id",
            "area_type",
            "maz",
            "constituency_code",
            "constituency_name",
            "polling_station_name",
            "polling_station_address",
            "electorate_total",
            "eligible_voters_individual",
            "eligible_voters_list",
            "turnout_individual",
            "turnout_list",
            "turnout_rate_pct_individual",
            "turnout_rate_pct_list",
            "valid_votes_individual",
            "valid_votes_list",
        ]
        if c in df_raw.columns
    ]
    df_meta = df_raw.loc[df_votes.index, meta_cols].copy()

    return df_votes, total_votes, df_meta


def candidate_col_widget(vote_cols: List[str], key: str = "candidate_col") -> str:
    if not vote_cols:
        st.error("No vote columns available to choose from.")
        st.stop()

    default_idx = 0
    for i, c in enumerate(vote_cols):
        if "fidesz" in c.lower():
            default_idx = i
            break

    labels = {c: humanize_vote_col(c) for c in vote_cols}
    inv = {v: k for k, v in labels.items()}
    display_list = [labels[c] for c in vote_cols]

    chosen_display = st.selectbox("Party / list column", display_list, index=default_idx, key=key)
    return inv[chosen_display]


# -----------------------------
# Urban / Rural classification (data-driven, constituency-level)
# -----------------------------
URBAN_SETTLEMENTS = [
    "Budapest",
    "Baja",
    "Békéscsaba",
    "Debrecen",
    "Dunaújváros",
    "Eger",
    "Érd",
    "Esztergom",
    "Győr",
    "Hódmezővásárhely",
    "Kaposvár",
    "Kecskemét",
    "Miskolc",
    "Nagykanizsa",
    "Nyíregyháza",
    "Pécs",
    "Salgótarján",
    "Sopron",
    "Szeged",
    "Szekszárd",
    "Székesfehérvár",
    "Szolnok",
    "Szombathely",
    "Tatabánya",
    "Veszprém",
    "Zalaegerszeg",
]


def _strip_accents(s: str) -> str:
    s = "" if s is None else str(s)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _norm(s: str) -> str:
    return _strip_accents(s).casefold().strip()


URBAN_NORM = set(_norm(x) for x in URBAN_SETTLEMENTS)


def extract_settlement(address: str, station_name: str = "") -> str:
    """
    Extract settlement name from address / station name.
    Handles:
      - "1011 Budapest 01, Bem rkp. 2. ..."
      - " Budapest 01, Úri u. 38. ..."
      - "4025 Debrecen, ..."
      - "Bocskaikert, Debreceni út 85. ..."
    """
    s = address if isinstance(address, str) and address.strip() else (station_name or "")
    s = s.strip()
    if not s:
        return ""

    head = s.split(",")[0].strip()

    # drop postcode if present
    m = re.match(r"^(\d{4})\s+(.*)$", head)
    if m:
        head = m.group(2).strip()

    # Budapest district formats
    if re.match(r"^(budapest)\s+\d{1,2}\b", head, flags=re.IGNORECASE):
        return "Budapest"

    # Sometimes "Town 01" without comma
    head = re.sub(r"\s+\d{1,2}$", "", head).strip()

    return head


@st.cache_data(show_spinner=False)
def classify_station_urban_rural(df_raw: pd.DataFrame) -> pd.Series:
    # station-level classification
    addr = (
        df_raw["polling_station_address"]
        if "polling_station_address" in df_raw.columns
        else pd.Series("", index=df_raw.index)
    )
    name = (
        df_raw["polling_station_name"]
        if "polling_station_name" in df_raw.columns
        else pd.Series("", index=df_raw.index)
    )

    settlements = [
        extract_settlement(a, n)
        for a, n in zip(addr.fillna("").astype(str).tolist(), name.fillna("").astype(str).tolist())
    ]
    settlements = pd.Series(settlements, index=df_raw.index)

    is_urban = settlements.map(lambda x: _norm(x) in URBAN_NORM)
    return pd.Series(np.where(is_urban, "Urban", "Rural"), index=df_raw.index)


@st.cache_data(show_spinner=False)
def classify_oevk_urban_rural(df_raw: pd.DataFrame) -> pd.Series:
    """
    OEVK = (maz, constituency_code). Label the OEVK Urban if >=50% of electorate_total
    (or eligible_voters_list fallback) is in Urban settlements.
    """
    if not {"maz", "constituency_code"}.issubset(df_raw.columns):
        return pd.Series("Unknown", index=df_raw.index)

    station_area = classify_station_urban_rural(df_raw)

    # weight by electorate size if present
    if "electorate_total" in df_raw.columns:
        w = pd.to_numeric(df_raw["electorate_total"], errors="coerce").fillna(0)
        w = w.where(w > 0, 1.0)
    elif "eligible_voters_list" in df_raw.columns:
        w = pd.to_numeric(df_raw["eligible_voters_list"], errors="coerce").fillna(0)
        w = w.where(w > 0, 1.0)
    else:
        w = pd.Series(1.0, index=df_raw.index)

    oevk_id = (
        df_raw["maz"].astype(int).astype(str).str.zfill(2) + "-" + df_raw["constituency_code"].astype(int).astype(str).str.zfill(2)
    )

    tmp = pd.DataFrame({"oevk_id": oevk_id, "w": w, "urban": (station_area == "Urban").astype(int)})
    g = tmp.groupby("oevk_id")
    # weighted share
    urban_share = (
        g.apply(lambda x: float((x["w"] * x["urban"]).sum()) / float(x["w"].sum()) if float(x["w"].sum()) > 0 else 0.0)
        .fillna(0.0)
    )

    oevk_area = urban_share.map(lambda s: "Urban" if s >= 0.5 else "Rural")

    # map back to stations
    return oevk_id.map(oevk_area).fillna("Unknown")


# -----------------------------
# Core computations
# -----------------------------
def compute_vote_share(df_votes: pd.DataFrame, total_votes: pd.Series, candidate_col: str) -> Tuple[pd.Series, pd.Series]:
    if candidate_col not in df_votes.columns:
        st.error(f"Column '{candidate_col}' not found.")
        st.stop()

    cand_votes = df_votes[candidate_col]
    pct = (cand_votes / total_votes) * 100.0
    return cand_votes, pct


def baselines(df_meta: pd.DataFrame, total_votes: pd.Series, cand_votes: pd.Series) -> Tuple[float, pd.Series]:
    # National baseline share
    nat_share = 100.0 * float(cand_votes.sum()) / float(total_votes.sum()) if float(total_votes.sum()) > 0 else 0.0

    # Constituency baseline share (constituency_code)
    if "constituency_code" in df_meta.columns:
        tmp = pd.DataFrame({"constituency_code": df_meta["constituency_code"], "V": total_votes, "C": cand_votes})
        grp = tmp.groupby("constituency_code")[["V", "C"]].sum()
        cons_share = 100.0 * grp["C"] / grp["V"]
        cons_share = cons_share.replace([np.inf, -np.inf], 0).fillna(0)
        cons_share_station = df_meta["constituency_code"].map(cons_share).fillna(nat_share)
    else:
        cons_share_station = pd.Series(nat_share, index=df_meta.index)

    return nat_share, cons_share_station


def render_summary_header(
    df_votes: pd.DataFrame,
    total_votes: pd.Series,
    cand_votes: pd.Series,
    pct: pd.Series,
    candidate_col: str,
    label_suffix: str = "",
) -> float:
    n = len(df_votes)
    V = float(total_votes.sum())
    C = float(cand_votes.sum())
    overall_share = 100.0 * C / V if V > 0 else 0.0

    st.markdown(f"### Summary{label_suffix}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Polling stations (valid > 0 votes)", f"{n:,}")
    c2.metric("Total valid votes", f"{int(round(V)):,}")
    c3.metric(f"Total votes ({humanize_vote_col(candidate_col)})", f"{int(round(C)):,}")

    c4, c5, c6 = st.columns(3)
    c4.metric(f"{humanize_vote_col(candidate_col)} share (overall)", f"{overall_share:.1f}%")
    c5.metric("Mean station share", f"{float(pct.mean()):.1f}%")
    c6.metric("Median station share", f"{float(pct.median()):.1f}%")

    return overall_share


def plot_hist_with_normal(
    pct: pd.Series,
    total_votes: pd.Series,
    title: str,
    *,
    fit_index: Optional[pd.Index] = None,
    weight_hist: bool = False,
):
    weights = total_votes.loc[pct.index] if weight_hist else None

    fig, ax = plt.subplots()
    ax.hist(pct, bins=30, density=True, alpha=0.6, weights=weights)

    p_fit = pct if fit_index is None else pct.loc[fit_index]
    if len(p_fit) > 1:
        med = float(np.median(p_fit))
        mad = float(np.median(np.abs(p_fit - med)))
        sigma = 1.4826 * mad if mad > 0 else (float(np.std(p_fit, ddof=1)) if len(p_fit) > 1 else 0.0)

        if sigma > 0:
            x = np.linspace(float(pct.min()), float(pct.max()), 200)
            y = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - med) / sigma) ** 2)
            ax.plot(x, y, label="Robust normal fit")

        ax.axvline(med, linestyle="--", label=f"Median (fit) {med:.1f}%")
        ax.legend()

    ax.set_xlabel("Vote share (%)")
    ax.set_ylabel("Density" + (" (vote-weighted)" if weight_hist else ""))
    ax.set_title(title)

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def segment_masks(df_meta: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if "area_type" not in df_meta.columns:
        m_all = pd.Series(True, index=df_meta.index)
        return m_all, m_all.copy(), m_all.copy()
    m_all = pd.Series(True, index=df_meta.index)
    m_urban = df_meta["area_type"].eq("Urban")
    m_rural = df_meta["area_type"].eq("Rural")
    return m_all, m_urban, m_rural


def segment_stats_block(df_votes: pd.DataFrame, total_votes: pd.Series, df_meta: pd.DataFrame, candidate_col: str):
    cand_votes, pct = compute_vote_share(df_votes, total_votes, candidate_col)
    m_all, m_urban, m_rural = segment_masks(df_meta)

    st.markdown("### Urban vs Rural summary")
    cols = st.columns(2)

    # Urban
    with cols[0]:
        if m_urban.any():
            render_summary_header(
                df_votes.loc[m_urban],
                total_votes.loc[m_urban],
                cand_votes.loc[m_urban],
                pct.loc[m_urban],
                candidate_col,
                " (Urban)",
            )
        else:
            st.write("No Urban stations (per current classification).")

    # Rural
    with cols[1]:
        if m_rural.any():
            render_summary_header(
                df_votes.loc[m_rural],
                total_votes.loc[m_rural],
                cand_votes.loc[m_rural],
                pct.loc[m_rural],
                candidate_col,
                " (Rural)",
            )
        else:
            st.write("No Rural stations (per current classification).")


def segmented_histograms(
    pct: pd.Series,
    total_votes: pd.Series,
    df_meta: pd.DataFrame,
    candidate_col: str,
    *,
    weight_hist: bool = False,
):
    m_all, m_urban, m_rural = segment_masks(df_meta)

    st.markdown("### Vote share distributions: Urban vs Rural")
    c1, c2 = st.columns(2)

    with c1:
        if m_urban.any():
            plot_hist_with_normal(
                pct.loc[m_urban],
                total_votes.loc[m_urban],
                f"Urban: {humanize_vote_col(candidate_col)} vote share",
                fit_index=None,
                weight_hist=weight_hist,
            )
        else:
            st.write("No Urban stations to plot.")

    with c2:
        if m_rural.any():
            plot_hist_with_normal(
                pct.loc[m_rural],
                total_votes.loc[m_rural],
                f"Rural: {humanize_vote_col(candidate_col)} vote share",
                fit_index=None,
                weight_hist=weight_hist,
            )
        else:
            st.write("No Rural stations to plot.")


# -----------------------------
# Threshold analysis with station tables + baseline excess
# -----------------------------
def threshold_cap_analysis(df_votes: pd.DataFrame, total_votes: pd.Series, df_meta: pd.DataFrame, candidate_col: str):
    st.subheader("Threshold cap analysis (with baseline excess)")

    cand_votes, pct = compute_vote_share(df_votes, total_votes, candidate_col)

    # Show Urban/Rural stats and segmented hist
    segment_stats_block(df_votes, total_votes, df_meta, candidate_col)

    weight_hist = st.checkbox("Weight histograms by total votes", value=False, key="thr_weight_hist")
    segmented_histograms(pct, total_votes, df_meta, candidate_col, weight_hist=weight_hist)

    # User chooses analysis segment
    seg = st.radio("Run threshold analysis on:", ["All", "Urban", "Rural"], horizontal=True, key="thr_seg")
    m_all, m_urban, m_rural = segment_masks(df_meta)
    m_seg = {"All": m_all, "Urban": m_urban, "Rural": m_rural}[seg]
    if not m_seg.any():
        st.write("No stations in selected segment.")
        return

    dfv = df_votes.loc[m_seg]
    V = total_votes.loc[m_seg]
    meta = df_meta.loc[m_seg]
    C = cand_votes.loc[m_seg]
    p = pct.loc[m_seg]

    # Baselines for THIS segment (so comparisons are coherent)
    nat_share, cons_share_station = baselines(meta, V, C)

    render_summary_header(dfv, V, C, p, candidate_col, f" ({seg} segment)")
    st.write(f"National baseline (within segment) = {nat_share:.2f}%")

    # Threshold slider
    T = st.slider("Threshold (%)", min_value=50.0, max_value=100.0, value=70.0, step=1.0, key="thr_T")

    # Fit only on stations <= T
    fit_index = p.index[p <= T]
    plot_hist_with_normal(
        p,
        V,
        f"{humanize_vote_col(candidate_col)} vote share (analysis segment: {seg}; fit on stations ≤ {T:.0f}%)",
        fit_index=fit_index,
        weight_hist=weight_hist,
    )

    # Stations above threshold
    above = p > T
    if not above.any():
        st.write("No polling stations above the threshold.")
        return

    # Excess vs cap T (your original measure)
    p_cap = p.clip(upper=T)
    excess_vs_cap_votes = np.clip((p - p_cap) * V / 100.0, 0, None)

    # Excess vs baseline
    excess_vs_national_votes = np.clip((p - nat_share) * V / 100.0, 0, None)
    excess_vs_const_votes = np.clip((p - cons_share_station) * V / 100.0, 0, None)

    st.markdown("### Polling stations above threshold")
    st.metric("Count above threshold", f"{int(above.sum()):,}")

    # totals
    st.markdown("### Excess vote totals (above-threshold stations only)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Excess vs cap T", f"{int(round(float(excess_vs_cap_votes.loc[above].sum()))):,}")
    c2.metric("Excess vs national baseline", f"{int(round(float(excess_vs_national_votes.loc[above].sum()))):,}")
    c3.metric("Excess vs constituency baseline", f"{int(round(float(excess_vs_const_votes.loc[above].sum()))):,}")

    out = (
        pd.DataFrame(
            {
                "constituency_code": meta.get("constituency_code", pd.Series(index=meta.index, dtype=object)),
                "constituency_name": meta.get("constituency_name", pd.Series(index=meta.index, dtype=object)),
                "area_type": meta.get("area_type", pd.Series(index=meta.index, dtype=object)),
                "polling_station_name": meta.get("polling_station_name", pd.Series(index=meta.index, dtype=object)),
                "polling_station_address": meta.get("polling_station_address", pd.Series(index=meta.index, dtype=object)),
                "total_votes": V,
                "candidate_votes": C,
                "candidate_share_pct": p,
                "national_baseline_pct": nat_share,
                "constituency_baseline_pct": cons_share_station,
                "excess_vs_cap_votes": excess_vs_cap_votes,
                "excess_vs_national_votes": excess_vs_national_votes,
                "excess_vs_constituency_votes": excess_vs_const_votes,
            }
        )
        .loc[above]
        .copy()
    )

    # sort by "excess vs constituency" (usually the stricter benchmark than national)
    sort_choice = st.selectbox(
        "Sort table by",
        [
            "excess_vs_constituency_votes",
            "excess_vs_national_votes",
            "excess_vs_cap_votes",
            "candidate_share_pct",
            "total_votes",
        ],
        index=0,
        key="thr_sort",
    )
    out = out.sort_values(sort_choice, ascending=False)

    st.dataframe(out, use_container_width=True)


# -----------------------------
# Polls integration
# -----------------------------
POLL_INLINE = """Date\tPolling Firm\tFidesz\tUnited for Hungary\tMKKP\tOur Homeland\tOthers\tSample Size
31-Mar-22\tMedian\t49\t41\t4.5\t4.5\t1\t1531
31-Mar-22\tPublicus\t47\t47\t2\t3\t1\t1025
30-Mar-22\tTarsadalomkutato\t50\t40\t4\t6\t0\t1000
29-Mar-22\tRepublikon\t49.3\t46.5\t0\t0\t4.2\t800
28-Mar-22\tIDEA\t50\t45\t2\t3\t0\t1800
27-Mar-22\tSzazadveg\t49\t44\t3\t3\t1\t1000
27-Mar-22\tZavecz Research\t50\t46\t1\t3\t0\t1000
26-Mar-22\tMedian\t50\t40\t4\t4\t2\t1096
25-Mar-22\tNezopont\t49\t43.8\t3.1\t3.1\t1\t1000
25-Mar-22\tZavecz Research\t48.8\t46.4\t0\t0\t4.8\t800
25-Mar-22\tPublicus\t48.2\t45.9\t2.4\t2.4\t1.2\t1000
24-Mar-22\tReal-PR 93\t49\t41\t3\t5\t2\t1000
23-Mar-22\tTarsadalomkutato\t52\t41\t3\t3\t1\t1000
21-Mar-22\tNezopont\t51.5\t43\t3\t2\t0\t1000
18-Mar-22\tRepublikon\t49\t46\t2\t3\t0\t1000
11-Mar-22\tPublicus\t48.5\t45.5\t1.4\t2.9\t1.4\t1001
11-Mar-22\tIDEA\t50\t43\t3\t4\t0\t2000
5-Mar-22\te-benchmark\t54\t46\t0\t0\t0\t1000
26-Feb-22\tMedian\t49\t43\t3\t4\t1\t1100
24-Feb-22\tRepublikon\t48\t46\t3\t3\t0\t1000
16-Feb-22\tNezopont\t50\t43\t3\t3\t1\t1000
14-Feb-22\tPublicus\t45.7\t48.5\t1.4\t2.8\t1.4\t1000
11-Feb-22\tReal-PR 93\t54\t41\t2\t3\t0\t1000
10-Feb-22\tZavecz Research\t49\t46\t2\t3\t0\t1000
9-Feb-22\tIDEA\t49\t44\t3\t4\t0\t2000
26-Jan-22\tTarsadalomkutato\t51\t43\t3\t3\t0\t1000
25-Jan-22\tRepublikon\t47\t47\t3\t3\t0\t1000
19-Jan-22\tAlapjogokert Kozpont\t49\t44\t3\t4\t0\t1000
14-Jan-22\tIDEA\t48\t44\t4\t4\t0\t2000
5-Jan-22\tNezopont\t50\t43\t3\t3\t1\t1000
23-Dec-21\tReal-PR 93\t54\t45\t0\t0\t1\t1000
15-Dec-21\tNezopont\t55\t43\t0\t0\t2\t1000
14-Dec-21\tRepublikon\t43\t48\t3\t4\t2\t1000
13-Dec-21\tIDEA\t47\t46\t3\t4\t0\t2000
13-Dec-21\tZavecz Research\t47\t47\t2\t3\t1\t1000
11-Dec-21\tIranytu\t52.2\t36.3\t4.5\t5.6\t1.4\t2000
7-Dec-21\tMedian\t44\t45\t6\t5\t0\t1000
30-Nov-21\tTarsadalomkutato\t51.2\t45.3\t0\t0\t3.5\t1000
30-Nov-21\tPublicus\t46\t53\t0\t1\t0\t1003
24-Nov-21\tZavecz Research\t48\t49\t2\t3\t2\t1000
23-Nov-21\tRepublikon\t41\t48\t4\t3\t4\t1000
12-Nov-21\te-benchmark\t52.3\t46.5\t0\t0\t1.2\t1000
12-Nov-21\tZavecz Research\t41.5\t46\t4.4\t4.4\t3.3\t1000
11-Nov-21\tReal-PR 93\t55\t44\t0\t0\t1\t1000
5-Nov-21\tIDEA\t48\t45\t3\t3\t1\t2000
3-Nov-21\tNezopont\t56\t42\t0\t0\t2\t1000
29-Oct-21\tRepublikon\t42\t51\t2\t3\t4\t1000
"""


@st.cache_data(show_spinner=False)
def load_polls(polls_bytes: Optional[bytes], path: str, sheet: str) -> pd.DataFrame:
    p = Path(path)
    if polls_bytes is not None:
        dfp = pd.read_excel(io.BytesIO(polls_bytes), sheet_name=sheet)
    elif p.exists():
        dfp = pd.read_excel(path, sheet_name=sheet)
    else:
        # fallback so the app runs even without the xlsx present
        dfp = pd.read_csv(io.StringIO(POLL_INLINE), sep="\t")

    # normalize
    dfp = dfp.rename(columns={"Polling Firm": "PollingFirm", "Sample Size": "SampleSize"})
    dfp["Date"] = pd.to_datetime(dfp["Date"], dayfirst=True, errors="coerce")
    for c in ["Fidesz", "United for Hungary", "MKKP", "Our Homeland", "Others", "SampleSize"]:
        if c in dfp.columns:
            dfp[c] = safe_numeric(dfp[c])

    # drop rows without date
    dfp = dfp.dropna(subset=["Date"]).sort_values("Date")
    return dfp


def actual_national_shares_for_polls(df_votes_list: pd.DataFrame, total_votes: pd.Series) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    # map poll parties to vote columns (defaults for your dataset)
    mapping_defaults: Dict[str, Optional[str]] = {
        "Fidesz": "votes_list_comp_fidesz",
        "United for Hungary": "votes_list_comp_united_for_hungary",
        "MKKP": "votes_list_party_two_tailed_dog",
        "Our Homeland": "votes_list_party_our_homeland",
        "Others": None,  # computed as residual
    }

    actual: Dict[str, float] = {}
    used_sum = pd.Series(0.0, index=df_votes_list.index)
    for party, col in mapping_defaults.items():
        if party == "Others":
            continue
        if col and col in df_votes_list.columns:
            v = df_votes_list[col]
            actual[party] = 100.0 * float(v.sum()) / float(total_votes.sum()) if float(total_votes.sum()) > 0 else np.nan
            used_sum += v
        else:
            actual[party] = np.nan

    # Others = residual if possible
    if float(total_votes.sum()) > 0:
        residual_votes = total_votes.sum() - used_sum.sum()
        actual["Others"] = 100.0 * float(residual_votes) / float(total_votes.sum())
    else:
        actual["Others"] = np.nan

    return actual, mapping_defaults


def polls_analysis(df_raw: pd.DataFrame):
    st.subheader("Polls vs results (national + constituency sensitivity)")
    st.caption("Uses the LIST contest by default (polls reflect list vote intention).")

    # Use LIST contest for comparison (polls are national list vote intention)
    vote_cols = contest_vote_cols(df_raw, "List (party lists)")
    total_col = TOTAL_COL_BY_CONTEST["List (party lists)"]
    df_votes, total_votes, df_meta = build_votes_df(df_raw, vote_cols, total_col)

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    polls_path = c1.text_input("Polls file path (optional)", value="polls.xlsx", key="polls_path")
    polls_sheet = c2.text_input("Sheet", value="2022", key="polls_sheet")
    polls_upload = c3.file_uploader("Upload polls .xlsx (optional)", type=["xlsx"], key="polls_upload")

    polls_bytes = polls_upload.getvalue() if polls_upload is not None else None
    dfp = load_polls(polls_bytes, polls_path, polls_sheet)

    actual, mapping = actual_national_shares_for_polls(df_votes, total_votes)

    st.markdown("### Actual national shares (from results file)")
    st.dataframe(pd.Series(actual).to_frame("Actual %"), use_container_width=True)

    st.markdown("### Poll dataset")
    st.dataframe(dfp.sort_values("Date", ascending=False), use_container_width=True)

    # Choose which poll estimate to benchmark (default: latest "Median" if available, else latest row)
    dfp_sorted = dfp.sort_values("Date")
    if "PollingFirm" in dfp_sorted.columns:
        med = dfp_sorted[dfp_sorted["PollingFirm"].astype(str).str.lower().eq("median")]
        default_idx_pos = len(dfp_sorted) - 1
        if len(med) > 0:
            # pick the last Median row (by date)
            default_idx = med.index.max()
            default_idx_pos = list(dfp_sorted.index).index(default_idx)
        idx = st.selectbox("Select benchmark poll row", dfp_sorted.index.tolist(), index=default_idx_pos, key="polls_benchmark_idx")
    else:
        idx = st.selectbox("Select benchmark poll row", dfp_sorted.index.tolist(), index=len(dfp_sorted) - 1, key="polls_benchmark_idx")

    row = dfp_sorted.loc[idx]
    st.markdown("### Poll deviation vs actual (benchmark row)")
    dev = {}
    for party in POLL_PARTIES:
        if party in row.index and party in actual and pd.notna(actual[party]):
            dev[party] = float(actual[party]) - float(row[party])
    st.dataframe(pd.Series(dev).to_frame("Actual - Poll (pp)"), use_container_width=True)

    # Firm-level accuracy: absolute error (pp) vs actual
    st.markdown("### Polling-firm accuracy (absolute error, pp)")
    firm_col = "PollingFirm" if "PollingFirm" in dfp.columns else None
    if firm_col:
        rows = []
        for _, r in dfp.iterrows():
            firm = str(r[firm_col])
            if firm.lower() == "median":
                continue
            errs = {}
            for party in POLL_PARTIES:
                if party in r.index and party in actual and pd.notna(actual[party]):
                    errs[party] = abs(float(actual[party]) - float(r[party]))
            if errs:
                rows.append({"Date": r["Date"], "Firm": firm, **errs, "MeanAbsErr": float(np.mean(list(errs.values())))})
        if rows:
            firm_df = pd.DataFrame(rows).sort_values("MeanAbsErr")
            st.dataframe(firm_df, use_container_width=True)
        else:
            st.write("No firm rows available for accuracy table.")
    else:
        st.write("No polling firm column found in polls data.")

    # Constituency sensitivity to benchmark poll (focus on Fidesz by default)
    st.markdown("### Constituency sensitivity vs benchmark poll (Fidesz)")
    if mapping["Fidesz"] in df_votes.columns and "constituency_code" in df_meta.columns:
        f_votes = df_votes[mapping["Fidesz"]]
        tmp = pd.DataFrame({"constituency_code": df_meta["constituency_code"], "V": total_votes, "C": f_votes})
        grp = tmp.groupby("constituency_code")[["V", "C"]].sum()
        share = 100.0 * grp["C"] / grp["V"]
        poll_f = float(row["Fidesz"]) if "Fidesz" in row.index else np.nan

        st.dataframe(
            pd.DataFrame({"Fidesz_constituency_share_%": share, "Benchmark_poll_%": poll_f, "Deviation_pp": share - poll_f})
            .sort_values("Deviation_pp", key=lambda s: s.abs(), ascending=False)
            .head(25),
            use_container_width=True,
        )

        st.dataframe(
            pd.Series(
                {
                    "Mean absolute deviation (pp)": float((share - poll_f).abs().mean()),
                    "P10 deviation (pp)": float(np.percentile((share - poll_f).values, 10)),
                    "Median deviation (pp)": float(np.percentile((share - poll_f).values, 50)),
                    "P90 deviation (pp)": float(np.percentile((share - poll_f).values, 90)),
                    "Max abs deviation (pp)": float((share - poll_f).abs().max()),
                }
            ).to_frame("Value"),
            use_container_width=True,
        )
    else:
        st.write("Cannot compute constituency sensitivity (missing constituency_code or Fidesz list column).")


# =============================
# Seat projection (Hungary)
# =============================

def dhondt_allocate(votes: Dict[str, float], seats: int) -> Dict[str, int]:
    """D'Hondt highest averages. votes must be non-negative."""
    parties = [p for p, v in votes.items() if v > 0]
    alloc = {p: 0 for p in parties}
    if seats <= 0 or not parties:
        return alloc

    # iterative allocation
    for _ in range(seats):
        best_p = None
        best_q = -1.0
        for p in parties:
            q = votes[p] / (alloc[p] + 1)
            if q > best_q:
                best_q = q
                best_p = p
        if best_p is None:
            break
        alloc[best_p] += 1
    return alloc


def list_threshold_pct(coalition_size: int) -> float:
    # Hungary: 5% single; 10% for 2-party joint; 15% for 3+ joint
    if coalition_size <= 1:
        return 5.0
    if coalition_size == 2:
        return 10.0
    return 15.0


def hungary_preferential_quota(total_list_votes: float, list_seats: int = 93) -> float:
    # Preferential quota: (total votes / list seats) / 4  (approx 0.27% of votes in practice)
    return (total_list_votes / list_seats) / 4.0 if list_seats > 0 else 0.0


def compute_station_excess_votes(
    df_votes: pd.DataFrame,
    total_votes: pd.Series,
    df_meta: pd.DataFrame,
    candidate_col: str,
    threshold_T: float,
    baseline_mode: str,  # "National" or "Constituency"
) -> pd.Series:
    """
    Excess votes for candidate_col at station level:
    excess = max(0, share - baseline) * total_votes, but only when share > threshold_T.
    baseline is national (within current df) or constituency-level (within current df).
    """
    cand = df_votes[candidate_col]
    share = (cand / total_votes) * 100.0

    # baseline
    if baseline_mode == "Constituency" and "constituency_code" in df_meta.columns:
        tmp = pd.DataFrame({"cc": df_meta["constituency_code"], "V": total_votes, "C": cand})
        grp = tmp.groupby("cc")[["V", "C"]].sum()
        cons_share = (grp["C"] / grp["V"]).replace([np.inf, -np.inf], 0).fillna(0) * 100.0
        baseline = df_meta["constituency_code"].map(cons_share).fillna((cand.sum() / total_votes.sum()) * 100.0)
    else:
        baseline_val = (cand.sum() / total_votes.sum()) * 100.0 if float(total_votes.sum()) > 0 else 0.0
        baseline = pd.Series(baseline_val, index=df_votes.index)

    excess_pct = (share - baseline).clip(lower=0)
    excess_votes = (excess_pct * total_votes / 100.0)

    # only above user threshold
    excess_votes = excess_votes.where(share > threshold_T, 0.0)

    # cannot exceed candidate votes
    excess_votes = np.minimum(excess_votes, cand)

    return excess_votes.fillna(0.0)


def apply_candidate_vote_reduction(
    df_votes: pd.DataFrame,
    candidate_col: str,
    excess_votes: pd.Series,
    redistribute_to_others: bool,
) -> pd.DataFrame:
    """
    Reduce candidate_col by excess_votes.
    If redistribute_to_others=True: redistribute removed votes proportionally to other columns per station.
    Else: discard removed votes (total votes shrink).
    """
    df_adj = df_votes.copy()
    ex = excess_votes.reindex(df_adj.index).fillna(0.0)

    # remove from candidate
    df_adj[candidate_col] = (df_adj[candidate_col] - ex).clip(lower=0)

    if redistribute_to_others:
        others = [c for c in df_adj.columns if c != candidate_col]
        if not others:
            return df_adj

        other_sum = df_adj[others].sum(axis=1)

        # proportional redistribution; if other_sum==0, distribute evenly
        prop = df_adj[others].div(other_sum.replace(0, np.nan), axis=0)
        even = pd.DataFrame(1.0 / len(others), index=df_adj.index, columns=others)

        weights = prop.fillna(even)
        df_adj[others] = df_adj[others] + weights.mul(ex, axis=0)

    return df_adj


def compute_smd_winners_and_compensation(
    df_ind_votes: pd.DataFrame,
    df_meta_ind: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Returns:
      - constituency_result_df with winner/runner-up and margins
      - compensation_votes_by_party (losers + winner surplus)
    """
    if "constituency_code" not in df_meta_ind.columns:
        st.error("Missing constituency_code in data; cannot compute SMD seats.")
        st.stop()

    parties = list(df_ind_votes.columns)
    tmp = df_ind_votes.copy()
    tmp["cc"] = df_meta_ind["constituency_code"].astype(str)
    cons = tmp.groupby("cc")[parties].sum()

    # SMD winner per constituency
    winner_party = cons.idxmax(axis=1)
    winner_votes = cons.max(axis=1)

    # runner-up votes per constituency
    runner_votes = cons.apply(lambda r: np.sort(r.values)[-2] if len(r.values) >= 2 else 0.0, axis=1)
    runner_party = cons.apply(lambda r: r.index[np.argsort(r.values)[-2]] if len(r.values) >= 2 else r.index[0], axis=1)

    # compensation:
    # losers: all losing votes per party
    # winner surplus: winner_votes - (runner_votes + 1)
    comp = {p: 0.0 for p in parties}

    for cc, row in cons.iterrows():
        wp = winner_party.loc[cc]
        wv = float(winner_votes.loc[cc])
        rv = float(runner_votes.loc[cc])

        # losers
        for p in parties:
            v = float(row[p])
            if p != wp:
                comp[p] += v

        # winner surplus (can’t be negative)
        surplus = max(0.0, wv - (rv + 1.0))
        comp[wp] += surplus

    res_df = pd.DataFrame(
        {
            "constituency_code": cons.index,
            "winner_party": winner_party.values,
            "winner_votes": winner_votes.values,
            "runner_party": runner_party.values,
            "runner_votes": runner_votes.values,
            "margin_votes": (winner_votes - runner_votes).values,
        }
    ).set_index("constituency_code")

    return res_df, comp


def seat_projection_analysis(df_raw: pd.DataFrame):
    st.subheader("Seat projection (199 seats: 106 SMD + 93 list with vote transfer)")
    st.caption(
        "This is a *counterfactual* calculator: it reduces a chosen party’s station-level vote share above a threshold, "
        "then recomputes SMD winners + list allocation with vote transfer."
    )

    # Build INDIVIDUAL votes (SMD tier)
    ind_cols = contest_vote_cols(df_raw, "Individual (single-member district)")
    df_ind, _, meta_ind = build_votes_df(df_raw, ind_cols, "valid_votes_individual")
    # total votes for shares in fraud model: use sum across IND columns
    V_ind = df_ind.sum(axis=1).replace(0, np.nan).fillna(0)

    # Build LIST votes (PR tier)
    list_cols = contest_vote_cols(df_raw, "List (party + minority lists)")
    df_list, total_list, meta_list = build_votes_df(df_raw, list_cols, "valid_votes_list")

    st.markdown("### Map the party columns (SMD vs list)")
    c1, c2 = st.columns(2)
    with c1:
        f_ind_col = st.selectbox(
            "SMD column to reduce",
            df_ind.columns.tolist(),
            index=df_ind.columns.tolist().index("votes_individual_party_fidesz") if "votes_individual_party_fidesz" in df_ind.columns else 0,
            key="seat_f_ind_col",
        )
    with c2:
        f_list_col = st.selectbox(
            "List column to reduce",
            df_list.columns.tolist(),
            index=df_list.columns.tolist().index("votes_list_comp_fidesz") if "votes_list_comp_fidesz" in df_list.columns else 0,
            key="seat_f_list_col",
        )

    st.markdown("### Vote reduction model")
    c1, c2, c3 = st.columns(3)
    with c1:
        T = st.slider("Apply reduction only when station share > T (%)", 50.0, 100.0, 70.0, 1.0, key="seat_T")
    with c2:
        baseline_mode = st.selectbox("Baseline for 'excess' vs", ["National", "Constituency"], index=1, key="seat_baseline_mode")
    with c3:
        redistribute = st.checkbox("Redistribute removed votes to others (keep totals constant)", value=False, key="seat_redistribute")

    # Excess votes in IND and LIST separately (because both determine seats)
    ex_ind = compute_station_excess_votes(df_ind, V_ind, meta_ind, f_ind_col, T, baseline_mode)
    ex_list = compute_station_excess_votes(df_list, total_list, meta_list, f_list_col, T, baseline_mode)

    # Apply reductions
    df_ind_adj = apply_candidate_vote_reduction(df_ind, f_ind_col, ex_ind, redistribute)
    df_list_adj = apply_candidate_vote_reduction(df_list, f_list_col, ex_list, redistribute)

    # -----------------------------
    # 1) SMD seats (106)
    # -----------------------------
    smd_actual_df, comp_actual = compute_smd_winners_and_compensation(df_ind, meta_ind)
    smd_adj_df, comp_adj = compute_smd_winners_and_compensation(df_ind_adj, meta_ind)

    smd_seats_actual = smd_actual_df["winner_party"].value_counts().to_dict()
    smd_seats_adj = smd_adj_df["winner_party"].value_counts().to_dict()

    # Constituencies where winner changes under de-cheating
    flips = smd_actual_df.join(smd_adj_df[["winner_party", "margin_votes"]], rsuffix="_adj")
    flips = flips[flips["winner_party"] != flips["winner_party_adj"]].copy()

    st.markdown("### SMD seats (106)")
    st.dataframe(
        pd.DataFrame({"SMD_actual": pd.Series(smd_seats_actual), "SMD_decheated": pd.Series(smd_seats_adj)})
        .fillna(0)
        .astype(int)
        .sort_values("SMD_actual", ascending=False),
        use_container_width=True,
    )

    st.markdown("### SMD winner flips (if any)")
    st.dataframe(flips.sort_values("margin_votes", key=lambda s: s.abs()).head(30), use_container_width=True)

    # -----------------------------
    # 2) List tier seats (93) with vote transfer
    # -----------------------------
    st.markdown("### List tier (93 seats)")

    # Separate party lists vs minority lists in your file
    party_list_cols = [c for c in df_list.columns if c.startswith("votes_list_comp_") or c.startswith("votes_list_party_")]
    minority_cols = [c for c in df_list.columns if c.startswith("votes_list_minority_")]

    st.markdown("#### Threshold / coalition settings (list tier)")
    st.caption("Coalition size determines list threshold: 1 party=5%, 2 parties=10%, 3+=15%.")

    # default assumptions for 2022: Fidesz-KDNP as joint list of 2; United for Hungary as 6+
    coalition_sizes: Dict[str, int] = {}
    with st.expander("Set coalition sizes (advanced)", expanded=False):
        for col in party_list_cols:
            label = humanize_vote_col(col)
            default = 1
            if "fidesz" in col.lower():
                default = 2  # Fidesz–KDNP
            if "united_for_hungary" in col.lower():
                default = 6  # 3+ coalition threshold
            coalition_sizes[col] = st.number_input(
                f"Coalition size for '{label}'", min_value=1, max_value=10, value=default, step=1, key=f"coal_{col}"
            )

    def list_seats_from(df_list_votes: pd.DataFrame, comp_votes: Dict[str, float]) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, int]]:
        # base list votes
        base_votes = {col: float(df_list_votes[col].sum()) for col in party_list_cols}
        total_votes_all_lists = float(df_list_votes[party_list_cols + minority_cols].sum().sum())

        # minority preferential mandates
        quota = hungary_preferential_quota(total_votes_all_lists, 93)
        minority_seats: Dict[str, int] = {}
        seats_left = 93
        for mcol in minority_cols:
            mv = float(df_list_votes[mcol].sum())
            if mv >= quota and seats_left > 0:
                minority_seats[mcol] = 1
                seats_left -= 1

        # add compensation votes (from SMD tier) to matching party columns where possible
        combined_votes = base_votes.copy()
        for ind_party, cv in comp_votes.items():
            # find best matching list column
            ind_key = ind_party.lower().replace("votes_individual_party_", "")
            match = None
            for lcol in party_list_cols:
                if ind_key and ind_key in lcol.lower():
                    match = lcol
                    break
            if match:
                combined_votes[match] = combined_votes.get(match, 0.0) + float(cv)

        # thresholding
        eligible: Dict[str, float] = {}
        for lcol, v in combined_votes.items():
            pct = 100.0 * v / total_votes_all_lists if total_votes_all_lists > 0 else 0.0
            thr = list_threshold_pct(int(coalition_sizes.get(lcol, 1)))
            if pct >= thr:
                eligible[lcol] = v

        # allocate remaining list seats with D'Hondt
        alloc = dhondt_allocate(eligible, seats_left)

        return alloc, combined_votes, minority_seats

    list_alloc_actual, _, minority_actual = list_seats_from(df_list, comp_actual)
    list_alloc_adj, _, minority_adj = list_seats_from(df_list_adj, comp_adj)

    st.dataframe(
        pd.DataFrame({"List_actual": pd.Series(list_alloc_actual), "List_decheated": pd.Series(list_alloc_adj)})
        .fillna(0)
        .astype(int)
        .sort_values("List_actual", ascending=False),
        use_container_width=True,
    )

    # -----------------------------
    # 3) Total seats (199)
    # -----------------------------
    def add_dicts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0) + int(v)
        return out

    total_actual = add_dicts(smd_seats_actual, list_alloc_actual)
    total_adj = add_dicts(smd_seats_adj, list_alloc_adj)

    # include minority preferential mandates as separate “lists”
    total_actual = add_dicts(total_actual, minority_actual)
    total_adj = add_dicts(total_adj, minority_adj)

    seats_df = pd.DataFrame({"Total_actual": pd.Series(total_actual), "Total_decheated": pd.Series(total_adj)}).fillna(0).astype(int)
    seats_df["Delta"] = seats_df["Total_decheated"] - seats_df["Total_actual"]

    st.markdown("### Total seats (199)")
    st.dataframe(seats_df.sort_values("Total_actual", ascending=False), use_container_width=True)

    # show how many votes were removed
    st.markdown("### Removed votes summary")
    c1, c2 = st.columns(2)
    c1.metric("Removed votes (SMD tier)", f"{int(round(float(ex_ind.sum()))):,}")
    c2.metric("Removed votes (List tier)", f"{int(round(float(ex_list.sum()))):,}")


# -----------------------------
# Public entrypoint: Streamlit tab renderer
# -----------------------------
def render_stealing_analysis_tab():
    st.subheader("Station-level anomaly / “stealing” analysis")
    st.caption(
        "Upload a polling-station results CSV (e.g., 2022 station-level results). "
        "This tab provides: (1) threshold/cap diagnostics by station, (2) polls-vs-results diagnostics, "
        "and (3) a counterfactual seat calculator that reduces excess votes above a threshold."
    )

    with st.expander("Data input", expanded=True):
        c1, c2 = st.columns([1.25, 1.0])
        with c1:
            upload = st.file_uploader("Upload polling_station_results.csv", type=["csv"], key="steal_csv_upload")
            path = st.text_input("…or local path (if running locally)", value="data/polling_station_results.csv", key="steal_csv_path")
        with c2:
            st.markdown("**Expected columns**")
            st.write(
                "- votes_individual_party_* (SMD tier)\n"
                "- votes_list_comp_* / votes_list_party_* / votes_list_minority_* (list tier)\n"
                "- maz, constituency_code (recommended)\n"
                "- polling_station_address / polling_station_name (optional; for urban/rural classification)"
            )

    df_raw = None
    try:
        if upload is not None:
            df_raw = load_raw_from_bytes(upload.getvalue())
        else:
            if Path(path).exists():
                df_raw = load_raw_from_path(path)
            else:
                st.info("Upload a CSV above (or provide a valid local path) to run this analysis.")
                return
    except Exception as e:
        st.error(f"Failed to load results CSV: {e}")
        return

    # add OEVK id and constituency-level urban/rural mapping
    df_raw = df_raw.copy()

    if {"maz", "constituency_code"}.issubset(df_raw.columns):
        df_raw["oevk_id"] = (
            df_raw["maz"].astype(int).astype(str).str.zfill(2) + "-" + df_raw["constituency_code"].astype(int).astype(str).str.zfill(2)
        )
    else:
        df_raw["oevk_id"] = ""

    df_raw["area_type"] = classify_oevk_urban_rural(df_raw)

    st.markdown("### Loaded dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Polling stations", f"{len(df_raw):,}")
    c2.metric("Columns", f"{len(df_raw.columns):,}")
    c3.metric("Urban stations (classified)", f"{int((df_raw['area_type'] == 'Urban').sum()):,}")

    analysis_type = st.selectbox(
        "Select analysis type",
        ["Threshold cap (with baselines + station table)", "Polls vs results", "Seat projection"],
        index=0,
        key="steal_analysis_type",
    )

    if analysis_type == "Threshold cap (with baselines + station table)":
        contest = st.selectbox(
            "Contest",
            ["Individual (single-member district)", "List (party lists)", "List (party + minority lists)"],
            index=0,
            key="steal_contest",
        )
        vote_cols = contest_vote_cols(df_raw, contest)
        total_col = TOTAL_COL_BY_CONTEST.get(contest)
        df_votes, total_votes, df_meta = build_votes_df(df_raw, vote_cols, total_col)

        candidate_col = candidate_col_widget(vote_cols, key="steal_candidate_col")
        st.write(f"Using column: **{humanize_vote_col(candidate_col)}**")

        threshold_cap_analysis(df_votes, total_votes, df_meta, candidate_col)

    elif analysis_type == "Polls vs results":
        polls_analysis(df_raw)

    elif analysis_type == "Seat projection":
        seat_projection_analysis(df_raw)
