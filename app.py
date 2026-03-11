
import json
import os
import re
import io
import hashlib
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import hungary_model as hm
from hungary_model.constants import DB_REQUIRED_FILES, POLL_REQUIRED_FILES
from stealing_analysis import render_stealing_analysis_tab


# ============================================================
# Page
# ============================================================

st.set_page_config(
    page_title="Hungary Election Model (2026) — Streamlit",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 2.0rem;}
div[data-testid="stMetric"] {background: rgba(0,0,0,0.03); padding: 0.75rem; border-radius: 0.6rem;}
div[data-testid="stMetric"] > div {justify-content: center;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Cache
# ============================================================

@st.cache_resource(show_spinner=False)
def cached_load_model_data(db_dir: Optional[str], poll_dir: Optional[str]) -> hm.ModelData:
    paths = hm.resolve_model_paths(
        db_dir=Path(db_dir) if db_dir else None,
        poll_dir=Path(poll_dir) if poll_dir else None,
    )
    return hm.load_model_data(paths)


@st.cache_data(show_spinner=False, max_entries=40)
def cached_run_scenario(db_dir: Optional[str], poll_dir: Optional[str], cfg_json: str) -> hm.ScenarioResults:
    data = cached_load_model_data(db_dir_override, poll_dir_override)
    cfg_dict = json.loads(cfg_json)
    cfg = hm.ScenarioConfig(**cfg_dict)
    return hm.run_scenario(data, cfg)


@st.cache_data(show_spinner=False, max_entries=10)
def cached_run_monte_carlo(
    db_dir: Optional[str],
    poll_dir: Optional[str],
    scenario_cfg_json: str,
    mc_json: str,
) -> hm.MonteCarloResults:
    data = cached_load_model_data(db_dir, poll_dir)
    cfg_dict = json.loads(scenario_cfg_json)
    cfg = hm.ScenarioConfig(**cfg_dict)
    base = hm.run_scenario(data, cfg)
    mc_dict = json.loads(mc_json)
    mc = hm.MonteCarloConfig(**mc_dict)
    return hm.run_monte_carlo(data, base, mc)


# ============================================================
# Streamlit Cloud helpers (optional)
# ============================================================

@st.cache_data(show_spinner=False)
def _extract_zip_to_tmp(zip_bytes: bytes) -> str:
    """Extract a user-uploaded zip to a stable tmp folder (keyed by content hash)."""
    h = hashlib.sha256(zip_bytes).hexdigest()[:12]
    out_dir = Path(tempfile.gettempdir()) / f"hungary_model_bundle_{h}"
    if out_dir.exists():
        return str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(out_dir)
    return str(out_dir)


def _find_dir_with_files(base: Path, required_files: List[str], max_depth: int = 6) -> Optional[Path]:
    """Return the first directory under base that contains all required_files."""
    base = base.resolve()
    for root, dirs, _files in os.walk(base):
        rel = Path(root).resolve().relative_to(base)
        if len(rel.parts) > max_depth:
            # don't descend further
            dirs[:] = []
            continue
        if all((Path(root) / f).exists() for f in required_files):
            return Path(root).resolve()
    return None



# ============================================================
# Helpers
# ============================================================

def pct(x: float, decimals: int = 2) -> float:
    return round(100.0 * float(x), decimals)


def normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in raw.items() if float(v) >= 0}
    s = float(sum(w.values()))
    if s <= 0:
        return {k: 0.0 for k in raw}
    return {k: v / s for k, v in w.items()}


def share_sum_widget(label: str, s: float, target: float, tol: float = 1e-6) -> None:
    delta = s - target
    if abs(delta) <= tol:
        st.success(f"{label}: {s:.4f}")
    else:
        st.warning(f"{label}: {s:.4f} (target {target:.4f})")


def build_dhondt_plot(dhondt: pd.DataFrame, n_show: int = 250) -> go.Figure:
    d = dhondt.head(n_show).copy()
    d["rank"] = np.arange(1, len(d) + 1)
    fig = px.scatter(d, x="rank", y="quotient", color="party", hover_data=["divisor", "wins_seat"])
    if (dhondt["wins_seat"] == True).any():
        cutoff = float(dhondt.loc[dhondt["wins_seat"] == True, "quotient"].min())
        fig.add_hline(y=cutoff, line_dash="dash")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def compute_mail_votes(total: int, pct_shares: Dict[str, float]) -> Dict[str, int]:
    """
    Convert total mail votes and per-party pct (0..100) into integer vote counts.
    Rounds to nearest integer and fixes rounding drift by allocating remainder to the largest party share.
    """
    total = int(max(0, total))
    if total == 0:
        return {p: 0 for p in hm.MODEL_PARTIES}

    w = {p: float(pct_shares.get(p, 0.0)) for p in hm.MODEL_PARTIES}
    s = sum(w.values())
    if s <= 0:
        out = {p: 0 for p in hm.MODEL_PARTIES}
        out["OTHER"] = total
        return out
    w = {p: 100.0 * v / s for p, v in w.items()}

    raw = {p: total * (w[p] / 100.0) for p in hm.MODEL_PARTIES}
    out = {p: int(round(raw[p])) for p in hm.MODEL_PARTIES}
    drift = total - sum(out.values())
    if drift != 0:
        pmax = max(w.keys(), key=lambda k: w[k])
        out[pmax] += drift
    return out


def join_evk_names(index: pd.Index, evk_meta: pd.DataFrame) -> pd.DataFrame:
    """Return a small frame mapping evk_id -> evk_name (if available)."""
    if evk_meta is None or evk_meta.empty:
        return pd.DataFrame(index=index)
    cols = [c for c in ["evk_name", "evk_no", "county_id"] if c in evk_meta.columns]
    m = evk_meta.reindex(index)[cols].copy()
    return m


# ============================================================
# Sidebar: data (auto-detected; override optional)
# ============================================================

st.sidebar.header("Scope")
exclude_budapest = st.sidebar.toggle(
    "Exclude Budapest",
    value=False,
    help=(
        "Treat poll inputs as 'Hungary ex Budapest'. The model will compute coefficients excluding Budapest, "
        "add Budapest list votes back via a historical estimate, and force all Budapest SMC winners to TISZA."
    ),
)

st.sidebar.markdown("---")

st.sidebar.header("Data")

db_dir_override: Optional[str] = None
poll_dir_override: Optional[str] = None

with st.sidebar.expander("Inputs (repo / upload / override)", expanded=False):
    st.caption(
        "Streamlit Community Cloud runs `streamlit run` from the *repository root*.\n"
        "Recommended repo layout:\n"
        "  • `data/Hungary Election Results/` for the DB files\n"
        "  • `data/` for the poll CSVs\n"
        "The app can also auto-detect `./Hungary Election Results/` + `./` if you prefer."
    )

    data_source = st.radio(
        "Data source",
        ["Auto-detect from repo", "Upload data bundle (.zip)", "Manual override paths"],
        index=0,
    )

    if data_source == "Upload data bundle (.zip)":
        bundle = st.file_uploader(
            "Upload model data bundle (.zip)",
            type=["zip"],
            help=(
                "The zip should contain the DB directory (e.g. 'Hungary Election Results/' with the *.db files) "
                "and the poll CSVs (2018_poll_hungary.csv, ..., 2026_poll_hungary.csv)."
            ),
            key="model_data_bundle_zip",
        )
        if bundle is not None:
            base_dir = Path(_extract_zip_to_tmp(bundle.getvalue()))
            db_dir = _find_dir_with_files(base_dir, DB_REQUIRED_FILES)
            poll_dir = _find_dir_with_files(base_dir, POLL_REQUIRED_FILES)

            if db_dir is None or poll_dir is None:
                st.error(
                    "Could not find the required DB/poll files inside the uploaded zip. "
                    "Expected DB files: "
                    + ", ".join(DB_REQUIRED_FILES)
                    + " | Expected poll files: "
                    + ", ".join(POLL_REQUIRED_FILES)
                )
                st.stop()

            db_dir_override = str(db_dir)
            poll_dir_override = str(poll_dir)

            st.success("Using uploaded data bundle ✅")
            st.caption(f"DB dir: `{db_dir_override}`")
            st.caption(f"Poll dir: `{poll_dir_override}`")
        else:
            st.info("Upload a .zip to use data that isn't committed to the repo.")

    elif data_source == "Manual override paths":
        app_root = Path(__file__).resolve().parent
        db_dir_override = st.text_input("DB directory", value=str(app_root / "Hungary Election Results"))
        poll_dir_override = st.text_input("Poll CSV directory", value=str(app_root))
        st.caption("Tip: on Community Cloud, absolute paths will look like `/mount/src/<repo>/...`")

    else:
        st.caption("Auto-detect will search `./`, `./Hungary Election Results/`, and `./data/…`")

    if st.button("Clear cache", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()

try:
    data = cached_load_model_data(db_dir_override, poll_dir_override)
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.caption(f"DB dir: `{data.paths.db_dir}`")
st.sidebar.caption(f"Poll dir: `{data.paths.poll_dir}`")
if getattr(data, "evk_meta", None) is not None and not data.evk_meta.empty:
    st.sidebar.caption("EVK mapping: using 2026 polling-station → EVK map (names available).")
else:
    st.sidebar.caption("EVK mapping: using EVK derived from historical district name (no 2026 mapping DB found).")


# ============================================================
# Sidebar: scenario inputs
# ============================================================

st.sidebar.header("Scenario")
cfg0 = hm.default_config()

# --- turnout target
turnout_target = st.sidebar.slider(
    "Turnout target (% of registered)",
    min_value=40.0,
    max_value=90.0,
    value=float(cfg0.turnout_target * 100.0),
    step=0.1,
) / 100.0

# --- turnout model
st.sidebar.subheader("Turnout model")

turnout_model_labels = {
    "uniform": "Uniform (same turnout rate everywhere)",
    "scaled_reference": "Scale reference turnout map (e.g., 2022) to target",
    "logit_slope_original": "Logit-slope (station, no EP offset)",
    "logit_slope_ep_offset": "Logit-slope + EP offset (grouped)",
    "relative_offset_parl": "Relative offset (parl-only)",
    "logit_slope_reserve_adjusted": "Logit-slope + reserve adjustment",
    "baseline_plus_marginal": "Baseline + marginal turnout (parl baseline)",
    "baseline_ref_plus_elastic_delta": "Baseline (fixed %) + elasticity delta (surplus/deficit)",
}
turnout_model = st.sidebar.selectbox(
    "Turnout model",
    options=hm.TURNOUT_MODELS,
    format_func=lambda k: turnout_model_labels.get(k, k),
    index=hm.TURNOUT_MODELS.index(cfg0.turnout_model) if cfg0.turnout_model in hm.TURNOUT_MODELS else 0,
)

turnout_granularity = st.sidebar.selectbox(
    "Turnout granularity",
    options=list(hm.TURNOUT_GRANULARITIES.keys()),
    index=list(hm.TURNOUT_GRANULARITIES.keys()).index(cfg0.turnout_granularity) if cfg0.turnout_granularity in hm.TURNOUT_GRANULARITIES else 1,
    help="Grouping level for turnout model (used in EP-offset / reserve-adjusted / baseline+marginal).",
)

turnout_reference_election = st.sidebar.selectbox(
    "Turnout baseline election",
    options=["parl_2022_list", "parl_2018_list"],
    index=0 if cfg0.turnout_reference_election == "parl_2022_list" else 1,
)

turnout_baseline_level = float(getattr(cfg0, "turnout_baseline_level", 0.70))
if turnout_model == "baseline_ref_plus_elastic_delta":
    turnout_baseline_level = (
        st.sidebar.slider(
            "Baseline turnout level (%)",
            min_value=20.0,
            max_value=90.0,
            value=float(turnout_baseline_level * 100.0),
            step=1.0,
            help="Used only for 'Baseline (fixed %) + elasticity delta' turnout model.",
        )
        / 100.0
    )

clip_lo, clip_hi = cfg0.turnout_clip if isinstance(cfg0.turnout_clip, (list, tuple)) else (0.20, 0.95)
with st.sidebar.expander("Turnout caps and advanced", expanded=False):
    turnout_clip_lo = st.slider("Turnout clip low", min_value=0.10, max_value=0.60, value=float(clip_lo), step=0.01)
    turnout_clip_hi = st.slider("Turnout clip high", min_value=0.70, max_value=0.99, value=float(clip_hi), step=0.01)

    reserve_adjust_power = st.slider(
        "Reserve adjust power",
        min_value=0.0,
        max_value=3.0,
        value=float(getattr(cfg0, "reserve_adjust_power", 1.0)),
        step=0.05,
        help="Only used for 'logit_slope_reserve_adjusted'. Higher => re-allocates turnout change more to groups with more headroom.",
    )

    marginal_concentration = st.slider(
        "Marginal concentration",
        min_value=0.0,
        max_value=4.0,
        value=float(getattr(cfg0, "marginal_concentration", 1.0)),
        step=0.05,
        help="Only used for 'baseline_plus_marginal'. Higher => concentrates marginal turnout change into high marginal-propensity areas.",
    )


# --- vote allocation model
st.sidebar.subheader("Vote model")

vote_model_labels = {
    "ipf": "IPF (fixed national vote shares; geography-only allocation)",
    "baseline_marginal": "Baseline+marginal (turnout composition; national totals endogenous)",
}

vote_allocation_model = st.sidebar.radio(
    "Approach",
    options=["ipf", "baseline_marginal"],
    format_func=lambda k: vote_model_labels.get(k, k),
    index=0 if str(cfg0.vote_allocation_model).strip().lower() == "ipf" else 1,
)

if bool(exclude_budapest) and str(vote_allocation_model).strip().lower() != "ipf":
    st.sidebar.warning("Exclude Budapest mode currently works only with IPF vote allocation. Switch Approach to IPF.")

if vote_allocation_model == "ipf":
    st.sidebar.markdown(
        """
**IPF** keeps the *national vote shares* fixed.
- You choose national voter shares (from polls or manual input).
- You choose turnout geography (turnout model → votes per area).
- IPF allocates votes across areas so that **row totals** (area votes) and **column totals** (national party totals) both match.

Use this when you want **polls to fully determine national totals**, and geography only affects EVK winners.
"""
    )
else:
    st.sidebar.markdown(
        """
**Baseline+marginal** treats polls as *population shares* (incl. undecided pool).
- Turnout target determines how many people vote.
- Mobilization/reserves determine who captures **marginal turnout**.
- National vote shares become an **output**, not an input.

Use this when you want turnout changes to shift **composition** (who shows up), not just geography.
"""
    )

# Controls that depend on vote model
if vote_allocation_model == "baseline_marginal":
    with st.sidebar.expander("Baseline+marginal settings", expanded=False):
        undecided_geo_model = st.selectbox(
            "Undecided geography",
            options=["uniform", "elasticity", "low_turnout"],
            index=["uniform", "elasticity", "low_turnout"].index(str(getattr(cfg0, "undecided_geo_model", "uniform"))),
            help="Where undecided supporters are located (affects marginal turnout composition).",
        )

        undecided_local_lean_strength = st.slider(
            "Undecided local lean strength",
            min_value=0.0,
            max_value=3.0,
            value=float(getattr(cfg0, "undecided_local_lean_strength", 0.0)),
            step=0.05,
            help="Tilts marginal undecided voting toward the locally stronger side (Fidesz vs Tisza).",
        )

    modelling_unit = "station"  # enforced
else:
    undecided_geo_model = str(getattr(cfg0, "undecided_geo_model", "uniform"))
    undecided_local_lean_strength = float(getattr(cfg0, "undecided_local_lean_strength", 0.0))

    modelling_unit = st.sidebar.radio(
        "IPF raking unit",
        options=["station", "evk"],
        index=0 if str(cfg0.modelling_unit).strip().lower() == "station" else 1,
        horizontal=True,
        help="station = most detailed (slower). evk = faster (rakes at EVK level).",
    )

# --- party preference input mode
st.sidebar.subheader("Party preference")

pref_mode = st.sidebar.radio(
    "Input mode",
    options=["Use polls", "Manual input"],
    index=0,
    horizontal=True,
)

poll_type = str(cfg0.poll_type)
pollster_filter: Optional[List[str]] = None
last_n_days = int(cfg0.last_n_days)
half_life = float(cfg0.poll_decay_half_life_days)
manual_poll_override: Optional[Dict[str, float]] = None

avg_poll_preview = pd.Series(dtype=float)

if pref_mode == "Use polls":
    poll_types = sorted(data.polls_all.loc[data.polls_all["year"] == 2026, "poll_type"].dropna().unique().tolist())
    if not poll_types:
        poll_types = ["raw", "decided"]

    # If baseline+marginal, force raw
    if vote_allocation_model == "baseline_marginal":
        poll_type = "raw"
        st.sidebar.info("Baseline+marginal requires raw polls (undecided pool). Poll type is forced to 'raw'.")
    else:
        poll_type = st.sidebar.selectbox(
            "Poll type (2026)",
            options=poll_types,
            index=poll_types.index(cfg0.poll_type) if cfg0.poll_type in poll_types else 0,
        )

    pollsters_2026 = sorted(data.polls_all.loc[data.polls_all["year"] == 2026, "pollster"].dropna().unique().tolist())
    pollster_filter = st.sidebar.multiselect("Pollsters (2026)", options=pollsters_2026, default=pollsters_2026)

    last_n_days = st.sidebar.slider("Polling window (days)", min_value=7, max_value=365, value=int(cfg0.last_n_days), step=1)
    half_life = st.sidebar.slider("Recency decay half-life (days, 0 = off)", min_value=0.0, max_value=180.0, value=float(cfg0.poll_decay_half_life_days), step=1.0)

    if pollster_filter is not None and len(pollster_filter) == 0:
        st.sidebar.warning("No pollster selected; using all pollsters.")
        pollster_filter = None

    # preview avg poll
    try:
        _tmp_cfg = hm.ScenarioConfig(
            turnout_target=float(turnout_target),
            poll_type=str(poll_type),
            last_n_days=int(last_n_days),
            pollster_filter=pollster_filter,
            poll_decay_half_life_days=float(half_life),
            manual_poll_override=None,
        )
        avg_poll_preview = hm.get_poll_average_2026(data.polls_all, _tmp_cfg)
    except Exception:
        avg_poll_preview = pd.Series(dtype=float)

else:
    include_undecided = st.sidebar.toggle("Manual input includes UNDECIDED share", value=True)
    st.sidebar.caption("Enter shares as fractions (0..1). A sum counter is shown; the model uses a normalized vector.")

    keys = ["FIDESZ", "TISZA", "DK", "MH", "MKKP", "OTHER"] + (["UNDECIDED"] if include_undecided else [])
    defaults_share = {k: float(avg_poll_preview.get(k, 0.0) or 0.0) / 100.0 for k in keys} if not avg_poll_preview.empty else {k: 0.0 for k in keys}

    manual_shares: Dict[str, float] = {}
    cols = st.sidebar.columns(2)
    for i, k in enumerate(keys):
        with cols[i % 2]:
            manual_shares[k] = float(
                st.number_input(
                    f"{k} share",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(defaults_share.get(k, 0.0)),
                    step=0.001,
                    format="%.3f",
                    key=f"man_{k}",
                )
            )

    s_manual = float(sum(manual_shares.values()))
    with st.sidebar.container():
        share_sum_widget("Manual shares sum", s_manual, target=1.0, tol=1e-3)

    if s_manual > 0:
        manual_shares_norm = {k: v / s_manual for k, v in manual_shares.items()}
    else:
        manual_shares_norm = {k: 0.0 for k in manual_shares}

    manual_poll_override = {k: 100.0 * float(manual_shares_norm.get(k, 0.0)) for k in keys}
    for k in ["FIDESZ", "TISZA", "DK", "MH", "MKKP", "OTHER", "UNDECIDED"]:
        manual_poll_override.setdefault(k, 0.0)

    poll_type = "raw" if include_undecided else "decided"
    pollster_filter = None
    last_n_days = int(cfg0.last_n_days)
    half_life = float(cfg0.poll_decay_half_life_days)

# --- undecided and nonresponse
st.sidebar.subheader("Undecided & nonresponse")

nonresponse_nonvoter_pct = st.sidebar.slider(
    "% not in polls and not voting (added to undecided pool in raw mode)",
    min_value=0.0,
    max_value=30.0,
    value=float(cfg0.nonresponse_nonvoter_pct * 100.0),
    step=0.1,
) / 100.0

uF = st.sidebar.slider(
    "Share of voting undecideds → Fidesz",
    min_value=0.0,
    max_value=1.0,
    value=float(cfg0.undecided_to_fidesz),
    step=0.01,
)

uT_default = float(getattr(cfg0, "undecided_to_tisza", 1.0 - float(cfg0.undecided_to_fidesz)))
uT = st.sidebar.slider(
    "Share of voting undecideds → Tisza",
    min_value=0.0,
    max_value=1.0,
    value=float(uT_default),
    step=0.01,
)

u_sum = uF + uT
if u_sum > 1.0:
    st.sidebar.warning("Fidesz+Tisza split > 1. The model rescales them to sum to 1.")
u_rem = max(0.0, 1.0 - min(1.0, u_sum))
st.sidebar.caption(f"Remainder to DK/MH/MKKP/OTHER (pro-rata): {u_rem:.2f}")

# --- mobilization model
st.sidebar.subheader("Mobilization / turnout composition")

use_mobilization_model = st.sidebar.toggle(
    "Use mobilization/reserve tilt",
    value=bool(getattr(cfg0, "use_mobilization_model", True)),
    help=(
        "Affects only raw-poll conversions and baseline+marginal vote allocation. "
        "If disabled, undecided is allocated using the manual split without reserve tilt."
    ),
)

mobilization_all_parties = st.sidebar.toggle(
    "Model turnout propensities for ALL parties",
    value=bool(getattr(cfg0, "mobilization_all_parties", False)),
    help=(
        "If enabled, mobilization_rates (and reserve_strength) apply to all MODEL_PARTIES. "
        "This allows turnout changes to shift composition beyond just the undecided pool."
    ),
)

with st.sidebar.expander("Mobilization inputs", expanded=False):
    mob0 = getattr(cfg0, "mobilization_rates", {}) or {}
    res0 = getattr(cfg0, "reserve_strength", {}) or {}

    mobilization_rates: Dict[str, float] = {}
    reserve_strength: Dict[str, float] = {}

    # Default sliders always include the big 2
    mobilization_rates["FIDESZ"] = float(
        st.slider(
            "FIDESZ mobilization rate",
            min_value=0.0,
            max_value=1.0,
            value=float(mob0.get("FIDESZ", 0.70)),
            step=0.01,
        )
    )
    mobilization_rates["TISZA"] = float(
        st.slider(
            "TISZA mobilization rate",
            min_value=0.0,
            max_value=1.0,
            value=float(mob0.get("TISZA", 0.90)),
            step=0.01,
        )
    )

    st.markdown("---")

    reserve_strength["FIDESZ"] = float(
        st.slider(
            "FIDESZ reserve strength",
            min_value=0.0,
            max_value=2.0,
            value=float(res0.get("FIDESZ", 1.0)),
            step=0.05,
        )
    )
    reserve_strength["TISZA"] = float(
        st.slider(
            "TISZA reserve strength",
            min_value=0.0,
            max_value=2.0,
            value=float(res0.get("TISZA", 1.0)),
            step=0.05,
        )
    )

    if mobilization_all_parties:
        st.markdown("---")
        st.caption("All-party settings (optional; default 1.0 mobilization, 1.0 reserve).")

        for p in ["DK", "MH", "MKKP", "OTHER"]:
            mobilization_rates[p] = float(
                st.slider(
                    f"{p} mobilization rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(mob0.get(p, 1.0)),
                    step=0.01,
                )
            )
        st.markdown("---")
        for p in ["DK", "MH", "MKKP", "OTHER"]:
            reserve_strength[p] = float(
                st.slider(
                    f"{p} reserve strength",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(res0.get(p, 1.0)),
                    step=0.05,
                )
            )
    else:
        # Ensure keys exist for stable config JSON
        for p in ["DK", "MH", "MKKP", "OTHER"]:
            mobilization_rates.setdefault(p, float(mob0.get(p, 1.0)))
            reserve_strength.setdefault(p, float(res0.get(p, 1.0)))

# --- coefficients
st.sidebar.header("Coefficients")

available_elections = ["ep_2024", "parl_2022_list", "ep_2019", "parl_2018_list"]
default_sel = [k for k in available_elections if k in (cfg0.election_weights or {})]
if not default_sel:
    default_sel = available_elections[:]
sel_elections = st.sidebar.multiselect("Elections used", options=available_elections, default=default_sel)
if not sel_elections:
    sel_elections = default_sel

with st.sidebar.expander("Election weights (auto-normalized)", expanded=False):
    w_raw: Dict[str, float] = {}
    for k in sel_elections:
        w_raw[k] = float(
            st.number_input(
                f"{k} weight",
                min_value=0.0,
                value=float((cfg0.election_weights or {}).get(k, 0.0) or 0.0),
                step=0.01,
                format="%.3f",
                key=f"w_{k}",
            )
        )
    election_weights = normalize_weights(w_raw)

with st.sidebar.expander("Geography weights (auto-normalized)", expanded=False):
    g0 = cfg0.geo_weights or {}
    g_default = {
        "station": float(g0.get("station", 0.0)),
        "location": float(g0.get("location", 0.60)),
        "settlement": float(g0.get("settlement", 0.25)),
        "evk": float(g0.get("evk", 0.15)),
    }
    g_raw: Dict[str, float] = {}
    labels = {
        "station": "Polling station (station_id)",
        "location": "Polling location (same name)",
        "settlement": "Settlement (maz-taz)",
        "evk": "Single constituency (EVK)",
    }
    for k in ["station", "location", "settlement", "evk"]:
        g_raw[k] = float(
            st.number_input(
                f"{labels[k]} weight",
                min_value=0.0,
                value=float(g_default.get(k, 0.0)),
                step=0.01,
                format="%.3f",
                key=f"g_{k}",
            )
        )
    geo_weights = normalize_weights(g_raw)

undecided_elasticity_link_strength = st.sidebar.slider(
    "Elasticity→geo tilt strength (Fidesz vs Tisza)",
    min_value=-2.0,
    max_value=2.0,
    value=float(getattr(cfg0, "undecided_elasticity_link_strength", 0.0)),
    step=0.05,
    help="Used in IPF initial matrix; also used by undecided_geo_model='elasticity' in baseline+marginal.",
)

# --- seats
st.sidebar.header("Seats")

smc_plus1_minus1 = st.sidebar.toggle("Apply +1pp/-1pp SMC adjustment", value=bool(cfg0.smc_plus1_minus1))
winner_two_party_only = st.sidebar.toggle("Winner among Fidesz vs Tisza only", value=bool(cfg0.winner_two_party_only))

threshold = st.sidebar.slider("List threshold", min_value=0.0, max_value=0.10, value=float(cfg0.threshold), step=0.005, format="%.3f")
nationality_seat_to_fidesz = st.sidebar.toggle("Allocate nationality seat to Fidesz (reduces list seats by 1)", value=bool(cfg0.nationality_seat_to_fidesz))

# --- Mail votes
st.sidebar.subheader("Mail-in votes")
with st.sidebar.expander("Mail-in vote assumptions", expanded=False):
    dv0 = cfg0.diaspora_votes or {}
    baseline_total = int(sum(int(v) for v in dv0.values())) if dv0 else 0
    mail_total = int(st.number_input("Total mail-in votes", min_value=0, value=int(baseline_total), step=1000))

    pct_defaults = {}
    if baseline_total > 0:
        pct_defaults = {p: 100.0 * float(dv0.get(p, 0)) / baseline_total for p in hm.MODEL_PARTIES}
    else:
        pct_defaults = {p: (100.0 if p == "FIDESZ" else 0.0) for p in hm.MODEL_PARTIES}

    pct_inputs: Dict[str, float] = {}
    cols = st.columns(2)
    for i, p in enumerate(hm.MODEL_PARTIES):
        with cols[i % 2]:
            pct_inputs[p] = float(
                st.number_input(
                    f"{p} mail vote %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(pct_defaults.get(p, 0.0)),
                    step=0.1,
                    format="%.1f",
                    key=f"mail_pct_{p}",
                )
            )

    pct_sum = float(sum(pct_inputs.values()))
    share_sum_widget("Mail vote % sum", pct_sum, target=100.0, tol=0.5)

diaspora_votes = compute_mail_votes(mail_total, pct_inputs)

# ============================================================
# Build cfg dict and run (reactive)
# ============================================================

cfg_dict = dict(
    exclude_budapest=bool(exclude_budapest),
    turnout_target=float(turnout_target),
    turnout_model=str(turnout_model),
    turnout_granularity=str(turnout_granularity),
    turnout_reference_election=str(turnout_reference_election),
    turnout_baseline_level=float(turnout_baseline_level),
    turnout_clip=(float(turnout_clip_lo), float(turnout_clip_hi)),
    reserve_adjust_power=float(reserve_adjust_power),
    marginal_concentration=float(marginal_concentration),

    poll_type=str(poll_type),
    last_n_days=int(last_n_days),
    pollster_filter=pollster_filter if pollster_filter else None,
    poll_decay_half_life_days=float(half_life),
    manual_poll_override=manual_poll_override,

    nonresponse_nonvoter_pct=float(nonresponse_nonvoter_pct),
    undecided_to_fidesz=float(uF),
    undecided_to_tisza=float(uT),

    election_weights=election_weights,
    geo_weights=geo_weights,
    undecided_elasticity_link_strength=float(undecided_elasticity_link_strength),

    use_mobilization_model=bool(use_mobilization_model),
    mobilization_all_parties=bool(mobilization_all_parties),
    mobilization_rates=mobilization_rates,
    reserve_strength=reserve_strength,

    vote_allocation_model=str(vote_allocation_model),
    undecided_geo_model=str(undecided_geo_model),
    undecided_local_lean_strength=float(undecided_local_lean_strength),

    smc_plus1_minus1=bool(smc_plus1_minus1),
    winner_two_party_only=bool(winner_two_party_only),

    diaspora_votes=diaspora_votes,
    nationality_seat_to_fidesz=bool(nationality_seat_to_fidesz),
    threshold=float(threshold),
    modelling_unit=str(modelling_unit),
)

cfg_json = json.dumps(cfg_dict, sort_keys=True)

with st.spinner("Running scenario…"):
    try:
        res = cached_run_scenario(
            db_dir_override,
            poll_dir_override,
            cfg_json,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()


# ============================================================
# Main
# ============================================================

st.title("Hungary Election Model — 2026")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Turnout target", f"{pct(res.cfg.turnout_target, 1):.1f}%")
m2.metric("FIDESZ seats", int(res.seats_table.loc["FIDESZ", "Total"]))
m3.metric("TISZA seats", int(res.seats_table.loc["TISZA", "Total"]))
m4.metric("MH seats", int(res.seats_table.loc["MH", "Total"]))
maj = "YES" if int(res.seats_table.loc["FIDESZ", "Total"]) >= 100 or int(res.seats_table.loc["TISZA", "Total"]) >= 100 else "NO"
m5.metric("Any single-party majority?", maj)

tabs = st.tabs(
    [
        "Seats",
        "National list",
        "Single constituencies",
        "Polls",
        "Coefficients",
        "Backtest 2022",
        "Monte Carlo",
        "Stealing analysis",
        "Methodology",
    ]
)

# ============================================================
# Tab: Seats
# ============================================================

with tabs[0]:
    st.subheader("Seat projection")
    st.dataframe(res.seats_table, use_container_width=True)

    focus = ["FIDESZ", "TISZA", "MH"]
    seat_focus = res.seats_table.loc[focus, ["SMC_seats", "List_seats", "Nationality", "Total"]].copy()
    long = seat_focus.reset_index().melt(id_vars="index", value_vars=["SMC_seats", "List_seats", "Nationality"], var_name="component", value_name="seats")
    long = long.rename(columns={"index": "party"})
    fig = px.bar(long, x="party", y="seats", color="component", barmode="stack", title="Seats (focus parties)")
    fig.add_hline(y=100, line_dash="dash")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="Seats")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Turnout diagnostics")
    turnout = res.turnout_rate_station.copy()
    st.caption("Station turnout rate distribution (predicted, after caps & rescaling).")

    figt = px.histogram(turnout.reset_index(), x="turnout_rate_2026", nbins=60, title="Station turnout rate distribution")
    figt.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Turnout rate", yaxis_title="Count")
    st.plotly_chart(figt, use_container_width=True)


# ============================================================
# Tab: National list
# ============================================================

with tabs[1]:
    st.subheader("National list votes and list seats")

    votes_df = pd.DataFrame(
        {
            "Domestic list": res.domestic_list_votes,
            "Loser comp": res.loser_comp,
            "Winner comp": res.winner_comp,
            "Mail/Diaspora": res.diaspora_votes,
        }
    )
    votes_df["Total list votes"] = votes_df.sum(axis=1)
    st.dataframe(votes_df.round(0).astype(int), use_container_width=True)

    fig_votes = px.bar(
        votes_df.reset_index(),
        x="index",
        y=["Domestic list", "Loser comp", "Winner comp", "Mail/Diaspora"],
        barmode="stack",
        title="List votes decomposition",
    )
    fig_votes.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Party", yaxis_title="Votes")
    st.plotly_chart(fig_votes, use_container_width=True)

    st.markdown("---")

    st.subheader("D'Hondt allocation (list seats)")
    st.caption(f"Eligible parties (threshold {pct(res.cfg.threshold, 1)}%): {', '.join(res.eligible_parties) if res.eligible_parties else '(none)'}")
    st.plotly_chart(build_dhondt_plot(res.dhondt_table, n_show=250), use_container_width=True)
    st.dataframe(res.dhondt_table.head(250), use_container_width=True)


# ============================================================
# Tab: Single constituencies
# ============================================================

with tabs[2]:
    st.subheader("Single constituencies (EVK)")

    def clean_evk_name(x: object) -> object:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return x
        s = str(x).strip()
        return re.sub(
            r"\s*(?:országgyűlési\s*)?(?:számú\s*)?egyéni\s+választókerület\s*$",
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()

    k = st.slider("Show closest EVKs (by absolute margin, votes)", min_value=5, max_value=50, value=15, step=1)

    winners = res.winners.copy()
    # winners returned by the model may already include evk_name; avoid duplicate join.
    if getattr(data, "evk_meta", None) is not None and not data.evk_meta.empty:
        _evk_names = join_evk_names(winners.index, data.evk_meta)
        winners = winners.drop(columns=[c for c in _evk_names.columns if c in winners.columns], errors="ignore").join(_evk_names)

    # Enhancements: margin as % of voters and party % columns (share of EVK SMC votes)
    evk_votes_raw = res.evk_smc_votes.copy()
    total_votes = evk_votes_raw.sum(axis=1).replace(0, np.nan)
    winners["total_votes"] = total_votes.reindex(winners.index).values
    winners["margin_pct_of_voters"] = (winners["margin_votes"] / winners["total_votes"].replace(0, np.nan) * 100.0).fillna(0.0)
    winners["FIDESZ_%"] = (evk_votes_raw["FIDESZ"] / total_votes * 100.0).reindex(winners.index).fillna(0.0)
    winners["TISZA_%"] = (evk_votes_raw["TISZA"] / total_votes * 100.0).reindex(winners.index).fillna(0.0)

    if "evk_name" in winners.columns:
        winners["evk_name"] = winners["evk_name"].map(clean_evk_name)

    closest = winners.sort_values("abs_margin_votes").head(int(k)).copy()
    closest_cols = [
        c
        for c in [
            "evk_name",
            "winner",
            "runner_up",
            "margin_votes",
            "margin_pct_of_voters",
            "FIDESZ_%",
            "TISZA_%",
            "total_votes",
        ]
        if c in closest.columns
    ]
    st.dataframe(closest[closest_cols], use_container_width=True)

    st.markdown("---")
    st.caption("EVK vote table (SMC)")
    evk_votes = res.evk_smc_votes.copy()
    if getattr(data, "evk_meta", None) is not None and not data.evk_meta.empty:
        m = join_evk_names(evk_votes.index, data.evk_meta)
        evk_votes = m.join(evk_votes.drop(columns=[c for c in m.columns if c in evk_votes.columns], errors="ignore"))

    # Add FIDESZ% and TISZA% for convenience.
    evk_votes_disp = evk_votes.copy()
    party_cols = [c for c in hm.MODEL_PARTIES if c in evk_votes_disp.columns]
    tot = evk_votes_disp[party_cols].sum(axis=1).replace(0, np.nan)
    if "FIDESZ" in evk_votes_disp.columns:
        evk_votes_disp["FIDESZ_%"] = (evk_votes_disp["FIDESZ"] / tot * 100.0).fillna(0.0)
    if "TISZA" in evk_votes_disp.columns:
        evk_votes_disp["TISZA_%"] = (evk_votes_disp["TISZA"] / tot * 100.0).fillna(0.0)
    if "evk_name" in evk_votes_disp.columns:
        evk_votes_disp["evk_name"] = evk_votes_disp["evk_name"].map(clean_evk_name)

    # Keep evk_name as text; cast only party vote columns to int for display.
    for c in party_cols:
        evk_votes_disp[c] = evk_votes_disp[c].round(0).astype(int)
    if "FIDESZ_%" in evk_votes_disp.columns:
        evk_votes_disp["FIDESZ_%"] = evk_votes_disp["FIDESZ_%"].round(2)
    if "TISZA_%" in evk_votes_disp.columns:
        evk_votes_disp["TISZA_%"] = evk_votes_disp["TISZA_%"].round(2)
    st.dataframe(evk_votes_disp, use_container_width=True)


# ============================================================
# Tab: Polls
# ============================================================

with tabs[3]:
    st.subheader("Poll inputs, population shares, and implied voter shares")

    c1, c2, c3 = st.columns(3)

    avg_poll = res.avg_poll_2026.copy().reindex(["FIDESZ", "TISZA", "DK", "MH", "MKKP", "OTHER", "UNDECIDED"])
    c1.dataframe(avg_poll.to_frame("avg_poll_pp").round(2), use_container_width=True)

    pop = res.population_shares_2026.copy().reindex(["FIDESZ", "TISZA", "DK", "MH", "MKKP", "OTHER", "UNDECIDED_TRUE"])
    c2.dataframe((pop * 100.0).round(2).to_frame("population_share_%"), use_container_width=True)

    c3.dataframe((res.national_shares_2026 * 100.0).round(2).to_frame("voter_share_%"), use_container_width=True)

    st.markdown("---")
    st.subheader("Pollster performance (backtests)")
    st.caption("Bias is shown in pp (predicted - actual). Overall error uses Fidesz+Challenger only; UNDECIDED is also evaluated (raw mode).")

    bias_cols = st.columns([2, 2, 1, 1])
    with bias_cols[0]:
        bias_years = st.multiselect("Years", options=[2018, 2019, 2022, 2024], default=[2022, 2024])
    with bias_cols[1]:
        bias_poll_type = st.selectbox("Poll type", options=sorted(data.polls_all["poll_type"].dropna().unique().tolist()), index=0)
    with bias_cols[2]:
        bias_window = st.number_input("Last N days", min_value=7, max_value=180, value=60, step=1)
    with bias_cols[3]:
        agg_level = st.selectbox("Aggregate", options=["none", "pollster", "year", "overall"], index=1)

    if bias_years:
        panel = hm.pollster_bias_panel(data, years=[int(y) for y in bias_years], poll_type=str(bias_poll_type), last_n_days=int(bias_window))
        if panel is None or panel.empty:
            st.info("No bias table available for the selected settings (no polls in window).")

        else:
            if agg_level == "none":
                st.dataframe(panel, use_container_width=True)
            else:
                agg = hm.aggregate_pollster_bias(panel, by=str(agg_level))
                st.dataframe(agg, use_container_width=True)

                if agg_level in ["pollster", "year"] and "RMSE_pp" in agg.columns:
                    fig = px.bar(agg, x=agg_level, y="RMSE_pp", title=f"RMSE (pp) — aggregated by {agg_level}")
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="RMSE (pp)")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one year.")


# ============================================================
# Tab: Coefficients
# ============================================================

with tabs[4]:
    st.subheader("Coefficients diagnostics (2026)")

    coef = res.coefs_2026.copy()
    st.caption("Coefficients are multiplicative geography factors in party space (mapped from BLOCK coefficients).")

    stats = pd.DataFrame(
        {
            "mean": coef.mean(),
            "median": coef.median(),
            "p10": coef.quantile(0.10),
            "p90": coef.quantile(0.90),
        }
    ).round(3)
    st.dataframe(stats, use_container_width=True)

    party = st.selectbox("Show coefficient distribution for party", options=hm.MODEL_PARTIES, index=hm.MODEL_PARTIES.index("FIDESZ"))
    fig = px.histogram(coef.reset_index(), x=party, nbins=60, title=f"{party} coefficient distribution")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Coefficient", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Tab: Monte Carlo
# ============================================================

with tabs[5]:
    st.subheader("Backtest 2022 (dirty)")
    st.caption(
        "Diagnostic only. 2022 station results are mapped into 2026 EVKs (via the 2026 station→EVK mapping). "
        "OPP is approximated as TISZA (DK=0). This is intentionally 'dirty' but useful to gauge how realistic the "
        "current coefficient settings are."
    )

    try:
        bt = hm.backtest_2022_dirty(data, res.cfg)
    except Exception as e:
        bt = None
        st.error(str(e))

    if bt is not None and bt.n_evks > 0 and bt.table is not None and not bt.table.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EVKs evaluated", int(bt.n_evks))
        c2.metric("MAE FIDESZ (pp)", f"{bt.mae_fidesz_pp:.2f}")
        c3.metric("MAE TISZA (pp)", f"{bt.mae_tisza_pp:.2f}")
        c4.metric("Winner accuracy (F vs T)", f"{100.0 * bt.winner_accuracy_ft:.1f}%")

        table = bt.table.copy()
        if "evk_name" in table.columns:
            table["evk_name"] = table["evk_name"].map(
                lambda x: re.sub(
                    r"\s*(?:országgyűlési\s*)?(?:számú\s*)?egyéni\s+választókerület\s*$",
                    "",
                    str(x).strip(),
                    flags=re.IGNORECASE,
                )
                if pd.notna(x)
                else x
            )
        st.dataframe(table, use_container_width=True)

        st.markdown('---')
        st.subheader('2022 national vote → seats (under current model rules)')
        st.caption(
            'This runs the seat rules with the *same* national 2022 totals, but geography comes from the current coefficient settings (via IPF). ' 
            'List seats can change via compensation (SMC winners), and SMC seats change directly via predicted winners.'
        )

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown('**Actual (mapped party-space) seats**')
            st.dataframe(bt.seats_actual.reset_index().rename(columns={'index': 'party'}), use_container_width=True)
        with s2:
            st.markdown('**Predicted seats (current parameters)**')
            st.dataframe(bt.seats_pred.reset_index().rename(columns={'index': 'party'}), use_container_width=True)
        with s3:
            st.markdown('**Difference (Pred − Actual)**')
            st.dataframe(bt.seats_diff.reset_index().rename(columns={'index': 'party'}), use_container_width=True)
    else:
        st.info("Backtest not available (no EVK-level 2022 data after mapping, or an error occurred).")


with tabs[6]:
    st.subheader("Monte Carlo")
    st.caption(
        "Runs many draws of the full scenario pipeline with uncertainty on turnout, polls, coefficients and mail votes. "
        "Monte Carlo is executed at EVK level (106 constituencies) for speed."
    )

    # -------------------------
    # Helpers
    # -------------------------
    def _dist_spec_widget(label: str, *, default: float, min_hard: float, max_hard: float, key: str, unit: str = ""):
        """Return (spec_dict, display_desc). Values are in native units."""

        dist = st.selectbox(
            f"{label} distribution",
            ["Fixed", "Uniform (min-max)", "Normal (mean, sd)"],
            key=f"{key}_dist",
        )

        if dist == "Fixed":
            v = st.number_input(
                f"{label}{unit}",
                value=float(default),
                min_value=float(min_hard),
                max_value=float(max_hard),
                key=f"{key}_fixed",
            )
            return {"dist": "fixed", "value": float(v)}, f"fixed={v}{unit}"

        if dist == "Uniform (min-max)":
            c1, c2 = st.columns(2)
            lo = c1.number_input(
                f"{label} min{unit}",
                value=float(default),
                min_value=float(min_hard),
                max_value=float(max_hard),
                key=f"{key}_min",
            )
            hi = c2.number_input(
                f"{label} max{unit}",
                value=float(default),
                min_value=float(min_hard),
                max_value=float(max_hard),
                key=f"{key}_max",
            )
            return {"dist": "uniform", "min": float(lo), "max": float(hi)}, f"uniform=[{lo},{hi}]{unit}"

        # Normal
        c1, c2 = st.columns(2)
        mu = c1.number_input(
            f"{label} mean{unit}",
            value=float(default),
            min_value=float(min_hard),
            max_value=float(max_hard),
            key=f"{key}_mean",
        )
        sd = c2.number_input(
            f"{label} sd{unit}",
            value=float((max_hard - min_hard) * 0.02),
            min_value=0.0,
            key=f"{key}_sd",
        )
        return {"dist": "normal", "mean": float(mu), "sd": float(sd), "min": float(min_hard), "max": float(max_hard)}, f"normal=({mu}±{sd}){unit}"

    # -------------------------
    # Core settings
    # -------------------------

    c1, c2, c3, c4 = st.columns(4)
    n_sims = c1.number_input("Simulations", min_value=1_000, max_value=500_000, value=25_000, step=5_000)
    seed = c2.number_input("Random seed", min_value=0, max_value=10_000_000, value=1234, step=1)
    n_workers = c3.number_input("Workers", min_value=1, max_value=32, value=min(4, os.cpu_count() or 4), step=1)
    chunk_size = c4.number_input("Chunk size", min_value=200, max_value=10_000, value=2_000, step=200)

    backend = st.selectbox("Parallel backend", ["processes", "threads"], index=1)
    st.caption("Tip: on Streamlit Community Cloud, **threads** is usually more stable than **processes** (no big pickles / less RAM duplication).")

    st.divider()

    # -------------------------
    # Turnout
    # -------------------------

    st.markdown("### Turnout")

    # Turnout target is stored as fraction (0..1), but we show % in UI.
    turnout_default_pct = float(res.cfg.turnout_target * 100.0)
    spec_turnout_pct, _ = _dist_spec_widget(
        "Turnout target",
        default=turnout_default_pct,
        min_hard=20.0,
        max_hard=95.0,
        key="mc_turnout",
        unit="%",
    )
    turnout_target_spec = {
        **spec_turnout_pct,
        "value": float(spec_turnout_pct.get("value", turnout_default_pct)) / 100.0,
        "min": float(spec_turnout_pct.get("min", 20.0)) / 100.0,
        "max": float(spec_turnout_pct.get("max", 95.0)) / 100.0,
    }
    if spec_turnout_pct.get("dist") == "uniform":
        turnout_target_spec["min"] = float(spec_turnout_pct["min"]) / 100.0
        turnout_target_spec["max"] = float(spec_turnout_pct["max"]) / 100.0
        turnout_target_spec.pop("value", None)
    if spec_turnout_pct.get("dist") == "normal":
        turnout_target_spec["mean"] = float(spec_turnout_pct["mean"]) / 100.0
        turnout_target_spec["sd"] = float(spec_turnout_pct["sd"]) / 100.0

    st.caption("Each draw samples a national turnout target and then applies a turnout model to distribute votes geographically.")

    turnout_models = st.multiselect(
        "Turnout model(s) to sample",
        hm.TURNOUT_MODELS,
        default=[res.cfg.turnout_model],
    )

    turnout_grans = st.multiselect(
        "Turnout granularity(ies) to sample",
        list(hm.TURNOUT_GRANULARITIES.keys()),
        default=[res.cfg.turnout_granularity],
    )

    st.divider()

    # -------------------------
    # Polls
    # -------------------------

    st.markdown("### Polls")

    poll_mode_ui = st.selectbox(
        "Poll mode",
        [
            "Use scenario setting",
            "Decided (no undecided / no nonresponse)",
            "Raw (includes undecided)",
        ],
        index=0,
    )
    poll_mode = {"Use scenario setting": "scenario", "Decided (no undecided / no nonresponse)": "decided", "Raw (includes undecided)": "raw"}[poll_mode_ui]

    sampling_ui = st.selectbox(
        "Poll sampling method",
        ["Multinomial (n respondents)", "Dirichlet-multinomial (extra variance)", "Gaussian noise (pp)"],
        index=0,
        help=(
            "Multinomial: draws a poll outcome from n respondents. "
            "Dirichlet-multinomial: adds extra dispersion for house effects / correlated error. "
            "Gaussian: adds independent Normal noise in percentage points then renormalizes."
        ),
    )

    if sampling_ui.startswith("Gaussian"):
        national_share_sampling = "gaussian"
        nat_sigma_pp = st.number_input("Gaussian sigma (pp)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        poll_n = st.number_input("Poll n (unused for Gaussian)", min_value=200, max_value=50_000, value=1_200, step=100)
        dirichlet_conc = 200.0
    elif sampling_ui.startswith("Dirichlet"):
        national_share_sampling = "dirichlet_multinomial"
        poll_n = st.number_input("Poll n", min_value=200, max_value=50_000, value=1_200, step=100)
        dirichlet_conc = st.number_input(
            "Dirichlet concentration",
            min_value=10.0,
            max_value=2_000.0,
            value=200.0,
            step=10.0,
            help="Higher = closer to multinomial. Lower = more variance (house effects / correlated error).",
        )
        nat_sigma_pp = 1.0
    else:
        national_share_sampling = "multinomial"
        poll_n = st.number_input("Poll n", min_value=200, max_value=50_000, value=1_200, step=100)
        dirichlet_conc = 200.0
        nat_sigma_pp = 1.0

    st.divider()

    # -------------------------
    # Raw poll -> voter conversion
    # -------------------------

    st.markdown("### Undecided & nonresponse (raw polls)")

    if poll_mode == "raw" or (poll_mode == "scenario" and str(res.cfg.poll_type).lower().startswith("raw")):
        nonresp_default = float(res.cfg.nonresponse_nonvoter_pct * 100.0)
        spec_nonresp_pct, _ = _dist_spec_widget(
            "Not in polls & not voting",
            default=nonresp_default,
            min_hard=0.0,
            max_hard=50.0,
            key="mc_nonresp",
            unit="%",
        )
        nonresponse_nonvoter_spec = {
            **spec_nonresp_pct,
            "value": float(spec_nonresp_pct.get("value", nonresp_default)) / 100.0,
            "min": float(spec_nonresp_pct.get("min", 0.0)) / 100.0,
            "max": float(spec_nonresp_pct.get("max", 50.0)) / 100.0,
        }
        if spec_nonresp_pct.get("dist") == "uniform":
            nonresponse_nonvoter_spec["min"] = float(spec_nonresp_pct["min"]) / 100.0
            nonresponse_nonvoter_spec["max"] = float(spec_nonresp_pct["max"]) / 100.0
            nonresponse_nonvoter_spec.pop("value", None)
        if spec_nonresp_pct.get("dist") == "normal":
            nonresponse_nonvoter_spec["mean"] = float(spec_nonresp_pct["mean"]) / 100.0
            nonresponse_nonvoter_spec["sd"] = float(spec_nonresp_pct["sd"]) / 100.0

        split_default = float(res.cfg.undecided_to_fidesz)
        undecided_to_fidesz_spec, _ = _dist_spec_widget(
            "Undecided -> FIDESZ share",
            default=split_default,
            min_hard=0.0,
            max_hard=1.0,
            key="mc_und_split",
        )

        # Mobilization/reserve tilt
        mob_ui = st.selectbox(
            "Mobilization / reserve tilt",
            ["Use scenario setting", "Always ON", "Always OFF", "Random 50/50"],
            index=0,
        )
        if mob_ui == "Use scenario setting":
            use_mobilization_choices = None
        elif mob_ui == "Always ON":
            use_mobilization_choices = [True]
        elif mob_ui == "Always OFF":
            use_mobilization_choices = [False]
        else:
            use_mobilization_choices = [True, False]

        st.markdown("#### Mobilization rates & reserve strengths")
        st.caption("Only used when poll mode is raw and turnout exceeds the decided share.")

        mob_rates0 = getattr(res.cfg, "mobilization_rates", None) or {}
        res_strength0 = getattr(res.cfg, "reserve_strength", None) or {}

        mob_f0 = float(mob_rates0.get("FIDESZ", 0.70))
        mob_t0 = float(mob_rates0.get("TISZA", 0.90))
        res_f0 = float(res_strength0.get("FIDESZ", 1.0))
        res_t0 = float(res_strength0.get("TISZA", 1.0))

        mob_f_spec, _ = _dist_spec_widget(
            "Mobilization rate (FIDESZ)",
            default=mob_f0,
            min_hard=0.0,
            max_hard=1.0,
            key="mc_mob_f",
        )
        mob_t_spec, _ = _dist_spec_widget(
            "Mobilization rate (TISZA)",
            default=mob_t0,
            min_hard=0.0,
            max_hard=1.0,
            key="mc_mob_t",
        )
        res_f_spec, _ = _dist_spec_widget(
            "Reserve strength (FIDESZ)",
            default=res_f0,
            min_hard=0.0,
            max_hard=1.0,
            key="mc_res_f",
        )
        res_t_spec, _ = _dist_spec_widget(
            "Reserve strength (TISZA)",
            default=res_t0,
            min_hard=0.0,
            max_hard=1.0,
            key="mc_res_t",
        )

    else:
        st.info("Raw-poll conversion parameters are inactive unless poll mode is Raw.")
        nonresponse_nonvoter_spec = None
        undecided_to_fidesz_spec = None
        use_mobilization_choices = None
        mob_f_spec = None
        mob_t_spec = None
        res_f_spec = None
        res_t_spec = None

    st.divider()

    # -------------------------
    # Coefficients: uncertainty
    # -------------------------

    st.markdown("### Coefficients")

    coef_log_sigma = st.number_input(
        "Coefficient log-sigma",
        min_value=0.0,
        max_value=1.0,
        value=0.08,
        step=0.01,
        help="Multiplicative coefficient noise per draw (log-normal).",
    )

    with st.expander("Simulate election weights", expanded=False):
        do_elec_w = st.checkbox("Enable election-weight uncertainty", value=False, key="mc_elec_w_on")
        election_weight_minmax = None
        if do_elec_w:
            election_weight_minmax = {}
            for k, v in res.cfg.election_weights.items():
                lo, hi = st.slider(
                    f"{k} weight range",
                    min_value=0.0,
                    max_value=1.0,
                    value=(max(0.0, float(v) - 0.10), min(1.0, float(v) + 0.10)),
                    step=0.01,
                    key=f"mc_ew_{k}",
                )
                election_weight_minmax[k] = [float(lo), float(hi)]

    with st.expander("Simulate geography weights", expanded=False):
        do_geo_w = st.checkbox("Enable geography-weight uncertainty", value=False, key="mc_geo_w_on")
        geo_weight_minmax = None
        if do_geo_w:
            geo_weight_minmax = {}
            for k, v in res.cfg.geo_weights.items():
                lo, hi = st.slider(
                    f"{k} weight range",
                    min_value=0.0,
                    max_value=1.0,
                    value=(max(0.0, float(v) - 0.10), min(1.0, float(v) + 0.10)),
                    step=0.01,
                    key=f"mc_gw_{k}",
                )
                geo_weight_minmax[k] = [float(lo), float(hi)]

    st.divider()

    # -------------------------
    # Diaspora / mail votes
    # -------------------------

    st.markdown("### Mail-in votes")
    st.caption("Mail votes affect list seats only. In this model we assume mail votes are FIDESZ vs TISZA (the rest -> TISZA).")

    simulate_mail = st.checkbox("Simulate mail votes", value=False)

    diaspora_total_spec = None
    diaspora_fidesz_share_spec = None

    if simulate_mail:
        # Suggested defaults roughly aligned with 2018 and 2022 mail voting volumes
        tot_default = float(res.diaspora_votes.sum())
        tot_min = 225_000.0
        tot_max = 264_000.0

        spec_tot, _ = _dist_spec_widget(
            "Mail vote total",
            default=float(tot_default),
            min_hard=0.0,
            max_hard=600_000.0,
            key="mc_mail_total",
        )
        diaspora_total_spec = spec_tot

        spec_share, _ = _dist_spec_widget(
            "Mail vote FIDESZ share",
            default=0.95,
            min_hard=0.0,
            max_hard=1.0,
            key="mc_mail_share",
        )
        diaspora_fidesz_share_spec = spec_share

        st.caption(f"Suggested total range from 2018–2022: ~{int(tot_min):,}–{int(tot_max):,} votes.")

    diaspora_log_sigma = st.number_input(
        "Legacy diaspora log-sigma (ignored if simulating total+share)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
    )

    st.divider()

    # -------------------------
    # IPF controls
    # -------------------------

    st.markdown("### IPF")
    c1, c2 = st.columns(2)
    ipf_max_iter = c1.number_input("IPF max iter", min_value=50, max_value=1_000, value=250, step=25)
    ipf_tol = c2.number_input("IPF tolerance", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.1e")

    st.divider()

    # -------------------------
    # Build MC config
    # -------------------------

    mc_cfg = {
        "n_sims": int(n_sims),
        "seed": int(seed),
        "n_workers": int(n_workers),
        "backend": str(backend),
        "chunk_size": int(chunk_size),
        "force_modelling_unit": "evk",
        "turnout_target_spec": turnout_target_spec,
        "turnout_models": turnout_models or None,
        "turnout_granularities": turnout_grans or None,
        "poll_mode": poll_mode,
        "national_share_sampling": national_share_sampling,
        "nat_sigma_pp": float(nat_sigma_pp),
        "poll_n": int(poll_n),
        "dirichlet_concentration": float(dirichlet_conc),
        "nonresponse_nonvoter_spec": nonresponse_nonvoter_spec,
        "undecided_to_fidesz_spec": undecided_to_fidesz_spec,
        "use_mobilization_choices": use_mobilization_choices,
        "mobilization_rate_fidesz_spec": mob_f_spec,
        "mobilization_rate_tisza_spec": mob_t_spec,
        "reserve_strength_fidesz_spec": res_f_spec,
        "reserve_strength_tisza_spec": res_t_spec,
        "coef_log_sigma": float(coef_log_sigma),
        "election_weight_minmax": election_weight_minmax,
        "geo_weight_minmax": geo_weight_minmax,
        "diaspora_total_spec": diaspora_total_spec,
        "diaspora_fidesz_share_spec": diaspora_fidesz_share_spec,
        "diaspora_log_sigma": float(diaspora_log_sigma),
        "ipf_max_iter": int(ipf_max_iter),
        "ipf_tol": float(ipf_tol),
    }

    run_button = st.button("Run Monte Carlo")

    if run_button:
        progress = st.progress(0.0)

        def _cb(done: int, total: int):
            progress.progress(min(1.0, done / max(1, total)))

        mc_res = hm.run_monte_carlo(data, res, hm.MonteCarloConfig(**mc_cfg), progress_cb=_cb)
        progress.empty()

        # Persist across reruns so diagnostics remain visible when interacting with widgets.
        st.session_state["mc_res"] = mc_res

    mc_res = st.session_state.get("mc_res", None)

    if mc_res is not None:
        # -------------------------
        # Doom probability KPIs
        # -------------------------
        st.markdown("### Doom scenario probabilities")
        k1, k2, k3 = st.columns(3)
        k1.metric("P(FIDESZ ≥ 100 seats)", f"{mc_res.doom_prob_fidesz_majority * 100.0:.2f}%")
        k2.metric("P(FIDESZ + MH ≥ 100 seats)", f"{mc_res.doom_prob_fidesz_mh_majority * 100.0:.2f}%")
        k3.metric("P(Doom: either)", f"{mc_res.doom_prob_any * 100.0:.2f}%")

        # -------------------------
        # Summary tables
        # -------------------------
        with st.expander("Summary tables", expanded=False):
            st.markdown("#### Seat distribution summary")
            st.dataframe(mc_res.seat_summary, use_container_width=True)

            st.markdown("#### Majority probabilities (≥ 100 seats)")
            st.dataframe((mc_res.prob_majority * 100.0).round(2).rename("%"), use_container_width=True)

            st.markdown("#### National vote share (domestic) distribution")
            st.dataframe(
                pd.DataFrame(
                    {
                        "mean": mc_res.nat_share_draws.mean(),
                        "p05": mc_res.nat_share_draws.quantile(0.05),
                        "p95": mc_res.nat_share_draws.quantile(0.95),
                    }
                ).mul(100.0).round(2),
                use_container_width=True,
            )

        # -------------------------
        # Combined per-draw table (inputs + outcomes)
        # -------------------------
        draws = pd.concat(
            [
                mc_res.input_draws.reset_index(drop=True),
                mc_res.seat_draws.add_prefix("seats_").reset_index(drop=True),
                mc_res.nat_share_draws.add_prefix("nat_realized_").reset_index(drop=True),
            ],
            axis=1,
        )

        if "seats_FIDESZ" in draws.columns and "seats_MH" in draws.columns:
            draws["seats_FIDESZ_MH"] = draws["seats_FIDESZ"] + draws["seats_MH"]
        else:
            draws["seats_FIDESZ_MH"] = np.nan

        draws["doom_fidesz_majority"] = draws["seats_FIDESZ"] >= 100
        draws["doom_fidesz_mh_majority"] = draws["seats_FIDESZ_MH"] >= 100
        draws["doom_any"] = draws["doom_fidesz_majority"] | draws["doom_fidesz_mh_majority"]

        # -------------------------
        # Diagnostics tabs
        # -------------------------
        st.markdown("### Monte Carlo diagnostics")
        diag_tabs = st.tabs(
            [
                "Outcome histograms",
                "Input sensitivity",
                "Doom scenario assumptions",
                "Input connectivity",
                "Download",
            ]
        )

        # ------------------------------------------------------------
        # Tab: Outcome histograms
        # ------------------------------------------------------------
        with diag_tabs[0]:
            c1, c2 = st.columns(2)

            with c1:
                fig = px.histogram(draws, x="seats_FIDESZ", nbins=60, title="Seats: FIDESZ")
                fig.add_vline(x=100, line_dash="dash")
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Seats", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.histogram(draws, x="seats_FIDESZ_MH", nbins=60, title="Seats: FIDESZ + MH")
                fig.add_vline(x=100, line_dash="dash")
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Seats", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

            if "seats_TISZA" in draws.columns and "seats_FIDESZ" in draws.columns:
                winner = np.where(
                    draws["seats_FIDESZ"] > draws["seats_TISZA"],
                    "FIDESZ",
                    np.where(draws["seats_TISZA"] > draws["seats_FIDESZ"], "TISZA", "Tie"),
                )
                win_tab = pd.Series(winner, name="winner").value_counts().rename_axis("winner").reset_index(name="count")
                figw = px.bar(win_tab, x="winner", y="count", title="Winner frequency (by total seats)")
                figw.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="Draws")
                st.plotly_chart(figw, use_container_width=True)

        # ------------------------------------------------------------
        # Tab: Input sensitivity
        # ------------------------------------------------------------
        with diag_tabs[1]:
            num_inputs = list(mc_res.input_draws.select_dtypes(include=[np.number]).columns)

            if not num_inputs:
                st.info("No numeric inputs captured for sensitivity analysis.")
            else:
                sel = st.selectbox("Numeric input", num_inputs, index=0, key="mc_sens_input")
                n_bins = st.slider("Quantile bins", min_value=4, max_value=20, value=10, step=1, key="mc_sens_bins")

                d = draws[
                    [
                        sel,
                        "doom_any",
                        "doom_fidesz_majority",
                        "doom_fidesz_mh_majority",
                        "seats_FIDESZ",
                        "seats_FIDESZ_MH",
                    ]
                ].copy()
                d = d.dropna(subset=[sel])

                if d.empty or d[sel].nunique() < 2:
                    st.warning("Selected input has too little variation (or too many missing values) for sensitivity analysis.")
                else:
                    q = int(min(n_bins, max(2, int(d[sel].nunique()))))
                    try:
                        d["bin"] = pd.qcut(d[sel], q=q, duplicates="drop")
                    except Exception:
                        d["bin"] = pd.cut(d[sel], bins=q)

                    g = d.groupby("bin", observed=True)
                    sens = pd.DataFrame(
                        {
                            "n": g.size(),
                            "x_min": g[sel].min(),
                            "x_max": g[sel].max(),
                            "x_mean": g[sel].mean(),
                            "P(Doom)": g["doom_any"].mean(),
                            "P(FIDESZ ≥ 100)": g["doom_fidesz_majority"].mean(),
                            "P(FIDESZ+MH ≥ 100)": g["doom_fidesz_mh_majority"].mean(),
                            "Mean seats (FIDESZ)": g["seats_FIDESZ"].mean(),
                            "Mean seats (FIDESZ+MH)": g["seats_FIDESZ_MH"].mean(),
                        }
                    ).reset_index(drop=True)

                    st.dataframe(
                        sens.assign(
                            **{
                                "P(Doom)": (sens["P(Doom)"] * 100.0).round(2),
                                "P(FIDESZ ≥ 100)": (sens["P(FIDESZ ≥ 100)"] * 100.0).round(2),
                                "P(FIDESZ+MH ≥ 100)": (sens["P(FIDESZ+MH ≥ 100)"] * 100.0).round(2),
                            }
                        ).round(4),
                        use_container_width=True,
                    )

                    sens_sorted = sens.sort_values("x_mean")

                    figp = px.line(
                        sens_sorted,
                        x="x_mean",
                        y=["P(Doom)", "P(FIDESZ ≥ 100)", "P(FIDESZ+MH ≥ 100)"],
                        title="Doom probabilities vs input (binned)",
                    )
                    figp.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title=sel, yaxis_title="Probability")
                    st.plotly_chart(figp, use_container_width=True)

                    figs = px.line(
                        sens_sorted,
                        x="x_mean",
                        y=["Mean seats (FIDESZ)", "Mean seats (FIDESZ+MH)"],
                        title="Mean seats vs input (binned)",
                    )
                    figs.add_hline(y=100, line_dash="dash")
                    figs.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title=sel, yaxis_title="Seats")
                    st.plotly_chart(figs, use_container_width=True)

                    show_scatter = st.checkbox("Show scatter (downsampled)", value=False, key="mc_sens_scatter")
                    if show_scatter:
                        max_pts = 5_000
                        dd = d.sample(n=min(max_pts, len(d)), random_state=1) if len(d) > max_pts else d
                        figsc = px.scatter(dd, x=sel, y="seats_FIDESZ", opacity=0.6, title="Input vs seats_FIDESZ (downsampled)")
                        figsc.add_hline(y=100, line_dash="dash")
                        figsc.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title=sel, yaxis_title="Seats (FIDESZ)")
                        st.plotly_chart(figsc, use_container_width=True)

        # ------------------------------------------------------------
        # Tab: Doom scenario assumptions
        # ------------------------------------------------------------
        with diag_tabs[2]:
            doom_mask = draws["doom_any"].astype(bool)
            n_doom = int(doom_mask.sum())
            n_total = int(len(draws))
            st.caption(f"Doom draws: {n_doom:,} / {n_total:,} ({(n_doom / n_total * 100.0) if n_total else 0.0:.2f}%)")

            num_cols = list(mc_res.input_draws.select_dtypes(include=[np.number]).columns)

            rows = []
            for col in num_cols:
                a = draws.loc[doom_mask, col].dropna()
                b = draws.loc[~doom_mask, col].dropna()
                if len(a) < 10 or len(b) < 10:
                    continue
                m1, m0 = float(a.mean()), float(b.mean())
                s1, s0 = float(a.std(ddof=0)), float(b.std(ddof=0))
                pooled = float(np.sqrt((s1 * s1 + s0 * s0) / 2.0))
                smd = (m1 - m0) / pooled if pooled > 0 else 0.0
                rows.append(
                    {
                        "variable": col,
                        "doom_mean": m1,
                        "non_doom_mean": m0,
                        "doom_sd": s1,
                        "non_doom_sd": s0,
                        "smd": smd,
                        "abs_smd": abs(smd),
                        "doom_n": int(len(a)),
                        "non_doom_n": int(len(b)),
                    }
                )

            if rows:
                drivers = pd.DataFrame(rows).sort_values("abs_smd", ascending=False)
                st.markdown("#### Driver table (numeric inputs, ranked by |SMD|)")
                st.dataframe(drivers.drop(columns=["abs_smd"]).head(40), use_container_width=True)
            else:
                drivers = pd.DataFrame()
                st.info("Not enough numeric inputs (or not enough doom/non-doom samples) to compute a driver table.")

            st.markdown("#### Categorical breakdowns")
            cat_cols = [c for c in ["turnout_model", "turnout_granularity", "poll_type", "use_mobilization"] if c in mc_res.input_draws.columns]
            if cat_cols:
                cat_sel = st.selectbox("Categorical variable", cat_cols, index=0, key="mc_cat_var")
                dd = draws[[cat_sel, "doom_any"]].copy()
                dd[cat_sel] = dd[cat_sel].astype("object").where(dd[cat_sel].notna(), other="NA")
                tab = (
                    dd.groupby(cat_sel, observed=True)["doom_any"]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "P(Doom)", "count": "n"})
                    .sort_values("P(Doom)", ascending=False)
                )
                tab["P(Doom)"] = (tab["P(Doom)"] * 100.0).round(2)
                st.dataframe(tab, use_container_width=True)
            else:
                st.info("No categorical inputs available for breakdowns.")

            st.markdown("#### Conditional histograms (doom vs non-doom)")
            if not drivers.empty:
                top_defaults = drivers["variable"].head(8).tolist()
                var = st.selectbox(
                    "Numeric variable",
                    top_defaults + [c for c in num_cols if c not in top_defaults],
                    index=0,
                    key="mc_cond_hist_var",
                )
                ddh = draws[[var, "doom_any"]].dropna(subset=[var]).copy()
                ddh["doom_any"] = ddh["doom_any"].map({True: "Doom", False: "Non-doom"})
                figh = px.histogram(ddh, x=var, color="doom_any", nbins=60, barmode="overlay", opacity=0.6, title=f"{var}: doom vs non-doom")
                figh.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title=var, yaxis_title="Count")
                st.plotly_chart(figh, use_container_width=True)

        # ------------------------------------------------------------
        # Tab: Input connectivity
        # ------------------------------------------------------------
        with diag_tabs[3]:
            num_cols = list(mc_res.input_draws.select_dtypes(include=[np.number]).columns)
            default_cols = [
                c
                for c in [
                    "turnout_target",
                    "nat_target_FIDESZ",
                    "nat_target_TISZA",
                    "poll_undecided_share",
                    "undecided_true_share",
                    "undecided_to_fidesz",
                    "diaspora_total",
                    "diaspora_fidesz_share",
                ]
                if c in num_cols
            ]
            sel_cols = st.multiselect("Numeric inputs", num_cols, default=default_cols or num_cols[:8], key="mc_corr_cols")
            if len(sel_cols) < 2:
                st.info("Select at least 2 numeric inputs to compute correlations.")
            else:
                corr = draws[sel_cols].corr()
                figc = px.imshow(corr, text_auto=".2f", aspect="auto", title="Input connectivity (correlation heatmap)")
                figc.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(figc, use_container_width=True)

        # ------------------------------------------------------------
        # Tab: Download
        # ------------------------------------------------------------
        with diag_tabs[4]:
            st.caption("Download per-draw inputs + outcomes for offline analysis (CSV).")
            with st.expander("Download combined draws table", expanded=False):
                csv_bytes = draws.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="monte_carlo_draws.csv",
                    mime="text/csv",
                )




with tabs[7]:
    render_stealing_analysis_tab()


with tabs[8]:
    st.subheader("Methodology")

    st.markdown(
        """
## Pipeline (single run)

### 1) Inputs

Key scenario inputs (sidebar) feed into three parts of the model:

- **National vote shares** (column totals): from polls (decided or raw→voter conversion).
- **Turnout totals by geography** (row totals): from the turnout model + turnout target.
- **Geographic voting patterns** (priors): from historical coefficients blended by election and geography weights.

### 2) Turnout: national target → EVK totals

A **turnout target** (e.g. 72%) sets the total number of domestic votes:

- **Total domestic votes** = `turnout_target × total_registered_2026`.

A **turnout model** then distributes those votes across geography.


#### Turnout models (detailed)

All turnout models produce a station-level vote total `votes_i` such that:

- `votes_i = turnout_i × registered_i`
- `Σ_i votes_i = turnout_target × Σ_i registered_i` (exactly, after the final rescale step)
- `turnout_i` is clipped to `[turnout_clip_low, turnout_clip_high]`

Notation used below:

- `i` = polling station (2026 station universe)
- `g` = group at the selected granularity (station / location / settlement / evk)
- `registered_i` = 2026 registered voters at station `i`
- `T` = `turnout_target` (national turnout rate)
- `t_ref(i)` = turnout rate at station `i` in the selected **reference election** (aligned to 2026 station IDs via settlement fallback)
- `logit(x) = log(x/(1-x))`, `sigmoid(z)=1/(1+e^{-z})`

A final common post-processing step is applied to every model:

- Clip all station turnout rates to `[clip_lo, clip_hi]`
- Rescale all station vote totals by a common factor so that the national total matches `T × total_registered`
- Repeat a few iterations until both constraints are satisfied

Below is what each model specifically does before that final enforcement step.

---

##### 1) `uniform`

Same turnout rate everywhere.

- `turnout_i = T`
- `votes_i = T × registered_i`

Use when you want *no turnout geography*.

---

##### 2) `scaled_reference`

Preserves the **shape** of a reference turnout map (e.g., 2022 list), scaled to match the target national turnout.

1) Start from station baseline votes: `base_votes_i = t_ref(i) × registered_i`
2) Compute scaling factor: `k = (T × Σ registered_i) / (Σ base_votes_i)`
3) Set: `votes_i = k × base_votes_i`

Use when you believe 2026 turnout geography will look like a chosen historical election, just higher/lower uniformly.

---

##### 3) `logit_slope_original`

Fits a **station-specific turnout elasticity** in logit space using historical elections.

Estimation (per station `i`):

- For each election `e`, compute station turnout `t_{i,e}` and national turnout `t_{nat,e}`
- Regress in logit space:

  `logit(t_{i,e}) = α_i + β_i × logit(t_{nat,e})`

- Shrink `β_i` toward 1.0 (to reduce overfitting when there are few elections).

Prediction for 2026:

- `turnout_i = sigmoid( α_i + β_i × logit(T) )`
- `votes_i = turnout_i × registered_i`

Use when you want the most granular “elasticity” behaviour, but it can be noisy (because it is station-specific).

---

##### 4) `logit_slope_ep_offset`

Fits elasticity at the chosen granularity `g`, and also estimates an **EP vs parliamentary** offset.

Estimation (per group `g`):

`logit(t_{g,e}) = α_g + β_g × logit(t_{nat,e}) + γ_g × I(e is EP)`

Prediction for 2026 (parliamentary): set `I(EP)=0`:

- `turnout_g = sigmoid( α_g + β_g × logit(T) )`
- Convert to group vote totals: `votes_g = turnout_g × Σ_{i∈g} registered_i`

Distribute group votes back to stations using the reference map as within-group weights:

- weight: `w_i = registered_i × t_ref(i)`
- share: `s_i = w_i / Σ_{k∈g} w_k` (fallback to registered if needed)
- `votes_i = votes_g × s_i`

Use when you want elasticity behaviour but smoother than station-level, and you want EP elections to inform the fit without forcing EP-level turnout for parliamentary.

---

##### 5) `relative_offset_parl`

Computes a *parliamentary-only* relative turnout offset for each group and adds it to the target national logit.

For group `g` compute:

- `offset_g = mean over parliamentary elections e [ logit(t_{g,e}) - logit(t_{nat,e}) ]`

Then predict:

- `logit(turnout_g) = logit(T) + offset_g`
- `turnout_g = sigmoid(logit(turnout_g))`
- `votes_g = turnout_g × Σ_{i∈g} registered_i`

And distribute `votes_g → votes_i` within group using the same weighting scheme as `logit_slope_ep_offset`.

Use when you want a very interpretable model: “this group is typically X pp above/below the national turnout in parliamentary elections.”

---

##### 6) `logit_slope_reserve_adjusted`

Starts from the `logit_slope_ep_offset` prediction, but reallocates the change vs the reference map toward groups with more **headroom**.

Steps:

1) Compute raw group turnout prediction: `turnout_g_raw` (from `logit_slope_ep_offset`)
2) Compute baseline group turnout from reference map:

- `turnout_g_base = (Σ_{i∈g} registered_i × t_ref(i)) / (Σ_{i∈g} registered_i)`

3) Delta vs baseline:

- `Δ_g = turnout_g_raw - turnout_g_base`

4) Compute “reserve” (headroom) depending on whether turnout is going up or down:

- If `T` > national reference turnout: `reserve_g = clip_hi - turnout_g_base`
- Else: `reserve_g = turnout_g_base - clip_lo`

5) Scale delta by reserve (power = `reserve_adjust_power`):

- `scale_g = (reserve_g / mean(reserve)) ^ reserve_adjust_power`
- `turnout_g_adj = clip(turnout_g_base + Δ_g × scale_g)`

6) Convert to votes and distribute to stations within group.

Use when you want turnout changes to come more from places with more “room” to move (e.g., low-turnout areas when national turnout rises).

---

##### 7) `baseline_plus_marginal`

Anchors turnout to a reference map and allocates only the *net change* using a “marginal propensity” proxy.

1) Baseline votes from reference map:

- `base_votes_i = t_ref(i) × registered_i`

2) Delta total:

- `Δ_total = (T × Σ registered_i) - Σ base_votes_i`

3) Weight each station by:

- `mp_g = max(0, mean_parl(turnout_g) - mean_ep(turnout_g))` (computed historically)
- `mp_i = mp_{group(i)}`
- capacity/headroom:
  - if `Δ_total > 0`: `cap_i = clip_hi - t_ref(i)`
  - else: `cap_i = t_ref(i) - clip_lo`
- final weight:

  `w_i = (mp_i + ε) ^ marginal_concentration × (cap_i + ε)`

4) Allocate delta:

- `add_i = Δ_total × w_i / Σ w_i`
- `votes_i = base_votes_i + add_i`

Use when you want a “who turns out at the margin?” model: areas with bigger parl–EP gaps are treated as having more marginal voters.

---

##### 8) `baseline_ref_plus_elastic_delta`

This is the “baseline at X% + elasticity for surplus/deficit” model.

Inputs:

- `turnout_baseline_level = B` (e.g., 0.70)

Steps:

1) Build a baseline distribution at `B` using the reference map:

- `base_votes_i = t_ref(i) × registered_i`, scaled so `Σ base_votes_i = B × Σ registered_i`

2) Compute delta total:

- `Δ_total = (T - B) × Σ registered_i`

3) Compute an elasticity-implied *pattern* of change across groups using the logit-slope model:

- `turnout_g(B) = sigmoid( α_g + β_g × logit(B) )`
- `turnout_g(T) = sigmoid( α_g + β_g × logit(T) )`
- raw delta votes per group:

  `Δ_votes_g_raw = (turnout_g(T) - turnout_g(B)) × registered_g`

4) Rescale the delta pattern to match `Δ_total` exactly:

- `Δ_votes_g = Δ_votes_g_raw × (Δ_total / Σ Δ_votes_g_raw)` (fallback to proportional-to-registered if needed)

5) Distribute `Δ_votes_g` back to stations within group (same weights as the other grouped models), and add:

- `votes_i = base_votes_i + delta_votes_i`

Use when you want to:

- anchor “normal turnout geography” to a chosen election (e.g., 2022 at 70%), and
- use elasticities only for *what changes* as turnout rises/falls.



**Granularity** controls the modelling unit of the turnout map:

- `station` (polling station)
- `settlement`
- `evk`

Higher granularity means more local variation; lower granularity means smoother turnout.

### 3) Polls: poll shares → voter shares

The model supports two poll interpretations:

- **Decided** polls: poll numbers are treated as already excluding undecided/nonresponse. National voter shares are the poll shares (after sampling/normalizing).
- **Raw** polls: poll numbers include an **UNDECIDED** share and must be converted to voter shares using turnout and your undecided/nonresponse assumptions.

#### Raw polls → population shares

Let the raw poll share vector be:

- `p_party` (for each party)
- `u = p_UNDECIDED`

You can add an extra share **not in polls and not voting**:

- `nv = nonresponse_nonvoter_pct`
- `UNDECIDED_TRUE = u + nv`

To keep totals consistent, decided-party shares are scaled down so that:

- `sum(parties_true) = 1 - UNDECIDED_TRUE`

#### Population shares → voter shares (turnout composition)

Let:

- `D = sum(parties_true)` (decided mass)
- `T = turnout_target`

If `T ≤ D`, turnout is fully absorbed by already-decided voters and composition does **not** change:

- `voter_shares = parties_true / D`

If `T > D`, the extra turnout comes from the undecided pool. Define:

- `extra = min(T - D, UNDECIDED_TRUE)`

You specify how those **marginal** voters break between the two main parties:

- `uF = undecided_to_fidesz` (0.30 → 30% to Fidesz, 70% to Tisza)
- `uT = 1 - uF - undecided_to_others`

Then marginal turnout is added to party vote masses and renormalized to voter shares.

### 4) Mobilization / reserve tilt (raw polls only)

This is a *second layer* on top of the undecided split. It does **not** change the decided base; it only tilts the **marginal** allocation when `T > D`.

For each of the two main parties, define a “reserve factor”:

- `reserveF = (1 - mobilization_rate_fidesz) × reserve_strength_fidesz`
- `reserveT = (1 - mobilization_rate_tisza) × reserve_strength_tisza`

Intuition:

- Higher **mobilization_rate** → party is already better activated → *smaller* remaining reserve.
- Higher **reserve_strength** → more sympathetic people among the undecided pool.

When the tilt is enabled, the undecided split is reweighted:

- `uF_adj ∝ uF × reserveF`
- `uT_adj ∝ uT × reserveT`
- then normalized so `uF_adj + uT_adj = 1`.

So, compared to the “no-tilt” case, the model shifts marginal turnout toward the party with the larger reserve factor.

Important interaction:

- If polls are **decided**, undecided/nonresponse and mobilization settings do nothing.
- If polls are **raw** but `T ≤ D`, mobilization settings do nothing.
- Mobilization only matters when you are modelling *additional* turnout beyond the decided share.

### 5) Coefficients: geographic priors from history

The model builds priors from historical results using:

- **Election weights**: how much each historical election matters.
- **Geography weights**: how local vs how aggregate the coefficient should be.

Conceptually, for each unit `u` and party `p` the blended coefficient is:

- `coef(u,p) = exp( Σ_e Σ_g w_e × w_g × log_coef_e,g(u,p) )`

The “log then exp” form means weights combine multiplicatively in coefficient space.

### 6) Vote allocation via IPF (Iterative Proportional Fitting)

Given:

- Row totals = turnout model EVK totals
- Column totals = national voter shares
- Priors = coefficients

The model constructs an initial EVK×party matrix and then uses IPF to exactly match:

- each EVK total (row constraint)
- national party totals (column constraint)

### 7) Seats

From the simulated EVK vote matrix:

- Determine **SMC winners** per EVK (optionally restricted to Fidesz vs Tisza only).
- Compute **compensation votes** (loser votes + winner margin).
- Add **diaspora/mail votes** (list only) and the **nationality seat** (optional).
- Allocate list seats by **D’Hondt**, applying the threshold.

## Monte Carlo

Monte Carlo repeats the full pipeline many times, sampling uncertain inputs.

### What is sampled per draw

Depending on your Monte Carlo settings, each draw can sample:

- **Turnout target** (fixed / uniform range / normal distribution)
- **Turnout model** and **granularity** (random choice from your selected lists)
- **Poll shares** (using one of the poll sampling methods below)
- If poll mode is **raw**:
  - not-in-polls & not-voting share
  - undecided → Fidesz share (Tisza gets the remainder)
  - whether mobilization tilt is on/off (optional)
  - mobilization rates and reserve strengths
- **Election weights** and **geography weights** (sample each weight in a min–max range, then renormalize to sum to 1)
- **Coefficient noise** (multiplicative, log-normal)
- **Mail votes** (optional: total mail votes + Fidesz share; remainder → Tisza)

### Turnout precomputation (speed)

To support 100k+ draws, the Monte Carlo engine precomputes turnout EVK vote totals on a small grid of turnout targets for each selected turnout model/granularity, then linearly interpolates between those grid points for each draw.

### Poll sampling methods (Gaussian vs multinomial)

All methods start from a mean share vector and produce a random share vector.

- **Gaussian (pp)**
  - Adds independent Normal noise to each party share in *percentage points*.
  - Clips negatives to 0 and renormalizes.
  - Pros: simple; easy to express a “±pp” uncertainty.
  - Cons: treats parties as independent; not a true simplex distribution.

- **Multinomial (n respondents)**
  - Treats the poll as if it were a sample of `n` respondents.
  - Draws counts ~ Multinomial(n, p) then converts to shares.
  - Pros: respects the simplex; induces realistic negative correlation (if one party is up, others must be down).
  - Cons: variance is fully determined by `n` (may be too optimistic if there are house effects).

- **Dirichlet-multinomial**
  - First draws `p*` from a Dirichlet distribution around `p`, then draws counts from Multinomial(n, p*).
  - The **Dirichlet concentration** controls extra variance:
    - higher concentration → closer to multinomial
    - lower concentration → more dispersion (house effects / correlated polling errors)

## How the key “turnout vs undecided vs mobilization” knobs interact

- **Turnout target** affects:
  - the total domestic vote volume (always)
  - *and* the national voter shares (only in raw-poll mode via marginal undecided allocation)

- **Turnout model / granularity** affects:
  - where votes show up geographically (row totals)
  - it does not directly change national shares

- **Undecided & nonresponse** settings affect:
  - the size of the undecided pool available for marginal turnout

- **Undecided split (Fidesz vs Tisza)** affects:
  - the baseline direction of marginal voters

- **Mobilization / reserve tilt** affects:
  - the *effective* split of marginal voters when turnout rises above the decided mass
  - it creates a mixture distribution if you allow tilt on/off across simulations

## Configuration recipe: “Use 2022 as baseline 70% turnout, allocate above 70%, and infer composition changes”

To do exactly this you need both:

1) a turnout model that allocates the *extra* turnout geographically, and
2) raw-poll conversion so that higher turnout changes the national composition.

Suggested setup:

### Sidebar (scenario)

- **Turnout reference election**: `parl_2022_list`
- **Turnout baseline level**: `0.70`
- **Turnout model**: `baseline_ref_plus_elastic_delta` (baseline map from 2022, then allocate Δturnout using elasticities)
- **Turnout target**: set to the level you want to evaluate (or keep at 70% and let Monte Carlo vary it)

- **Poll type**: `raw`
- Set **nonresponse_nonvoter_pct** and **undecided_to_fidesz** (and optionally enable mobilization tilt) so that when turnout is above the decided mass, marginal voters are allocated and the national shares shift.

### Monte Carlo tab

- Turnout target distribution: e.g. Normal with mean 72% and sd 1% (or a uniform range 70–75%)
- Turnout models to sample: include `baseline_ref_plus_elastic_delta` (and optionally alternatives)
- Poll mode: `Raw` (or “Use scenario setting” if the scenario is raw)
- Poll sampling: Multinomial with your best estimate of `n`, or Dirichlet-multinomial if you want extra correlated uncertainty
- Add uncertainty on:
  - nonresponse_nonvoter_pct
  - undecided → Fidesz share
  - mobilization tilt on/off and its parameters
  - election/geography weights if you want robustness to historical-weight choices
  - mail votes if you want diaspora uncertainty
"""
    )

    # Polling-station universe differences (2026 has fewer/different station IDs than older elections)
    if getattr(data, "station_meta", None) is not None and getattr(data, "parl22_list_i", None) is not None:
        s26 = set(data.station_meta.index.astype(str))
        s22 = set(data.parl22_list_i["station_id"].astype(str).unique())
        overlap = len(s26 & s22)
        st.caption(
            "Station universe coverage (2026 vs 2022): "
            f"2026 stations={len(s26):,}, 2022 stations={len(s22):,}, "
            f"overlap={overlap:,}, 2022-only={len(s22 - s26):,}, 2026-only={len(s26 - s22):,}."
        )
        st.caption(
            "Model runs are produced on the 2026 station universe. Where a 2026 station has no exact historical "
            "station match, station-level turnout/coefficients fall back to higher-level geographies "
            "(settlement/EVK), so totals remain consistent while station-level detail is smoothed."
        )
