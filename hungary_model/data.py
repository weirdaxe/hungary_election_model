
from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import (
    BLOCK_BUCKETS,
    DB_REQUIRED_FILES,
    EVK_MAPPING_DB_CANDIDATES,
    POLL_REQUIRED_FILES,
    TURNOUT_GRANULARITIES,
)
from .math_utils import logit
from .polls import load_polls
from .turnout import compute_marginal_propensity, compute_relative_offset_parl, estimate_turnout_model_logit_slope
from .types import ModelData, ModelPaths


# ============================================================
# Paths
# ============================================================

def _all_exist(folder: Path, files: Iterable[str]) -> bool:
    return all((folder / f).exists() for f in files)


def _validate_paths(db_dir: Path, poll_dir: Path) -> None:
    missing_db = [f for f in DB_REQUIRED_FILES if not (db_dir / f).exists()]
    missing_poll = [f for f in POLL_REQUIRED_FILES if not (poll_dir / f).exists()]
    if missing_db or missing_poll:
        msg = []
        if missing_db:
            msg.append(f"Missing DB files in {db_dir}: {missing_db}")
        if missing_poll:
            msg.append(f"Missing poll files in {poll_dir}: {missing_poll}")
        raise FileNotFoundError("\n".join(msg))


def resolve_model_paths(
    db_dir: Optional[Path] = None,
    poll_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> ModelPaths:
    """
    Resolve data locations without requiring the user to pick directories.

    Default expectation:
      - DBs in ./Hungary Election Results/
      - Poll CSVs in ./

    If not found, tries a handful of common alternatives.
    """
    base = (base_dir or Path.cwd()).resolve()

    # Explicit overrides
    if db_dir is not None or poll_dir is not None:
        dbd = (db_dir or (base / "Hungary Election Results")).resolve()
        pold = (poll_dir or base).resolve()
        _validate_paths(dbd, pold)
        return ModelPaths(db_dir=dbd, poll_dir=pold, evk_mapping_db=_find_evk_mapping_db(dbd))

    candidates: List[Tuple[Path, Path]] = [
        (base / "Hungary Election Results", base),
        (base, base),
        (base / "data" / "Hungary Election Results", base / "data"),
        (base / "inputs" / "Hungary Election Results", base / "inputs"),
        (Path("/mnt/data") / "Hungary Election Results", Path("/mnt/data")),
        (Path("/mnt/data"), Path("/mnt/data")),
    ]

    for dbd, pold in candidates:
        if _all_exist(dbd, DB_REQUIRED_FILES) and _all_exist(pold, POLL_REQUIRED_FILES):
            return ModelPaths(db_dir=dbd.resolve(), poll_dir=pold.resolve(), evk_mapping_db=_find_evk_mapping_db(dbd))

    # As a last attempt, if a "./Hungary Election Results" exists relative to this package, try that.
    script_base = Path(__file__).resolve().parent
    candidates2 = [
        (script_base / "Hungary Election Results", script_base),
        (script_base, script_base),
        (script_base / "data" / "Hungary Election Results", script_base / "data"),
    ]
    for dbd, pold in candidates2:
        if _all_exist(dbd, DB_REQUIRED_FILES) and _all_exist(pold, POLL_REQUIRED_FILES):
            return ModelPaths(db_dir=dbd.resolve(), poll_dir=pold.resolve(), evk_mapping_db=_find_evk_mapping_db(dbd))

    searched = "\n".join([f"- DB: {dbd} | Polls: {pold}" for dbd, pold in (candidates + candidates2)])
    raise FileNotFoundError(
        "Could not auto-detect required inputs.\n\n"
        "Expected:\n"
        f"  DB files ({len(DB_REQUIRED_FILES)}): {', '.join(DB_REQUIRED_FILES)}\n"
        f"  Poll files ({len(POLL_REQUIRED_FILES)}): {', '.join(POLL_REQUIRED_FILES)}\n\n"
        "Searched these (db_dir | poll_dir) candidates:\n"
        f"{searched}\n\n"
        "Fix: place DBs in './Hungary Election Results/' and poll CSVs in './', "
        "or set overrides in the Streamlit app."
    )


def _find_evk_mapping_db(db_dir: Path) -> Optional[Path]:
    for name in EVK_MAPPING_DB_CANDIDATES:
        p = (db_dir / name)
        if p.exists():
            return p
    return None


# ============================================================
# SQLite helpers
# ============================================================

def sqlite_list_tables(db_path: Path) -> List[str]:
    with sqlite3.connect(db_path) as con:
        q = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        return pd.read_sql(q, con)["name"].tolist()


def read_sqlite_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(f"SELECT * FROM {table};", con)


def read_sqlite_best_table(db_path: Path, table_hints: List[str], required_cols: List[str]) -> pd.DataFrame:
    """
    Robust table selection:
      - Scores each table by number of required_cols present
      - Adds small score for hint matches in name
      - Loads best scoring table
    """
    tables = sqlite_list_tables(db_path)
    if not tables:
        raise ValueError(f"No tables in DB: {db_path}")

    best = None
    best_score = -1
    best_cols = None

    for t in tables:
        df0 = read_sqlite_table(db_path, t)
        cols = set(df0.columns)
        score = sum(1 for c in required_cols if c in cols)
        name_u = t.lower()
        score += sum(0.1 for h in table_hints if h.lower() in name_u)
        if score > best_score:
            best = t
            best_score = score
            best_cols = cols

    if best is None:
        raise ValueError(f"Could not select a table from DB: {db_path}")

    df = read_sqlite_table(db_path, best)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Selected table '{best}' from {db_path.name} but missing required cols: {missing}.\n"
            f"Available cols: {sorted(list(best_cols))[:60]}"
        )
    return df


# ============================================================
# Party / label mapping (BLOCK buckets)
# ============================================================

def normalize_party_label(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*-\s*", "-", s)
    return s.upper()


FORCE_OTHER_EXACT = {
    "MEMO",
    "NORMÁLIS PÁRT",
    "NORMALIS PART",
    "MUNKÁSPÁRT",
    "MUNKASPART",
    "MUNKÁSPÁRT-ISZOMM",
    "MUNKASPART-ISZOMM",
    "2RK PÁRT",
    "2RK PART",
    "MMN",
}
FORCE_OTHER_PATTERNS = ["FÜGGETLEN", "FUGGETLEN", "INDEPENDENT"]


def build_block_party_mapping(
    df_long: pd.DataFrame,
    opp_min_share: float = 0.01,
    force_other_exact: Optional[set[str]] = None,
    force_other_patterns: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Build mapping raw party -> BLOCK bucket.

    Logic:
      1) Detect FIDESZ / MH / MKKP by keyword
      2) Forced OTHER list / patterns
      3) Everything else defaults to OPP
      4) Tiny OPP parties (< opp_min_share, vote-weighted) -> OTHER
    """
    force_other_exact = set(force_other_exact or FORCE_OTHER_EXACT)
    force_other_patterns = list(force_other_patterns or FORCE_OTHER_PATTERNS)

    tmp = df_long[["party", "votes"]].copy()
    tmp["party_norm"] = tmp["party"].map(normalize_party_label)
    tmp["votes"] = pd.to_numeric(tmp["votes"], errors="coerce").fillna(0.0)

    agg = tmp.groupby("party_norm", as_index=False)["votes"].sum()
    total_votes = float(agg["votes"].sum())
    agg["share"] = np.where(total_votes > 0, agg["votes"] / total_votes, 0.0)

    def _forced_other(pn: str) -> bool:
        if pn in force_other_exact:
            return True
        for pat in force_other_patterns:
            if pat in pn:
                return True
        return False

    buckets = []
    for pn, share in zip(agg["party_norm"].tolist(), agg["share"].tolist()):
        if "FIDESZ" in pn:
            b = "FIDESZ"
        elif "MI HAZ" in pn or "MIHAZ" in pn:
            b = "MH"
        elif "MKKP" in pn:
            b = "MKKP"
        elif _forced_other(pn):
            b = "OTHER"
        else:
            b = "OPP"

        if b == "OPP" and share < opp_min_share:
            b = "OTHER"
        buckets.append(b)

    agg["bucket"] = buckets
    mapping = dict(zip(agg["party_norm"], agg["bucket"]))

    audit = agg.sort_values("votes", ascending=False).reset_index(drop=True)
    audit["votes_pct"] = 100 * audit["share"]
    return mapping, audit


# ============================================================
# Station ids and basic parsing
# ============================================================

def make_station_id(df: pd.DataFrame, maz_col: str = "maz", taz_col: str = "taz", sorsz_col: str = "sorsz") -> pd.Series:
    return (
        df[maz_col].astype(str).str.zfill(2) + "-" +
        df[taz_col].astype(str).str.zfill(3) + "-" +
        df[sorsz_col].astype(str).str.zfill(3)
    )


def normalize_station_name(name: str) -> str:
    if name is None:
        return "NA"
    s = str(name).strip()
    if s == "":
        return "NA"

    s = re.sub(r"\s*\(.*?\)\s*$", "", s).strip()
    s = re.sub(r"\s+", " ", s).upper()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    return s if s else "NA"


def parse_district_number(district_name: str) -> Optional[int]:
    if district_name is None:
        return None
    m = re.search(r"(\d+)\.", str(district_name))
    return int(m.group(1)) if m else None


# ============================================================
# Long -> station-wide tables
# ============================================================

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def build_station_wide_from_list_like(
    df_long: pd.DataFrame,
    election: str,
    kind: str,
    table_type: str,  # 'list' or 'ep'
    party_map_norm: Dict[str, str],
    station_to_evk: Optional[pd.DataFrame] = None,
    buckets: Optional[List[str]] = None,
) -> pd.DataFrame:
    buckets = buckets or list(BLOCK_BUCKETS)

    df = df_long.copy()
    df = _ensure_columns(
        df,
        [
            "maz", "taz", "sorsz",
            "district_name", "polling_station_name",
            "registered_voters", "voters_appeared", "valid_ballots", "invalid_ballots",
            "party", "votes",
        ],
    )

    df["station_id"] = make_station_id(df)

    for col in ["registered_voters", "voters_appeared", "valid_ballots", "invalid_ballots"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    v_calc = df["valid_ballots"].fillna(0) + df["invalid_ballots"].fillna(0)
    df.loc[(df["voters_appeared"].fillna(0) <= 0) & (v_calc > 0), "voters_appeared"] = v_calc
    df.loc[df["registered_voters"].fillna(0) <= 0, "registered_voters"] = np.nan

    df["party_norm"] = df["party"].map(normalize_party_label)
    df["bucket"] = df["party_norm"].map(party_map_norm).fillna("OTHER")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(float)

    pivot = df.pivot_table(index="station_id", columns="bucket", values="votes", aggfunc="sum", fill_value=0.0)
    for b in buckets:
        if b not in pivot.columns:
            pivot[b] = 0.0
    pivot = pivot[buckets]

    meta_cols = [
        "maz", "taz", "sorsz", "district_name", "polling_station_name",
        "registered_voters", "voters_appeared", "valid_ballots", "invalid_ballots",
    ]
    meta = df.groupby("station_id")[meta_cols].first()

    out = meta.join(pivot, how="left").reset_index()

    out["valid_votes_calc"] = out[buckets].sum(axis=1)
    out["invalid_ballots"] = out["invalid_ballots"].fillna(0.0)
    out["voters_appeared_calc"] = out["voters_appeared"].fillna(0.0)
    out.loc[out["voters_appeared_calc"] <= 0, "voters_appeared_calc"] = out["valid_votes_calc"] + out["invalid_ballots"]

    out["settlement_id"] = out["maz"].astype(str).str.zfill(2) + "-" + out["taz"].astype(str).str.zfill(3)
    out["county_id"] = out["maz"].astype(str).str.zfill(2)

    out["station_name_key"] = out["polling_station_name"].map(normalize_station_name)
    out["station_name_id"] = out["settlement_id"] + "-" + out["station_name_key"]

    if table_type == "list":
        out["district_no"] = out["district_name"].map(parse_district_number)
        out["evk_id"] = out["county_id"] + "-" + out["district_no"].fillna(-1).astype(int).astype(str).str.zfill(2)
    else:
        out["district_no"] = np.nan
        out["evk_id"] = np.nan

    if table_type == "ep" and station_to_evk is not None:
        out = out.merge(
            station_to_evk[["station_id", "evk_id"]],
            on="station_id",
            how="left",
            suffixes=(None, "_mapped"),
        )
        out["evk_id"] = out["evk_id_mapped"]
        out.drop(columns=["evk_id_mapped"], inplace=True)

    out["election"] = election
    out["kind"] = kind
    return out


def build_station_wide_from_smc(
    df_long: pd.DataFrame,
    election: str,
    kind: str,
    party_map_norm: Dict[str, str],
    buckets: Optional[List[str]] = None,
) -> pd.DataFrame:
    buckets = buckets or list(BLOCK_BUCKETS)

    df = df_long.copy()
    df = _ensure_columns(
        df,
        [
            "maz", "taz", "sorsz",
            "district_name", "polling_station_name",
            "registered_voters", "voters_appeared", "valid_ballots", "invalid_ballots",
            "party", "votes",
        ],
    )

    df["station_id"] = make_station_id(df)

    for col in ["registered_voters", "voters_appeared", "valid_ballots", "invalid_ballots"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    v_calc = df["valid_ballots"].fillna(0) + df["invalid_ballots"].fillna(0)
    df.loc[(df["voters_appeared"].fillna(0) <= 0) & (v_calc > 0), "voters_appeared"] = v_calc
    df.loc[df["registered_voters"].fillna(0) <= 0, "registered_voters"] = np.nan

    df["party_norm"] = df["party"].map(normalize_party_label)
    df["bucket"] = df["party_norm"].map(party_map_norm).fillna("OTHER")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(float)

    pivot = df.pivot_table(index="station_id", columns="bucket", values="votes", aggfunc="sum", fill_value=0.0)
    for b in buckets:
        if b not in pivot.columns:
            pivot[b] = 0.0
    pivot = pivot[buckets]

    meta_cols = [
        "maz", "taz", "sorsz", "district_name", "polling_station_name",
        "registered_voters", "voters_appeared", "valid_ballots", "invalid_ballots",
    ]
    meta = df.groupby("station_id")[meta_cols].first()

    out = meta.join(pivot, how="left").reset_index()
    out["valid_votes_calc"] = out[buckets].sum(axis=1)

    out["invalid_ballots"] = out["invalid_ballots"].fillna(0.0)
    out["voters_appeared_calc"] = out["voters_appeared"].fillna(0.0)
    out.loc[out["voters_appeared_calc"] <= 0, "voters_appeared_calc"] = out["valid_votes_calc"] + out["invalid_ballots"]

    out["settlement_id"] = out["maz"].astype(str).str.zfill(2) + "-" + out["taz"].astype(str).str.zfill(3)
    out["county_id"] = out["maz"].astype(str).str.zfill(2)

    out["station_name_key"] = out["polling_station_name"].map(normalize_station_name)
    out["station_name_id"] = out["settlement_id"] + "-" + out["station_name_key"]

    out["district_no"] = out["district_name"].map(parse_district_number)
    out["evk_id"] = out["county_id"] + "-" + out["district_no"].fillna(-1).astype(int).astype(str).str.zfill(2)

    out["election"] = election
    out["kind"] = kind
    return out


# ============================================================
# EVK 2026 mapping
# ============================================================

def load_evk_mapping_2026(mapping_db: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load station -> EVK mapping and EVK names from the 2026 mapping DB.

    Expected table: polling_stations with columns like:
      - maz, taz, sorszam (or sorsz)
      - evk (EVK number within county)
      - evk_nev (human-readable constituency name)

    Returns:
      - station_map: columns [station_id, evk_id]
      - evk_meta: index evk_id with column [evk_name]
    """
    with sqlite3.connect(mapping_db) as con:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;", con)["name"].tolist()
        if "polling_stations" not in tables:
            # Try first table if naming differs
            if not tables:
                raise ValueError(f"No tables in mapping DB: {mapping_db}")
            table = tables[0]
        else:
            table = "polling_stations"

        df = pd.read_sql(f"SELECT * FROM {table};", con)

    # Find columns with minimal robustness
    cols = {c.lower(): c for c in df.columns}

    def _col(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_maz = _col("maz")
    c_taz = _col("taz")
    c_s = _col("sorsz", "sorszam", "sorszám", "sorszám")

    c_evk = _col("evk", "evk_szam", "evk_szám")
    c_evk_name = _col("evk_nev", "evk_név", "evk_name")

    missing = [("maz", c_maz), ("taz", c_taz), ("sorszam", c_s), ("evk", c_evk)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise ValueError(f"Mapping DB missing required columns: {missing}. Available: {list(df.columns)}")

    tmp = df.copy()
    tmp["station_id"] = make_station_id(tmp, maz_col=c_maz, taz_col=c_taz, sorsz_col=c_s)
    tmp["evk_id"] = tmp[c_maz].astype(str).str.zfill(2) + "-" + tmp[c_evk].astype(str).str.zfill(2)

    station_map = tmp[["station_id", "evk_id"]].drop_duplicates().copy()

    if c_evk_name is not None:
        evk_meta = tmp[["evk_id", c_evk_name]].drop_duplicates().rename(columns={c_evk_name: "evk_name"})
        evk_meta = evk_meta.set_index("evk_id").sort_index()
    else:
        evk_meta = pd.DataFrame(index=sorted(station_map["evk_id"].unique()), data={"evk_name": ""})

    return station_map, evk_meta


def load_station_universe_2026(mapping_db: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the 2026 polling-station universe from the 2026 EVK mapping DB.

    The mapping DB is treated as *source of truth* for:
      - which polling stations exist in 2026 (station universe)
      - station_id = maz-taz-sorsz/sorszam (stable across elections)
      - 2026 EVK assignment (maz-evk) for every station
      - registered voters per station (typically `num_voters`)

    Returns:
      station_meta_2026:
        index = station_id
        columns = station_name_id, settlement_id, county_id, evk_id
      evk_meta:
        index = evk_id
        columns = evk_name
      registered_2026:
        Series indexed by station_id (registered voters)
    """
    with sqlite3.connect(mapping_db) as con:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
            con,
        )["name"].tolist()
        if not tables:
            raise ValueError(f"No tables in mapping DB: {mapping_db}")

        table = "polling_stations" if "polling_stations" in tables else tables[0]
        df = pd.read_sql(f"SELECT * FROM {table};", con)

    cols = {c.lower(): c for c in df.columns}

    def _col(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_maz = _col("maz")
    c_taz = _col("taz")
    c_s = _col("sorsz", "sorszam", "sorszám", "sorszam")
    c_evk = _col("evk", "evk_szam", "evk_szám")

    c_evk_name = _col("evk_nev", "evk_név", "evk_name")
    c_num_voters = _col("num_voters", "registered_voters", "registered", "n_voters", "num_registered")
    c_station_name = _col("station_name", "polling_station_name", "polling_station", "szavazokor_nev", "szavazokor")

    missing = [("maz", c_maz), ("taz", c_taz), ("sorszam", c_s), ("evk", c_evk)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise ValueError(f"Mapping DB missing required columns: {missing}. Available: {list(df.columns)}")

    tmp = df.copy()
    tmp["station_id"] = make_station_id(tmp, maz_col=c_maz, taz_col=c_taz, sorsz_col=c_s)

    maz2 = tmp[c_maz].astype(str).str.zfill(2)
    taz3 = tmp[c_taz].astype(str).str.zfill(3)

    tmp["county_id"] = maz2
    tmp["settlement_id"] = maz2 + "-" + taz3

    if c_station_name is not None:
        tmp["station_name_key"] = tmp[c_station_name].map(normalize_station_name)
    else:
        tmp["station_name_key"] = "NA"

    tmp["station_name_id"] = tmp["settlement_id"] + "-" + tmp["station_name_key"]
    tmp["evk_id"] = maz2 + "-" + tmp[c_evk].astype(str).str.zfill(2)

    station_meta_2026 = (
        tmp.groupby("station_id")[["station_name_id", "settlement_id", "county_id", "evk_id"]]
        .first()
        .copy()
    )
    station_meta_2026.index.name = "station_id"

    if c_num_voters is not None:
        tmp["_registered"] = pd.to_numeric(tmp[c_num_voters], errors="coerce")
        registered_2026 = tmp.groupby("station_id")["_registered"].max()
    else:
        registered_2026 = pd.Series(np.nan, index=station_meta_2026.index)

    registered_2026.name = "registered_2026"

    # EVK meta (names)
    if c_evk_name is not None:
        evk_meta = (
            tmp[["evk_id", c_evk_name]]
            .drop_duplicates()
            .rename(columns={c_evk_name: "evk_name"})
            .set_index("evk_id")
            .sort_index()
        )
    else:
        evk_meta = pd.DataFrame(index=sorted(station_meta_2026["evk_id"].dropna().unique()), data={"evk_name": ""})

    return station_meta_2026, evk_meta, registered_2026




def apply_evk_mapping_2026(
    df_station: pd.DataFrame,
    station_map: pd.DataFrame,
    *,
    strict: bool = True,
    settlement_to_evk: Optional[pd.Series] = None,
    settlement_col: str = "settlement_id",
) -> pd.DataFrame:
    """Map station rows to 2026 EVK IDs using the 2026 mapping DB.

    - Primary key: station_id -> evk_id_2026 (exact match).
    - Fallback: settlement_id -> evk_id, but only if the settlement maps to a single EVK in 2026.

    When strict=False, rows that still cannot be mapped keep evk_id as NaN (evk_id_legacy is preserved).
    """
    # Accept either (station_id, evk_id_2026) or (station_id, evk_id)
    if "evk_id_2026" not in station_map.columns:
        if "evk_id" in station_map.columns:
            station_map = station_map.rename(columns={"evk_id": "evk_id_2026"})
        else:
            raise ValueError("station_map must contain an evk_id_2026 (or evk_id) column")

    out = df_station.merge(station_map, on="station_id", how="left")

    # Preserve legacy EVK, then overwrite with 2026 mapping.
    out["evk_id_legacy"] = out.get("evk_id", np.nan)
    out["evk_id"] = out["evk_id_2026"]

    if settlement_to_evk is not None and settlement_col in out.columns:
        out["evk_id"] = out["evk_id"].combine_first(out[settlement_col].map(settlement_to_evk))

    if strict and out["evk_id"].isna().any():
        missing = out.loc[out["evk_id"].isna(), "station_id"].astype(str).unique()
        raise ValueError(
            "Some station_ids cannot be mapped to 2026 EVK IDs (and no unique settlement fallback exists). "
            f"Example missing station_ids: {missing[:10].tolist()}"
        )

    out = out.drop(columns=["evk_id_2026"])
    return out

def build_registered_base(*station_tables: pd.DataFrame) -> pd.Series:
    reg = []
    for t in station_tables:
        reg.append(t[["station_id", "registered_voters"]].copy())
    all_reg = pd.concat(reg, ignore_index=True).dropna()
    all_reg = all_reg[all_reg["registered_voters"] > 0]
    return all_reg.groupby("station_id")["registered_voters"].max()


def apply_registered_imputation(df: pd.DataFrame, reg_base: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(reg_base.rename("registered_base"), on="station_id", how="left")
    out["registered_voters_imp"] = out["registered_voters"]
    out.loc[out["registered_voters_imp"].isna(), "registered_voters_imp"] = out["registered_base"]
    out["turnout_rate"] = out["voters_appeared_calc"] / out["registered_voters_imp"]
    out.loc[out["turnout_rate"] > 1.0, "turnout_rate"] = 1.0
    return out


def _compute_actual_block_shares_valid(df_station_wide: pd.DataFrame) -> pd.Series:
    """National shares among valid votes in BLOCK-space aggregation."""
    v = df_station_wide[list(BLOCK_BUCKETS)].sum()
    total = float(v.sum())
    if total <= 0:
        return pd.Series({"FIDESZ": 0.0, "CHALLENGER": 0.0, "OTHER": 0.0})

    f = float(v.get("FIDESZ", 0.0))
    opp = float(v.get("OPP", 0.0))
    other = float(v.get("MH", 0.0) + v.get("MKKP", 0.0) + v.get("OTHER", 0.0))
    return pd.Series({"FIDESZ": f / total, "CHALLENGER": opp / total, "OTHER": other / total})


def _compute_actual_block_shares_population(df_station_wide: pd.DataFrame) -> pd.Series:
    """
    National shares among registered population.

    - Party shares are (party valid votes) / registered_total.
    - UNDECIDED_TRUE is 1 - (valid_votes_total / registered_total) and includes nonvoters + invalid ballots.
    """
    v = df_station_wide[list(BLOCK_BUCKETS)].sum()
    reg_total = float(df_station_wide["registered_voters_imp"].sum())
    valid_total = float(v.sum())

    if reg_total <= 0:
        return pd.Series({"FIDESZ": 0.0, "CHALLENGER": 0.0, "OTHER": 0.0, "UNDECIDED_TRUE": 0.0})

    f = float(v.get("FIDESZ", 0.0))
    opp = float(v.get("OPP", 0.0))
    other = float(v.get("MH", 0.0) + v.get("MKKP", 0.0) + v.get("OTHER", 0.0))

    undec = max(0.0, 1.0 - valid_total / reg_total)

    return pd.Series(
        {
            "FIDESZ": f / reg_total,
            "CHALLENGER": opp / reg_total,
            "OTHER": other / reg_total,
            "UNDECIDED_TRUE": undec,
        }
    )


# ============================================================
# Main loader
# ============================================================

def load_model_data(paths: Optional[ModelPaths] = None) -> ModelData:
    paths = paths or resolve_model_paths()
    _validate_paths(paths.db_dir, paths.poll_dir)

    # Optional but strongly recommended: 2026 EVK mapping DB.
    #
    # IMPORTANT: When provided, this DB is treated as *source of truth* for:
    #   - the 2026 polling-station universe (station_id set)
    #   - station->EVK assignment under 2026 boundaries
    #   - registered voters per station in 2026 (if available)
    station_map_2026: Optional[pd.DataFrame] = None
    station_meta_2026: Optional[pd.DataFrame] = None
    registered_2026_from_mapping: Optional[pd.Series] = None
    settlement_to_evk: Optional[pd.Series] = None
    evk_meta = pd.DataFrame()

    station_universe: Optional[pd.Index] = None
    if paths.evk_mapping_db and Path(paths.evk_mapping_db).exists():
        station_meta_2026, evk_meta, registered_2026_from_mapping = load_station_universe_2026(Path(paths.evk_mapping_db))
        station_map_2026 = station_meta_2026.reset_index()[["station_id", "evk_id"]].copy()
        # settlement->EVK fallback (only when settlement is fully contained in a single EVK in 2026)
        _sett_evk_n = station_meta_2026.groupby("settlement_id")["evk_id"].nunique()
        settlement_to_evk = station_meta_2026.groupby("settlement_id")["evk_id"].first()
        settlement_to_evk = settlement_to_evk.loc[_sett_evk_n[_sett_evk_n == 1].index]
        station_universe = station_meta_2026.index

    db = paths.db_files()

    required_long_cols = ["maz", "taz", "sorsz", "party", "votes"]
    parl22_list_long = read_sqlite_best_table(db["parl_list_2022"], table_hints=["national", "list"], required_cols=required_long_cols)
    parl22_smc_long = read_sqlite_best_table(db["parl_smc_2022"], table_hints=["const", "smc"], required_cols=required_long_cols)
    parl18_list_long = read_sqlite_best_table(db["parl_list_2018"], table_hints=["national", "list"], required_cols=required_long_cols)
    parl18_smc_long = read_sqlite_best_table(db["parl_smc_2018"], table_hints=["const", "smc"], required_cols=required_long_cols)
    ep24_long = read_sqlite_best_table(db["ep_2024"], table_hints=["ep"], required_cols=required_long_cols)
    ep19_long = read_sqlite_best_table(db["ep_2019"], table_hints=["ep"], required_cols=required_long_cols)

    # Party mappings
    map_parl18_list, _ = build_block_party_mapping(parl18_list_long, opp_min_share=0.01)
    map_parl22_list, _ = build_block_party_mapping(parl22_list_long, opp_min_share=0.01)
    map_ep19, _ = build_block_party_mapping(ep19_long, opp_min_share=0.01)
    map_ep24, _ = build_block_party_mapping(ep24_long, opp_min_share=0.01)
    map_parl18_smc, _ = build_block_party_mapping(parl18_smc_long, opp_min_share=0.01)
    map_parl22_smc, _ = build_block_party_mapping(parl22_smc_long, opp_min_share=0.01)

    # Build SMC station-wide first (for legacy EVK mapping; but we'll override with 2026 mapping if provided)
    parl18_smc = build_station_wide_from_smc(parl18_smc_long, election="parl_2018_smc", kind="smc", party_map_norm=map_parl18_smc)
    parl22_smc = build_station_wide_from_smc(parl22_smc_long, election="parl_2022_smc", kind="smc", party_map_norm=map_parl22_smc)

    # If a 2026 mapping DB is provided, it defines the station universe.
    # Apply 2026 mapping override if present
    if station_map_2026 is not None:
        parl18_smc = apply_evk_mapping_2026(parl18_smc, station_map_2026, strict=False, settlement_to_evk=settlement_to_evk)
        parl22_smc = apply_evk_mapping_2026(parl22_smc, station_map_2026, strict=False, settlement_to_evk=settlement_to_evk)

    # For EP tables, mapping is always needed. Prefer 2026 mapping if available; else fall back to SMC-based mapping.
    if station_map_2026 is not None:
        station_to_evk_2018 = station_map_2026
        station_to_evk_2022 = station_map_2026
    else:
        station_to_evk_2018 = parl18_smc.groupby("station_id")[["evk_id"]].first().reset_index()
        station_to_evk_2022 = parl22_smc.groupby("station_id")[["evk_id"]].first().reset_index()

    # Station-wide tables (LIST + EP)
    parl18_list = build_station_wide_from_list_like(
        parl18_list_long, election="parl_2018_list", kind="list", table_type="list", party_map_norm=map_parl18_list
    )
    parl22_list = build_station_wide_from_list_like(
        parl22_list_long, election="parl_2022_list", kind="list", table_type="list", party_map_norm=map_parl22_list
    )
    ep19 = build_station_wide_from_list_like(
        ep19_long, election="ep_2019", kind="ep", table_type="ep", party_map_norm=map_ep19, station_to_evk=station_to_evk_2018
    )
    ep24 = build_station_wide_from_list_like(
        ep24_long, election="ep_2024", kind="ep", table_type="ep", party_map_norm=map_ep24, station_to_evk=station_to_evk_2022
    )

    # Apply EVK 2026 mapping to all list tables too
    if station_map_2026 is not None:
        parl18_list = apply_evk_mapping_2026(parl18_list, station_map_2026, strict=False, settlement_to_evk=settlement_to_evk)
        parl22_list = apply_evk_mapping_2026(parl22_list, station_map_2026, strict=False, settlement_to_evk=settlement_to_evk)
        ep19 = apply_evk_mapping_2026(ep19, station_map_2026, strict=False, settlement_to_evk=settlement_to_evk)
        ep24 = apply_evk_mapping_2026(ep24, station_map_2026, strict=False, settlement_to_evk=settlement_to_evk)

    # Registered voter base and imputation
    registered_base = build_registered_base(parl18_list, parl22_list, ep19, ep24, parl18_smc, parl22_smc)

    parl18_list_i = apply_registered_imputation(parl18_list, registered_base)
    parl22_list_i = apply_registered_imputation(parl22_list, registered_base)
    ep19_i = apply_registered_imputation(ep19, registered_base)
    ep24_i = apply_registered_imputation(ep24, registered_base)

    # Universal station metadata
    #
    # If a 2026 mapping DB is provided, it defines the station universe and EVK assignment.
    # We still *prefer* station_name_id / settlement_id computed from election result tables
    # (they are used in turnout model estimation), but we reindex to the 2026 universe and
    # fill missing metadata from the mapping DB.
    station_meta_hist = ep24_i.set_index("station_id")[["station_name_id", "settlement_id", "county_id", "evk_id"]].combine_first(
        parl22_list_i.set_index("station_id")[["station_name_id", "settlement_id", "county_id", "evk_id"]]
    ).combine_first(
        parl18_list_i.set_index("station_id")[["station_name_id", "settlement_id", "county_id", "evk_id"]]
    )
    station_meta_hist.index.name = "station_id"

    if station_meta_2026 is not None:
        # Reindex to the 2026 station universe and force 2026 EVK mapping.
        station_meta = station_meta_hist.combine_first(station_meta_2026)
        station_meta = station_meta.reindex(station_meta_2026.index)

        # EVK is always taken from 2026 mapping (no fallback).
        station_meta["evk_id"] = station_meta_2026["evk_id"].reindex(station_meta.index).values
    else:
        station_meta = station_meta_hist

    # Election tables for coefficients & turnout panel
    election_tables = {
        "parl_2018_list": parl18_list_i,
        "parl_2022_list": parl22_list_i,
        "ep_2019": ep19_i,
        "ep_2024": ep24_i,
    }

    # Coefficients by election (BLOCK)
    coef_by_election: Dict[str, Dict[str, pd.DataFrame]] = {}
    coef_by_election_no_budapest: Dict[str, Dict[str, pd.DataFrame]] = {}
    for e, df in election_tables.items():
        buckets = [b for b in BLOCK_BUCKETS if df[b].sum() > 0]
        coef_by_election[e] = {
            "buckets": buckets,  # for diagnostics
            "station": clip_and_fill_coef(compute_group_coefficients(df, "station_id", buckets)),
            "location": clip_and_fill_coef(compute_group_coefficients(df, "station_name_id", buckets)),
            "settlement": clip_and_fill_coef(compute_group_coefficients(df, "settlement_id", buckets)),
            "evk": clip_and_fill_coef(compute_group_coefficients(df, "evk_id", buckets)),
        }

        # Same computation but excluding Budapest (county_id == '01') from the
        # national baseline.
        if "county_id" in df.columns:
            df_nb = df.loc[df["county_id"].astype(str) != "01"].copy()
        else:
            df_nb = df.copy()
        buckets_nb = [b for b in BLOCK_BUCKETS if df_nb[b].sum() > 0]
        coef_by_election_no_budapest[e] = {
            "buckets": buckets_nb,
            "station": clip_and_fill_coef(compute_group_coefficients(df_nb, "station_id", buckets_nb)),
            "location": clip_and_fill_coef(compute_group_coefficients(df_nb, "station_name_id", buckets_nb)),
            "settlement": clip_and_fill_coef(compute_group_coefficients(df_nb, "settlement_id", buckets_nb)),
            "evk": clip_and_fill_coef(compute_group_coefficients(df_nb, "evk_id", buckets_nb)),
        }

    # Turnout panel
    turnout_panel = pd.concat(
        [
            df[["station_id", "station_name_id", "settlement_id", "evk_id", "registered_voters_imp", "voters_appeared_calc", "turnout_rate"]]
            .assign(election=e, is_ep=str(e).startswith("ep_"), is_parl=str(e).startswith("parl_"))
            for e, df in election_tables.items()
        ],
        ignore_index=True,
    )

    # National turnout by election
    nat_turnout = pd.Series({e: float(df["voters_appeared_calc"].sum() / df["registered_voters_imp"].sum()) for e, df in election_tables.items()}).sort_index()
    nat_turnout_logit = pd.Series(logit(nat_turnout.values), index=nat_turnout.index)

    # Turnout model components
    elasticity_station_original = estimate_turnout_model_logit_slope(
        turnout_panel, nat_turnout_logit, group_col="station_id", include_ep_dummy=False, k_shrink=4.0
    )

    elasticity_ep_offset: Dict[str, pd.DataFrame] = {}
    offset_parl: Dict[str, pd.Series] = {}
    marginal_propensity: Dict[str, pd.Series] = {}

    for g, col in TURNOUT_GRANULARITIES.items():
        elasticity_ep_offset[g] = estimate_turnout_model_logit_slope(
            turnout_panel, nat_turnout_logit, group_col=col, include_ep_dummy=True, k_shrink=4.0
        )
        offset_parl[g] = compute_relative_offset_parl(turnout_panel, nat_turnout_logit, group_col=col)
        marginal_propensity[g] = compute_marginal_propensity(turnout_panel, group_col=col)

    baseline_turnout_station = parl22_list_i.set_index("station_id")["turnout_rate"]
    baseline_turnout_station.name = "baseline_turnout_parl22"

    # Registered baseline for 2026
    #
    # If a 2026 mapping DB is provided and contains registered-voter counts, use that as the
    # primary source. Otherwise, fall back to imputed registered voters from recent elections.
    if registered_2026_from_mapping is not None:
        registered_2026 = registered_2026_from_mapping.copy()
        # align to station universe (station_meta index) if available
        if station_meta is not None and not station_meta.empty:
            registered_2026 = registered_2026.reindex(station_meta.index)

        # Fill missing values from historical registered imputation (rare).
        reg_2024 = ep24_i.set_index("station_id")["registered_voters_imp"]
        reg_2022 = parl22_list_i.set_index("station_id")["registered_voters_imp"]
        fallback = reg_2024.combine_first(reg_2022).combine_first(registered_base)
        registered_2026 = registered_2026.combine_first(fallback.reindex(registered_2026.index))
    else:
        reg_2024 = ep24_i.set_index("station_id")["registered_voters_imp"]
        reg_2022 = parl22_list_i.set_index("station_id")["registered_voters_imp"]
        registered_2026 = reg_2024.combine_first(reg_2022).combine_first(registered_base)

    registered_2026.name = "registered_2026"


    # Polls
    polls_all = pd.concat([load_polls(paths, y) for y in ["2018", "2019", "2022", "2024", "2026"]], ignore_index=True)

    # Actual BLOCK shares by year
    actual_block_shares_valid = {
        2018: _compute_actual_block_shares_valid(parl18_list_i),
        2019: _compute_actual_block_shares_valid(ep19_i),
        2022: _compute_actual_block_shares_valid(parl22_list_i),
        2024: _compute_actual_block_shares_valid(ep24_i),
    }
    actual_block_shares_population = {
        2018: _compute_actual_block_shares_population(parl18_list_i),
        2019: _compute_actual_block_shares_population(ep19_i),
        2022: _compute_actual_block_shares_population(parl22_list_i),
        2024: _compute_actual_block_shares_population(ep24_i),
    }

    return ModelData(
        paths=paths,
        parl18_list_i=parl18_list_i,
        parl22_list_i=parl22_list_i,
        ep19_i=ep19_i,
        ep24_i=ep24_i,
        station_meta=station_meta,
        evk_meta=evk_meta,
        registered_2026=registered_2026,
        coef_by_election=coef_by_election,
        coef_by_election_no_budapest=coef_by_election_no_budapest,
        turnout_panel=turnout_panel,
        nat_turnout=nat_turnout,
        nat_turnout_logit=nat_turnout_logit,
        elasticity_station_original=elasticity_station_original,
        elasticity_ep_offset=elasticity_ep_offset,
        offset_parl=offset_parl,
        marginal_propensity=marginal_propensity,
        baseline_turnout_station=baseline_turnout_station,
        polls_all=polls_all,
        actual_block_shares_valid=actual_block_shares_valid,
        actual_block_shares_population=actual_block_shares_population,
    )


# ============================================================
# Coefficients helpers (BLOCK)
# ============================================================

def compute_group_coefficients(stations: pd.DataFrame, group_col: str, buckets: List[str]) -> pd.DataFrame:
    df = stations.copy()
    g_votes = df.groupby(group_col)[buckets].sum()
    g_total = g_votes.sum(axis=1)
    g_shares = g_votes.div(g_total.replace(0, np.nan), axis=0)

    nat_votes = df[buckets].sum()
    nat_shares = (nat_votes / nat_votes.sum()).replace(0, np.nan)

    coef = g_shares.div(nat_shares, axis=1).replace([np.inf, -np.inf], np.nan)
    return coef


def clip_and_fill_coef(coef: pd.DataFrame, lo: float = 0.05, hi: float = 20.0) -> pd.DataFrame:
    return coef.clip(lower=lo, upper=hi).fillna(1.0)
