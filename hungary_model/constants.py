
from __future__ import annotations

from typing import Dict, List

# ============================================================
# Public constants
# ============================================================

# Standardized historical BLOCK buckets (used for coefficients)
BLOCK_BUCKETS: List[str] = ["FIDESZ", "OPP", "MH", "MKKP", "OTHER"]

# 2026 output parties (seat calculator output layer)
MODEL_PARTIES: List[str] = ["FIDESZ", "TISZA", "DK", "MH", "MKKP", "OTHER"]

DB_REQUIRED_FILES: List[str] = [
    "national_result_2018.db",
    "national_result_2022.db",
    "single_constituency_result_2018.db",
    "single_constituency_result_2022.db",
    "ep_results_2019.db",
    "ep_results_2024.db",
]

POLL_REQUIRED_FILES: List[str] = [
    "2018_poll_hungary.csv",
    "2019_poll_hungary.csv",
    "2022_poll_hungary.csv",
    "2024_poll_hungary.csv",
    "2026_poll_hungary.csv",
]

# EVK (single constituency) mapping DB for 2026
# The repository/user may provide one of these files in the DB directory.
EVK_MAPPING_DB_CANDIDATES: List[str] = [
    "vtr_ogy2026_fffffff.db",
    "vtr_ogy2026_mapping.db",
    "evk_mapping_2026.db",
]

# Turnout modelling
TURNOUT_MODELS: List[str] = [
    "uniform",
    "scaled_reference",
    "logit_slope_original",
    "logit_slope_ep_offset",
    "relative_offset_parl",
    "logit_slope_reserve_adjusted",
    "baseline_plus_marginal",
    "baseline_ref_plus_elastic_delta",
]

TURNOUT_GRANULARITIES: Dict[str, str] = {
    "station": "station_id",
    "location": "station_name_id",
    "settlement": "settlement_id",
    "evk": "evk_id",
}
