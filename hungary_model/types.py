
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import pandas as pd


# ============================================================
# Paths + data containers
# ============================================================

@dataclass(frozen=True)
class ModelPaths:
    """Resolved paths to input data."""
    db_dir: Path
    poll_dir: Path
    evk_mapping_db: Optional[Path] = None  # 2026 EVK mapping for polling stations (optional but recommended)

    def db_files(self) -> Dict[str, Path]:
        return {
            "parl_list_2018": self.db_dir / "national_result_2018.db",
            "parl_smc_2018":  self.db_dir / "single_constituency_result_2018.db",
            "parl_list_2022": self.db_dir / "national_result_2022.db",
            "parl_smc_2022":  self.db_dir / "single_constituency_result_2022.db",
            "ep_2019":        self.db_dir / "ep_results_2019.db",
            "ep_2024":        self.db_dir / "ep_results_2024.db",
        }

    def poll_files(self) -> Dict[str, Path]:
        return {
            "2018": self.poll_dir / "2018_poll_hungary.csv",
            "2019": self.poll_dir / "2019_poll_hungary.csv",
            "2022": self.poll_dir / "2022_poll_hungary.csv",
            "2024": self.poll_dir / "2024_poll_hungary.csv",
            "2026": self.poll_dir / "2026_poll_hungary.csv",
        }


@dataclass
class ModelData:
    paths: ModelPaths

    # Station-wide tables (imputed)
    parl18_list_i: pd.DataFrame
    parl22_list_i: pd.DataFrame
    ep19_i: pd.DataFrame
    ep24_i: pd.DataFrame

    # Station metadata (index station_id)
    # Required columns: station_name_id, settlement_id, county_id, evk_id
    # Optional: evk_name (2026 constituency name)
    station_meta: pd.DataFrame

    # EVK metadata (index evk_id), optional but preferred for UI
    evk_meta: pd.DataFrame  # columns: evk_name

    # Registered voters baseline for 2026 (station-level)
    registered_2026: pd.Series  # index station_id

    # Coefficients by election (BLOCK space), multi-level
    coef_by_election: Dict[str, Dict[str, pd.DataFrame]]

    # Same coefficient objects, but computed with Budapest excluded from the
    # national baseline (county_id != '01'). Used when running the model in
    # "Exclude Budapest" mode.
    coef_by_election_no_budapest: Dict[str, Dict[str, pd.DataFrame]]

    # Turnout panel (station-level, 4 elections)
    turnout_panel: pd.DataFrame  # columns include station_id, group cols, election, is_ep, turnout_rate, registered, votes

    nat_turnout: pd.Series       # index election -> national turnout rate
    nat_turnout_logit: pd.Series # index election -> logit(national turnout)

    # Turnout model components
    elasticity_station_original: pd.DataFrame  # index station_id; columns alpha, beta
    elasticity_ep_offset: Dict[str, pd.DataFrame]  # granularity -> (alpha,beta,gamma)
    offset_parl: Dict[str, pd.Series]              # granularity -> logit offset
    marginal_propensity: Dict[str, pd.Series]      # granularity -> Parl-EP gap proxy (0..)

    baseline_turnout_station: pd.Series  # station-level baseline turnout map (parl_2022_list)

    # Polls
    polls_all: pd.DataFrame

    # Actual national BLOCK shares for backtests
    # Shares among VALID votes: FIDESZ, CHALLENGER, OTHER
    actual_block_shares_valid: Dict[int, pd.Series]
    # Shares among REGISTERED: FIDESZ, CHALLENGER, OTHER, UNDECIDED (nonvoters)
    actual_block_shares_population: Dict[int, pd.Series]


# ============================================================
# Scenario + results
# ============================================================

@dataclass
class ScenarioConfig:
    # Geographic scope toggle
    # - False (default): run on full Hungary
    # - True: treat poll inputs as "Hungary ex Budapest" and add Budapest back via
    #         a simple historical estimate; also force Budapest SMC winners to TISZA.
    exclude_budapest: bool = False

    # turnout (registered voter share)
    turnout_target: float = 0.72

    # turnout model selection (per notebook v6)
    turnout_model: str = "logit_slope_ep_offset"     # one of TURNOUT_MODELS
    turnout_granularity: str = "location"            # station|location|settlement|evk
    turnout_reference_election: str = "parl_2022_list"
    # baseline turnout level used by turnout_model="baseline_ref_plus_elastic_delta"
    turnout_baseline_level: float = 0.70
    turnout_clip: Tuple[float, float] = (0.20, 0.95)
    reserve_adjust_power: float = 1.0
    marginal_concentration: float = 1.0

    # polls
    poll_type: str = "raw"                 # 'raw' | 'decided' | 'pártválasztók'
    last_n_days: int = 60
    pollster_filter: Optional[List[str]] = None
    poll_decay_half_life_days: float = 0.0  # 0 => equal weights
    manual_poll_override: Optional[Dict[str, float]] = None  # percent points

    # pool not in polls and not voting (added to undecided pool in raw interpretation)
    nonresponse_nonvoter_pct: float = 0.10

    # undecided split (among voting undecideds)
    undecided_to_fidesz: float = 0.40
    undecided_to_tisza: float = 0.60

    # coefficient blending
    election_weights: Optional[Dict[str, float]] = None
    geo_weights: Optional[Dict[str, float]] = None

    # elasticity→geo tilt (used in IPF initial matrix) AND in undecided_geo_model='elasticity'
    undecided_elasticity_link_strength: float = 0.0

    # mobilization model (raw polls only): converts population shares to voter shares endogenously
    use_mobilization_model: bool = True

    # If True, allow mobilization/reserve parameters for ALL parties (not only Fidesz/Tisza).
    # This is primarily used by the baseline+marginal model. For IPF, national totals are fixed.
    mobilization_all_parties: bool = False

    mobilization_rates: Optional[Dict[str, float]] = None  # party -> [0..1]
    reserve_strength: Optional[Dict[str, float]] = None    # party -> multiplier

    # vote allocation model
    vote_allocation_model: str = "ipf"  # 'ipf' | 'baseline_marginal'
    undecided_geo_model: str = "uniform"  # uniform | elasticity | low_turnout (baseline+marginal)
    undecided_local_lean_strength: float = 0.0

    # SMC correction
    smc_plus1_minus1: bool = True
    winner_two_party_only: bool = True

    # diaspora votes (list only; absolute vote counts)
    diaspora_votes: Optional[Dict[str, int]] = None

    # nationality seat toggle (takes 1 of list seats)
    nationality_seat_to_fidesz: bool = True

    # list threshold
    threshold: float = 0.05

    # modelling unit: 'station' (exact) or 'evk' (faster). baseline_marginal requires 'station'.
    modelling_unit: str = "station"


@dataclass
class ScenarioResults:
    cfg: ScenarioConfig

    # polls
    avg_poll_2026: pd.Series               # percent points
    population_shares_2026: pd.Series      # includes UNDECIDED_TRUE if raw
    national_shares_2026: pd.Series        # vote shares among voters (MODEL_PARTIES)

    # turnout
    turnout_rate_station: pd.Series        # station turnout rate prediction
    votes_station: pd.Series               # station vote totals (appeared)

    # coefficients (MODEL_PARTIES space; index=unit ids)
    coefs_2026: pd.DataFrame

    # list/SMC unit matrices (votes)
    unit_list_votes: pd.DataFrame          # index = unit ids; cols MODEL_PARTIES
    evk_list_votes: pd.DataFrame           # index evk_id; cols MODEL_PARTIES
    evk_smc_votes: pd.DataFrame            # index evk_id; cols MODEL_PARTIES

    # EVK winners (index evk_id)
    winners: pd.DataFrame                  # columns winner, runner_up, margin_votes, abs_margin_votes, evk_name(optional)

    # compensation breakdown
    loser_comp: pd.Series
    winner_comp: pd.Series
    total_comp: pd.Series

    # list totals
    domestic_list_votes: pd.Series
    diaspora_votes: pd.Series
    list_total_votes: pd.Series
    list_shares: pd.Series
    eligible_parties: List[str]

    # seats
    smc_seats: pd.Series
    list_seats: pd.Series
    nationality_seats: pd.Series
    seats_table: pd.DataFrame

    # D'Hondt quotients (top ranks)
    dhondt_table: pd.DataFrame


# ============================================================
# Monte Carlo
# ============================================================


NationalShareSampling = Literal["gaussian", "multinomial", "dirichlet_multinomial"]


ParallelBackend = Literal["auto", "threads", "processes"]


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations.

    Notes on distributions
    ----------------------
    Several fields accept a *distribution spec* dictionary. Supported forms:

    - Fixed value:
        {"dist": "fixed", "value": 0.72}

    - Uniform range:
        {"dist": "uniform", "min": 0.68, "max": 0.76}

    - Truncated normal:
        {"dist": "normal", "mean": 0.72, "sd": 0.01, "min": 0.20, "max": 0.95}

    If a spec is omitted, the model falls back to the scenario value (or to
    legacy *_sigma_pp fields where applicable).
    """

    # core
    n_sims: int = 10_000
    # Parallel backend:
    # - "threads": safer in Streamlit/Windows (no pickling issues)
    # - "processes": faster for heavy CPU but requires picklable payload
    # - "auto": choose a sensible default in the runner
    backend: ParallelBackend = "auto"
    n_workers: int = 0
    chunk_size: int = 500
    seed: int = 1234

    # modelling unit override for MC (None -> use scenario)
    force_modelling_unit: Optional[str] = None

    # --------------------------------------------------------
    # Turnout uncertainty
    # --------------------------------------------------------

    # If provided, MC will sample turnout targets using this spec.
    # Otherwise it falls back to legacy turnout_sigma_pp around the scenario target.
    turnout_target_spec: Optional[Dict[str, float]] = None

    # Legacy (backward compatible): SD in percentage points (pp)
    turnout_sigma_pp: float = 0.5

    # Optional: sample turnout model / granularity from these lists.
    # If None/empty -> use the scenario values.
    turnout_models: Optional[List[str]] = None
    turnout_granularities: Optional[List[str]] = None

    # --------------------------------------------------------
    # Poll / national share sampling
    # --------------------------------------------------------

    # Whether to treat poll inputs as the scenario poll type, or force decided/raw.
    poll_mode: Literal["scenario", "decided", "raw"] = "scenario"

    # How to sample poll/national shares
    national_share_sampling: NationalShareSampling = "multinomial"

    # Gaussian sampling (SD in pp)
    nat_sigma_pp: float = 1.0

    # Multinomial / Dirichlet-multinomial sampling (effective sample size)
    poll_n: int = 1200

    # Dirichlet-multinomial overdispersion control.
    # Larger values -> closer to multinomial. Typical values: 50..500.
    dirichlet_concentration: float = 200.0

    # --------------------------------------------------------
    # Undecided / nonresponse (only meaningful for raw poll mode)
    # --------------------------------------------------------

    # Additional "not in polls and not voting" share added to undecided (as fraction of registered)
    nonresponse_nonvoter_spec: Optional[Dict[str, float]] = None

    # Share of undecided voters who choose FIDESZ (remainder to TISZA unless reserve tilt redirects)
    undecided_to_fidesz_spec: Optional[Dict[str, float]] = None

    # --------------------------------------------------------
    # Mobilization / reserve tilt (raw poll -> voter shares)
    # --------------------------------------------------------

    # Optionally sample whether mobilization tilt is enabled.
    # If None -> follow scenario config.
    use_mobilization_choices: Optional[List[bool]] = None

    mobilization_rate_fidesz_spec: Optional[Dict[str, float]] = None
    mobilization_rate_tisza_spec: Optional[Dict[str, float]] = None

    reserve_strength_fidesz_spec: Optional[Dict[str, float]] = None
    reserve_strength_tisza_spec: Optional[Dict[str, float]] = None

    # --------------------------------------------------------
    # Coefficient uncertainty
    # --------------------------------------------------------

    # Multiplicative noise applied to coefficients (log-space SD)
    coef_log_sigma: float = 0.08

    # Sample election weights and/or geography weights.
    # Each entry is a [min,max] interval for the weight; sampled weights are renormalized to sum to 1.
    election_weight_minmax: Optional[Dict[str, List[float]]] = None
    geo_weight_minmax: Optional[Dict[str, List[float]]] = None

    # --------------------------------------------------------
    # Diaspora / mail votes uncertainty
    # --------------------------------------------------------

    # Total mail-in votes (absolute count)
    diaspora_total_spec: Optional[Dict[str, float]] = None

    # FIDESZ share of mail-in votes (0..1). Remainder goes to TISZA.
    diaspora_fidesz_share_spec: Optional[Dict[str, float]] = None

    diaspora_log_sigma: float = 0.10  # legacy multiplicative noise on existing diaspora vector

    # --------------------------------------------------------
    # IPF controls
    # --------------------------------------------------------

    ipf_max_iter: int = 250
    ipf_tol: float = 1e-4


@dataclass
class MonteCarloResults:
    cfg: MonteCarloConfig

    # Seats by party for each draw (rows = sims, cols = parties)
    seat_draws: pd.DataFrame

    # Summary statistics over draws (mean, p05, p50, p95)
    seat_summary: pd.DataFrame

    # Probability of > 100 seats for each party
    prob_majority: pd.Series

    # Pairwise winner probabilities (P(seats_i > seats_j))
    prob_winner: pd.DataFrame

    # National vote share draws (among *domestic* valid voters)
    nat_share_draws: pd.DataFrame

    # Turnout target draws
    turnout_draws: pd.Series

    # Per-draw sampled inputs used in the simulation (one row per draw)
    input_draws: pd.DataFrame

    # Optional: EVK-level winner draws (rows=sims, cols=evk_id, values=party)
    evk_winner_draws: Optional[pd.DataFrame] = None

    # Convenience: doom scenario probabilities
    # - FIDESZ majority: FIDESZ >= 100 seats
    # - FIDESZ+MH majority: (FIDESZ + MH) >= 100 seats
    # - any: either of the above
    doom_prob_fidesz_majority: float = 0.0
    doom_prob_fidesz_mh_majority: float = 0.0
    doom_prob_any: float = 0.0
