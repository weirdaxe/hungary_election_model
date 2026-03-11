from __future__ import annotations

import numpy as np

from .constants import MODEL_PARTIES
from .mc_distributions import sample_float, sample_int
from .types import MonteCarloConfig


def sample_diaspora_votes_with_meta(
    base_diaspora: np.ndarray, mc: MonteCarloConfig, rng: np.random.Generator
) -> tuple[np.ndarray, int, float]:
    """Sample diaspora/mail-in vote vector and return the realized assumptions.

    Returns
    -------
    votes : np.ndarray
        Length = n_parties (MODEL_PARTIES), vote counts (float).
    diaspora_total : int
        Realized total diaspora/mail votes used in the draw (rounded to int).
    diaspora_fidesz_share : float
        Realized FIDESZ share of diaspora votes (0..1). If total==0 -> NaN.

    Notes
    -----
    Two modes:

    1) If diaspora_total_spec or diaspora_fidesz_share_spec is provided, we treat the
       diaspora as *only* FIDESZ vs TISZA:
         - total mail votes is sampled (absolute)
         - FIDESZ share is sampled (0..1)
         - remainder goes to TISZA
         - other parties get 0

    2) Otherwise, fall back to legacy multiplicative log-normal noise around the
       scenario's diaspora vector.
    """

    base = np.asarray(base_diaspora, dtype=float)

    idx_f = MODEL_PARTIES.index("FIDESZ")
    idx_t = MODEL_PARTIES.index("TISZA")

    # Explicit total + share mode
    if (mc.diaspora_total_spec is not None) or (mc.diaspora_fidesz_share_spec is not None):
        base_total = float(base.sum())
        base_total_int = int(round(base_total))

        base_share = 0.75
        if base_total > 0:
            base_share = float(base[idx_f] / base_total)

        total = sample_int(rng, mc.diaspora_total_spec, base_total_int, hard_min=0)
        share_f = sample_float(rng, mc.diaspora_fidesz_share_spec, base_share, hard_min=0.0, hard_max=1.0)

        f_votes = int(round(total * share_f))
        t_votes = int(total - f_votes)

        out = np.zeros_like(base, dtype=float)
        out[idx_f] = float(f_votes)
        out[idx_t] = float(t_votes)

        # If total==0, share_f is still whatever was drawn; keep it for diagnostics.
        return out, int(total), float(share_f)

    # Legacy mode: multiplicative noise on full vector
    sigma = float(mc.diaspora_log_sigma)
    if sigma <= 0:
        out = base
    else:
        noise = rng.normal(0.0, sigma, size=base.shape)
        out = base * np.exp(noise)
        out = np.clip(out, 0.0, None)

    total_f = float(out.sum())
    if total_f > 0:
        share_f = float(out[idx_f] / total_f)
    else:
        share_f = float("nan")
    return out, int(round(total_f)), share_f


def sample_diaspora_votes(base_diaspora: np.ndarray, mc: MonteCarloConfig, rng: np.random.Generator) -> np.ndarray:
    """Backward-compatible wrapper: return only the vote vector."""

    votes, _, _ = sample_diaspora_votes_with_meta(base_diaspora, mc, rng)
    return votes
