from __future__ import annotations

import numpy as np

from .types import MonteCarloConfig


def _clip_and_normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(p) / max(1, len(p))
    return p / s


def sample_share_vector(base_probs: np.ndarray, mc: MonteCarloConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample a probability vector.

    Used for both:
      - national vote shares among voters (decided mode)
      - raw poll shares including UNDECIDED (raw mode)

    The sampling method is controlled by mc.national_share_sampling.
    """

    base = _clip_and_normalize(np.asarray(base_probs, dtype=float))

    method = str(mc.national_share_sampling)

    if method == "gaussian":
        # Add noise in percentage points then renormalize
        sigma = float(mc.nat_sigma_pp) / 100.0
        draw = base + rng.normal(0.0, sigma, size=base.shape)
        return _clip_and_normalize(draw)

    if method == "dirichlet_multinomial":
        n = int(mc.poll_n)
        conc = float(getattr(mc, "dirichlet_concentration", 200.0))
        conc = max(conc, 1e-6)
        alpha = base * conc
        # Dirichlet requires strictly positive alphas
        alpha = np.clip(alpha, 1e-6, None)
        p_draw = rng.dirichlet(alpha)
        counts = rng.multinomial(max(1, n), p_draw)
        return counts / counts.sum()

    # default: multinomial
    n = int(mc.poll_n)
    counts = rng.multinomial(max(1, n), base)
    return counts / counts.sum()
