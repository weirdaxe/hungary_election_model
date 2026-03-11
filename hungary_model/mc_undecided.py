from __future__ import annotations

import numpy as np

from .types import MonteCarloConfig


def sample_undecided_split(base_u_fidesz: float, mc: MonteCarloConfig, rng: np.random.Generator) -> float:
    """Sample the share of voting undecideds going to Fidesz (0..1)."""
    sigma = float(getattr(mc, "undecided_split_sigma", 0.0) or 0.0)
    u = float(base_u_fidesz) + rng.normal(0.0, sigma)
    return float(np.clip(u, 0.0, 1.0))
