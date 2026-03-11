"""Aggregation helpers.

This module exists to centralize common aggregation utilities that are used
across scenario, Monte Carlo, and backtests.

Notes
-----
We intentionally support both:
- station-level vote *Series* (e.g. turnout totals)
- station-level vote *DataFrames* with party columns

so callers don't need to duplicate EVK aggregation logic.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from .constants import MODEL_PARTIES


def aggregate_station_to_evk(
    station_votes: Union[pd.Series, pd.DataFrame],
    station_meta: pd.DataFrame,
) -> Union[pd.Series, pd.DataFrame]:
    """Aggregate station-level objects to EVK.

    Parameters
    ----------
    station_votes:
        Either a Series indexed by station_id (e.g. turnout vote totals)
        or a DataFrame indexed by station_id with party columns.

    station_meta:
        DataFrame indexed by station_id with an 'evk_id' column.

    Returns
    -------
    Series or DataFrame indexed by evk_id.

    Raises
    ------
    ValueError
        If 'evk_id' is missing from station_meta.
    TypeError
        If station_votes is not a Series or DataFrame.
    """

    if "evk_id" not in station_meta.columns:
        raise ValueError("station_meta must contain an 'evk_id' column")

    evk_map = station_meta["evk_id"].reindex(station_votes.index)

    if isinstance(station_votes, pd.Series):
        out = station_votes.groupby(evk_map).sum()
        out = out[out.index.notna()].copy()
        out.index.name = "evk_id"
        return out

    if isinstance(station_votes, pd.DataFrame):
        df = station_votes.copy()
        df["evk_id"] = evk_map.values
        # Only sum party columns that exist
        party_cols = [c for c in MODEL_PARTIES if c in df.columns]
        out = df.groupby("evk_id")[party_cols].sum()
        out = out.loc[out.index.notna()].copy()
        out.index.name = "evk_id"
        return out

    raise TypeError(f"station_votes must be a pandas Series or DataFrame, got {type(station_votes)}")
