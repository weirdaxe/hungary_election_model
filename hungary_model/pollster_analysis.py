
from __future__ import annotations

from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd

from .types import ModelData
from .polls import weighted_mean


AggLevel = Literal["pollster", "year", "overall"]


def _actual_targets_pp(data: ModelData, year: int, poll_type: str) -> pd.Series:
    """
    Choose the appropriate 'actual' target vector depending on poll_type.

    - raw polls are treated as population shares, so we compare against
      actual_block_shares_population (which includes UNDECIDED_TRUE).
    - decided polls are compared against actual_block_shares_valid (shares among valid votes).

    Returned series is in percent points.
    """
    pt = str(poll_type).strip().lower()
    if pt == "raw":
        actual = data.actual_block_shares_population.get(int(year))
        if actual is None or actual.empty:
            return pd.Series(dtype=float)
        out = actual.copy() * 100.0
        # Normalize naming for the output table
        if "UNDECIDED_TRUE" in out.index and "UNDECIDED" not in out.index:
            out = out.rename({"UNDECIDED_TRUE": "UNDECIDED"})
        return out
    else:
        actual = data.actual_block_shares_valid.get(int(year))
        if actual is None or actual.empty:
            return pd.Series(dtype=float)
        out = actual.copy() * 100.0
        # In decided comparison, undecided is by definition 0 in the target.
        out["UNDECIDED"] = 0.0
        return out


def pollster_bias_table(
    data: ModelData,
    year: int,
    poll_type: str = "raw",
    last_n_days: int = 60,
) -> pd.DataFrame:
    """
    Backtest pollster averages vs actual election result in BLOCK space.

    Output columns:
      - FIDESZ_bias_pp, CHALLENGER_bias_pp, OTHER_bias_pp, UNDECIDED_bias_pp
      - MSE_pp (computed using FIDESZ and CHALLENGER only)
      - MAE_pp (computed using FIDESZ and CHALLENGER only)

    Notes:
      - For poll_type='raw', UNDECIDED is evaluated against UNDECIDED_TRUE (share of registered without a valid vote).
      - For poll_type != 'raw', UNDECIDED target is 0.
    """
    df = data.polls_all[
        (data.polls_all["year"] == int(year))
        & (data.polls_all["poll_type"] == str(poll_type).strip().lower())
    ].copy()
    if df.empty:
        return pd.DataFrame()

    max_date = df["date"].max()
    start = max_date - pd.Timedelta(days=int(last_n_days))
    df = df[df["date"] >= start]
    if df.empty:
        return pd.DataFrame()

    actual_pp = _actual_targets_pp(data, int(year), poll_type=str(poll_type))
    if actual_pp.empty:
        return pd.DataFrame()

    cols = [c for c in ["FIDESZ", "CHALLENGER", "OTHER", "UNDECIDED"] if c in df.columns]

    out_rows = []
    for pollster, g in df.groupby("pollster"):
        w = pd.Series(1.0, index=g.index)

        f = weighted_mean(g.get("FIDESZ"), w) if "FIDESZ" in cols else float("nan")
        ch = weighted_mean(g.get("CHALLENGER"), w) if "CHALLENGER" in cols else float("nan")
        oth = weighted_mean(g.get("OTHER"), w) if "OTHER" in cols else float("nan")
        u = weighted_mean(g.get("UNDECIDED"), w) if "UNDECIDED" in cols else float("nan")

        # Back out missing components if possible (conservative).
        # Priority: infer CHALLENGER if absent (many poll formats only give Fidesz + Other + Undecided).
        if np.isnan(ch) and not np.isnan(f):
            comps = []
            if not np.isnan(oth):
                comps.append(oth)
            if not np.isnan(u):
                comps.append(u)
            ch = max(0.0, 100.0 - f - sum(comps))

        # If OTHER missing but F and CH known, infer OTHER as leftover (ignoring undecided if missing).
        if np.isnan(oth) and (not np.isnan(f)) and (not np.isnan(ch)):
            comps = [f, ch]
            if not np.isnan(u):
                comps.append(u)
            oth = max(0.0, 100.0 - sum(comps))

        pred = pd.Series({"FIDESZ": f, "CHALLENGER": ch, "OTHER": oth, "UNDECIDED": u}).astype(float)

        # Bias vectors (pp)
        err = (pred - actual_pp.reindex(pred.index)).rename(lambda k: f"{k}_bias_pp")

        # Score using F + Challenger only, as requested
        err_fc = np.array([err.get("FIDESZ_bias_pp", np.nan), err.get("CHALLENGER_bias_pp", np.nan)], dtype=float)
        mse = float(np.nanmean(err_fc ** 2))
        mae = float(np.nanmean(np.abs(err_fc)))

        out_rows.append(
            {
                "pollster": pollster,
                "n_polls": int(len(g)),
                **err.to_dict(),
                "FIDESZ_abs_err_pp": float(abs(err.get("FIDESZ_bias_pp", float("nan")))),
                "CHALLENGER_abs_err_pp": float(abs(err.get("CHALLENGER_bias_pp", float("nan")))),
                "UNDECIDED_abs_err_pp": float(abs(err.get("UNDECIDED_bias_pp", float("nan")))),
                "MAE_pp": mae,
                "MSE_pp": mse,
                "RMSE_pp": float(np.sqrt(mse)) if not np.isnan(mse) else np.nan,
            }
        )

    out = pd.DataFrame(out_rows)

    # Sort by RMSE (preferred), then MAE as tie-breaker
    if not out.empty:
        out = out.sort_values(["RMSE_pp", "MAE_pp"], ascending=[True, True]).reset_index(drop=True)
    return out


def pollster_bias_panel(
    data: ModelData,
    years: Iterable[int],
    poll_type: str = "raw",
    last_n_days: int = 60,
) -> pd.DataFrame:
    """Convenience: concat pollster_bias_table across years."""
    rows = []
    for y in years:
        t = pollster_bias_table(data, year=int(y), poll_type=str(poll_type), last_n_days=int(last_n_days))
        if t is not None and not t.empty:
            t = t.copy()
            t["year"] = int(y)
            rows.append(t)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_pollster_bias(panel: pd.DataFrame, by: AggLevel = "pollster") -> pd.DataFrame:
    """
    Aggregate a pollster bias panel.

    by='pollster': average metrics across years (each pollster-year equally weighted).
    by='year': average metrics across pollsters in the panel.
    by='overall': average across all rows.

    This is intentionally simple; if you want poll-count weighting, add it here.
    """
    if panel is None or panel.empty:
        return pd.DataFrame()

    metric_cols = [c for c in panel.columns if c.endswith("_bias_pp") or c in ["MAE_pp", "MSE_pp", "RMSE_pp"]]

    if by == "pollster":
        out = panel.groupby("pollster", as_index=False)[metric_cols].mean()
        return out.sort_values(["RMSE_pp", "MAE_pp"], ascending=[True, True]).reset_index(drop=True)

    if by == "year":
        out = panel.groupby("year", as_index=False)[metric_cols].mean()
        return out.sort_values(["RMSE_pp", "MAE_pp"], ascending=[True, True]).reset_index(drop=True)

    # overall
    d = panel[metric_cols].mean(numeric_only=True).to_frame("avg").T
    d.insert(0, "scope", "overall")
    return d