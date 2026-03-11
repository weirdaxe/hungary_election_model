
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def determine_winner_row(v: pd.Series, two_party_only: bool = True) -> Tuple[str, str]:
    """Determine winner and runner-up in an EVK."""
    if two_party_only:
        a = v[["FIDESZ", "TISZA"]]
        winner = a.idxmax()
        runner = a.drop(index=winner).idxmax()
        return winner, runner
    winner = v.idxmax()
    runner = v.drop(index=winner).idxmax()
    return winner, runner


def dhondt_alloc(votes: pd.Series, seats: int) -> Tuple[pd.Series, pd.DataFrame]:
    """
    D'Hondt allocation returning:
      - seat counts
      - quotient table (sorted) with seat-winner flags
    """
    parties = votes.index.tolist()
    rows = []
    for p in parties:
        v = float(votes[p])
        for d in range(1, int(seats) + 1):
            rows.append({"party": p, "divisor": d, "quotient": v / d})
    q = pd.DataFrame(rows).sort_values("quotient", ascending=False).reset_index(drop=True)
    q["wins_seat"] = False
    if seats > 0 and len(q) > 0:
        q.loc[: int(seats) - 1, "wins_seat"] = True
    out = pd.Series(0, index=parties, dtype=int)
    for _, r in q.head(int(seats)).iterrows():
        out[r["party"]] += 1
    return out, q
