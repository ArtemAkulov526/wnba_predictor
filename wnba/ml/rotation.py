"""Who plays how much: appearance-weighted minutes and the playoff rotation.

Shared by the props projections (wnba.ml.props) and the injury adjustment of
the winner model (wnba.ml.predict.missing_share).

Two corrections to naive "average minutes":

1. Appearance weighting. A player's minutes are averaged over the games she
   PLAYED, then weighted by how often she appeared in the team's recent games.
   Without this, garbage-time players who show up in 3 of 10 games take a full
   slice of the 200 team minutes and starters come out ~10% low.

2. Playoff rotation. Rotations shorten in the playoffs. Measured on 64
   team-playoff runs (2018-2025), minutes per game in the playoffs vs the last
   10 regular-season games, by regular-season rotation rank:

       rank   1     2     3     4     5     6     7     8     9     10
       mult  1.06  1.08  1.06  1.00  1.07  1.00  0.91  0.82  0.65  0.76

   Smoothed below to be non-increasing down the rotation. Once a team has
   PLAYOFF_MIN_GAMES playoff games, its own playoff minutes are used instead.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
from wnba.dates import us_today

RECENT_TEAM_GAMES = 10
PLAYOFF_MIN_GAMES = 3
MINUTES_HALFLIFE = 5
PLAYOFF_SEASON_TYPE = 3

PLAYOFF_ROTATION_MULT = {1: 1.07, 2: 1.07, 3: 1.07, 4: 1.03, 5: 1.03, 6: 1.00,
                         7: 0.91, 8: 0.82}
PLAYOFF_DEEP_BENCH_MULT = 0.70      # rank 9 and below


# Last regular-season day per season. Needed because postseason box scores
# reach the data a day or more after the games, so "any playoff rows yet?"
# is still False when predicting game 2. Add each season's date once known.
REGULAR_SEASON_END = {2026: date(2026, 9, 24)}


def is_postseason(game_date: Optional[date] = None,
                  player_logs: Optional[pd.DataFrame] = None) -> bool:
    """True for a game after its season's last regular-season day, or once
    postseason games of the latest season appear in the data."""
    d = game_date or us_today()
    end = REGULAR_SEASON_END.get(d.year)
    if end is not None and d > end:
        return True
    return playoffs_underway(player_logs) if player_logs is not None else False


def playoffs_underway(player_logs: pd.DataFrame) -> bool:
    """True once any postseason game of the latest season is in the data."""
    if player_logs.empty or "season_type" not in player_logs:
        return False
    latest = player_logs["season"].max()
    return bool(((player_logs["season"] == latest)
                 & (player_logs["season_type"] == PLAYOFF_SEASON_TYPE)).any())


def rotation_minutes(team_logs: pd.DataFrame, playoff: bool) -> pd.DataFrame:
    """Per player: P(appears), minutes if she plays, and expected minutes.

    team_logs: one team's player rows for the current season, minutes > 0,
    with columns date, athlete_id, minutes, season_type.
    Returns a frame indexed by athlete_id with p_appear, min_if_play,
    expected (= p_appear * min_if_play) and basis.
    """
    t = team_logs.sort_values("date")
    playoff_dates = sorted(t.loc[t["season_type"] == PLAYOFF_SEASON_TYPE, "date"].unique())
    use_playoff_games = playoff and len(playoff_dates) >= PLAYOFF_MIN_GAMES
    dates = (playoff_dates if use_playoff_games else sorted(t["date"].unique()))[-RECENT_TEAM_GAMES:]
    window = t[t["date"].isin(dates)]

    g = window.groupby("athlete_id")
    out = pd.DataFrame({
        "p_appear": g.size() / len(dates),
        "min_if_play": g["minutes"].apply(
            lambda s: s.ewm(halflife=MINUTES_HALFLIFE).mean().iloc[-1]),
    })
    out["expected"] = out["p_appear"] * out["min_if_play"]
    out["basis"] = "playoff games" if use_playoff_games else "recent games"

    if playoff and not use_playoff_games:
        rank = out["expected"].rank(ascending=False, method="first").astype(int)
        mult = rank.map(lambda r: PLAYOFF_ROTATION_MULT.get(r, PLAYOFF_DEEP_BENCH_MULT))
        out["min_if_play"] *= mult
        out["expected"] *= mult
        out["basis"] = "recent games x playoff rotation"
    return out
