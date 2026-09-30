"""Elo ratings for WNBA teams.

Elo is a strong, cheap baseline: it updates game by game, needs nothing but
results, and is naturally symmetric. It also doubles as the sanity check for
the full model -- if the box-score features cannot beat Elo plus home court,
they are not earning their place.

Ratings are produced strictly chronologically, and the rating attached to a
game is the one computed BEFORE that game was played, so nothing leaks.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
# ~70 Elo points is the standard home edge; the WNBA home win rate sits a
# little above the NBA's, and this is re-fit implicitly by the logistic layer.
HOME_ADVANTAGE = 70.0
# Carry ratings across seasons but pull them a quarter of the way back to the
# mean: rosters churn, and expansion teams should not inherit a void.
SEASON_REGRESSION = 0.25


def expected_score(rating: float, opponent_rating: float) -> float:
    """Probability `rating` beats `opponent_rating` (home edge already applied)."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def _mov_multiplier(margin: int, rating_diff: float) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier.

    Blowouts move ratings more than one-point games, but the denominator damps
    the effect for teams that were already heavy favourites, which stops good
    teams from running away with the rating.
    """
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * rating_diff)


def compute_elo(
    games: pd.DataFrame,
    k: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
    season_regression: float = SEASON_REGRESSION,
) -> pd.DataFrame:
    """Walk games in date order and return pre-game Elo for both sides.

    Args:
        games: one row per GAME (not per team-game) with columns
            game_key, date, season, home_team_id, away_team_id,
            home_pts, away_pts, home_won.

    Returns:
        DataFrame keyed by game_key with home_elo_pre / away_elo_pre, i.e.
        each team's rating going into that game.
    """
    games = games.sort_values(["date", "game_key"]).reset_index(drop=True)

    ratings: Dict[int, float] = {}
    records: List[dict] = []
    current_season: int | None = None

    for row in games.itertuples(index=False):
        if current_season is not None and row.season != current_season:
            # New season: regress every rating toward the league mean.
            for team_id, rating in ratings.items():
                ratings[team_id] = (
                    rating * (1.0 - season_regression)
                    + INITIAL_RATING * season_regression
                )
        current_season = row.season

        home = ratings.setdefault(row.home_team_id, INITIAL_RATING)
        away = ratings.setdefault(row.away_team_id, INITIAL_RATING)

        records.append({
            "game_key": row.game_key,
            "home_elo_pre": home,
            "away_elo_pre": away,
        })

        # Update using the rating the home team effectively plays at.
        home_effective = home + home_advantage
        expected_home = expected_score(home_effective, away)
        actual_home = 1.0 if row.home_won else 0.0

        margin = int(row.home_pts) - int(row.away_pts)
        # Rating gap from the winner's perspective, per 538's formulation.
        winner_diff = (home_effective - away) if margin > 0 else (away - home_effective)
        multiplier = _mov_multiplier(margin, winner_diff)

        change = k * multiplier * (actual_home - expected_home)
        ratings[row.home_team_id] = home + change
        ratings[row.away_team_id] = away - change

    return pd.DataFrame.from_records(records)


def final_ratings(games: pd.DataFrame, **kwargs) -> Dict[int, float]:
    """Ratings after every game in `games` -- used to predict upcoming games."""
    games = games.sort_values(["date", "game_key"]).reset_index(drop=True)
    pre = compute_elo(games, **kwargs).set_index("game_key")

    k = kwargs.get("k", K_FACTOR)
    home_advantage = kwargs.get("home_advantage", HOME_ADVANTAGE)

    ratings: Dict[int, float] = {}
    for row in games.itertuples(index=False):
        home = pre.at[row.game_key, "home_elo_pre"]
        away = pre.at[row.game_key, "away_elo_pre"]
        ratings[row.home_team_id] = home
        ratings[row.away_team_id] = away

        home_effective = home + home_advantage
        expected_home = expected_score(home_effective, away)
        margin = int(row.home_pts) - int(row.away_pts)
        winner_diff = (home_effective - away) if margin > 0 else (away - home_effective)
        change = k * _mov_multiplier(margin, winner_diff) * (
            (1.0 if row.home_won else 0.0) - expected_home)
        ratings[row.home_team_id] = home + change
        ratings[row.away_team_id] = away - change

    return ratings
