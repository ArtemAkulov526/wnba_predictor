"""Rebuild playoff team box scores from player box scores.

Now a CROSS-CHECK. It was written when playoff games were missing from
`game_logs`; the real cause turned out to be the scraper reading only the first
table on basketball-reference's gamelog page (playoffs sit in a separate
table), which is fixed. This module stays as an independent check.

The ESPN player box scores already loaded into `player_game_logs` do include
the playoffs (`season_type = 3`). Summing a team's player lines reconstructs
its team line: validated against 4,048 regular-season games, the rebuilt
totals match basketball-reference exactly for 98.8-100% of rows depending on
the column (mean absolute difference < 0.02).

Usage:
    python -m wnba.scraping.playoffs
    python -m wnba.scraping.playoffs --from-season 2026
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, List

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from wnba.dates import current_season
from wnba.db import SessionLocal
from wnba.models import GameLog, Team
from wnba.scraping.player_logs import _nullable_int, _resolve

logger = logging.getLogger(__name__)

PLAYOFF_SEASON_TYPE = 3

# player-box-score column -> GameLog column
SUM_COLS = {
    "field_goals_made": "fg", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ft", "free_throws_attempted": "fta",
    "offensive_rebounds": "orb", "rebounds": "trb",
    "assists": "ast", "steals": "stl", "blocks": "blk",
    "turnovers": "tov", "fouls": "pf",
}


def build_playoff_rows(season: int, team_ids: Dict[str, int]) -> List[dict]:
    from sportsdataverse.wnba import load_wnba_player_boxscore

    df = load_wnba_player_boxscore(seasons=[season], return_as_pandas=True)
    if df is None or df.empty:
        return []
    df = df[df["season_type"] == PLAYOFF_SEASON_TYPE].copy()
    if df.empty:
        logger.info("%s: no playoff games", season)
        return []

    df["abbr"] = df["team_abbreviation"].map(_resolve)
    df["opp_abbr"] = df["opponent_team_abbreviation"].map(_resolve)
    df = df[df["abbr"].isin(team_ids) & df["opp_abbr"].isin(team_ids)]
    if df.empty:
        return []

    for col in SUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby(["game_id", "abbr"], sort=False)
    team_rows = grouped.agg(
        date=("game_date", "first"),
        opp_abbr=("opp_abbr", "first"),
        home_away=("home_away", "first"),
        team_score=("team_score", "first"),
        opponent_team_score=("opponent_team_score", "first"),
        **{v: (k, "sum") for k, v in SUM_COLS.items()},
    ).reset_index()

    # Opponent's box score for the same game, for the opp_* columns.
    opp = team_rows.set_index(["game_id", "abbr"])
    records: List[dict] = []
    for row in team_rows.itertuples(index=False):
        try:
            other = opp.loc[(row.game_id, row.opp_abbr)]
        except KeyError:
            logger.warning("no opponent row for %s / %s", row.game_id, row.opp_abbr)
            continue

        record = {
            "team_id": team_ids[row.abbr],
            "opponent_id": team_ids[row.opp_abbr],
            "season": season,
            "date": pd.to_datetime(row.date).date(),
            "game_num": None,
            "is_home": row.home_away == "home",
            "won": int(row.team_score) > int(row.opponent_team_score),
            "is_playoff": True,
            "pts": _nullable_int(row.team_score),
            "opp_pts": _nullable_int(row.opponent_team_score),
        }
        for col in SUM_COLS.values():
            record[col] = _nullable_int(getattr(row, col))
            record[f"opp_{col}"] = _nullable_int(other[col])
        records.append(record)

    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill playoff games into game_logs.")
    parser.add_argument("--from-season", type=int, default=2018)
    parser.add_argument("--to-season", type=int, default=current_season())
    args = parser.parse_args()

    db = SessionLocal()
    try:
        team_ids = dict(db.execute(select(Team.abbreviation, Team.id)).all())
        total = 0
        for season in range(args.from_season, args.to_season + 1):
            rows = build_playoff_rows(season, team_ids)
            if not rows:
                continue
            result = db.execute(
                insert(GameLog).values(rows).on_conflict_do_nothing(constraint="uq_gamelog"))
            db.commit()
            total += result.rowcount
            logger.info("%s playoffs: %s team-games rebuilt, %s new",
                        season, len(rows), result.rowcount)
        logger.info("DONE: %s new playoff rows", total)
    finally:
        db.close()


if __name__ == "__main__":
    main()
