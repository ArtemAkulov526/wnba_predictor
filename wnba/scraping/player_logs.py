"""Ingest per-player box scores from the sportsdataverse ESPN datasets.

This is not a scraper: sportsdataverse publishes pre-built season files, so
this is a download rather than a crawl. It is the cheapest route to player
data -- no API key, no rate limit, no terms-of-service question.

Why this matters: team box scores cannot tell you WHO was on the floor. These
rows can, which is what lets `features.add_availability_features` reconstruct
availability for every past game without an injury feed.

Usage:
    python -m wnba.scraping.player_logs                      # 2018..current
    python -m wnba.scraping.player_logs --from-season 2026
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
from wnba.models import PlayerGameLog, Team

logger = logging.getLogger(__name__)

# sportsdataverse/ESPN abbreviations -> the codes used in the teams table.
ABBR_ALIASES: Dict[str, str] = {
    "GS": "GSV",     # Golden State Valkyries
    "LA": "LAS",     # Los Angeles Sparks
    "LV": "LVA",     # Las Vegas Aces
    "NY": "NYL",     # New York Liberty
    "PHX": "PHO",    # Phoenix Mercury
    "WSH": "WAS",    # Washington Mystics
    "CONN": "CON",   # Connecticut Sun
}

# All-Star and exhibition rosters that are not real franchises.
NON_FRANCHISE = {"CLA", "COL", "COOP", "DEL", "PAR", "SPO", "USA", "WNBASTARS",
                 "WIL", "STE", "ALL"}  # Team Wilson / Team Stewart / generic

COLUMN_MAP = {
    "points": "pts",
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m",
    "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "orb", "defensive_rebounds": "drb",
    "rebounds": "reb", "assists": "ast", "steals": "stl",
    "blocks": "blk", "turnovers": "tov", "fouls": "pf",
    "plus_minus": "plus_minus",
}
INT_COLS = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
            "orb", "drb", "reb", "ast", "stl", "blk", "tov", "pf"]


def _resolve(abbr: str) -> str:
    return ABBR_ALIASES.get(abbr, abbr)


def _nullable_int(value):
    """ESPN uses '--' and '' as placeholders, so coerce rather than cast."""
    value = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(value) else int(value)


def _nullable_float(value):
    value = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(value) else float(value)


def _nullable_bool(value):
    return None if pd.isna(value) else bool(value)


def _nullable_str(value):
    """pandas hands back pd.NA for missing strings, which psycopg2 cannot adapt."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_season(db, season: int, team_ids: Dict[str, int]) -> int:
    """Download and store one season of player box scores."""
    from sportsdataverse.wnba import load_wnba_player_boxscore

    try:
        df = load_wnba_player_boxscore(seasons=[season], return_as_pandas=True)
    except Exception as exc:
        logger.warning("season %s failed: %s: %s", season, type(exc).__name__, exc)
        return 0
    if df is None or df.empty:
        logger.warning("season %s returned no rows", season)
        return 0

    df = df.copy()
    df["abbr"] = df["team_abbreviation"].map(_resolve)
    df["opp_abbr"] = df["opponent_team_abbreviation"].map(_resolve)

    dropped = df[~df["abbr"].isin(team_ids)]
    if not dropped.empty:
        skipped = sorted(set(dropped["team_abbreviation"].dropna()))
        unexpected = [s for s in skipped if _resolve(s) not in NON_FRANCHISE]
        if unexpected:
            logger.warning("season %s: unmapped team codes %s", season, unexpected)
    df = df[df["abbr"].isin(team_ids)]
    if df.empty:
        return 0

    rows: List[dict] = []
    for r in df.itertuples(index=False):
        record = {
            "game_id": _nullable_str(r.game_id),
            "date": pd.to_datetime(r.game_date).date(),
            "season": int(r.season),
            "season_type": _nullable_int(getattr(r, "season_type", None)),
            "team_id": team_ids[r.abbr],
            "opponent_id": team_ids.get(r.opp_abbr),
            "athlete_id": _nullable_str(r.athlete_id),
            "athlete_name": _nullable_str(getattr(r, "athlete_display_name", None)),
            "position": _nullable_str(getattr(r, "athlete_position_abbreviation", None)),
            "minutes": _nullable_float(r.minutes),
            "did_not_play": _nullable_bool(getattr(r, "did_not_play", None)),
            "reason": _nullable_str(getattr(r, "reason", None)),
            "starter": _nullable_bool(getattr(r, "starter", None)),
        }
        for source, target in COLUMN_MAP.items():
            value = getattr(r, source, None)
            record[target] = (_nullable_int(value) if target in INT_COLS
                              else _nullable_float(value))
        rows.append(record)

    written = 0
    # Chunked so a season does not build one enormous statement.
    for start in range(0, len(rows), 2000):
        chunk = rows[start:start + 2000]
        result = db.execute(
            insert(PlayerGameLog).values(chunk)
            .on_conflict_do_nothing(constraint="uq_player_game"))
        written += result.rowcount
        db.commit()

    logger.info("%s: %s rows read, %s new", season, len(rows), written)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Load WNBA player box scores.")
    parser.add_argument("--from-season", type=int, default=2018)
    parser.add_argument("--to-season", type=int, default=current_season())
    args = parser.parse_args()

    db = SessionLocal()
    try:
        team_ids = dict(db.execute(select(Team.abbreviation, Team.id)).all())
        total = sum(load_season(db, s, team_ids)
                    for s in range(args.from_season, args.to_season + 1))
        logger.info("DONE: %s new player-game rows", total)
    finally:
        db.close()


if __name__ == "__main__":
    main()
