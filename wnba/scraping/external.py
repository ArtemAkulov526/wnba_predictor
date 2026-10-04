"""Load closing betting lines from a CSV into the `odds` table.

`train.py` uses these for its flat-stake betting evaluation. (Injury reports
are not loaded here: they have no history, so they never feed training; live
predictions read the official/ESPN report archives directly.)

Odds CSV columns (see data/samples/odds.example.csv):
    date,home_team,away_team,home_moneyline,away_moneyline,spread,total,book
    2026-06-14,MIN,LVA,-165,+140,-4.5,163.5,pinnacle
  Moneylines are American odds. Closing lines are the meaningful ones --
  opening lines are much easier to beat and will flatter the model.

Usage:
    python -m wnba.scraping.external --odds data/odds.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from wnba.db import SessionLocal
from wnba.models import Odds, Team

logger = logging.getLogger(__name__)

def _team_map(db) -> Dict[str, int]:
    return dict(db.execute(select(Team.abbreviation, Team.id)).all())


def load_odds(db, path: Path) -> int:
    df = pd.read_csv(path)
    required = {"date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")

    teams = _team_map(db)
    unknown = sorted((set(df["home_team"]) | set(df["away_team"])) - set(teams))
    if unknown:
        raise SystemExit(f"{path}: unknown team codes {unknown}")

    def opt(row, name):
        value = getattr(row, name, None)
        return None if value is None or pd.isna(value) else value

    rows = [{
        "date": pd.to_datetime(r.date).date(),
        "home_team_id": teams[r.home_team],
        "away_team_id": teams[r.away_team],
        "home_moneyline": opt(r, "home_moneyline"),
        "away_moneyline": opt(r, "away_moneyline"),
        "spread": opt(r, "spread"),
        "total": opt(r, "total"),
        "book": opt(r, "book"),
    } for r in df.itertuples(index=False)]

    result = db.execute(
        insert(Odds).values(rows).on_conflict_do_nothing(constraint="uq_odds"))
    db.commit()
    logger.info("odds: %d rows read, %d new", len(rows), result.rowcount)
    return result.rowcount


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odds", type=Path, required=True)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        load_odds(db, args.odds)
    finally:
        db.close()


if __name__ == "__main__":
    main()
