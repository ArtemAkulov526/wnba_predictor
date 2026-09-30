"""Load injury reports and closing betting lines from CSV into Postgres.

Neither of these is available from basketball-reference, so this is the seam
where you plug in whatever source you end up using. The feature pipeline reads
both tables when they have rows and degrades to neutral values when they do
not, so nothing breaks while these stay empty.

Injuries CSV columns:
    date,team,player,status,minutes_share
    2026-06-14,MIN,Napheesa Collier,out,0.28
  `team` is the basketball-reference abbreviation, `status` is one of
  out/doubtful/questionable/probable, and `minutes_share` is the fraction of
  the team's minutes that player normally takes (0-1). Only out/doubtful are
  counted as missing; minutes_share is what makes a star's absence weigh more
  than a bench player's.

Odds CSV columns:
    date,home_team,away_team,home_moneyline,away_moneyline,spread,total,book
    2026-06-14,MIN,LVA,-165,+140,-4.5,163.5,pinnacle
  Moneylines are American odds. Closing lines are the meaningful ones --
  opening lines are much easier to beat and will flatter the model.

Usage:
    python load_external.py --injuries data/injuries.csv
    python load_external.py --odds data/odds.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from wnba.db import SessionLocal
from wnba.models import Injury, Odds, Team

VALID_STATUS = {"out", "doubtful", "questionable", "probable"}


def _team_map(db) -> Dict[str, int]:
    return dict(db.execute(select(Team.abbreviation, Team.id)).all())


def load_injuries(db, path: Path) -> int:
    df = pd.read_csv(path)
    required = {"date", "team", "player", "status"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")

    teams = _team_map(db)
    unknown = sorted(set(df["team"]) - set(teams))
    if unknown:
        raise SystemExit(f"{path}: unknown team codes {unknown}")

    bad_status = sorted(set(df["status"].str.lower()) - VALID_STATUS)
    if bad_status:
        raise SystemExit(f"{path}: status must be one of {sorted(VALID_STATUS)}; "
                         f"got {bad_status}")

    rows = [{
        "team_id": teams[r.team],
        "date": pd.to_datetime(r.date).date(),
        "player": r.player,
        "status": str(r.status).lower(),
        "minutes_share": float(getattr(r, "minutes_share", 0.0) or 0.0),
    } for r in df.itertuples(index=False)]

    result = db.execute(
        insert(Injury).values(rows).on_conflict_do_nothing(constraint="uq_injury"))
    db.commit()
    print(f"injuries: {len(rows)} rows read, {result.rowcount} new")
    return result.rowcount


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
    print(f"odds: {len(rows)} rows read, {result.rowcount} new")
    return result.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--injuries", type=Path)
    parser.add_argument("--odds", type=Path)
    args = parser.parse_args()

    if not args.injuries and not args.odds:
        parser.error("give --injuries and/or --odds")

    db = SessionLocal()
    try:
        if args.injuries:
            load_injuries(db, args.injuries)
        if args.odds:
            load_odds(db, args.odds)
    finally:
        db.close()


if __name__ == "__main__":
    main()
