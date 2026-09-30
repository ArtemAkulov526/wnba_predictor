"""Create new tables and add the uniqueness constraints that make the
scrapers idempotent. Safe to run repeatedly."""

import sys
from pathlib import Path

# Allow both `python scripts/<name>.py` and `python -m scripts.<name>`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from wnba.db import Base, engine
from wnba import models  # noqa: F401  (registers tables on Base.metadata)

CONSTRAINTS = [
    ("games", "uq_game_team_date_opp", "UNIQUE (team_id, season, date, opponent)"),
    ("stats", "uq_stat_team_season", "UNIQUE (team_id, season)"),
]


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("tables ensured:", ", ".join(sorted(Base.metadata.tables)))

    with engine.begin() as conn:
        for table, name, defn in CONSTRAINTS:
            exists = conn.execute(text(
                "SELECT 1 FROM pg_constraint WHERE conname = :n"
            ), {"n": name}).first()
            if exists:
                print(f"  constraint {name} already present")
                continue
            dupes = conn.execute(text(
                f"SELECT count(*) FROM (SELECT 1 FROM {table} "
                f"GROUP BY {defn.split('(')[1].rstrip(') ')} HAVING count(*) > 1) d"
            )).scalar()
            if dupes:
                print(f"  SKIP {name}: {dupes} duplicate group(s) in {table}; "
                      f"dedupe first")
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD CONSTRAINT {name} {defn}"))
            print(f"  added {name} on {table}")


if __name__ == "__main__":
    main()
