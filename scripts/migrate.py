"""Create the tables (with the uniqueness constraints that make the scrapers
idempotent). Safe to run repeatedly.

    python scripts/migrate.py                 # create missing tables
    python scripts/migrate.py --drop-legacy   # also drop the retired tables

Retired tables: `games` and `stats` (old UI data, replaced by `game_logs`) and
`injuries` (injury reports are archived to CSV instead). They are only
dropped when asked, since dropping deletes their rows.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow both `python scripts/<name>.py` and `python -m scripts.<name>`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from wnba.db import Base, engine
from wnba import models  # noqa: F401  (registers tables on Base.metadata)

logger = logging.getLogger(__name__)

LEGACY_TABLES = ["games", "stats", "injuries"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Create tables; optionally drop retired ones.")
    parser.add_argument("--drop-legacy", action="store_true",
                        help=f"drop the retired tables: {', '.join(LEGACY_TABLES)}")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    logger.info("tables ensured: %s", ", ".join(sorted(Base.metadata.tables)))

    with engine.begin() as conn:
        present = [t for t in LEGACY_TABLES if conn.execute(
            text("SELECT to_regclass(:t)"), {"t": f"public.{t}"}).scalar()]
        if not present:
            return
        if args.drop_legacy:
            for t in present:
                conn.execute(text(f"DROP TABLE {t}"))
                logger.info("dropped %s", t)
        else:
            logger.info("retired tables still present: %s (no longer used; --drop-legacy removes them)",
                        ", ".join(present))


if __name__ == "__main__":
    main()
