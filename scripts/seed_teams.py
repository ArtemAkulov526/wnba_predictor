"""Seed the teams table. Idempotent: safe to re-run after expansion."""

import sys
from pathlib import Path

# Allow both `python scripts/<name>.py` and `python -m scripts.<name>`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session
from wnba.db import SessionLocal
from wnba.models import Team

# Basketball-Reference abbreviations. LVA/DAL resolve correctly even though
# BBRef's canonical franchise codes are SAS (Las Vegas) and TUL (Dallas) --
# these are the codes that appear in opp_id on the gamelog pages.
TEAMS_DATA = [
    {"name": "Minnesota Lynx",         "abbreviation": "MIN"},
    {"name": "New York Liberty",       "abbreviation": "NYL"},
    {"name": "Atlanta Dream",          "abbreviation": "ATL"},
    {"name": "Phoenix Mercury",        "abbreviation": "PHO"},
    {"name": "Indiana Fever",          "abbreviation": "IND"},
    {"name": "Las Vegas Aces",         "abbreviation": "LVA"},
    {"name": "Seattle Storm",          "abbreviation": "SEA"},
    {"name": "Los Angeles Sparks",     "abbreviation": "LAS"},
    {"name": "Washington Mystics",     "abbreviation": "WAS"},
    {"name": "Chicago Sky",            "abbreviation": "CHI"},
    {"name": "Dallas Wings",           "abbreviation": "DAL"},
    {"name": "Connecticut Sun",        "abbreviation": "CON"},
    {"name": "Golden State Valkyries", "abbreviation": "GSV"},  # expansion 2025
    {"name": "Toronto Tempo",          "abbreviation": "TOR"},  # expansion 2026
    {"name": "Portland Fire",          "abbreviation": "POR"},  # expansion 2026
]


def seed(db: Session) -> None:
    existing = {a for (a,) in db.query(Team.abbreviation).all()}
    added = [t for t in TEAMS_DATA if t["abbreviation"] not in existing]
    for team in added:
        db.add(Team(**team))
    db.commit()
    print(f"teams: {len(existing)} existing, {len(added)} added "
          f"({', '.join(t['abbreviation'] for t in added) or 'none'})")


if __name__ == "__main__":
    db = SessionLocal()
    try:
        seed(db)
    finally:
        db.close()
