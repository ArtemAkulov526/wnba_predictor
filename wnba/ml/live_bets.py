"""In-game (live) bets, kept apart from the pre-game prediction log.

The prediction log holds one pre-game pick per game; a live bet on the same
game would overwrite it. Live bets go here and are graded from that game's
result in the prediction log (`tracking.py`).

    python -m wnba.ml.live_bets add --home ATL --away NYL --date 2026-10-04 --pick NYL \\
           --odds 1.94 --state "HT NYL 40-37" --note "Gray ruled out at halftime"
    python -m wnba.ml.live_bets report
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from wnba.ml import tracking
from wnba.paths import DATA_DIR
from wnba.csvio import write_csv

logger = logging.getLogger(__name__)

LIVE_PATH = DATA_DIR / "live_bets.csv"
COLUMNS = ["game_date", "away", "home", "pick", "odds", "user_prob", "state",
           "tags", "note", "created_at"]


def read_live() -> pd.DataFrame:
    if not LIVE_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(LIVE_PATH, dtype={"game_date": str})


def add(home: str, away: str, game_date: date, pick: str, odds: float,
        user_prob: Optional[float] = None, state: Optional[str] = None,
        tags: Optional[str] = None, note: Optional[str] = None) -> None:
    pick = pick.upper()
    if pick not in (home.upper(), away.upper()):
        raise ValueError("pick must be the home or the away team")
    df = read_live()
    row = {"game_date": game_date.isoformat(), "away": away.upper(), "home": home.upper(),
           "pick": pick, "odds": odds, "user_prob": user_prob, "state": state,
           "tags": tracking.normalize_tags(tags), "note": note,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_csv(pd.concat([df, pd.DataFrame([row])], ignore_index=True), LIVE_PATH)


def _scraped_home_won(game_date: str, home: str, away: str) -> Optional[bool]:
    """Result from game_logs, for live bets on games never logged pre-game."""
    from wnba.db import SessionLocal
    db = SessionLocal()
    try:
        row = pd.read_sql("""SELECT g.won FROM game_logs g
                             JOIN teams t ON t.id = g.team_id JOIN teams o ON o.id = g.opponent_id
                             WHERE g.date = %(d)s AND t.abbreviation = %(h)s
                               AND o.abbreviation = %(a)s AND g.is_home""",
                          db.bind, params={"d": game_date, "h": home, "a": away})
    finally:
        db.close()
    return None if row.empty else bool(row.iloc[0]["won"])


def report() -> None:
    df = read_live()
    if df.empty:
        print("no live bets")
        return
    games = tracking.read_log()
    games["game_date"] = games["game_date"].astype(str)
    total = 0.0
    n = 0
    for r in df.itertuples():
        g = games[(games.game_date == r.game_date) & (games.home == r.home) & (games.away == r.away)]
        home_won = None
        if len(g) and not pd.isna(g.iloc[0].home_won):
            home_won = bool(g.iloc[0].home_won)
        else:                                  # not in the prediction log: use game_logs
            home_won = _scraped_home_won(r.game_date, r.home, r.away)
        won = None if home_won is None else home_won == (r.pick == r.home)
        status = "pending" if won is None else ("WON" if won else "lost")
        units = None if won is None else (r.odds - 1 if won else -1.0)
        print(f"{r.game_date} {r.away}@{r.home}  {r.pick} @ {r.odds:.2f}  [{r.state}]  -> {status}"
              + (f"  {units:+.2f} units" if units is not None else ""))
        if units is not None:
            total += units
            n += 1
    if n:
        print(f"\nlive bets settled: {n}, {total:+.2f} units")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Live (in-game) bets.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    for k in ("--home", "--away", "--pick"):
        a.add_argument(k, required=True)
    a.add_argument("--date", type=date.fromisoformat, required=True, help="US game date")
    a.add_argument("--odds", type=float, required=True)
    a.add_argument("--confidence", type=float)
    a.add_argument("--state", help='game state when the bet was made, e.g. "HT NYL 40-37"')
    a.add_argument("--tags")
    a.add_argument("--note")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "add":
        add(args.home, args.away, args.date, args.pick, args.odds, args.confidence,
            args.state, args.tags, args.note)
        logger.info("logged live bet: %s @ %s (%s@%s)", args.pick.upper(), args.odds,
                    args.away.upper(), args.home.upper())
    else:
        report()


if __name__ == "__main__":
    main()
