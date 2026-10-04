"""Game odds (moneyline, spread, total) from ESPN's public API.

ESPN shows one sportsbook's lines (DraftKings) for every game, with the
opening and the current price. After the game the last price stays in the game
summary, so for finished games "close" is the closing line. Coverage: the 2026
season onward (earlier seasons have no odds in the feed).

This is a market reference -- the benchmark for the model and for closing line
value -- not necessarily the book you bet at. Your own bet prices are still
logged with each pick.

Output: data/espn_odds.csv, one row per game (re-running updates the row).

    python -m wnba.scraping.espn_odds                       # yesterday .. +3 days
    python -m wnba.scraping.espn_odds --from 2026-05-07 --to 2026-10-04
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import requests

from wnba.paths import DATA_DIR
from wnba.scraping.player_logs import ABBR_ALIASES

logger = logging.getLogger(__name__)

ESPN_API = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ODDS_PATH = DATA_DIR / "espn_odds.csv"
DELAY = 1.0                                    # seconds between requests
COLUMNS = ["game_date", "away", "home", "event_id", "status", "provider",
           "home_ml_open", "away_ml_open", "home_ml", "away_ml",
           "home_spread_open", "home_spread", "home_spread_odds", "away_spread_odds",
           "total_open", "total", "over_odds", "under_odds", "fetched_at"]


def american_to_decimal(odds) -> Optional[float]:
    """'-192' / '+160' / 160 -> decimal odds; None if missing."""
    if odds is None or odds == "" or (isinstance(odds, float) and pd.isna(odds)):
        return None
    try:
        a = float(str(odds).replace("+", ""))
    except ValueError:
        return None                            # e.g. "EVEN" is handled below
    if a == 0:
        return None
    return round(1 + (a / 100 if a > 0 else 100 / -a), 4)


def _line(text) -> Optional[float]:
    """'-4.5' / 'o172.5' -> float."""
    if text is None:
        return None
    try:
        return float(str(text).lstrip("ou").replace("+", ""))
    except ValueError:
        return None


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parse_pick(p: dict) -> dict:
    """One ESPN 'pickcenter' / 'odds' entry -> flat row (decimal odds)."""
    ml, ps, tot = p.get("moneyline") or {}, p.get("pointSpread") or {}, p.get("total") or {}
    even = lambda v: "+100" if str(v).upper() == "EVEN" else v
    return {
        "provider": _get(p, "provider", "name"),
        "home_ml_open": american_to_decimal(even(_get(ml, "home", "open", "odds"))),
        "away_ml_open": american_to_decimal(even(_get(ml, "away", "open", "odds"))),
        "home_ml": american_to_decimal(even(_get(ml, "home", "close", "odds"))
                                       or _get(p, "homeTeamOdds", "moneyLine")),
        "away_ml": american_to_decimal(even(_get(ml, "away", "close", "odds"))
                                       or _get(p, "awayTeamOdds", "moneyLine")),
        "home_spread_open": _line(_get(ps, "home", "open", "line")),
        "home_spread": _line(_get(ps, "home", "close", "line")) if ps else p.get("spread"),
        "home_spread_odds": american_to_decimal(_get(ps, "home", "close", "odds")),
        "away_spread_odds": american_to_decimal(_get(ps, "away", "close", "odds")),
        "total_open": _line(_get(tot, "over", "open", "line")),
        "total": _line(_get(tot, "over", "close", "line")) if tot else p.get("overUnder"),
        "over_odds": american_to_decimal(_get(tot, "over", "close", "odds")),
        "under_odds": american_to_decimal(_get(tot, "under", "close", "odds")),
    }


def fetch_day(day: date, session: requests.Session) -> List[dict]:
    code = lambda a: ABBR_ALIASES.get(a, a)
    board = session.get(f"{ESPN_API}/scoreboard", params={"dates": day.strftime("%Y%m%d")},
                        timeout=20).json()
    rows = []
    for e in board.get("events", []):
        comp = e["competitions"][0]
        teams = {c["homeAway"]: code(c["team"]["abbreviation"]) for c in comp["competitors"]}
        time.sleep(DELAY)
        summary = session.get(f"{ESPN_API}/summary", params={"event": e["id"]}, timeout=20).json()
        picks = summary.get("pickcenter") or comp.get("odds") or []
        if not picks:
            continue
        status = "close" if comp["status"]["type"]["completed"] else "current"
        rows.append({"game_date": day.isoformat(), "away": teams["away"], "home": teams["home"],
                     "event_id": e["id"], "status": status, **parse_pick(picks[0]),
                     "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return rows


def read_odds() -> pd.DataFrame:
    if not ODDS_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(ODDS_PATH, dtype={"game_date": str, "event_id": str})


def save(rows: List[dict]) -> int:
    if not rows:
        return 0
    df = pd.concat([read_odds(), pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    # a finished game's closing row is final: never replace it with an older snapshot
    df["_rank"] = df["status"].map({"close": 1, "current": 0})
    df = (df.sort_values(["_rank", "fetched_at"])
            .drop_duplicates(["game_date", "home", "away"], keep="last")
            .drop(columns="_rank").sort_values(["game_date", "home"]))
    tmp = ODDS_PATH.with_name(ODDS_PATH.name + ".tmp")
    df[COLUMNS].to_csv(tmp, index=False)
    for attempt in range(20):                  # OneDrive may briefly lock the file
        try:
            os.replace(tmp, ODDS_PATH)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)
    return len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="ESPN (DraftKings) game odds.")
    us_today = (datetime.now(timezone.utc) - timedelta(hours=5)).date()
    ap.add_argument("--from", dest="start", type=date.fromisoformat, default=us_today - timedelta(days=1))
    ap.add_argument("--to", dest="end", type=date.fromisoformat, default=us_today + timedelta(days=3))
    args = ap.parse_args()
    session = requests.Session()
    n, day = 0, args.start
    while day <= args.end:
        rows = fetch_day(day, session)
        n += save(rows)
        if rows:
            logger.info("%s: %d games", day, len(rows))
        day += timedelta(days=1)
        time.sleep(DELAY)
    logger.info("done: %d game rows -> %s", n, ODDS_PATH)


if __name__ == "__main__":
    main()
