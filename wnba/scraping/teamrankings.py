"""Closing point spreads per game from TeamRankings.

Source: https://www.teamrankings.com/wnba/team/<team>/ats-results -- one table
per team with the closing line for every game of the CURRENT season. Allowed
by robots.txt, which asks for a 10-second Crawl-delay (respected below).

Limits found while building this (2026-09-21):
  * spreads only -- there is no moneyline table;
  * current season only -- the `?season=` parameter is ignored, and older
    seasons sit behind /ajax/, which robots.txt disallows.
So each season has to be captured while it is live. Lines are written to
`data/closing_lines.csv` (a file, like the prediction log) because they cannot
be fetched again once the season rolls over.

Every game appears on both teams' pages; the two rows are cross-checked (the
lines must be exact negatives) and stored once, from the home side.

Usage:
    python -m wnba.scraping.teamrankings
"""
from __future__ import annotations

import logging
import os
import re
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

from wnba.paths import DATA_DIR

logger = logging.getLogger(__name__)

BASE = "https://www.teamrankings.com"
CRAWL_DELAY = 10
LINES_PATH = DATA_DIR / "closing_lines.csv"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36")

# TeamRankings URL slugs and the city names it uses for opponents.
TEAMS: Dict[str, tuple[str, str]] = {
    "ATL": ("atlanta-dream", "Atlanta"),
    "CHI": ("chicago-sky", "Chicago"),
    "CON": ("connecticut-sun", "Connecticut"),
    "DAL": ("dallas-wings", "Dallas"),
    "GSV": ("golden-state-valkyries", "Golden State"),
    "IND": ("indiana-fever", "Indiana"),
    "LAS": ("los-angeles-sparks", "Los Angeles"),
    "LVA": ("las-vegas-aces", "Las Vegas"),
    "MIN": ("minnesota-lynx", "Minnesota"),
    "NYL": ("new-york-liberty", "New York"),
    "PHO": ("phoenix-mercury", "Phoenix"),
    "POR": ("portland-fire", "Portland"),
    "SEA": ("seattle-storm", "Seattle"),
    "TOR": ("toronto-tempo", "Toronto"),
    "WAS": ("washington-mystics", "Washington"),
}
CITY_TO_ABBR = {city: abbr for abbr, (_, city) in TEAMS.items()}
COLUMNS = ["game_date", "home", "away", "neutral_site", "home_spread",
           "home_pts_margin", "source"]


def parse_line(text: str) -> Optional[float]:
    t = (text or "").strip().upper()
    if t in ("PK", "PICK", "EVEN"):
        return 0.0
    try:
        return float(t)
    except ValueError:
        return None                      # '--' or blank: no line posted


def parse_margin(result: str) -> Optional[int]:
    """'W by 4' -> 4, 'L by 1' -> -1."""
    m = re.match(r"\s*([WL])\s+by\s+(\d+)", result or "")
    if not m:
        return None
    return int(m.group(2)) * (1 if m.group(1) == "W" else -1)


def parse_team_page(html: str, team: str, season: int) -> pd.DataFrame:
    """One team's ATS table -> rows from that team's perspective. Pure."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows: List[dict] = []
    if table is None:
        return pd.DataFrame(rows)
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) < 6:
            continue
        mm, dd = cells[0].split("/")
        opp = CITY_TO_ABBR.get(cells[2])
        if opp is None:
            raise ValueError(f"{team}: unknown opponent name {cells[2]!r}")
        rows.append({
            "game_date": date(season, int(mm), int(dd)).isoformat(),
            "team": team, "opp": opp,
            "location": cells[1].lower(),          # home / away / neutral
            "line": parse_line(cells[4]),
            "margin": parse_margin(cells[5]),
        })
    return pd.DataFrame(rows)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def collect(season: int) -> pd.DataFrame:
    frames = []
    for i, (abbr, (slug, _)) in enumerate(sorted(TEAMS.items())):
        if i:
            time.sleep(CRAWL_DELAY)
        frames.append(parse_team_page(
            fetch(f"{BASE}/wnba/team/{slug}/ats-results"), abbr, season))
        logger.info("%s: %d games", abbr, len(frames[-1]))
    return pd.concat(frames, ignore_index=True)


def to_games(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two team-perspective rows into one row per game.

    Checks that both sides agree (lines are exact negatives, margins too);
    a disagreement raises rather than silently picking one.
    """
    home = team_rows[team_rows["location"].isin(["home", "neutral"])]
    away = team_rows[team_rows["location"].isin(["away", "neutral"])]
    m = home.merge(away, left_on=["game_date", "team", "opp"],
                   right_on=["game_date", "opp", "team"], suffixes=("", "_other"))
    # neutral games appear twice in each role; keep one orientation
    m = m[~((m["location"] == "neutral") & (m["team"] > m["opp"]))]

    both = m["line"].notna() & m["line_other"].notna()
    bad = m[both & ((m["line"] + m["line_other"]).abs() > 1e-9)]
    if not bad.empty:
        raise ValueError(f"line mismatch between team pages:\n{bad}")

    return pd.DataFrame({
        "game_date": m["game_date"],
        "home": m["team"], "away": m["opp"],
        "neutral_site": m["location"] == "neutral",
        "home_spread": m["line"].where(m["line"].notna(), -m["line_other"]),
        "home_pts_margin": m["margin"],
        "source": "teamrankings-close",
    })[COLUMNS].sort_values(["game_date", "home"]).reset_index(drop=True)


def save(games: pd.DataFrame, path: Path = LINES_PATH) -> int:
    """Merge into the file; existing rows for the same game are replaced."""
    path = Path(path)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    key = ["game_date", "home", "away"]
    old = old[~old.set_index(key).index.isin(games.set_index(key).index)]
    out = pd.concat([old, games], ignore_index=True).sort_values(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(games)


# --------------------------------------------------------------------------
# closing game totals (over/under) -- same page layout, /over-under-results
# --------------------------------------------------------------------------
TOTALS_PATH = DATA_DIR / "closing_totals.csv"
TOTALS_COLUMNS = ["game_date", "home", "away", "neutral_site", "total",
                  "home_pts", "away_pts", "source"]


def parse_totals_page(html: str, team: str, season: int) -> pd.DataFrame:
    """One team's over/under table -> rows from that team's perspective. Pure.

    Columns on the page: Date, H/A/N, Opponent, Opp Rank, Total,
    Score ("team-opp"), Result (Over/Under/Push), Diff.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows: List[dict] = []
    if table is None:
        return pd.DataFrame(rows)
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) < 6:
            continue
        mm, dd = cells[0].split("/")
        opp = CITY_TO_ABBR.get(cells[2])
        if opp is None:
            raise ValueError(f"{team}: unknown opponent name {cells[2]!r}")
        m = re.match(r"\s*(\d+)\s*-\s*(\d+)", cells[5])
        rows.append({
            "game_date": date(season, int(mm), int(dd)).isoformat(),
            "team": team, "opp": opp, "location": cells[1].lower(),
            "total": parse_line(cells[4]),
            "pts": int(m.group(1)) if m else None,
            "opp_pts": int(m.group(2)) if m else None,
        })
    return pd.DataFrame(rows)


def to_totals_games(team_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per game; both teams' pages must show the same total."""
    home = team_rows[team_rows["location"].isin(["home", "neutral"])]
    away = team_rows[team_rows["location"].isin(["away", "neutral"])]
    m = home.merge(away, left_on=["game_date", "team", "opp"],
                   right_on=["game_date", "opp", "team"], suffixes=("", "_other"))
    m = m[~((m["location"] == "neutral") & (m["team"] > m["opp"]))]
    both = m["total"].notna() & m["total_other"].notna()
    mismatch = both & ((m["total"] - m["total_other"]).abs() > 1e-9)
    if mismatch.any():
        # Seen for 3 of ~320 games in 2026 (e.g. GSV-POR 2026-06-02): the two
        # team pages disagree. Neither can be trusted, so drop those games
        # rather than guess -- and rather than abort the whole collection.
        for r in m[mismatch].itertuples(index=False):
            logger.warning("dropped %s %s@%s: totals disagree (%s vs %s)",
                           r.game_date, r.opp, r.team, r.total, r.total_other)
        m = m[~mismatch]
    return pd.DataFrame({
        "game_date": m["game_date"], "home": m["team"], "away": m["opp"],
        "neutral_site": m["location"] == "neutral",
        "total": m["total"].where(m["total"].notna(), m["total_other"]),
        "home_pts": m["pts"], "away_pts": m["opp_pts"],
        "source": "teamrankings-close",
    })[TOTALS_COLUMNS].sort_values(["game_date", "home"]).reset_index(drop=True)


def collect_totals(season: int) -> pd.DataFrame:
    frames = []
    for i, (abbr, (slug, _)) in enumerate(sorted(TEAMS.items())):
        if i:
            time.sleep(CRAWL_DELAY)
        frames.append(parse_totals_page(
            fetch(f"{BASE}/wnba/team/{slug}/over-under-results"), abbr, season))
        logger.info("%s: %d games", abbr, len(frames[-1]))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="TeamRankings closing lines.")
    parser.add_argument("--market", choices=["spreads", "totals", "both"], default="both")
    args = parser.parse_args()
    season = date.today().year

    if args.market in ("spreads", "both"):
        logger.info("collecting %s closing spreads (%ss between requests)", season, CRAWL_DELAY)
        games = to_games(collect(season))
        n = save(games)
        logger.info("%d games with lines -> %s (%d without a posted line)",
                    n, LINES_PATH, int(games["home_spread"].isna().sum()))
    if args.market in ("totals", "both"):
        if args.market == "both":
            time.sleep(CRAWL_DELAY)
        logger.info("collecting %s closing totals (%ss between requests)", season, CRAWL_DELAY)
        totals = to_totals_games(collect_totals(season))
        n = save(totals, TOTALS_PATH)
        logger.info("%d games with totals -> %s (%d without a posted total)",
                    n, TOTALS_PATH, int(totals["total"].isna().sum()))


if __name__ == "__main__":
    main()
