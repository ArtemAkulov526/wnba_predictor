"""Scrape WNBA team game logs (per-game box scores, regular season and
playoffs) from basketball-reference into Postgres.

The game logs are the modelling source of truth and also feed the web UI's
team pages. Writes are idempotent (ON CONFLICT DO NOTHING), so re-running only
fills gaps. Respects a 3s crawl delay, matching basketball-reference robots.txt.

    python -m wnba.scraping.basketball_reference                       # 2018..now
    python -m wnba.scraping.basketball_reference --from-season 2026
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from bs4 import BeautifulSoup
from playwright.async_api import Page, async_playwright
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from wnba.dates import current_season
from wnba.db import SessionLocal
from wnba.models import GameLog, Team

logger = logging.getLogger(__name__)

BASE = "https://www.basketball-reference.com/wnba/teams"
CRAWL_DELAY = 3  # seconds, per basketball-reference robots.txt
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)

# First season each franchise is in scope for this project.
TEAM_DEBUT: Dict[str, int] = {
    "MIN": 2018, "NYL": 2018, "ATL": 2018, "PHO": 2018, "IND": 2018,
    "LVA": 2018, "SEA": 2018, "LAS": 2018, "WAS": 2018, "CHI": 2018,
    "DAL": 2018, "CON": 2018,
    "GSV": 2025,   # expansion
    "TOR": 2026,   # expansion
    "POR": 2026,   # expansion (revived franchise)
}

BOX_COLS = [
    "fg", "fga", "fg3", "fg3a", "ft", "fta",
    "orb", "trb", "ast", "stl", "blk", "tov", "pf",
]


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value not in (None, "", "None") else None
    except ValueError:
        return None


async def _fetch(page: Page, url: str) -> Optional[BeautifulSoup]:
    try:
        response = await page.goto(url, timeout=45000, wait_until="domcontentloaded")
    except Exception as exc:                      # network / timeout
        logger.warning("fetch failed %s: %s", url, type(exc).__name__)
        return None
    await asyncio.sleep(CRAWL_DELAY)
    if response is None or response.status != 200:
        logger.warning("fetch %s -> status %s", url, response and response.status)
        return None
    return BeautifulSoup(await page.content(), "html.parser")


# --------------------------------------------------------------------------
# game logs -- per-game box scores
# --------------------------------------------------------------------------
async def get_team_gamelog(
    db: Session, page: Page, team_abbr: str, season: int, team_ids: Dict[str, int]
) -> int:
    """Scrape one team-season game log. Returns the number of new rows."""
    soup = await _fetch(page, f"{BASE}/{team_abbr}/{season}/gamelog/")
    if soup is None:
        return 0

    # The page carries TWO tables: `wnba_tgl_basic` (captioned "Regular Season
    # Table") and `wnba_tgl_basic_p` ("Playoffs Table"). Reading only the
    # first silently dropped every postseason game.
    tables = [("wnba_tgl_basic", False), ("wnba_tgl_basic_p", True)]
    found = [(soup.find("table", {"id": tid}), playoff) for tid, playoff in tables]
    if all(tbl is None or tbl.tbody is None for tbl, _ in found):
        logger.warning("no gamelog table for %s %s", team_abbr, season)
        return 0

    rows: List[dict] = []
    for table, is_playoff in found:
        if table is None or table.tbody is None:
            continue                              # team missed the playoffs
        for tr in table.tbody.find_all("tr"):
            cells = {td.get("data-stat"): td.text.strip() for td in tr.find_all("td")}
            if not cells or not cells.get("date_game") or not cells.get("opp_id"):
                continue                          # spacer / repeated header
            if not cells.get("pts"):
                continue                          # unplayed game
            opponent_id = team_ids.get(cells["opp_id"])
            if opponent_id is None:
                logger.warning("unknown opponent %s for %s %s",
                               cells["opp_id"], team_abbr, season)
                continue

            record = {
                "team_id": team_ids[team_abbr],
                "opponent_id": opponent_id,
                "season": season,
                "date": datetime.strptime(cells["date_game"], "%Y-%m-%d").date(),
                "game_num": _int(cells.get("game_num")),
                # '@' marks an away game; blank means home
                "is_home": cells.get("game_location", "") != "@",
                "won": cells.get("game_result") == "W",
                "is_playoff": is_playoff,
                "pts": _int(cells.get("pts")),
                "opp_pts": _int(cells.get("opp_pts")),
            }
            for col in BOX_COLS:
                record[col] = _int(cells.get(col))
                record[f"opp_{col}"] = _int(cells.get(f"opp_{col}"))
            rows.append(record)

    if not rows:
        return 0

    stmt = insert(GameLog).values(rows).on_conflict_do_nothing(constraint="uq_gamelog")
    result = db.execute(stmt)
    db.commit()
    playoff_n = sum(1 for r in rows if r["is_playoff"])
    logger.info("%s %s gamelog: %s scraped (%s playoff), %s new",
                team_abbr, season, len(rows), playoff_n, result.rowcount)
    return result.rowcount


# --------------------------------------------------------------------------
async def collect(db: Session, page: Page, seasons: Iterable[int]) -> None:
    team_ids = dict(db.execute(select(Team.abbreviation, Team.id)).all())
    missing = set(TEAM_DEBUT) - set(team_ids)
    if missing:
        raise SystemExit(f"teams missing from DB: {sorted(missing)} -- run scripts/seed_teams.py")

    new = 0
    for season in seasons:
        for abbr, debut in sorted(TEAM_DEBUT.items()):
            if season >= debut:
                new += await get_team_gamelog(db, page, abbr, season, team_ids)
    logger.info("DONE new game-log rows: %s", new)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Scrape WNBA team game logs into Postgres.")
    parser.add_argument("--from-season", type=int, default=2018)
    parser.add_argument("--to-season", type=int, default=current_season())
    args = parser.parse_args()

    seasons = range(args.from_season, args.to_season + 1)
    db = SessionLocal()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                extra_http_headers={"User-Agent": USER_AGENT})
            page = await context.new_page()
            try:
                await collect(db, page, seasons)
            finally:
                await browser.close()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
