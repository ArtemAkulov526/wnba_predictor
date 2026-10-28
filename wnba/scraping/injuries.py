"""Snapshot ESPN's WNBA injury report, archive it, and feed it to predictions.

Source: https://www.espn.com/wnba/injuries -- allowed by ESPN's robots.txt
(checked 2026-09-21). This is the FALLBACK source: the official league report
(wnba.scraping.official_injuries, read from the PDFs it links to) is used first
for every team that has filed, and ESPN fills in teams that have not.

The report is CURRENT STATUS ONLY. It is for live prediction, never training
(see features.build_dataset): the model learned what missing minutes are worth
from box-score history, and this module only says who will be missing.

Every snapshot is archived to `data/injury_reports.csv`. Like the prediction
log, a past report cannot be re-scraped, so the archive is the only way to
build history -- e.g. to measure how often "Day-To-Day" players actually sit
instead of assuming 50/50.

Usage:
    python -m wnba.scraping.injuries              # snapshot, archive, print all
    python -m wnba.scraping.injuries --team PHO   # just one team
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

from wnba.paths import DATA_DIR
from wnba.dates import us_today
from wnba.csvio import write_csv

ESPN_URL = "https://www.espn.com/wnba/injuries"
ARCHIVE_PATH = DATA_DIR / "injury_reports.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)
ARCHIVE_COLUMNS = ["snapshot_date", "first_seen_at", "team", "team_name", "player",
                   "position", "est_return", "status", "availability", "comment"]


def parse_return(est_return: str | None, reference: date) -> Optional[date]:
    """'Sep 22' -> a date. ESPN omits the year; a date well before the
    reference (e.g. 'May 1' seen in September) means NEXT season."""
    if not est_return:
        return None
    try:
        parsed = datetime.strptime(f"{est_return} {reference.year}", "%b %d %Y").date()
    except ValueError:
        return None
    if parsed < reference - timedelta(days=30):
        parsed = parsed.replace(year=reference.year + 1)
    return parsed


def classify(status: str, comment: str | None = None,
             est_return: str | None = None,
             game_date: date | None = None) -> str:
    """ESPN listing -> 'out', 'questionable' or 'probable' for a given game.

    The status column lags the comment. Real example (2026-09-21): Kahleah
    Copper listed "Out" while the comment dated the day before said she was
    "probable for Monday's meeting with the Wings". So, in priority order:

      1. the comment's own game-status words (probable / questionable /
         doubtful / game-time decision),
      2. a "Day-To-Day" status,
      3. an estimated return on or before the game date -> the listing is
         probably stale (it may describe a game already played), so treat as
         questionable rather than certainly out,
      4. otherwise out.
    """
    c = (comment or "").lower()
    # Known issues (audit 6.3/6.4, fix after the season -- they change
    # predictions): "doubtful" is treated as questionable (plays ~43% instead of
    # ~5%), "probable" is a substring match, and "ruled out" later in a comment
    # does not override an earlier "probable".
    if "probable" in c:
        return "probable"
    if any(w in c for w in ("questionable", "doubtful", "game-time", "game time")):
        return "questionable"
    if "day" in (status or "").lower():
        return "questionable"
    if game_date is not None:
        back = parse_return(est_return, game_date)
        if back is not None and back <= game_date:
            return "questionable"
    return "out"


def parse_injuries(html: str) -> pd.DataFrame:
    """ESPN injuries page -> one row per listed player. Pure; no network."""
    soup = BeautifulSoup(html, "html.parser")
    rows: List[dict] = []
    for title in soup.select(".Table__Title"):
        table = title.find_next("table")
        if table is None or table.tbody is None:
            continue
        team_name = title.get_text(strip=True)
        for tr in table.tbody.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 4 or not cells[0]:
                continue
            rows.append({
                "team_name": team_name,
                "player": cells[0],
                "position": cells[1],
                "est_return": cells[2],
                "status": cells[3],
                "availability": None,          # set per game date below
                "comment": cells[4] if len(cells) > 4 else None,
            })
    df = pd.DataFrame(rows, columns=["team_name", "player", "position",
                                     "est_return", "status", "availability",
                                     "comment"])
    return with_availability(df, us_today())


def with_availability(report: pd.DataFrame, game_date: date) -> pd.DataFrame:
    """(Re)classify every listing for a specific game date."""
    report = report.copy()
    report["availability"] = [
        classify(s, c, r, game_date) for s, c, r in
        zip(report["status"], report["comment"], report["est_return"])]
    return report


def fetch_html() -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_context(
                extra_http_headers={"User-Agent": USER_AGENT}).new_page()
            page.goto(ESPN_URL, timeout=45000, wait_until="domcontentloaded")
            page.wait_for_selector(".Table__Title", timeout=20000)
            return page.content()
        finally:
            browser.close()


def team_abbreviations(db) -> Dict[str, str]:
    """Full team name -> abbreviation, from the teams table."""
    return dict(pd.read_sql("SELECT name, abbreviation FROM teams", db.bind)
                .itertuples(index=False, name=None))


def fetch_report(db) -> pd.DataFrame:
    """Live report with team abbreviations. Unknown team names raise."""
    report = parse_injuries(fetch_html())
    names = team_abbreviations(db)
    unknown = sorted(set(report["team_name"]) - set(names))
    if unknown:
        raise ValueError(f"ESPN team names not in teams table: {unknown}")
    report.insert(0, "team", report["team_name"].map(names))
    return report


def archive(report: pd.DataFrame, path: Path = ARCHIVE_PATH,
            now: datetime | None = None) -> int:
    """Append a snapshot. Identical rows on the same day are not duplicated,
    but a status change during the day is kept as a new row."""
    now = now or datetime.now(timezone.utc)
    snap = report.copy()
    snap["snapshot_date"] = now.date().isoformat()
    snap["first_seen_at"] = now.isoformat(timespec="seconds")
    snap = snap[ARCHIVE_COLUMNS]

    path = Path(path)
    old = (pd.read_csv(path, dtype=str, keep_default_na=False)
           if path.exists() else pd.DataFrame(columns=ARCHIVE_COLUMNS))
    key = ["snapshot_date", "team", "player", "status", "comment"]
    snap = snap.fillna("").astype(str)
    seen = set(map(tuple, old[key].astype(str).values.tolist()))
    new = snap[[tuple(r) not in seen for r in snap[key].values.tolist()]]
    if new.empty:
        return 0

    write_csv(pd.concat([old, new], ignore_index=True), path)
    return len(new)


def split_for_team(report: pd.DataFrame, team: str,
                   game_date: date) -> Dict[str, List[str]]:
    """{'out': [...], 'questionable': [...], 'probable': [...]} for one team."""
    sub = with_availability(report[report["team"] == team], game_date)
    return {k: sub.loc[sub["availability"] == k, "player"].tolist()
            for k in ("out", "questionable", "probable")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot ESPN's WNBA injury report.")
    parser.add_argument("--team", help="only print this team (abbreviation)")
    args = parser.parse_args()

    from wnba.db import SessionLocal
    db = SessionLocal()
    try:
        report = fetch_report(db)
    finally:
        db.close()

    added = archive(report)
    print(f"{len(report)} players listed across {report['team'].nunique()} teams; "
          f"{added} new rows archived to {ARCHIVE_PATH}")
    shown = report[report["team"] == args.team.upper()] if args.team else report
    for team, sub in shown.groupby("team"):
        print(f"\n{team}")
        for r in sub.itertuples(index=False):
            print(f"  {r.availability:12} {r.player:24} (ESPN: {r.status}, "
                  f"return {r.est_return})  {(r.comment or '')[:60]}")
    if args.team and shown.empty:
        print(f"\n{args.team.upper()}: nobody listed -- fully available")


if __name__ == "__main__":
    main()
