"""The league's official WNBA injury report, parsed from its PDFs.

How it is reached: https://www.wnba.com/wnba-injury-report is only an index,
filled in by JavaScript and behind bot protection. The PDFs it links to live on
a separate host that robots.txt allows (`/referee/wnba_injury/`, Crawl-Delay 1),
under a fixed name per 15-minute slot, US Eastern time:

    https://ak-static.cms.nba.com/referee/wnba_injury/Injury-Report_2026-09-21_02_00PM.pdf

So this module builds the URL for a slot directly and never touches the
protected page. Old slots stay hosted: the whole 2026 season is available from
2026-05-07 (nothing earlier exists at this address).

Compared with ESPN (wnba.scraping.injuries): these are the statuses teams file
with the league for a specific game (Out / Doubtful / Questionable / Probable),
so there are no stale comments to second-guess. Long-term absences (e.g. Azzi
Fudd, out for the season) are NOT listed -- harmless, since a player out for
weeks has ~0 recent minutes share. Teams that have not filed yet appear as
"NOT YET SUBMITTED"; for those, fall back to ESPN.

Usage:
    python -m wnba.scraping.official_injuries                    # latest report
    python -m wnba.scraping.official_injuries --backfill 2026-05-07 2026-09-21
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from wnba.paths import DATA_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://ak-static.cms.nba.com/referee/wnba_injury/Injury-Report_{}.pdf"
ARCHIVE_PATH = DATA_DIR / "official_injury_reports.csv"
ET = ZoneInfo("America/New_York")
CRAWL_DELAY = 1.0          # robots.txt Crawl-Delay on ak-static.cms.nba.com
USER_AGENT = "Mozilla/5.0 (wnba_predictor; personal research)"

COLUMNS = ["report_time", "game_date", "game_time", "matchup", "team_name",
           "player", "status", "reason"]
# Status -> how predictions treat it. Doubtful players mostly sit but not
# always, so they stay a scenario like Questionable until the archive can
# measure the real rates (see played_rates()).
AVAILABILITY = {"out": "out", "doubtful": "questionable",
                "questionable": "questionable", "probable": "probable",
                "available": "available"}


# --------------------------------------------------------------------------
# URLs and download
# --------------------------------------------------------------------------
def slot_name(ts: datetime) -> str:
    """Eastern-time datetime -> '2026-09-21_02_00PM' (rounded down to 15 min)."""
    ts = ts.replace(minute=ts.minute - ts.minute % 15, second=0, microsecond=0)
    return ts.strftime("%Y-%m-%d_%I_%M%p")


def download(slot: str) -> Optional[bytes]:
    """PDF bytes for a slot, or None if that slot does not exist."""
    req = urllib.request.Request(BASE_URL.format(slot), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):          # the host answers 403 for missing files
            return None
        raise
    finally:
        time.sleep(CRAWL_DELAY)
    return body if body[:5] == b"%PDF-" else None


def latest_report(now: Optional[datetime] = None,
                  look_back_hours: int = 6) -> Optional[tuple[str, bytes]]:
    """Most recent published slot at or before `now` (Eastern)."""
    now = (now or datetime.now(ET)).astimezone(ET)
    for i in range(look_back_hours * 4):
        slot = slot_name(now - timedelta(minutes=15 * i))
        body = download(slot)
        if body:
            return slot, body
    return None


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
HEADERS = [("Game", "Date", "game_date"), ("Game", "Time", "game_time"),
           ("Matchup", None, "matchup"), ("Team", None, "team_name"),
           ("Player", "Name", "player"), ("Current", "Status", "status"),
           ("Reason", None, "reason")]


def _column_starts(words: List[dict]) -> Optional[List[tuple[float, str]]]:
    """x-position where each column starts, read from the header row."""
    starts = []
    for first, second, name in HEADERS:
        hits = [w for w in words if w["text"] == first]
        if second:
            hits = [w for w in hits if any(v["text"] == second and abs(v["top"] - w["top"]) < 2
                                           and 0 < v["x0"] - w["x1"] < 10 for v in words)]
        if not hits:
            return None
        starts.append((min(h["x0"] for h in hits), name))
    return sorted(starts)


def parse_pdf(body: bytes, report_time: str = "") -> pd.DataFrame:
    """Official injury report PDF -> one row per listed player.

    pdfplumber's table extraction returns only the header on these files, and
    plain text glues words together ("AtlantaDream"), so rows are rebuilt
    from word positions: each word goes to the column whose header starts at
    or left of it. Date, time, matchup and team are printed only when they
    change, so they carry forward to the rows below.
    """
    import pdfplumber

    rows: List[dict] = []
    carry = {"game_date": None, "game_time": None, "matchup": None, "team_name": None}
    with pdfplumber.open(io.BytesIO(body)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5, keep_blank_chars=False)
            starts = _column_starts(words)
            if starts is None:
                continue
            header_top = min(w["top"] for w in words if w["text"] == "Reason")
            lines: dict = {}
            for w in words:
                if w["top"] <= header_top + 2:
                    continue
                lines.setdefault(round(w["top"]), []).append(w)

            for top in sorted(lines):
                cells = {name: [] for _, name in starts}
                for w in sorted(lines[top], key=lambda w: w["x0"]):
                    col = starts[0][1]
                    for x, name in starts:
                        if w["x0"] >= x - 3:
                            col = name
                    cells[col].append(w["text"])
                cell = {k: " ".join(v) for k, v in cells.items()}
                full = " ".join(v for v in cell.values() if v)
                if re.match(r"Page\s*\d+\s*of\s*\d+", full):
                    continue                          # footer

                context = any(cell[k] for k in carry)
                if not context and not cell["player"] and not cell["status"]:
                    if cell["reason"] and rows:       # wrapped reason text
                        rows[-1]["reason"] = " ".join(
                            x for x in (rows[-1]["reason"], cell["reason"]) if x)
                    continue

                for k in carry:
                    if cell[k]:
                        carry[k] = cell[k]
                        if k == "game_date":          # new date resets the rest
                            carry.update(game_time=None, matchup=None, team_name=None)
                        if k == "matchup":
                            carry["team_name"] = None

                player, status = cell["player"], cell["status"]
                if "NOT YET SUBMITTED" in full.upper():
                    player, status = None, "Not Yet Submitted"
                    cell["reason"] = ""
                elif not player and not status:
                    continue                          # context-only line
                elif "," in player:                   # "Copper, Kahleah" -> "Kahleah Copper"
                    last, first = [s.strip() for s in player.split(",", 1)]
                    player = f"{first} {last}"
                rows.append({"report_time": report_time, **carry, "player": player,
                             "status": status, "reason": cell["reason"] or None})

    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    return df


def availability(status: str) -> str:
    return AVAILABILITY.get((status or "").strip().lower(), "unknown")


# --------------------------------------------------------------------------
# archive
# --------------------------------------------------------------------------
def archive(df: pd.DataFrame, path: Path = ARCHIVE_PATH) -> int:
    """Append parsed rows, skipping rows already stored for the same report."""
    if df.empty:
        return 0
    path = Path(path)
    old = (pd.read_csv(path, dtype=str, keep_default_na=False)
           if path.exists() else pd.DataFrame(columns=COLUMNS))
    new = df.fillna("").astype(str)
    key = ["report_time", "game_date", "team_name", "player", "status"]
    seen = set(map(tuple, old[key].values.tolist()))
    new = new[[tuple(r) not in seen for r in new[key].values.tolist()]]
    if new.empty:
        return 0
    out = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def backfill(start: date, end: date, slot_time: str = "12_00PM") -> None:
    """One report per day, taken at `slot_time` Eastern -- before nearly every
    tip-off, so it records pre-game status. Falls back a few slots if that
    exact file is missing. Days with no games have no report and are skipped."""
    fallback = ["12_00PM", "11_45AM", "11_30AM", "12_15PM", "12_30PM"]
    order = [slot_time] + [s for s in fallback if s != slot_time]
    day, added, found = start, 0, 0
    while day <= end:
        for s in order:
            slot = f"{day.isoformat()}_{s}"
            body = download(slot)
            if body:
                added += archive(parse_pdf(body, slot))
                found += 1
                break
        day += timedelta(days=1)
    logger.info("backfill %s..%s: %d daily reports, %d new rows -> %s", start, end, found, added, ARCHIVE_PATH)


# --------------------------------------------------------------------------
# how often each status actually plays
# --------------------------------------------------------------------------
# Measured on the 2026 season: 1,105 official listings checked against box
# scores (164 of 170 listed players matched by name; the rest never played
# for that team). Used when the archive or database is unavailable.
DEFAULT_PLAY_RATES = {"Out": 0.01, "Doubtful": 0.07, "Questionable": 0.42,
                      "Probable": 0.87, "Available": 1.0}


def measure_play_rates(db, path: Path = ARCHIVE_PATH) -> dict:
    """P(played | status) from the archive joined to player box scores.

    The latest pre-game listing per player-game is used. A player absent from
    the box score counts as not having played. Rates are smoothed with one
    played and one sat pseudo-observation, so small groups (Doubtful) are not
    pinned to exactly 0 or 1.
    """
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_PLAY_RATES)
    rep = pd.read_csv(path)
    rep = rep[rep["player"].notna() & rep["status"].isin(list(DEFAULT_PLAY_RATES))]
    if rep.empty:
        return dict(DEFAULT_PLAY_RATES)
    rep["rt"] = pd.to_datetime(rep["report_time"], format="%Y-%m-%d_%I_%M%p", errors="coerce")
    rep = rep.sort_values("rt").drop_duplicates(["game_date", "team_name", "player"], keep="last")

    teams = pd.read_sql("SELECT id AS team_id, name AS team_name FROM teams", db.bind)
    logs = pd.read_sql("SELECT date, team_id, athlete_name, minutes FROM player_game_logs "
                       "WHERE date >= %(d)s", db.bind, params={"d": rep["game_date"].min()})
    if logs.empty:
        return dict(DEFAULT_PLAY_RATES)
    logs["game_date"] = logs["date"].astype(str)
    logs["played"] = pd.to_numeric(logs["minutes"], errors="coerce").fillna(0) > 0
    have = set(zip(logs["game_date"], logs["team_id"]))

    rep = rep.merge(teams, on="team_name", how="inner")
    rep = rep[[(d, t) in have for d, t in zip(rep["game_date"], rep["team_id"])]]
    m = rep.merge(logs.rename(columns={"athlete_name": "player"})[
        ["game_date", "team_id", "player", "played"]],
        on=["game_date", "team_id", "player"], how="left")
    m["played"] = m["played"].fillna(False).astype(bool)

    rates = dict(DEFAULT_PLAY_RATES)
    for status, sub in m.groupby("status"):
        if status != "Available":
            rates[status] = (sub["played"].sum() + 1) / (len(sub) + 2)
    return rates


# --------------------------------------------------------------------------
# one report for a game date: official where filed, ESPN otherwise
# --------------------------------------------------------------------------
ESPN_TO_STATUS = {"out": "Out", "questionable": "Questionable", "probable": "Probable"}


def combined_report(db, game_date: date) -> pd.DataFrame:
    """Columns: team, player, status, source.

    Official statuses are used for every team that has filed for `game_date`.
    Teams marked NOT YET SUBMITTED, or missing from the official report, fall
    back to ESPN's listing (classified by wnba.scraping.injuries). Both raw
    reports are archived.
    """
    from wnba.scraping import injuries as espn

    names = espn.team_abbreviations(db)
    rows = []
    filed = set()

    hit = latest_report()
    if hit is not None:
        slot, body = hit
        off = parse_pdf(body, slot)
        archive(off)
        off = off[off["game_date"] == game_date.isoformat()]
        for r in off.itertuples(index=False):
            team = names.get(r.team_name)
            if team is None or r.status == "Not Yet Submitted" or not r.player:
                continue
            filed.add(team)
            rows.append({"team": team, "player": r.player, "status": r.status,
                         "source": "official"})

    live = espn.fetch_report(db)
    espn.archive(live)
    live = espn.with_availability(live, game_date)
    for r in live.itertuples(index=False):
        if r.team in filed:
            continue
        rows.append({"team": r.team, "player": r.player,
                     "status": ESPN_TO_STATUS[r.availability], "source": "espn"})
    return pd.DataFrame(rows, columns=["team", "player", "status", "source"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Official WNBA injury report (PDF).")
    parser.add_argument("--backfill", nargs=2, type=date.fromisoformat,
                        metavar=("START", "END"))
    args = parser.parse_args()

    if args.backfill:
        backfill(*args.backfill)
        return

    hit = latest_report()
    if hit is None:
        logger.warning("no report published in the last 6 hours")
        return
    slot, body = hit
    df = parse_pdf(body, slot)
    added = archive(df)
    logger.info("report %s: %d rows, %d new archived", slot, len(df), added)
    for (gd, mu), sub in df.groupby(["game_date", "matchup"], sort=False):
        print(f"\n{gd}  {mu}")
        for r in sub.itertuples(index=False):
            who = r.player or "-"
            print(f"  {r.team_name:24} {who:24} {r.status:18} {r.reason or ''}")


if __name__ == "__main__":
    main()
