"""Read player-prop lines from a bookmaker event page YOU saved (Ctrl+S), or
from the text that `vbet_collect.js` gathers in your own Chrome (Claude in
Chrome runs it on the game page; one market per line, see LADDER_CODES).

Nothing is fetched: the page is one you opened yourself as a normal visitor,
so the bookmaker's terms and bot protection are not an issue. Written for
VBET Ukraine's layout (text like
"Бріанна Стюарт (НЙ Ліберті) Очки | Більше | Менше | 23.5 | 1.85 | 1.85"),
but it only relies on that text pattern.

Ukrainian player names are transliterated and matched to the two rosters in
player_game_logs; every match is printed with a similarity score so a wrong
match is visible before anything is logged.

Usage:
    python -m wnba.scraping.saved_lines data/vbet_2026-09-29_LVA@IND_pts.html --home IND --away LVA --date 2026-09-29
    python -m wnba.scraping.saved_lines data/vbet_2026-09-30_ATL@WAS.txt --home WAS --away ATL --date 2026-09-30 --log --auto-injuries
"""
from __future__ import annotations

import argparse
import difflib
import logging
import pandas as pd
import re
from datetime import date
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# market title (lower-case fragment) -> our stat key
MARKETS = {"очки": "pts", "підбирання": "reb", "асист": "ast", "ассист": "ast",
           "передач": "ast", "триочков": "fg3m",
           # VBET shows a few markets in English
           "points": "pts", "rebounds": "reb", "assists": "ast", "three": "fg3m"}
# ladder codes written by vbet_collect.js: "#R7" = "7 or more rebounds" = over 6.5
LADDER_CODES = {"P": "pts", "R": "reb", "A": "ast", "T": "fg3m"}
# bookmaker name -> roster name, where spelling alone can't find her
ALIASES = {"Ніа Коффі": "Nia Brodie",          # Nia Coffey, listed as Brodie from 2026
           "Джекі Янг": "Jackie Young"}

TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ye",
    "ж": "zh", "з": "z", "и": "y", "і": "i", "ї": "yi", "й": "y", "к": "k", "л": "l",
    "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ь": "",
    "ю": "yu", "я": "ya", "'": "", "’": "",
}
PROP_RE = re.compile(
    r"^(?P<name>[^()|]+?)\s*\((?P<team>[^)]+)\)\s*(?P<market>[^|]+?)\s*\|\s*Більше\s*\|\s*Менше\s*\|"
    r"\s*(?P<line>\d+(?:\.\d+)?)\s*\|\s*(?P<over>\d+(?:\.\d+)?)\s*\|\s*(?P<under>\d+(?:\.\d+)?)")


def translit(text: str) -> str:
    text = text.lower().replace("дж", "j")      # Джонс -> jons, not dzhons
    return "".join(TRANSLIT.get(c, c) for c in text)


def page_text(path: Path) -> str:
    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8", errors="replace"), "html.parser")
    for s in soup(["script", "style", "noscript", "svg"]):
        s.decompose()
    lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
    return " | ".join(lines)


def parse_props(path: Path) -> pd.DataFrame:
    """Every 'Player (Team) Market | Більше | Менше | line | over | under' block."""
    text = page_text(path)
    rows = []
    # split so each chunk starts at a player-market header
    for chunk in re.split(r"\s\|\s(?=[А-ЯІЇЄҐ][^|()]+\([^)]+\)\s*[^|]+\|\s*Більше)", text):
        m = PROP_RE.match(chunk.strip())
        if not m:
            continue
        stat = market_stat(m["market"])
        if stat is None:
            continue
        rows.append({"name_ua": m["name"].strip(), "team_ua": m["team"].strip(),
                     "stat": stat, "line": float(m["line"]),
                     "over_odds": float(m["over"]), "under_odds": float(m["under"])})
    return pd.DataFrame(rows)


def market_stat(title: str):
    t = title.strip().lower()
    if ("очки" in t and "підбир" in t and ("асист" in t or "ассист" in t)) or             ("points" in t and "rebounds" in t and "assists" in t):
        return "pra"
    if " та " in t or " and " in t or "," in t:
        return None                                    # other combos: not modelled
    return next((v for k, v in MARKETS.items() if k in t), None)


def _odds(tok: str):
    try:
        v = float(tok)
    except ValueError:
        return None
    return v if v > 1.0 else None


def parse_collected(path: Path) -> pd.DataFrame:
    """Text from vbet_collect.js: '#R7;Name;1.08;Name;2.75' ladders and
    'Name (Team) Market;Більше;Менше;line;over;under' over/unders."""
    rows = []
    names: list = []                                   # "#N;name0;name1" -> "@0", "@1"
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        tok = [t.strip() for t in raw.split(";") if t.strip()]
        if not tok:
            continue
        if tok[0] == "#N":
            names = tok[1:]
            continue
        tok = [names[int(t[1:])] if re.fullmatch(r"@\d+", t) else t for t in tok]
        lad = re.fullmatch(r"#([PRAT])(\d+)", tok[0])
        if lad:
            stat, n = LADDER_CODES[lad[1]], int(lad[2])
            i = 1
            while i + 1 < len(tok):
                name, odds = tok[i], _odds(tok[i + 1])
                if _odds(name) is None and odds is not None:
                    rows.append({"name_ua": name, "team_ua": None, "stat": stat,
                                 "line": n - 0.5, "over_odds": odds, "under_odds": None})
                    i += 2
                else:
                    i += 1                             # malformed pair: skip a token
            continue
        m = re.fullmatch(r"(?P<name>[^()]+?)\s*\((?P<team>[^)]+)\)\s*(?P<market>.+)", tok[0])
        if m and len(tok) >= 6 and tok[1] == "Більше" and tok[2] == "Менше":
            stat = market_stat(m["market"])
            over, under = _odds(tok[4]), _odds(tok[5])
            if stat and over and under:
                rows.append({"name_ua": m["name"].strip(), "team_ua": m["team"].strip(),
                             "stat": stat, "line": float(tok[3]),
                             "over_odds": over, "under_odds": under})
    return pd.DataFrame(rows)


def match_names(props: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Map Ukrainian names to roster names of the two teams (current season)."""
    from wnba.db import SessionLocal
    db = SessionLocal()
    try:
        roster = pd.read_sql("""
            SELECT DISTINCT p.athlete_name, t.abbreviation team
            FROM player_game_logs p JOIN teams t ON t.id = p.team_id
            WHERE t.abbreviation IN (%(h)s, %(a)s)
              AND p.season = (SELECT max(season) FROM player_game_logs)""",
            db.bind, params={"h": home, "a": away})
    finally:
        db.close()
    names = roster["athlete_name"].tolist()
    keys = [n.lower().replace("-", " ") for n in names]
    out = []
    for r in props.itertuples(index=False):
        t = translit(r.name_ua).replace("-", " ")
        scored = sorted(((difflib.SequenceMatcher(None, t, k).ratio(), n)
                         for k, n in zip(keys, names)), reverse=True)
        score, best = scored[0] if scored else (0.0, None)
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        weak = False
        if 0.45 <= score < 0.6 and score - runner_up >= 0.2:
            score, weak = 0.6, True         # weak spelling match, but no rival name close
        if r.name_ua in ALIASES and ALIASES[r.name_ua] in names:
            score, best, weak = 1.0, ALIASES[r.name_ua], False
        out.append({**r._asdict(), "player": best, "match": round(score, 2),
                    "weak": weak, "translit": t})
    return pd.DataFrame(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Read prop lines from a saved bookmaker page.")
    ap.add_argument("html", type=Path)
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--date", type=date.fromisoformat, required=True, help="US game date")
    ap.add_argument("--min-match", type=float, default=0.6,
                    help="lowest name-similarity accepted (others are skipped)")
    ap.add_argument("--ladder-odds", type=float, nargs=2, default=(1.30, 5.00),
                    metavar=("MIN", "MAX"),
                    help="keep ladder rungs priced in this range (default 1.30 5.00)")
    ap.add_argument("--log", action="store_true", help="add the props to the props log")
    ap.add_argument("--auto-injuries", action="store_true",
                    help="project with the latest injury report (official, ESPN fallback)")
    ap.add_argument("--closing", action="store_true",
                    help="store these prices as closing odds for props already logged")
    args = ap.parse_args()

    props = parse_collected(args.html) if args.html.suffix == ".txt" else parse_props(args.html)
    if props.empty:
        raise SystemExit("no player props found in that page")
    m = match_names(props, args.home.upper(), args.away.upper())
    logger.info("%d props found", len(m))
    if m["weak"].any():
        logger.warning("weak name matches (check the table): %s",
                       ", ".join(sorted(set(m.loc[m["weak"], "name_ua"]))))
    names = m.drop_duplicates("name_ua")
    print(f"{'bookmaker name':26} {'-> roster':22} {'match':>5}  props")
    for r in names.itertuples():
        flag = ("   <- SKIPPED, check" if r.match < args.min_match
                else "   <- weak match, check" if r.weak else "")
        n = int((m["name_ua"] == r.name_ua).sum())
        print(f"{r.name_ua[:26]:26} {str(r.player)[:22]:22} {r.match:5.2f}  {n:5d}{flag}")

    lo, hi = args.ladder_odds
    ladder = m["under_odds"].isna()
    keep = ~ladder | m["over_odds"].between(lo, hi)
    good = m[(m["match"] >= args.min_match) & keep]
    if ladder.any():
        logger.info("ladder rungs kept: %d of %d (odds %.2f-%.2f)",
                    int((ladder & keep).sum()), int(ladder.sum()), lo, hi)
    suffix = "_closing" if args.closing else ""
    csv = args.html.with_name(f"props_{args.date}_{args.away.upper()}@{args.home.upper()}{suffix}.csv")
    pd.DataFrame({"home": args.home.upper(), "away": args.away.upper(), "date": args.date.isoformat(),
                  "player": good["player"], "stat": good["stat"], "line": good["line"],
                  "over_odds": good["over_odds"], "under_odds": good["under_odds"]}).to_csv(csv, index=False)
    logger.info("%d written to %s", len(good), csv)
    if args.closing:
        from wnba.ml import props_log
        done, skipped = props_log.closing_file(csv)
        logger.info("closing odds stored for %d logged props; %d not in the log "
                    "(new, or the line moved)", done, skipped)
    elif args.log:
        from wnba.ml import props_log
        props_log.add_file(csv, auto_injuries=args.auto_injuries)
        print()
        props_log.rank_pending()


if __name__ == "__main__":
    main()
