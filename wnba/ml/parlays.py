"""Parlays built from the picks already logged, compared with the same legs as singles.

A parlay here is a bundle of your logged picks: game picks from the prediction
log (`tracking.py`) and prop picks from the props log (`props_log.py`). Its odds
are the product of the odds each leg was logged at, so it is graded from the
same results as the singles -- nothing is entered twice.

Caveat: a bookmaker prices a same-game parlay ("bet builder") itself, usually
below the product of the single odds, because legs in one game are correlated
(WAS winning and WAS players going over move together). The product here is
the ordinary multi-game accumulator price; for same-game legs it is optimistic.

    python -m wnba.ml.parlays add --date 2026-09-30            # all your picks that day
    python -m wnba.ml.parlays add --date 2026-09-30 --games-only
    python -m wnba.ml.parlays add --date 2026-10-04 --name "ATL same game" \
           --leg "NYL@ATL:ATL" --leg "Angel Reese:reb"
    python -m wnba.ml.parlays report
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from wnba.ml import props_log, tracking
from wnba.paths import DATA_DIR

logger = logging.getLogger(__name__)

PARLAY_PATH = DATA_DIR / "parlays.csv"
COLUMNS = ["parlay_id", "name", "dates", "legs", "created_at"]


def user_legs(dates: List[str], games_only: bool = False) -> pd.DataFrame:
    """Every logged pick on these US dates: odds taken, model probability, outcome
    (True won, False lost, None void or not settled yet)."""
    rows = []
    g = tracking.read_log()
    g = g[g["game_date"].astype(str).isin(dates) & g["user_pick"].notna() & g["home_odds"].notna()]
    for r in g.itertuples():
        home = r.user_pick == r.home
        won = None if pd.isna(r.home_won) else bool(r.home_won) == home
        rows.append({"leg": f"game:{r.game_date}:{r.away}@{r.home}:{r.user_pick}",
                     "label": f"{r.user_pick} win ({r.away}@{r.home})",
                     "odds": r.home_odds if home else r.away_odds,
                     "model_p": r.model_home_prob if home else 1 - r.model_home_prob,
                     "won": won})
    if not games_only:
        p = props_log.read_log()
        p = p[p["game_date"].isin(dates) & p["user_side"].notna()]
        for r in p.itertuples():
            over = r.user_side == "over"
            won = None if r.result in (None, "void", "push") else r.result == r.user_side
            rows.append({"leg": f"prop:{r.game_date}:{r.player}:{r.stat}:{r.line:g}:{r.user_side}",
                         "label": f"{r.player} {r.stat} {r.user_side} {r.line:g}",
                         "odds": r.over_odds if over else r.under_odds,
                         "model_p": r.model_p_over if over else 1 - r.model_p_over,
                         "won": won})
    return pd.DataFrame(rows)


def read_parlays(path: Path = PARLAY_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype=str)


def pick_legs(legs: pd.DataFrame, selectors: List[str]) -> pd.DataFrame:
    """Choose legs by text, e.g. "NYL@ATL:ATL" or "Angel Reese:reb";
    each selector must match exactly one logged pick (case-insensitive)."""
    chosen = []
    for sel in selectors:
        hit = legs[legs["leg"].str.lower().str.contains(sel.lower(), regex=False)]
        if len(hit) != 1:
            found = hit["leg"].tolist() or "nothing"
            raise ValueError(f"{sel!r} matched {found}; make it more specific")
        chosen.append(hit.index[0])
    return legs.loc[chosen]


def add(dates: List[str], games_only: bool = False, name: Optional[str] = None,
        selectors: Optional[List[str]] = None, path: Path = PARLAY_PATH) -> dict:
    legs = user_legs(dates, games_only)
    if selectors:
        legs = pick_legs(legs, selectors)
    if len(legs) < 2:
        raise ValueError("a parlay needs at least two logged picks")
    df = read_parlays(path)
    row = {"parlay_id": str(len(df) + 1),
           "name": name or f"{'+'.join(dates)} {'games' if games_only else 'all picks'}",
           "dates": ";".join(dates), "legs": "|".join(legs["leg"]),
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    df.to_csv(tmp, index=False)
    for attempt in range(20):                  # OneDrive may briefly lock the file
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)
    return row


def settle(legs: pd.DataFrame) -> dict:
    """Parlay outcome at 1 unit and the same legs as 1-unit singles."""
    live = legs[legs["won"].notna()]
    pending = legs["won"].isna().any() and not (live["won"] == False).any()  # noqa: E712
    odds = math.prod(legs["odds"])
    model_p = math.prod(legs["model_p"])
    if (live["won"] == False).any():                                       # noqa: E712
        parlay = -1.0
    elif pending:
        parlay = None
    else:
        parlay = odds - 1
    singles = float(sum((r.odds - 1) if r.won else -1.0 for r in live.itertuples()))
    return {"odds": odds, "model_p": model_p, "model_ev": model_p * odds - 1,
            "parlay": parlay, "singles": singles, "n_settled": len(live), "n": len(legs)}


def report(path: Path = PARLAY_PATH) -> None:
    df = read_parlays(path)
    if df.empty:
        print("no parlays logged")
        return
    tot_p = tot_s = staked_s = 0.0
    n_p = 0
    for r in df.itertuples():
        all_legs = user_legs(r.dates.split(";"), games_only=False)
        legs = all_legs[all_legs["leg"].isin(r.legs.split("|"))]
        s = settle(legs)
        status = "pending" if s["parlay"] is None else ("WON" if s["parlay"] > 0 else "lost")
        print(f"\n#{r.parlay_id} {r.name}: {s['n']} legs, odds {s['odds']:.2f}, "
              f"model P(all hit) {s['model_p']:.1%} (EV {s['model_ev']:+.0%}) -> {status}")
        for l in legs.itertuples():
            mark = {True: "won", False: "lost", None: "-"}[l.won]
            print(f"    {l.label:34} {l.odds:5.2f}  model {l.model_p:4.0%}  {mark}")
        if s["parlay"] is not None:
            print(f"    1 unit as a parlay: {s['parlay']:+.2f}    "
                  f"same legs as {s['n_settled']} 1-unit singles: {s['singles']:+.2f} "
                  f"({s['singles'] / s['n_settled']:+.0%} per unit)")
            tot_p += s["parlay"]
            n_p += 1
            tot_s += s["singles"]
            staked_s += s["n_settled"]
    if n_p:
        print(f"\nTOTAL  parlays {tot_p:+.2f} on {n_p} units ({tot_p / n_p:+.0%} per unit)   "
              f"same legs as singles {tot_s:+.2f} on {staked_s:.0f} units ({tot_s / staked_s:+.0%} per unit; a leg in several parlays counts once per parlay)")
        print("note: model P(all hit) assumes legs are independent; same-game legs are not.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Parlays from logged picks vs the same singles.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--date", nargs="+", required=True, help="US game date(s), YYYY-MM-DD")
    a.add_argument("--games-only", action="store_true", help="only the game (moneyline) picks")
    a.add_argument("--name")
    a.add_argument("--leg", action="append", dest="legs",
                   help='pick one leg by text, repeatable: --leg "NYL@ATL:ATL" --leg "Angel Reese:reb"')
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "add":
        row = add(args.date, args.games_only, args.name, args.legs)
        logger.info("parlay #%s logged: %s (%d legs)", row["parlay_id"], row["name"],
                    row["legs"].count("|") + 1)
    else:
        report()


if __name__ == "__main__":
    main()
