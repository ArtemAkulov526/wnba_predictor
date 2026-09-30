"""Player-prop log: lines, the model's view, your pick, results.

The prop counterpart of wnba.ml.tracking. One row per (game, player, stat,
line). Stored in data/props_log.csv for the same reason as the game log: a
pre-game view cannot be re-created later.

A prop is VOID if the player does not play (sportsbooks refund those), and
does not count toward any record.

Usage:
    # one prop
    python -m wnba.ml.props_log add --home NYL --away MIN --date 2026-09-29 \\
        --player "Breanna Stewart" --stat pts --line 21.5 \\
        --over-odds 1.87 --under-odds 1.87 --pick over --confidence 0.55

    # many props from a CSV you fill in from your bookmaker:
    #   home,away,date,player,stat,line,over_odds,under_odds[,pick,confidence,tags]
    python -m wnba.ml.props_log add-file data/props_tonight.csv

    python -m wnba.ml.props_log rank            # pending props by model EV
    python -m wnba.ml.props_log closing --date 2026-09-29 --player "Breanna Stewart" \\
        --stat pts --line 21.5 --over-odds 1.80 --under-odds 1.95
    python -m wnba.ml.props_log grade           # results from player box scores
    python -m wnba.ml.props_log report
    python -m wnba.ml.props_log list

Stats: pts (points), reb (rebounds), ast (assists), fg3m (threes),
pra (points + rebounds + assists).

Ladder markets ("10 or more rebounds", one price only) are stored as an over
at line N - 0.5 with no under odds; their market % includes the margin.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from wnba.ml.tracking import expected_value, market_probability
from wnba.paths import DATA_DIR

LOG_PATH = DATA_DIR / "props_log.csv"
STAT_ALIASES = {"pts": "pts", "points": "pts", "reb": "reb", "rebounds": "reb",
                "ast": "ast", "assists": "ast", "fg3m": "fg3m", "threes": "fg3m",
                "3pm": "fg3m", "3s": "fg3m", "pra": "pra", "par": "pra"}
COLUMNS = ["game_date", "home", "away", "team", "player", "stat", "line",
           "over_odds", "under_odds", "odds_at",
           "model_mean", "model_p_over",
           "user_side", "user_prob", "tags", "note",
           "closing_over_odds", "closing_under_odds",
           "actual", "result", "graded_at", "created_at"]
NUMERIC = ["line", "over_odds", "under_odds", "model_mean", "model_p_over",
           "user_prob", "closing_over_odds", "closing_under_odds", "actual"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm_stat(stat: str) -> str:
    key = stat.strip().lower()
    if key not in STAT_ALIASES:
        raise ValueError(f"unknown stat {stat!r}; use pts, reb, ast, fg3m or pra")
    return STAT_ALIASES[key]


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def read_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS].copy()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in set(COLUMNS) - set(NUMERIC):
        df[c] = pd.Series([None if (v is None or (isinstance(v, float) and math.isnan(v)))
                           else str(v) for v in df[c]], index=df.index, dtype="object")
    return df


def write_log(df: pd.DataFrame, path: Path = LOG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[COLUMNS].sort_values(["game_date", "player", "stat"], kind="stable")
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    # OneDrive (or an editor) briefly locks the file while syncing it; retry
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def _find(df: pd.DataFrame, game_date: str, player: str, stat: str, line: float,
          ladder: Optional[bool] = None) -> Optional[int]:
    """ladder=True/False tells a one-sided rung ("5+ rebounds" = over 4.5) from
    the over/under at the same line; None matches either."""
    mask = ((df["game_date"] == game_date) & (df["player"].str.lower() == player.lower())
            & (df["stat"] == stat) & ((df["line"] - line).abs() < 1e-9))
    if ladder is not None:
        mask &= df["under_odds"].isna() == ladder
    hit = df.index[mask]
    return int(hit[0]) if len(hit) else None


# --------------------------------------------------------------------------
# model view
# --------------------------------------------------------------------------
_PROJ_CACHE: Dict[Tuple[str, str, str], pd.DataFrame] = {}


def model_view(home: str, away: str, game_date: date, player: str, stat: str,
               line: float, auto_injuries: bool = False) -> Tuple[str, str, float, float]:
    """(canonical player name, team, projected mean, P(over line))."""
    from wnba.ml import props
    key = (home, away, game_date.isoformat())
    if key not in _PROJ_CACHE:
        report = rates = None
        if auto_injuries:
            from wnba.db import SessionLocal
            from wnba.scraping.official_injuries import combined_report, measure_play_rates
            db = SessionLocal()
            try:
                report, rates = combined_report(db, game_date), measure_play_rates(db)
            finally:
                db.close()
        _PROJ_CACHE[key] = props.project_game(home, away, game_date, report, rates)
    res = _PROJ_CACHE[key]
    names = res["athlete_name"].str.lower()
    q = player.strip().lower()
    match = res[names == q]
    if match.empty:
        match = res[names.str.contains(q, regex=False)]
    if len(match) != 1:
        found = match["athlete_name"].tolist()
        raise ValueError(f"{player!r} matched {found or 'nobody'} in {away}@{home}")
    row = match.iloc[0]
    mean = float(row[f"proj_{stat}"])
    p_over = res.attrs["models"][stat].p_over(mean, line)
    return row["athlete_name"], row["team"], mean, p_over


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------
def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def market_over(over_odds: float, under_odds: Optional[float]) -> float:
    """Market P(over): no-vig with both sides, raw 1/odds for a one-sided ladder."""
    if _missing(under_odds):
        return 1.0 / over_odds
    return market_probability(over_odds, under_odds)


def add_prop(home: str, away: str, game_date: date, player: str, stat: str, line: float,
             over_odds: float, under_odds: Optional[float], pick: Optional[str] = None,
             confidence: Optional[float] = None, tags: Optional[str] = None,
             note: Optional[str] = None, auto_injuries: bool = False,
             path: Path = LOG_PATH) -> dict:
    stat = norm_stat(stat)
    if pick is not None:
        pick = pick.strip().lower()
        if pick not in ("over", "under"):
            raise ValueError("--pick must be over or under")
    if confidence is not None and not 0.5 <= confidence <= 0.99:
        raise ValueError("confidence is your probability that YOUR side wins: 0.50-0.99")
    if _missing(under_odds):
        under_odds = None
        if pick == "under":
            raise ValueError("a one-sided (ladder) prop has no under to pick")
    market_over(over_odds, under_odds)                            # validates odds

    name, team, mean, p_over = model_view(home, away, game_date, player, stat, line, auto_injuries)
    df = read_log(path)
    i = _find(df, game_date.isoformat(), name, stat, line, ladder=under_odds is None)
    if i is not None and df.at[i, "result"] is not None:
        raise ValueError("that prop is already graded; it is locked")
    row = {
        "game_date": game_date.isoformat(), "home": home, "away": away, "team": team,
        "player": name, "stat": stat, "line": line,
        "over_odds": over_odds, "under_odds": under_odds, "odds_at": _now(),
        "model_mean": round(mean, 2), "model_p_over": round(p_over, 4),
        "user_side": pick, "user_prob": confidence,
        "tags": tags, "note": note,
        "closing_over_odds": None, "closing_under_odds": None,
        "actual": None, "result": None, "graded_at": None,
        "created_at": _now() if i is None else df.at[i, "created_at"],
    }
    if i is None:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        for k, v in row.items():
            if k in ("user_side", "user_prob", "tags", "note") and v is None:
                continue                        # keep an existing pick when re-adding
            if (k in ("over_odds", "under_odds", "odds_at") and pick is None
                    and df.at[i, "user_side"] is not None):
                continue                        # keep the odds the user's pick was taken at
            df.at[i, k] = v
    write_log(df, path)
    return row


def add_file(csv_path: Path, auto_injuries: bool = False, path: Path = LOG_PATH) -> int:
    lines = pd.read_csv(csv_path)
    n = 0
    for r in lines.itertuples(index=False):
        get = lambda k: (getattr(r, k) if hasattr(r, k) and not pd.isna(getattr(r, k)) else None)
        add_prop(r.home.upper(), r.away.upper(), date.fromisoformat(str(r.date)), r.player,
                 r.stat, float(r.line), float(r.over_odds), get("under_odds"),
                 pick=get("pick"), confidence=get("confidence"), tags=get("tags"),
                 auto_injuries=auto_injuries, path=path)
        n += 1
    return n


def record_closing(game_date: date, player: str, stat: str, line: float,
                   over_odds: float, under_odds: float, path: Path = LOG_PATH) -> None:
    df = read_log(path)
    i = _find(df, game_date.isoformat(), player, norm_stat(stat), line)
    if i is None:
        raise ValueError("no such prop in the log")
    df.at[i, "closing_over_odds"], df.at[i, "closing_under_odds"] = over_odds, under_odds
    write_log(df, path)


def closing_file(csv_path: Path, path: Path = LOG_PATH) -> Tuple[int, int]:
    """Closing odds for every logged prop found in a lines CSV (same columns as
    add-file). A prop whose line moved has no closing price at the logged line
    and is skipped. Returns (updated, skipped)."""
    lines = pd.read_csv(csv_path)
    df = read_log(path)
    done = skipped = 0
    for r in lines.itertuples(index=False):
        under = None if pd.isna(r.under_odds) else float(r.under_odds)
        i = _find(df, str(r.date), r.player, norm_stat(r.stat), float(r.line), ladder=under is None)
        if i is None or df.at[i, "result"] is not None:
            skipped += 1
            continue
        df.at[i, "closing_over_odds"], df.at[i, "closing_under_odds"] = float(r.over_odds), under
        done += 1
    write_log(df, path)
    return done, skipped


def settle(actual: Optional[float], line: float) -> str:
    if actual is None:
        return "void"
    if actual > line:
        return "over"
    if actual < line:
        return "under"
    return "push"                               # whole-number lines only


def grade(path: Path = LOG_PATH) -> Tuple[int, int]:
    """Fill results from player_game_logs. A player with no minutes -> void."""
    from wnba.db import SessionLocal
    df = read_log(path)
    pending = df.index[df["result"].isna()]
    if len(pending) == 0:
        return 0, 0
    db = SessionLocal()
    try:
        box = pd.read_sql("""SELECT to_char(date,'YYYY-MM-DD') game_date, lower(athlete_name) player,
                                    minutes, pts, reb, ast, fg3m, pts + reb + ast AS pra
                             FROM player_game_logs WHERE date >= %(d)s""", db.bind,
                          params={"d": df.loc[pending, "game_date"].min()})
        played_dates = set(pd.read_sql("SELECT DISTINCT to_char(date,'YYYY-MM-DD') d "
                                       "FROM player_game_logs", db.bind)["d"])
    finally:
        db.close()
    box = box.set_index(["game_date", "player"])
    graded = 0
    for i in pending:
        d, p, stat = df.at[i, "game_date"], df.at[i, "player"].lower(), df.at[i, "stat"]
        if d not in played_dates:
            continue                                  # box scores not posted yet
        actual = None
        if (d, p) in box.index:
            b = box.loc[(d, p)]
            b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
            if pd.notna(b["minutes"]) and b["minutes"] > 0:
                actual = float(b[stat])
        df.at[i, "actual"] = actual
        df.at[i, "result"] = settle(actual, df.at[i, "line"])
        df.at[i, "graded_at"] = _now()
        graded += 1
    write_log(df, path)
    return graded, len(pending) - graded


def record_result(game_date: date, player: str, stat: str, line: float,
                  actual: Optional[float], path: Path = LOG_PATH) -> str:
    """Grade by hand; actual=None means the player did not play (void)."""
    df = read_log(path)
    i = _find(df, game_date.isoformat(), player, norm_stat(stat), line)
    if i is None:
        raise ValueError("no such prop in the log")
    df.at[i, "actual"] = actual
    df.at[i, "result"] = settle(actual, df.at[i, "line"])
    df.at[i, "graded_at"] = _now()
    write_log(df, path)
    return df.at[i, "result"]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _ev(p_over: float, over_odds: float, under_odds: Optional[float]) -> Tuple[str, float]:
    eo = expected_value(p_over, over_odds)
    if _missing(under_odds):
        return "over", eo
    eu = expected_value(1 - p_over, under_odds)
    return ("over", eo) if eo >= eu else ("under", eu)


def rank_pending(path: Path = LOG_PATH) -> None:
    df = read_log(path)
    p = df[df["result"].isna()].copy()
    if p.empty:
        print("no pending props")
        return
    p[["best_side", "best_ev"]] = [list(_ev(r.model_p_over, r.over_odds, r.under_odds))
                                   for r in p.itertuples()]
    p = p.sort_values("best_ev", ascending=False)
    print(f"{'date':10} {'player':22} {'stat':4} {'line':>5} {'proj':>5} {'P(over)':>7} "
          f"{'mkt':>5} {'best':>5} {'odds':>5} {'EV':>7}  you")
    for r in p.itertuples():
        mkt = market_over(r.over_odds, r.under_odds)
        odds = r.over_odds if r.best_side == "over" else r.under_odds
        you = "" if r.user_side is None else f"{r.user_side} {r.user_prob or 0:.0%}"
        print(f"{r.game_date} {r.player[:22]:22} {r.stat:4} {r.line:5.1f} {r.model_mean:5.1f} "
              f"{r.model_p_over:7.1%} {mkt:5.1%} {r.best_side:>5} {odds:5.2f} {r.best_ev:+7.1%}  {you}")


def report(path: Path = LOG_PATH) -> None:
    df = read_log(path)
    done = df[df["result"].notna()]
    live = done[~done["result"].isin(["void", "push"])]
    print(f"logged {len(df)}   graded {len(done)}   void {int((done['result'] == 'void').sum())}   "
          f"push {int((done['result'] == 'push').sum())}   pending {len(df) - len(done)}")
    if live.empty:
        print("nothing settled yet")
        return

    def book(rows, side_of, prob_of=None):
        n = w = 0
        units = 0.0
        clv = []
        for r in rows.itertuples():
            side = side_of(r)
            if side is None or (side == "under" and _missing(r.under_odds)):
                continue
            odds = r.over_odds if side == "over" else r.under_odds
            won = r.result == side
            n += 1
            w += won
            units += (odds - 1) if won else -1.0
            if (pd.notna(r.closing_over_odds) and not _missing(r.under_odds)
                    and pd.notna(r.closing_under_odds)):
                taken = market_probability(r.over_odds, r.under_odds)
                close = market_probability(r.closing_over_odds, r.closing_under_odds)
                clv.append((close - taken) if side == "over" else (taken - close))
        clv_txt = f"   CLV {sum(clv) / len(clv):+.1%} on {len(clv)}" if clv else ""
        return f"{w}-{n - w}  {units:+.2f} units{clv_txt}" if n else "-"

    print("\n1 unit per prop at the logged odds")
    print(f"  model, side it favours         {book(live, lambda r: 'over' if r.model_p_over >= 0.5 else 'under')}")
    print(f"  model, positive-EV only        {book(live, lambda r: (_ev(r.model_p_over, r.over_odds, r.under_odds)[0] if _ev(r.model_p_over, r.over_odds, r.under_odds)[1] > 0 else None))}")
    print(f"  you, your side                 {book(live, lambda r: r.user_side)}")
    def user_ev(r):
        if r.user_side is None or pd.isna(r.user_prob):
            return None
        odds = r.over_odds if r.user_side == "over" else r.under_odds
        return r.user_side if expected_value(r.user_prob, odds) > 0 else None
    print(f"  you, positive-EV by your number {book(live, user_ev)}")

    over_rate = (live["result"] == "over").mean()
    print(f"\nover hit {over_rate:.0%} of settled props; model's mean P(over) was "
          f"{live['model_p_over'].mean():.0%}")
    if len(live) < 50:
        print(f"note: {len(live)} settled props -- far too few to judge; keep logging.")


def list_log(path: Path = LOG_PATH) -> None:
    df = read_log(path)
    if df.empty:
        print(f"props log is empty ({path})")
        return
    for r in df.itertuples():
        res = "pending" if r.result is None else f"{r.result} ({'' if pd.isna(r.actual) else int(r.actual)})"
        you = "-" if r.user_side is None else f"{r.user_side} {r.user_prob or 0:.0%}"
        print(f"{r.game_date} {r.player[:22]:22} {r.stat:4} {r.line:5.1f}  o{r.over_odds:.2f}/u{r.under_odds:.2f}  "
              f"model {r.model_mean:5.1f} P(o) {r.model_p_over:5.1%}  you {you:10} {res}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Player-prop log.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    for k in ("--home", "--away", "--player", "--stat"):
        a.add_argument(k, required=True)
    a.add_argument("--date", type=date.fromisoformat, default=date.today())
    a.add_argument("--line", type=float, required=True)
    a.add_argument("--over-odds", type=float, required=True)
    a.add_argument("--under-odds", type=float, help="omit for a one-sided ladder market")
    a.add_argument("--pick", choices=["over", "under"])
    a.add_argument("--confidence", type=float)
    a.add_argument("--tags")
    a.add_argument("--note")
    a.add_argument("--auto-injuries", action="store_true")

    f = sub.add_parser("add-file")
    f.add_argument("csv", type=Path)
    f.add_argument("--auto-injuries", action="store_true")

    c = sub.add_parser("closing")
    c.add_argument("--date", type=date.fromisoformat, required=True)
    c.add_argument("--player", required=True)
    c.add_argument("--stat", required=True)
    c.add_argument("--line", type=float, required=True)
    c.add_argument("--over-odds", type=float, required=True)
    c.add_argument("--under-odds", type=float, required=True)

    r = sub.add_parser("result", help="grade by hand; omit --actual if the player sat")
    r.add_argument("--date", type=date.fromisoformat, required=True)
    r.add_argument("--player", required=True)
    r.add_argument("--stat", required=True)
    r.add_argument("--line", type=float, required=True)
    r.add_argument("--actual", type=float)

    for name in ("grade", "rank", "report", "list"):
        sub.add_parser(name)
    args = ap.parse_args()

    if args.cmd == "add":
        row = add_prop(args.home.upper(), args.away.upper(), args.date, args.player, args.stat,
                       args.line, args.over_odds, args.under_odds, args.pick, args.confidence,
                       args.tags, args.note, args.auto_injuries)
        side, ev = _ev(row["model_p_over"], row["over_odds"], row["under_odds"])
        print(f"logged {row['player']} {row['stat']} {row['line']}: model projects {row['model_mean']:.1f}, "
              f"P(over) {row['model_p_over']:.1%}; best side {side} EV {ev:+.1%}")
    elif args.cmd == "add-file":
        print(f"logged {add_file(args.csv, args.auto_injuries)} props")
        rank_pending()
    elif args.cmd == "closing":
        record_closing(args.date, args.player, args.stat, args.line, args.over_odds, args.under_odds)
        print("closing odds stored")
    elif args.cmd == "result":
        print("result:", record_result(args.date, args.player, args.stat, args.line, args.actual))
    elif args.cmd == "grade":
        g, p = grade()
        print(f"graded {g}, still pending {p}")
    elif args.cmd == "rank":
        rank_pending()
    elif args.cmd == "report":
        report()
    elif args.cmd == "list":
        list_log()


if __name__ == "__main__":
    main()
