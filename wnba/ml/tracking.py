"""Prediction log: record the model's pick and yours BEFORE tip-off, grade after.

Five games cannot tell a good predictor from a lucky one. This keeps a running
record so that, after a few dozen games, you can see whether you or the model
is better -- and, through reason tags, which KINDS of your reasoning beat the
model (e.g. `fatigue`, `injury`) and which do not (e.g. `team-i-like`).

Storage is a plain CSV, `data/prediction_log.csv`, not the database. It is the
one dataset here that cannot be re-scraped: a prediction is only meaningful if
it was written down before the game. A file survives database resets, can be
read and edited by hand, and is tracked by git. Teams are stored by
abbreviation so the file does not depend on database ids.

Usage:
    # before the game
    python -m wnba.ml.tracking log --home IND --away WAS --date 2026-09-22 \\
        --pick WAS --confidence 0.55 --tags fatigue,team-i-like \\
        --note "Clark and Boston played the World Cup"

    python -m wnba.ml.tracking result --home IND --away WAS --date 2026-09-22 \\
        --home-pts 77 --away-pts 93             # enter a final score by hand
    python -m wnba.ml.tracking grade            # or fill results from game_logs
    python -m wnba.ml.tracking report           # model vs you, overall and by tag
    python -m wnba.ml.tracking list             # every logged game

`grade` reads results from `game_logs`, so run the gamelog scraper first:
    python -m wnba.scraping.basketball_reference --what gamelog --from-season 2026
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from wnba.paths import DATA_DIR

LOG_PATH = DATA_DIR / "prediction_log.csv"
EPS = 1e-6
SMALL_SAMPLE = 30   # below this, results are mostly luck

COLUMNS = [
    "game_date", "away", "home", "neutral_site",
    "model_home_prob", "model_feature_set",
    "user_pick", "user_home_prob", "tags", "note", "backfilled",
    # Decimal odds available when the pick was made -- the price you would
    # actually bet at. NOT the closing line: that is the last pre-game line at
    # tip-off, attached afterwards from data/closing_lines.csv.
    "home_odds", "away_odds", "odds_at",
    "closing_home_spread", "closing_home_odds", "closing_away_odds",
    "closing_home_prob",
    "home_pts", "away_pts", "home_won", "result_source",
    "created_at", "graded_at",
]
# Spread of final margins around the closing line, fitted on 2026 (12.7 pts);
# used when the lines file has too few finished games to fit its own.
DEFAULT_MARGIN_SIGMA = 12.7


# --------------------------------------------------------------------------
# scoring (pure functions, tested without a database)
# --------------------------------------------------------------------------
@dataclass
class Grade:
    n: int                      # games with a pick
    correct: int
    n_prob: int                 # games with a probability
    log_loss: Optional[float]
    brier: Optional[float]

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else float("nan")

    def line(self) -> str:
        acc = f"{self.correct}/{self.n} ({self.accuracy:.0%})" if self.n else "-"
        ll = f"{self.log_loss:.4f}" if self.log_loss is not None else "-"
        br = f"{self.brier:.4f}" if self.brier is not None else "-"
        return f"picks {acc:>12}   log loss {ll:>7}   brier {br:>7}   (prob on {self.n_prob})"


def grade(picks_home: Sequence[Optional[bool]],
          home_probs: Sequence[Optional[float]],
          home_won: Sequence[bool]) -> Grade:
    """Score picks and home-perspective probabilities against outcomes.

    `picks_home[i]` is True if the home team was picked, False for away, None
    for no pick. Probabilities may be None (pick given without confidence);
    those games count toward accuracy but not toward log loss / Brier.
    """
    picked = [(p, w) for p, w in zip(picks_home, home_won) if p is not None]
    correct = sum(p == w for p, w in picked)

    probs = [(p, w) for p, w in zip(home_probs, home_won) if p is not None]
    if probs:
        log_loss = sum(-math.log(min(max(p if w else 1 - p, EPS), 1.0))
                       for p, w in probs) / len(probs)
        brier = sum((p - (1.0 if w else 0.0)) ** 2 for p, w in probs) / len(probs)
    else:
        log_loss = brier = None
    return Grade(len(picked), correct, len(probs), log_loss, brier)


def normalize_tags(raw: Optional[str]) -> Optional[str]:
    """'Fatigue, team I like' -> 'fatigue,team-i-like'."""
    if not raw:
        return None
    tags = [t.strip().lower().replace(" ", "-") for t in raw.split(",")]
    tags = sorted({t for t in tags if t})
    return ",".join(tags) or None


def user_home_probability(pick_is_home: bool, confidence: Optional[float]) -> Optional[float]:
    """Convert 'I pick X with confidence c' into P(home wins)."""
    if confidence is None:
        return None
    if not 0.5 <= confidence <= 0.99:
        raise ValueError("confidence is your probability that YOUR pick wins: "
                         "use 0.50-0.99")
    return confidence if pick_is_home else 1.0 - confidence


def market_probability(home_odds: float, away_odds: float) -> float:
    """Decimal odds -> P(home wins) with the bookmaker's margin removed.

    1/odds overstates both sides (1/1.75 + 1/2.00 = 1.071, a 7.1% margin);
    normalising the two implied probabilities to sum to 1 removes it.
    """
    if home_odds <= 1.0 or away_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1.0")
    h, a = 1.0 / home_odds, 1.0 / away_odds
    return h / (h + a)


def bookmaker_margin(home_odds: float, away_odds: float) -> float:
    return 1.0 / home_odds + 1.0 / away_odds - 1.0


def expected_value(prob: float, odds: float) -> float:
    """Expected profit per unit staked at decimal `odds` if `prob` is right."""
    return prob * odds - 1.0


def spread_to_probability(home_spread: float, sigma: float = DEFAULT_MARGIN_SIGMA) -> float:
    """Closing point spread -> P(home wins): final margin ~ Normal(-spread, sigma)."""
    return 0.5 * (1.0 + math.erf((-home_spread / sigma) / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# file storage
# --------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_bool(value) -> Optional[bool]:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("", "nan", "none", "<na>"):
            return None
        return v in ("true", "1", "yes")
    return bool(value)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Give every column a predictable type after a CSV round trip."""
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS].copy()
    if len(df):
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    for col in ("neutral_site", "backfilled"):
        df[col] = [bool(_to_bool(v)) for v in df[col]]
    df["home_won"] = pd.array([_to_bool(v) for v in df["home_won"]], dtype="boolean")
    for col in ("model_home_prob", "user_home_prob", "home_odds", "away_odds",
                "closing_home_spread", "closing_home_odds", "closing_away_odds",
                "closing_home_prob"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("home_pts", "away_pts"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("away", "home", "user_pick", "model_feature_set", "tags", "note",
                "odds_at", "result_source", "created_at", "graded_at"):
        # Explicit object dtype: pandas 3 infers a string dtype from a plain
        # list and turns every None back into NaN, which then prints as "nan".
        df[col] = pd.Series(
            [None if (v is None or (isinstance(v, float) and math.isnan(v)))
             else str(v) for v in df[col]],
            index=df.index, dtype="object")
    return df


def read_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return _coerce(pd.DataFrame(columns=COLUMNS))
    return _coerce(pd.read_csv(path, dtype=str, keep_default_na=False,
                               na_values=[""]))


def write_log(df: pd.DataFrame, path: Path = LOG_PATH) -> None:
    """Atomic write: a crash mid-save cannot leave a half-written log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _coerce(df).sort_values(["game_date", "home"], kind="stable")
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _find(df: pd.DataFrame, game_date: date, home: str, away: str) -> Optional[int]:
    hit = df.index[(df["game_date"] == game_date.isoformat())
                   & (df["home"] == home) & (df["away"] == away)]
    return int(hit[0]) if len(hit) else None


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------
def _team_ids(db: Session) -> Dict[str, int]:
    from wnba.models import Team
    return dict(db.execute(select(Team.abbreviation, Team.id)).all())


def log_prediction(
    db: Session,
    home: str,
    away: str,
    game_date: date,
    *,
    neutral: bool = False,
    pick: Optional[str] = None,
    confidence: Optional[float] = None,
    tags: Optional[str] = None,
    note: Optional[str] = None,
    model_home_prob: Optional[float] = None,
    model_feature_set: Optional[str] = None,
    backfilled: Optional[bool] = None,
    home_out_players: Optional[List[str]] = None,
    away_out_players: Optional[List[str]] = None,
    injury_report: Optional[pd.DataFrame] = None,
    home_odds: Optional[float] = None,
    away_odds: Optional[float] = None,
    path: Path = LOG_PATH,
) -> dict:
    """Record a prediction. Re-logging an UNGRADED game updates it in place.

    Re-logging without new injury info keeps the stored model probability, so
    adding your own pick later cannot silently overwrite an injury-adjusted
    number with a full-strength one. The note is also kept unless a new one
    is given.
    """
    ids = _team_ids(db)
    for code in (home, away) + ((pick,) if pick else ()):
        if code not in ids:
            raise ValueError(f"unknown team code {code!r}")
    if home == away:
        raise ValueError("a team cannot play itself")
    if pick and pick not in (home, away):
        raise ValueError(f"--pick must be {home} or {away}")
    if confidence is not None and not pick:
        raise ValueError("--confidence needs --pick")
    if (home_odds is None) != (away_odds is None):
        raise ValueError("give both --home-odds and --away-odds")
    if home_odds is not None:
        market_probability(home_odds, away_odds)          # validates > 1.0

    df = read_log(path)
    i = _find(df, game_date, home, away)
    existing = df.loc[i].to_dict() if i is not None else None
    if existing is not None and not pd.isna(existing["home_won"]):
        raise ValueError("that game is already graded; its prediction is locked")

    injuries_given = bool(home_out_players or away_out_players) or injury_report is not None
    if model_home_prob is None and injury_report is not None:
        from wnba.ml.predict import build_predictor, injury_scenarios
        from wnba.scraping.official_injuries import measure_play_rates
        predictor = build_predictor(db)
        s = injury_scenarios(db, predictor, home, away, injury_report, neutral,
                             rates=measure_play_rates(db))
        model_home_prob, model_feature_set = s["headline"], predictor.feature_set
        parts = []
        for side in ("home", "away"):
            d = s["sides"][side]
            by_status: Dict[str, List[str]] = {}
            for x in d["players"]:
                by_status.setdefault(x["status"].lower(), []).append(x["player"])
            bits = [f"{k} " + ", ".join(v) for k, v in by_status.items()]
            sources = sorted({x["source"] for x in d["players"] if x["source"]})
            src = f" [{'/'.join(sources)}]" if sources else ""
            parts.append(f"{d['team']}{src}: " + ("; ".join(bits) or "full strength"))
        auto = "injuries -> " + " | ".join(parts)
        note = auto if note is None else f"{note} | {auto}"
    if model_home_prob is None and existing is not None and not injuries_given:
        model_home_prob = existing["model_home_prob"]        # keep what was logged
        model_feature_set = existing["model_feature_set"]
    if model_home_prob is None:
        from wnba.ml.predict import build_predictor, missing_share
        predictor = build_predictor(db)
        home_out = missing_share(db, home, home_out_players)[0] if home_out_players else 0.0
        away_out = missing_share(db, away, away_out_players)[0] if away_out_players else 0.0
        model_home_prob = predictor.predict(
            home, away, neutral=neutral,
            home_out=home_out, away_out=away_out).home_win_probability
        model_feature_set = predictor.feature_set

    if backfilled is None:
        # Logged after game day is not a clean pre-game prediction.
        backfilled = game_date < date.today()

    row = {
        "game_date": game_date.isoformat(), "away": away, "home": home,
        "neutral_site": neutral,
        "model_home_prob": round(float(model_home_prob), 4),
        "model_feature_set": model_feature_set,
        # Keep an existing pick/tags when re-logging only to refresh the model
        # number (e.g. with --auto-injuries); otherwise the pick was wiped.
        "user_pick": pick if pick else (existing or {}).get("user_pick"),
        "user_home_prob": (user_home_probability(pick == home, confidence) if pick
                           else (existing or {}).get("user_home_prob")),
        "tags": normalize_tags(tags) if tags else (existing or {}).get("tags"),
        "home_odds": home_odds if home_odds is not None else (existing or {}).get("home_odds"),
        "away_odds": away_odds if away_odds is not None else (existing or {}).get("away_odds"),
        "odds_at": _now() if home_odds is not None else (existing or {}).get("odds_at"),
        "closing_home_spread": (existing or {}).get("closing_home_spread"),
        "closing_home_prob": (existing or {}).get("closing_home_prob"),
        "note": note if note is not None else (existing or {}).get("note"),
        "backfilled": backfilled,
        "home_pts": None, "away_pts": None, "home_won": None,
        "result_source": None,
        "created_at": existing["created_at"] if existing else _now(),
        "graded_at": None,
    }
    if i is None:
        df = pd.concat([df, _coerce(pd.DataFrame([row]))], ignore_index=True)
    else:
        for k, v in row.items():
            df.at[i, k] = v
    write_log(df, path)
    return row


def record_result(home: str, away: str, game_date: date,
                  home_pts: int, away_pts: int, path: Path = LOG_PATH) -> dict:
    """Grade a logged game by hand, for when basketball-reference lags."""
    if home_pts == away_pts:
        raise ValueError("WNBA games cannot end tied")
    df = read_log(path)
    i = _find(df, game_date, home, away)
    if i is None:
        raise ValueError(f"no logged prediction for {away}@{home} on {game_date}")
    df.at[i, "home_pts"] = int(home_pts)
    df.at[i, "away_pts"] = int(away_pts)
    df.at[i, "home_won"] = home_pts > away_pts
    df.at[i, "result_source"] = "manual"
    df.at[i, "graded_at"] = _now()
    write_log(df, path)
    return {"home": home, "away": away, "home_won": home_pts > away_pts}


def amend_backfilled(home: str, away: str, game_date: date, confidence: float,
                     tags: Optional[str] = None, path: Path = LOG_PATH) -> dict:
    """Add your confidence (and tags) to a BACKFILLED row after grading.

    Graded predictions are otherwise locked. This is the one exception, and
    only for rows already flagged backfilled -- i.e. recorded after tip-off,
    which `report --clean-only` excludes. The pick itself cannot change.
    """
    df = read_log(path)
    i = _find(df, game_date, home, away)
    if i is None:
        raise ValueError(f"no logged prediction for {away}@{home} on {game_date}")
    if not df.at[i, "backfilled"]:
        raise ValueError("only backfilled rows can be amended after grading")
    pick = df.at[i, "user_pick"]
    if pick is None:
        raise ValueError("row has no pick to attach a confidence to")
    df.at[i, "user_home_prob"] = user_home_probability(pick == home, confidence)
    if tags:
        merged = ",".join(filter(None, [df.at[i, "tags"], tags]))
        df.at[i, "tags"] = normalize_tags(merged)
    write_log(df, path)
    return df.loc[i].to_dict()


def record_closing_odds(home: str, away: str, game_date: date,
                        home_odds: float, away_odds: float,
                        path: Path = LOG_PATH) -> dict:
    """Store the closing moneyline (last pre-game odds) for a logged game.

    Preferred over a spread for the closing probability: it is the market's
    own win price, with only the bookmaker's margin to remove.
    """
    df = read_log(path)
    i = _find(df, game_date, home, away)
    if i is None:
        raise ValueError(f"no logged prediction for {away}@{home} on {game_date}")
    df.at[i, "closing_home_odds"] = float(home_odds)
    df.at[i, "closing_away_odds"] = float(away_odds)
    df.at[i, "closing_home_prob"] = round(market_probability(home_odds, away_odds), 4)
    write_log(df, path)
    return df.loc[i].to_dict()


def attach_closing_lines(path: Path = LOG_PATH,
                         lines_path: Optional[Path] = None) -> int:
    """Fill the closing spread for logged games from data/closing_lines.csv.

    Only FINISHED games are used (a row with a final margin): for those the
    stored line is the true closing line; for an upcoming game it would just
    be the current line. Games listed home/away the other way round are
    matched too, with the spread's sign flipped. Returns rows updated.
    """
    lines_path = Path(lines_path or DATA_DIR / "closing_lines.csv")
    if not lines_path.exists():
        return 0
    lines = pd.read_csv(lines_path).dropna(subset=["home_spread", "home_pts_margin"])
    if lines.empty:
        return 0
    sigma = (float((lines["home_pts_margin"] + lines["home_spread"]).std(ddof=1))
             if len(lines) >= 50 else DEFAULT_MARGIN_SIGMA)
    spreads = {(r.game_date, r.home, r.away): float(r.home_spread)
               for r in lines.itertuples(index=False)}

    df = read_log(path)
    updated = 0
    for i in df.index:
        d, h, a = df.at[i, "game_date"], df.at[i, "home"], df.at[i, "away"]
        if (d, h, a) in spreads:
            s = spreads[(d, h, a)]
        elif (d, a, h) in spreads:
            s = -spreads[(d, a, h)]
        else:
            continue
        df.at[i, "closing_home_spread"] = s
        if pd.isna(df.at[i, "closing_home_odds"]):       # moneyline beats spread
            df.at[i, "closing_home_prob"] = round(spread_to_probability(s, sigma), 4)
        updated += 1
    if updated:
        write_log(df, path)
    return updated


def betting_summary(df: pd.DataFrame, who: str = "user") -> Optional[dict]:
    """Flat 1-unit results at the logged odds, for graded games with odds.

    who="user": back your pick every time, and separately only when your own
    probability says the bet has positive expected value (either side).
    who="model": the same using the model's probability.
    Also closing line value: how the closing probability of the side backed
    compares with the market probability at the time of the pick.
    """
    rows = df[df["home_won"].notna() & df["home_odds"].notna()]
    if who == "user":
        rows = rows[rows["user_pick"].notna()]
    if rows.empty:
        return None

    every = value = 0.0
    n_every = n_value = 0
    clv = []
    for r in rows.itertuples(index=False):
        won_home = bool(r.home_won)
        if who == "user":
            back_home = r.user_pick == r.home
            prob = r.user_home_prob
        else:
            back_home = r.model_home_prob >= 0.5
            prob = r.model_home_prob
        odds = r.home_odds if back_home else r.away_odds
        every += (odds - 1.0) if back_home == won_home else -1.0
        n_every += 1

        if prob is not None and not pd.isna(prob):
            ev_home = expected_value(prob, r.home_odds)
            ev_away = expected_value(1 - prob, r.away_odds)
            if max(ev_home, ev_away) > 0:
                side_home = ev_home >= ev_away
                o = r.home_odds if side_home else r.away_odds
                value += (o - 1.0) if side_home == won_home else -1.0
                n_value += 1

        if not pd.isna(r.closing_home_prob):
            taken = market_probability(r.home_odds, r.away_odds)
            taken = taken if back_home else 1 - taken
            close = r.closing_home_prob if back_home else 1 - r.closing_home_prob
            clv.append(close - taken)

    return {"n_every": n_every, "units_every": round(every, 2),
            "n_value": n_value, "units_value": round(value, 2),
            "n_clv": len(clv), "mean_clv": sum(clv) / len(clv) if clv else None}


def _scraped_results(db: Session, since: str) -> pd.DataFrame:
    """Home-side results from game_logs on or after `since`, keyed like the log."""
    return pd.read_sql("""
        SELECT to_char(g.date, 'YYYY-MM-DD') game_date, h.abbreviation home,
               a.abbreviation away, g.pts s_home_pts, g.opp_pts s_away_pts,
               g.won s_home_won
        FROM game_logs g JOIN teams h ON h.id = g.team_id
        JOIN teams a ON a.id = g.opponent_id
        WHERE g.date >= %(since)s""", db.bind, params={"since": since})


def grade_pending(db: Session, path: Path = LOG_PATH) -> tuple[int, int]:
    """Fill in results from game_logs. Returns (graded now, still pending)."""
    df = read_log(path)
    pending = df.index[df["home_won"].isna()]
    if len(pending) == 0:
        return 0, 0
    scraped = _scraped_results(db, df.loc[pending, "game_date"].min())
    scraped = scraped.set_index(["game_date", "home", "away"])
    graded = 0
    for i in pending:
        key = (df.at[i, "game_date"], df.at[i, "home"], df.at[i, "away"])
        if key not in scraped.index:
            continue
        s = scraped.loc[key]
        df.at[i, "home_pts"] = int(s["s_home_pts"])
        df.at[i, "away_pts"] = int(s["s_away_pts"])
        df.at[i, "home_won"] = bool(s["s_home_won"])
        df.at[i, "result_source"] = "scraped"
        df.at[i, "graded_at"] = _now()
        graded += 1
    write_log(df, path)
    return graded, len(pending) - graded


def scraped_conflicts(db: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Graded rows whose recorded score disagrees with game_logs."""
    done = df[df["home_won"].notna()]
    if done.empty:
        return done
    m = done.merge(_scraped_results(db, done["game_date"].min()),
                   on=["game_date", "home", "away"])
    bad = m[(m["home_pts"] != m["s_home_pts"]) | (m["away_pts"] != m["s_away_pts"])]
    return bad[["game_date", "away", "home", "away_pts", "home_pts",
                "s_away_pts", "s_home_pts"]]


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    df = read_log(path)
    df["model_pick"] = [h if p >= 0.5 else a for h, a, p in
                        zip(df["home"], df["away"], df["model_home_prob"])]
    return df


def _grade_frame(df: pd.DataFrame, who: str) -> Grade:
    won = [bool(w) for w in df["home_won"]]
    if who == "model":
        return grade([p >= 0.5 for p in df["model_home_prob"]],
                     list(df["model_home_prob"]), won)
    picks = [None if u is None else (u == h) for u, h in zip(df["user_pick"], df["home"])]
    probs = [None if pd.isna(p) else float(p) for p in df["user_home_prob"]]
    return grade(picks, probs, won)


def report(db: Optional[Session] = None, clean_only: bool = False,
           path: Path = LOG_PATH) -> None:
    df = load_log(path)
    if df.empty:
        print(f"prediction log is empty ({path}) -- start with: "
              "python -m wnba.ml.tracking log ...")
        return

    pending = df[df["home_won"].isna()]
    done = df[df["home_won"].notna()].copy()
    backfilled = int(sum(done["backfilled"]))
    if clean_only:
        done = done[[not b for b in done["backfilled"]]]

    print(f"logged {len(df)}   graded {len(df) - len(pending)}   pending {len(pending)}"
          + (f"   (backfilled graded: {backfilled}"
             f"{', excluded' if clean_only else ', included'})" if backfilled else ""))
    if done.empty:
        print("nothing graded yet -- enter scores with `result`, or scrape and `grade`")
        return

    print("\nOVERALL")
    print(f"  model  {_grade_frame(done, 'model').line()}")
    print(f"  you    {_grade_frame(done, 'user').line()}")
    mk = done[done["home_odds"].notna()]
    if not mk.empty:
        mp = [market_probability(h, a) for h, a in zip(mk["home_odds"], mk["away_odds"])]
        g = grade([x >= 0.5 for x in mp], mp, [bool(w) for w in mk["home_won"]])
        print(f"  market {g.line()}   <- odds at pick time")
    cl = done[done["closing_home_prob"].notna()]
    if not cl.empty:
        cp = list(cl["closing_home_prob"])
        g = grade([x >= 0.5 for x in cp], cp, [bool(w) for w in cl["home_won"]])
        print(f"  close  {g.line()}   <- closing line (moneyline if entered, else spread)")

    for who, label in (("user", "YOU"), ("model", "MODEL")):
        b = betting_summary(done, who)
        if not b:
            continue
        print(chr(10) + f"BETTING, {label} (1 unit per bet at the logged odds)")
        print(f"  back every pick      {b['n_every']:3d} bets   {b['units_every']:+.2f} units")
        print(f"  only positive-EV     {b['n_value']:3d} bets   {b['units_value']:+.2f} units")
        if b["mean_clv"] is not None:
            print(f"  closing line value   {b['mean_clv']:+.1%} avg on {b['n_clv']} bets "
                  "(positive = the market moved your way)")

    both = done[done["user_pick"].notna()]
    disagree = both[both["user_pick"] != both["model_pick"]]
    if not disagree.empty:
        you = sum(u == (h if w else a) for u, h, a, w in
                  zip(disagree["user_pick"], disagree["home"],
                      disagree["away"], disagree["home_won"]))
        print(f"\nWHEN YOU DISAGREED WITH THE MODEL ({len(disagree)} games)")
        print(f"  you were right {you}, the model was right {len(disagree) - you}")

    tagged = done[done["tags"].notna()].copy()
    if not tagged.empty:
        tagged["tag"] = tagged["tags"].str.split(",")
        tagged = tagged.explode("tag")
        print("\nBY REASON TAG (your picks vs the model on the same games)")
        for tag, sub in tagged.groupby("tag"):
            y, m = _grade_frame(sub, "user"), _grade_frame(sub, "model")
            print(f"  {tag:18} n={len(sub):3d}   you {y.correct}/{y.n}   "
                  f"model {m.correct}/{m.n}")

    if db is not None:
        conflicts = scraped_conflicts(db, df)
        if not conflicts.empty:
            print("\nWARNING: logged scores disagree with scraped game_logs:")
            print(conflicts.to_string(index=False))

    if len(done) < SMALL_SAMPLE:
        print(f"\nnote: {len(done)} graded games. Below ~{SMALL_SAMPLE}, "
              "differences are mostly luck -- keep logging.")


def list_log(path: Path = LOG_PATH) -> None:
    df = load_log(path)
    if df.empty:
        print(f"prediction log is empty ({path})")
        return
    for r in df.itertuples(index=False):
        side = max(r.model_home_prob, 1 - r.model_home_prob)
        if pd.isna(r.home_won):
            result = "pending"
        else:
            result = f"{r.away} {r.away_pts}-{r.home_pts} {r.home}"
        you = "-" if r.user_pick is None else r.user_pick
        if not pd.isna(r.user_home_prob):
            p = r.user_home_prob if r.user_pick == r.home else 1 - r.user_home_prob
            you += f" {p:.0%}"
        flag = " [backfilled]" if r.backfilled else ""
        odds = ""
        if not pd.isna(r.home_odds):
            odds = f"odds {r.away_odds:.2f}/{r.home_odds:.2f}"
            if r.user_pick is not None and not pd.isna(r.user_home_prob):
                back_home = r.user_pick == r.home
                p_pick = r.user_home_prob if back_home else 1 - r.user_home_prob
                o = r.home_odds if back_home else r.away_odds
                odds += f" EV(pick) {expected_value(p_pick, o):+.0%}"
        print(f"{r.game_date}  {r.away}@{r.home}   model {r.model_pick} {side:5.1%}"
              f"   you {you:9}  {result:18}  {odds:30} {r.tags or ''}{flag}")


def _live_report(db: Session, game_date: date) -> pd.DataFrame:
    """Official report where filed, ESPN otherwise; both are archived."""
    from wnba.scraping.official_injuries import combined_report
    return combined_report(db, game_date)


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction log.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_log = sub.add_parser("log", help="record a prediction before tip-off")
    p_log.add_argument("--home", required=True)
    p_log.add_argument("--away", required=True)
    p_log.add_argument("--date", type=date.fromisoformat, default=date.today())
    p_log.add_argument("--neutral", action="store_true")
    p_log.add_argument("--pick", help="team you think wins")
    p_log.add_argument("--confidence", type=float,
                       help="your probability that YOUR pick wins, 0.50-0.99")
    p_log.add_argument("--tags", help="comma-separated reasons, e.g. fatigue,injury")
    p_log.add_argument("--note")
    p_log.add_argument("--home-out", help='missing home players, e.g. "Satou Sabally"')
    p_log.add_argument("--away-out", help='missing away players, e.g. "Azzi Fudd,Alanna Smith"')
    p_log.add_argument("--home-odds", type=float, help="decimal odds on the home team, e.g. 2.00")
    p_log.add_argument("--away-odds", type=float, help="decimal odds on the away team, e.g. 1.75")
    p_log.add_argument("--auto-injuries", action="store_true",
                       help="use ESPN's current injury report for both teams")

    p_res = sub.add_parser("result", help="enter a final score by hand")
    p_res.add_argument("--home", required=True)
    p_res.add_argument("--away", required=True)
    p_res.add_argument("--date", type=date.fromisoformat, required=True)
    p_res.add_argument("--home-pts", type=int, required=True)
    p_res.add_argument("--away-pts", type=int, required=True)

    p_cl = sub.add_parser("closing", help="enter the closing moneyline by hand")
    p_cl.add_argument("--home", required=True)
    p_cl.add_argument("--away", required=True)
    p_cl.add_argument("--date", type=date.fromisoformat, required=True)
    p_cl.add_argument("--home-odds", type=float, required=True)
    p_cl.add_argument("--away-odds", type=float, required=True)

    sub.add_parser("grade", help="fill in results from game_logs")
    p_rep = sub.add_parser("report", help="model vs you")
    p_rep.add_argument("--clean-only", action="store_true",
                       help="exclude predictions logged after the game")
    sub.add_parser("list", help="show every logged game")

    args = parser.parse_args()

    # `result` and `list` only touch the file; the rest need the database.
    if args.cmd == "result":
        row = record_result(args.home.upper(), args.away.upper(), args.date,
                            args.home_pts, args.away_pts)
        print(f"graded: {row['home'] if row['home_won'] else row['away']} won")
        attach_closing_lines()
        return
    if args.cmd == "closing":
        r = record_closing_odds(args.home.upper(), args.away.upper(), args.date,
                                args.home_odds, args.away_odds)
        print(f"closing line stored: P({r['home']}) {r['closing_home_prob']:.1%}")
        return
    if args.cmd == "list":
        list_log()
        return

    from wnba.db import SessionLocal
    db = SessionLocal()
    try:
        if args.cmd == "log":
            row = log_prediction(
                db, args.home.upper(), args.away.upper(), args.date,
                neutral=args.neutral,
                pick=args.pick.upper() if args.pick else None,
                confidence=args.confidence, tags=args.tags, note=args.note,
                home_out_players=args.home_out.split(",") if args.home_out else None,
                away_out_players=args.away_out.split(",") if args.away_out else None,
                injury_report=_live_report(db, args.date) if args.auto_injuries else None,
                home_odds=args.home_odds, away_odds=args.away_odds)
            p = row["model_home_prob"]
            pick = row["home"] if p >= 0.5 else row["away"]
            print(f"logged {row['away']}@{row['home']} {row['game_date']}: "
                  f"model {pick} {max(p, 1 - p):.1%}"
                  + (" [backfilled: logged after game day]" if row["backfilled"] else ""))
        elif args.cmd == "grade":
            graded, pending = grade_pending(db)
            closing = attach_closing_lines()
            print(f"graded {graded}, still pending {pending}; closing lines attached to {closing}")
        elif args.cmd == "report":
            report(db, clean_only=args.clean_only)
    finally:
        db.close()


if __name__ == "__main__":
    main()
