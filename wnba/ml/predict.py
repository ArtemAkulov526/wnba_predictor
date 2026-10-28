"""Predict a single matchup from the trained model.

Loads the model once (see `get_predictor`, refreshed when the model or the
game data changes) rather than refitting per request,
and builds the same difference features the model was trained on from each
team's current form and Elo.

CLI:
    python -m wnba.ml.predict --home MIN --away LVA
    python -m wnba.ml.predict --home MIN --away LVA --neutral
"""
from __future__ import annotations

import argparse
from datetime import date
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from wnba.ml import elo
from wnba.ml import features
from wnba.db import SessionLocal
from wnba.ml.train import MODEL_PATH
from wnba.dates import us_today


@dataclass
class Prediction:
    home_team: str
    away_team: str
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str
    neutral_site: bool
    drivers: Dict[str, float]


class Predictor:
    """Trained model plus the current-form snapshot it needs at inference."""

    def __init__(self, bundle: dict, form: pd.DataFrame,
                 ratings: Dict[int, float], teams: pd.DataFrame):
        self.pipeline = bundle["pipeline"]
        self.feature_cols = bundle["feature_cols"]
        self.feature_set = bundle.get("feature_set", "unknown")
        self.form = form.set_index("team_id")
        self.ratings = ratings
        self.teams = teams
        self.by_id = dict(zip(teams.id, teams.name))
        self.by_abbr = dict(zip(teams.abbreviation, teams.id))

    def resolve(self, team: str | int) -> int:
        """Accept a team id, abbreviation or name."""
        if isinstance(team, (int, np.integer)):
            return int(team)
        if team in self.by_abbr:
            return self.by_abbr[team]
        match = self.teams[self.teams["name"].str.lower() == str(team).lower()]
        if match.empty:
            raise ValueError(f"unknown team: {team!r}")
        return int(match.iloc[0]["id"])

    def _row(self, home_id: int, away_id: int, neutral: bool,
             home_rest: Optional[float], away_rest: Optional[float],
             home_out: float = 0.0, away_out: float = 0.0) -> pd.DataFrame:
        if home_id not in self.form.index:
            raise ValueError(f"no game history for team id {home_id}")
        if away_id not in self.form.index:
            raise ValueError(f"no game history for team id {away_id}")

        home, away = self.form.loc[home_id], self.form.loc[away_id]
        row: Dict[str, float] = {}

        for metric, name in features.DIFF_METRICS:
            row[name] = home[f"r_{metric}"] - away[f"r_{metric}"]

        row["d_elo"] = (self.ratings.get(home_id, elo.INITIAL_RATING)
                        - self.ratings.get(away_id, elo.INITIAL_RATING))
        row["home_court"] = 0.0 if neutral else 1.0

        # With no scheduled date, assume both teams are equally rested.
        hr = 3.0 if home_rest is None else float(home_rest)
        ar = 3.0 if away_rest is None else float(away_rest)
        row["d_rest_days"] = hr - ar
        row["d_is_b2b"] = float(hr <= 1) - float(ar <= 1)
        row["d_games_last_7"] = 0.0
        # Share of each side's normal minutes that will be missing. Defaults to
        # 0 (full strength) because nothing free reports availability ahead of
        # tip-off; pass it in from the injury report via `missing_share()`.
        row[features.INJURY_FEATURE] = float(home_out) - float(away_out)

        return pd.DataFrame([{c: row.get(c, 0.0) for c in self.feature_cols}])

    def predict(self, home: str | int, away: str | int, neutral: bool = False,
                home_rest: Optional[float] = None,
                away_rest: Optional[float] = None,
                home_out: float = 0.0, away_out: float = 0.0) -> Prediction:
        home_id, away_id = self.resolve(home), self.resolve(away)
        if home_id == away_id:
            raise ValueError("a team cannot play itself")

        X = self._row(home_id, away_id, neutral, home_rest, away_rest,
                      home_out, away_out)
        prob = float(self.pipeline.predict_proba(X)[0, 1])

        # Contribution of each feature to the log-odds, for explainability.
        scaler = self.pipeline.named_steps["scale"]
        coefs = self.pipeline.named_steps["lr"].coef_[0]
        scaled = scaler.transform(X)[0]
        drivers = {name: round(float(z * c), 4)
                   for name, z, c in zip(self.feature_cols, scaled, coefs)}
        drivers = dict(sorted(drivers.items(), key=lambda kv: -abs(kv[1])))

        home_name = self.by_id[home_id]
        away_name = self.by_id[away_id]
        return Prediction(
            home_team=home_name,
            away_team=away_name,
            home_win_probability=round(prob, 4),
            away_win_probability=round(1.0 - prob, 4),
            predicted_winner=home_name if prob >= 0.5 else away_name,
            neutral_site=neutral,
            drivers=drivers,
        )


def missing_share(db: Session, team: str, players: List[str],
                  window: int = features.AVAILABILITY_WINDOW,
                  strict: bool = True) -> tuple[float, List[dict]]:
    """Share of `team`'s normal minutes held by the named players.

    Uses the same definition as training (`compute_availability`): each
    player's share of team minutes over the team's last `window` games. Names
    match case-insensitively on any part, e.g. "sabally" or "Satou Sabally".

    A player already out for most of the window has a small recent share, and
    that is intended: the team's recent form already reflects their absence,
    so counting them in full would double-count it.

    `strict=False` is for injury-report input: a listed player with no logs
    for this team (e.g. out all season, or just signed) counts as 0 minutes
    instead of raising. Ambiguous names always raise.
    """
    from wnba.ml import rotation

    df = pd.read_sql("""
        SELECT p.date, p.season, p.season_type, p.athlete_id, p.athlete_name, p.minutes
        FROM player_game_logs p JOIN teams t ON t.id = p.team_id
        WHERE t.abbreviation = %(team)s""", db.bind, params={"team": team})
    if df.empty:
        raise ValueError(f"no player logs for {team}")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)

    # A4: in the playoffs, use the team's playoff games once it has enough;
    # before that, reweight regular-season shares by the playoff rotation.
    league = pd.read_sql("SELECT DISTINCT season, season_type FROM player_game_logs", db.bind)
    playoff = rotation.is_postseason(None, league)      # live use: today's date
    season_now = df["season"].max()
    po_dates = sorted(df.loc[(df["season"] == season_now)
                             & (df["season_type"] == rotation.PLAYOFF_SEASON_TYPE), "date"].unique())
    use_po = playoff and len(po_dates) >= rotation.PLAYOFF_MIN_GAMES
    dates = (po_dates if use_po else sorted(df["date"].unique()))[-window:]
    recent = df[df["date"].isin(dates)]
    team_minutes = recent.groupby("date")["minutes"].sum()
    per_game = recent.pivot_table(index="date", columns="athlete_id",
                                  values="minutes", aggfunc="sum").reindex(dates).fillna(0.0)
    shares = per_game.div(team_minutes, axis=0).fillna(0.0).mean()
    if playoff and not use_po and shares.sum() > 0:
        rank = shares.rank(ascending=False, method="first").astype(int)
        mult = rank.map(lambda r: rotation.PLAYOFF_ROTATION_MULT.get(r, rotation.PLAYOFF_DEEP_BENCH_MULT))
        shares = shares * mult
        shares = shares / shares.sum() * per_game.div(team_minutes, axis=0).fillna(0.0).mean().sum()
    names = df.drop_duplicates("athlete_id").set_index("athlete_id")["athlete_name"]

    total, details = 0.0, []
    for query in players:
        q = query.strip().lower()
        exact = [a for a, n in names.items() if n and n.lower() == q]
        matches = exact or [a for a, n in names.items() if n and q in n.lower()]
        if len(matches) != 1:
            if not matches and not strict:
                details.append({"player": query.strip(), "share": 0.0,
                                "played_last_n": 0, "window": len(dates),
                                "unmatched": True})
                continue
            found = [names[a] for a in matches]
            raise ValueError(f"{team}: {query!r} matched {found or 'nobody'}")
        aid = matches[0]
        share = float(shares.get(aid, 0.0))
        played = int((per_game[aid] > 0).sum()) if aid in per_game else 0
        details.append({"player": names[aid], "share": share,
                        "played_last_n": played, "window": len(dates)})
        total += share
    return total, details


def expected_missing(players: List[dict]) -> float:
    """Expected share of minutes missing: each listed player's share weighted
    by their chance of sitting. Pure; tested without a database."""
    return sum(p["share"] * (1.0 - p["p_play"]) for p in players)


def injury_scenarios(db: Session, predictor: "Predictor", home: str, away: str,
                     report: pd.DataFrame, neutral: bool = False,
                     rates: Optional[dict] = None) -> dict:
    """Win probability under an injury report (columns team, player, status).

    Each listed player's minutes share is weighted by how often players with
    that official status actually played in 2026 (see
    official_injuries.measure_play_rates): Out ~1%, Doubtful ~7%,
    Questionable ~42%, Probable ~86%. The model is close to linear in missing
    minutes, so the headline uses the EXPECTED missing share per side.
    `low`/`high` bracket P(home) (see the return value).
    """
    from wnba.scraping.official_injuries import DEFAULT_PLAY_RATES
    rates = rates or DEFAULT_PLAY_RATES
    sides = {}
    for side, team in (("home", home), ("away", away)):
        players = []
        for r in report[report["team"] == team].itertuples(index=False):
            if r.status == "Available":
                continue
            share, det = missing_share(db, team, [r.player], strict=False)
            players.append({"player": r.player, "status": r.status,
                            "source": getattr(r, "source", ""), "share": share,
                            "p_play": float(rates.get(r.status, 0.5)),
                            "unmatched": det[0].get("unmatched", False)})
        sides[side] = {
            "team": team, "players": players,
            "expected": expected_missing(players),
            "best": sum(p["share"] for p in players if p["status"] in ("Out", "Doubtful")),
            "worst": sum(p["share"] for p in players),
        }

    def prob(h, a):
        return predictor.predict(home, away, neutral=neutral,
                                 home_out=h, away_out=a).home_win_probability

    return {"healthy": prob(0.0, 0.0),
            # Bracket for P(home): home's uncertain players sit while away's
            # play, and the reverse. An earlier "everyone plays / everyone
            # sits on both sides" pair was not a bracket -- with doubts on
            # both teams the headline could fall outside it.
            "low": prob(sides["home"]["worst"], sides["away"]["best"]),
            "high": prob(sides["home"]["best"], sides["away"]["worst"]),
            "headline": prob(sides["home"]["expected"], sides["away"]["expected"]),
            "sides": sides}


def format_scenarios(s: dict) -> str:
    """Human-readable breakdown for the CLI."""
    lines = []
    for side in ("home", "away"):
        d = s["sides"][side]
        lines.append(f"  {d['team']}   expected missing share {d['expected']:.3f}")
        for x in d["players"]:
            tag = "  (no recent minutes for this team)" if x["unmatched"] else ""
            lines.append(f"    {x['status']:12} {x['player']:24} share {x['share']:.3f}"
                         f"  plays {x['p_play']:.0%}  [{x['source']}]{tag}")
        if not d["players"]:
            lines.append("    nobody listed -- full strength")
    return chr(10).join(lines)


def build_predictor(db: Session) -> Predictor:
    import joblib
    if not MODEL_PATH.exists():
        raise SystemExit(f"no trained model at {MODEL_PATH} -- run: python -m wnba.ml.train")
    bundle = joblib.load(MODEL_PATH)
    form, ratings = features.team_form_snapshot(db)
    return Predictor(bundle, form, ratings, features.load_teams(db))


_CACHE: Dict[str, object] = {}


def get_predictor() -> Predictor:
    """Cached predictor, rebuilt when the saved model or the game data changes.

    The web app runs for a long time; a plain cache would keep serving the old
    model, Elo and team form after a retrain or a scrape until a restart. The
    cache key is the model file's timestamp plus the latest game date.
    """
    db = SessionLocal()
    try:
        key = (MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None,
               db.execute(text("SELECT max(date) FROM game_logs")).scalar())
        if _CACHE.get("key") != key:
            _CACHE["predictor"] = build_predictor(db)
            _CACHE["key"] = key
        return _CACHE["predictor"]
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one WNBA matchup.")
    parser.add_argument("--home", required=True, help="team abbreviation, name or id")
    parser.add_argument("--away", required=True)
    parser.add_argument("--neutral", action="store_true",
                        help="no home-court advantage")
    parser.add_argument("--home-rest", type=float, default=None)
    parser.add_argument("--away-rest", type=float, default=None)
    parser.add_argument("--auto-injuries", action="store_true",
                        help="apply the latest injury report (official PDF, ESPN fallback)")
    parser.add_argument("--date", type=date.fromisoformat, default=us_today(),
                        help="game date, used to judge stale injury listings")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        predictor = build_predictor(db)
        result = predictor.predict(args.home, args.away, args.neutral,
                                   args.home_rest, args.away_rest)
        # --home/--away accept a name or id too; the report is keyed by abbreviation
        abbr = {v: k for k, v in predictor.by_abbr.items()}
        home_abbr = abbr[predictor.resolve(args.home)]
        away_abbr = abbr[predictor.resolve(args.away)]
        if args.auto_injuries:
            from wnba.scraping.official_injuries import combined_report, measure_play_rates
            report = combined_report(db, args.date)
            s = injury_scenarios(db, predictor, home_abbr, away_abbr,
                                 report, args.neutral, rates=measure_play_rates(db))
    finally:
        db.close()

    if args.auto_injuries:
        print(f"\ninjury report for {args.date} (official where filed, else ESPN):")
        print(format_scenarios(s))
        print(f"\n  P({home_abbr} wins)  full strength {s['healthy']:.1%}   "
              f"range {s['low']:.1%} - {s['high']:.1%}")
        print(f"  headline (expected availability): {s['headline']:.1%}")

    if args.auto_injuries:
        print(chr(10) + "--- full-strength baseline (ignores the report above) ---")
    site = "neutral site" if result.neutral_site else f"at {result.home_team}"
    print(f"\n{result.away_team} @ {result.home_team}  ({site})")
    print(f"  {result.home_team:24} {result.home_win_probability:6.1%}")
    print(f"  {result.away_team:24} {result.away_win_probability:6.1%}")
    print(f"  predicted winner: {result.predicted_winner}")
    print("\n  top drivers (log-odds contribution):")
    for name, value in list(result.drivers.items())[:6]:
        print(f"    {name:20} {value:+.3f}")


if __name__ == "__main__":
    main()
