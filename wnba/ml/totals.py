"""Game and team totals: predicted points, and P(over) for a posted total.

Structure (standard in basketball analytics):

    points = possessions x points per possession

  * expected possessions = team pace + opponent pace - league pace
  * expected efficiency  = team offence + opponent defence - league efficiency
    (all per 100 possessions)

Team numbers are trailing 20-game averages (features.add_rolling_features,
shift-then-roll, so a game never sees itself). League numbers are trailing
averages over the previous ~25 game days. That league term matters: scoring
jumped ~10 points per game in 2026 (174 vs 158-166 in 2018-2025), and a model
anchored to older seasons would sit under every total all year.

A small linear layer then calibrates that structural estimate, RELATIVE to
the current league scoring level:

    team_pts - league_pts ~ a + b * (structural - league_pts) + c * home

Fitting absolute points instead (team_pts ~ a + b * structural) shrinks every
prediction toward the TRAINING seasons' average -- in the 2026 backtest that
left the model 6.9 points per game too low. Relative to the league level, the
shrinkage is toward today's scoring instead. Fitted walk-forward by season,
like the winner model. P(over line) treats the final total as Normal(prediction,
sigma), sigma taken from the training residuals.

Usage:
    python -m wnba.ml.totals --backtest            # 2026 vs closing totals
    python -m wnba.ml.totals --home LVA --away LAS --line 165.5
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from wnba.db import SessionLocal
from wnba.ml import features
from wnba.paths import DATA_DIR

logger = logging.getLogger(__name__)

LEAGUE_WINDOW_DAYS = 25        # game days in the league-environment average
TOTALS_PATH = DATA_DIR / "closing_totals.csv"
WIN, LOSE = 100 / 110, -1.0    # totals are usually priced at -110 both ways


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def league_environment(logs: pd.DataFrame,
                       window: int = LEAGUE_WINDOW_DAYS) -> pd.DataFrame:
    """Trailing league pace and efficiency per date, strictly before it."""
    daily = (logs.groupby("date")
                 .agg(poss=("possessions", "sum"), pts=("pts", "sum"),
                      n=("pts", "size"))
                 .sort_index())
    prior = daily.shift(1).rolling(window, min_periods=5).sum()
    return pd.DataFrame({
        "lg_pace": prior["poss"] / prior["n"],
        "lg_rtg": 100.0 * prior["pts"] / prior["poss"],
    }).reset_index()


def build_team_games(db) -> pd.DataFrame:
    """One row per team per game with everything needed to predict its points."""
    logs = features.load_game_logs(db)
    logs = features.add_efficiency_metrics(logs)
    logs = features.add_rolling_features(logs)
    logs = logs.merge(league_environment(logs), on="date", how="left")

    opp = logs[["date", "team_id", "r_off_rtg", "r_def_rtg", "r_pace"]].rename(
        columns={"team_id": "opponent_id", "r_off_rtg": "opp_r_off_rtg",
                 "r_def_rtg": "opp_r_def_rtg", "r_pace": "opp_r_pace"})
    df = logs.merge(opp, on=["date", "opponent_id"], how="inner")

    df["exp_poss"] = df["r_pace"] + df["opp_r_pace"] - df["lg_pace"]
    df["exp_rtg"] = df["r_off_rtg"] + df["opp_r_def_rtg"] - df["lg_rtg"]
    df["structural"] = df["exp_poss"] * df["exp_rtg"] / 100.0
    df["lg_pts"] = df["lg_pace"] * df["lg_rtg"] / 100.0      # league-average team score
    df["rel_structural"] = df["structural"] - df["lg_pts"]
    df["rel_pts"] = df["pts"] - df["lg_pts"]
    df["home"] = df["is_home"].astype(float)
    df.loc[df["season"].isin(features.NEUTRAL_SEASONS), "home"] = 0.0
    return df.dropna(subset=["structural", "pts"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@dataclass
class TotalsModel:
    reg: LinearRegression
    sigma_team: float
    sigma_game: float

    def team_points(self, structural: np.ndarray, home: np.ndarray,
                    lg_pts: np.ndarray) -> np.ndarray:
        rel = self.reg.predict(np.column_stack([structural - lg_pts, home]))
        return lg_pts + rel


def fit(train: pd.DataFrame) -> TotalsModel:
    X = train[["rel_structural", "home"]].to_numpy()
    reg = LinearRegression().fit(X, train["rel_pts"].to_numpy())
    pred = train["lg_pts"].to_numpy() + reg.predict(X)
    resid_team = train["pts"].to_numpy() - pred
    games = to_games(train.assign(pred=pred))
    return TotalsModel(reg, float(np.std(resid_team, ddof=1)),
                       float((games["total"] - games["pred_total"]).std(ddof=1)))


def to_games(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Pair the two team rows of each game into one game row (home side)."""
    h = team_rows[team_rows["is_home"]]
    a = team_rows[~team_rows["is_home"]]
    g = h.merge(a[["date", "team_id", "pts", "pred"]],
                left_on=["date", "opponent_id"], right_on=["date", "team_id"],
                suffixes=("", "_away"))
    return pd.DataFrame({
        "date": g["date"], "season": g["season"],
        "home_id": g["team_id"], "away_id": g["opponent_id"],
        "home_pts": g["pts"], "away_pts": g["pts_away"],
        "pred_home": g["pred"], "pred_away": g["pred_away"],
        "total": g["pts"] + g["pts_away"],
        "pred_total": g["pred"] + g["pred_away"],
    })


def p_over(pred_total: float, line: float, sigma: float) -> float:
    """P(final total > line) with total ~ Normal(pred_total, sigma)."""
    return 0.5 * (1.0 - math.erf((line - pred_total) / (sigma * math.sqrt(2.0))))


def walk_forward(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, TotalsModel]:
    train, test = df[df["season"] < season], df[df["season"] == season].copy()
    model = fit(train)
    test["pred"] = model.team_points(test["structural"].to_numpy(), test["home"].to_numpy(),
                                     test["lg_pts"].to_numpy())
    return to_games(test), model


# --------------------------------------------------------------------------
# backtest against the market
# --------------------------------------------------------------------------
def backtest(season: int = 2026) -> None:
    db = SessionLocal()
    try:
        df = build_team_games(db)
        teams = pd.read_sql("SELECT id, abbreviation FROM teams", db.bind)
    finally:
        db.close()
    games, model = walk_forward(df, season)
    abbr = dict(zip(teams["id"], teams["abbreviation"]))
    games["game_date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    games["home"] = games["home_id"].map(abbr)
    games["away"] = games["away_id"].map(abbr)

    err = games["pred_total"] - games["total"]
    print(f"{season}: {len(games)} games, model trained on seasons before {season}")
    print(f"  model:  MAE {err.abs().mean():5.2f}   bias {err.mean():+5.2f} pts "
          f"(negative = predicts too low)   sigma(game) {model.sigma_game:.1f}")

    # naive baseline: last season's average total
    prev = df[df["season"] == season - 1]
    naive = to_games(prev.assign(pred=0.0))["total"].mean()
    print(f"  naive (last season's average {naive:.1f}):  MAE "
          f"{(naive - games['total']).abs().mean():5.2f}   bias {(naive - games['total']).mean():+5.2f}")

    if not TOTALS_PATH.exists():
        logger.warning("no %s yet -- run: python -m wnba.scraping.teamrankings --market totals",
                       TOTALS_PATH.name)
        return
    lines = pd.read_csv(TOTALS_PATH)
    key = ["game_date", "home", "away"]
    m = games.merge(lines[key + ["total"]].rename(columns={"total": "line"}), on=key)
    flipped = games.merge(lines.rename(columns={"home": "away", "away": "home"})[key + ["total"]]
                          .rename(columns={"total": "line"}), on=key)
    m = pd.concat([m, flipped]).drop_duplicates(key).dropna(subset=["line"])
    le = m["line"] - m["total"]
    me = m["pred_total"] - m["total"]
    print(f"\nagainst the {len(m)} closing totals:")
    print(f"  closing line: MAE {le.abs().mean():5.2f}   bias {le.mean():+5.2f}")
    print(f"  model:        MAE {me.abs().mean():5.2f}   bias {me.mean():+5.2f}")
    blend = (m["pred_total"] + m["line"]) / 2 - m["total"]
    print(f"  50/50 blend:  MAE {blend.abs().mean():5.2f}")

    print("\nOVER/UNDER at -110 (break-even 52.4%): bet when |model - line| >= k points")
    print(f"  {'k':>3} {'bets':>5} {'won':>4} {'lost':>5} {'push':>5} {'win%':>6} {'units':>7}")
    edge = m["pred_total"] - m["line"]
    for k in (0, 2, 4, 6, 8):
        bet = edge.abs() >= k if k else edge != 0
        s = m[bet]
        over = edge[bet] > 0
        res = np.sign(s["total"] - s["line"])            # +1 over, -1 under, 0 push
        won = int(((res > 0) & over).sum() + ((res < 0) & ~over).sum())
        push = int((res == 0).sum())
        lost = len(s) - won - push
        units = won * WIN + lost * LOSE
        rate = won / (won + lost) if won + lost else float("nan")
        print(f"  {k:3d} {len(s):5d} {won:4d} {lost:5d} {push:5d} {rate:6.1%} {units:+7.2f}")


# --------------------------------------------------------------------------
# prediction for an upcoming game
# --------------------------------------------------------------------------
def predict_game(home: str, away: str, line: Optional[float] = None,
                 season: Optional[int] = None) -> dict:
    """Points for each team and P(over) for a matchup, using current form."""
    db = SessionLocal()
    try:
        df = build_team_games(db)
        teams = pd.read_sql("SELECT id, abbreviation FROM teams", db.bind)
        logs = features.add_rolling_features(
            features.add_efficiency_metrics(features.load_game_logs(db)))
    finally:
        db.close()
    model = fit(df)
    ids = dict(zip(teams["abbreviation"], teams["id"]))

    # latest trailing form per team, INCLUDING its most recent game
    form = {}
    for tid, g in logs.sort_values("date").groupby("team_id"):
        tail = g.tail(features.LONG_WINDOW)
        form[tid] = {"off": tail["off_rtg"].mean(), "def": tail["def_rtg"].mean(),
                     "pace": tail["pace"].mean()}
    env = league_environment(logs)
    recent = logs[logs["date"] > logs["date"].max() - pd.Timedelta(days=60)]
    lg_pace = recent["possessions"].mean()
    lg_rtg = 100.0 * recent["pts"].sum() / recent["possessions"].sum()
    del env

    def structural(team, opp):
        t, o = form[ids[team]], form[ids[opp]]
        return (t["pace"] + o["pace"] - lg_pace) * (t["off"] + o["def"] - lg_rtg) / 100.0

    lg_pts = np.array([lg_pace * lg_rtg / 100.0])
    ph = float(model.team_points(np.array([structural(home, away)]), np.array([1.0]), lg_pts)[0])
    pa = float(model.team_points(np.array([structural(away, home)]), np.array([0.0]), lg_pts)[0])
    out = {"home": home, "away": away, "home_pts": ph, "away_pts": pa,
           "total": ph + pa, "sigma": model.sigma_game,
           "league_pace": lg_pace, "league_rtg": lg_rtg}
    if line is not None:
        out["line"] = line
        out["p_over"] = p_over(ph + pa, line, model.sigma_game)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Game totals model.")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--line", type=float)
    args = parser.parse_args()
    if args.backtest:
        backtest(args.season)
        return
    if not (args.home and args.away):
        parser.error("give --backtest, or --home and --away")
    r = predict_game(args.home.upper(), args.away.upper(), args.line)
    print(f"{r['away']} @ {r['home']}:  {r['away']} {r['away_pts']:.1f}  "
          f"{r['home']} {r['home_pts']:.1f}  -> total {r['total']:.1f} "
          f"(+/- {r['sigma']:.1f})")
    if "p_over" in r:
        print(f"  line {r['line']}:  P(over) {r['p_over']:.1%}   P(under) {1 - r['p_over']:.1%}")


if __name__ == "__main__":
    main()
