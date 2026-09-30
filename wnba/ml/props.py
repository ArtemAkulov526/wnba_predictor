"""Player projections for props: points, rebounds, assists, threes.

    projection = expected minutes x per-minute rate x game adjustment

Expected minutes. Exponentially weighted average of the player's recent
appearances, then scaled so that the players AVAILABLE for the game fill the
team's ~200 minutes. When a starter sits, her minutes are shared out in
proportion to everyone else's role -- the injury effect that props books can
be slow to price. In the backtest "available" means "actually played", i.e.
it assumes the pre-game injury report was known, which the official report
mostly provides (late scratches aside).

Per-minute rate. Two estimates, both from strictly earlier games: a recent
exponentially weighted rate and the season-to-date rate. A walk-forward linear
calibration (train on earlier seasons, test on the next) decides how to weight
them and corrects systematic lag, as in the totals model.

Game adjustment. Expected possessions relative to league pace (from
wnba.ml.totals) for every stat; points also scale with the opponent's recent
defensive rating relative to the league.

Distribution. Counts are over-dispersed Poisson: variance = alpha * mean,
alpha fitted per stat on training residuals. P(over x.5) comes from the
matching negative binomial (Poisson when alpha <= 1).

Usage:
    python -m wnba.ml.props --backtest                       # 2026, per stat
    python -m wnba.ml.props --home NYL --away ATL --auto-injuries
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LinearRegression

from wnba.db import SessionLocal
from wnba.ml import totals

STATS = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes",
         "pra": "pts+reb+ast"}
TEAM_MINUTES = 200.0          # 5 players x 40 minutes, regulation
MINUTES_HALFLIFE = 5          # appearances
RATE_HALFLIFE = 10            # appearances
MIN_APPEARANCES = 3


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_player_games(db) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT date, season, season_type, team_id, opponent_id, athlete_id,
               athlete_name, minutes, pts, reb, ast, fg3m
        FROM player_game_logs
        WHERE minutes > 0""", db.bind)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["minutes", "pts", "reb", "ast", "fg3m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # modelled as its own stat so its spread (the three move together) is fitted
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    return df.sort_values(["athlete_id", "date"]).reset_index(drop=True)


def add_player_form(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing minutes and per-minute rates, strictly from earlier appearances."""
    df = df.sort_values(["athlete_id", "date"]).copy()
    g = df.groupby("athlete_id", sort=False)
    df["n_prior"] = g.cumcount()
    df["exp_min_raw"] = g["minutes"].transform(
        lambda s: s.shift(1).ewm(halflife=MINUTES_HALFLIFE, min_periods=1).mean())

    season_g = df.groupby(["athlete_id", "season"], sort=False)
    cum_min = season_g["minutes"].transform(lambda s: s.shift(1).cumsum())
    for stat in STATS:
        per_min = df[stat] / df["minutes"]
        # EWM of the rate, weighting each game by the minutes it came from
        num = g[stat].transform(lambda s: s.shift(1).ewm(halflife=RATE_HALFLIFE, min_periods=1).mean())
        den = g["minutes"].transform(lambda s: s.shift(1).ewm(halflife=RATE_HALFLIFE, min_periods=1).mean())
        df[f"rate_ewm_{stat}"] = num / den
        cum = season_g[stat].transform(lambda s: s.shift(1).cumsum())
        # season-to-date rate; early in a season fall back to the recent rate
        df[f"rate_season_{stat}"] = (cum / cum_min).where(cum_min >= 60, df[f"rate_ewm_{stat}"])
        del per_min
    # simple baselines for comparison
    for stat in STATS:
        df[f"base_last10_{stat}"] = g[stat].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    return df


def add_minutes_share(df: pd.DataFrame) -> pd.DataFrame:
    """Scale expected minutes so the players who were available fill the game."""
    df = df.copy()
    tot = df.groupby(["date", "team_id"])["exp_min_raw"].transform("sum")
    df["exp_min"] = (df["exp_min_raw"] * TEAM_MINUTES / tot).clip(upper=40.0)
    return df


def add_game_context(df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    ctx = team_games[["date", "team_id", "exp_poss", "lg_pace", "opp_r_def_rtg", "lg_rtg"]]
    df = df.merge(ctx, on=["date", "team_id"], how="left")
    df["pace_factor"] = (df["exp_poss"] / df["lg_pace"]).fillna(1.0)
    df["def_factor"] = (df["opp_r_def_rtg"] / df["lg_rtg"]).fillna(1.0)
    return df


def raw_projections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for stat in STATS:
        factor = df["pace_factor"] * (df["def_factor"] if stat == "pts" else 1.0)
        df[f"raw_ewm_{stat}"] = df["exp_min"] * df[f"rate_ewm_{stat}"] * factor
        df[f"raw_season_{stat}"] = df["exp_min"] * df[f"rate_season_{stat}"] * factor
    return df


def build_dataset(db) -> pd.DataFrame:
    team_games = totals.build_team_games(db)
    df = load_player_games(db)
    df = add_player_form(df)
    df = df[df["n_prior"] >= MIN_APPEARANCES]
    df = add_minutes_share(df)
    df = add_game_context(df, team_games)
    return raw_projections(df).dropna(subset=[f"raw_ewm_{s}" for s in STATS]).reset_index(drop=True)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@dataclass
class StatModel:
    stat: str
    reg: LinearRegression
    alpha: float                # variance = alpha * mean

    def mean(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[[f"raw_ewm_{self.stat}", f"raw_season_{self.stat}"]].to_numpy()
        return np.clip(self.reg.predict(X), 0.05, None)

    def p_over(self, mean: float, line: float) -> float:
        k = math.floor(line)                  # over x.5 means at least k + 1
        if self.alpha <= 1.0:
            return float(poisson.sf(k, mean))
        n, p = mean / (self.alpha - 1.0), 1.0 / self.alpha
        return float(nbinom.sf(k, n, p))


def fit(train: pd.DataFrame) -> Dict[str, StatModel]:
    models = {}
    for stat in STATS:
        X = train[[f"raw_ewm_{stat}", f"raw_season_{stat}"]].to_numpy()
        y = train[stat].to_numpy()
        reg = LinearRegression().fit(X, y)
        mu = np.clip(reg.predict(X), 0.05, None)
        # method of moments on binned means: Var(y) ~ alpha * mu
        alpha = float(np.sum((y - mu) ** 2) / np.sum(mu))
        models[stat] = StatModel(stat, reg, max(alpha, 1.0))
    return models


# --------------------------------------------------------------------------
# backtest (no market lines exist for past props, so: accuracy + calibration)
# --------------------------------------------------------------------------
def backtest(season: int = 2026) -> None:
    db = SessionLocal()
    try:
        df = build_dataset(db)
    finally:
        db.close()
    train, test = df[df["season"] < season], df[df["season"] == season].copy()
    models = fit(train)
    print(f"{season}: {len(test):,} player-games (trained on {len(train):,} from earlier seasons)\n")
    print(f"{'stat':8} {'model MAE':>10} {'last-10 avg':>12} {'season avg':>11} {'bias':>7} {'alpha':>6}")
    for stat, name in STATS.items():
        m = models[stat]
        test[f"pred_{stat}"] = m.mean(test)
        season_avg = test[f"rate_season_{stat}"] * test["exp_min_raw"]
        err = test[f"pred_{stat}"] - test[stat]
        print(f"{name:8} {err.abs().mean():10.2f} "
              f"{(test[f'base_last10_{stat}'] - test[stat]).abs().mean():12.2f} "
              f"{(season_avg - test[stat]).abs().mean():11.2f} {err.mean():+7.2f} {m.alpha:6.2f}")

    # Calibration against book-like lines. Books set props close to recent
    # averages, so use each player's last-10 average rounded to the nearest .5
    # as a stand-in line and check that "P(over) = 60%" happens ~60% of the time.
    print("\nCALIBRATION of P(over) against stand-in lines (last-10 average, to .5)")
    rows = []
    for stat in STATS:
        line = np.floor(test[f"base_last10_{stat}"]) + 0.5
        p = np.array([models[stat].p_over(mu, ln) for mu, ln in zip(test[f"pred_{stat}"], line)])
        rows.append(pd.DataFrame({"stat": stat, "p": p, "over": (test[stat] > line).astype(int)}))
    cal = pd.concat(rows)
    cal["bucket"] = pd.cut(cal["p"], [0, .3, .4, .5, .6, .7, 1.0])
    table = cal.groupby("bucket", observed=True).agg(n=("over", "size"), predicted=("p", "mean"),
                                                     actual=("over", "mean"))
    for b, r in table.iterrows():
        print(f"  {str(b):12} n={int(r.n):6,}  predicted {r.predicted:5.1%}  actual {r.actual:5.1%}")
    p = cal["p"].clip(1e-4, 1 - 1e-4)
    ll = -np.mean(cal["over"] * np.log(p) + (1 - cal["over"]) * np.log(1 - p))
    print(f"  log loss {ll:.4f}  vs 0.6931 for a coin flip (what 'line = recent average' implies)")


# --------------------------------------------------------------------------
# projections for an upcoming game
# --------------------------------------------------------------------------
def project_game(home: str, away: str, game_date: Optional[date] = None,
                 report: Optional[pd.DataFrame] = None,
                 rates: Optional[dict] = None,
                 playoff: Optional[bool] = None) -> pd.DataFrame:
    """Projected stats for both rosters, with injury-report availability.

    `report` (columns team, player, status) removes Out players and weights
    Questionable/Probable ones by how often that status actually played.
    `playoff` switches on the playoff rotation (wnba.ml.rotation); by default
    it is on once postseason games of the current season are in the data.
    """
    from wnba.ml import rotation
    from wnba.scraping.official_injuries import DEFAULT_PLAY_RATES
    rates = rates or DEFAULT_PLAY_RATES
    db = SessionLocal()
    try:
        df = build_dataset(db)
        full = add_player_form(load_player_games(db))
        teams = pd.read_sql("SELECT id, abbreviation FROM teams", db.bind)
        game = totals.predict_game(home, away)
    finally:
        db.close()
    models = fit(df)
    ids = dict(zip(teams["abbreviation"], teams["id"]))
    if playoff is None:
        playoff = rotation.is_postseason(game_date, full)

    out = []
    for team, opp, is_home in ((home, away, True), (away, home, False)):
        tid = ids[team]
        season_now = full["season"].max()
        team_logs = full[(full["team_id"] == tid) & (full["season"] == season_now)]
        # A5 + A4: appearance-weighted minutes, playoff rotation when relevant
        rot = rotation.rotation_minutes(team_logs, playoff)
        latest = (team_logs[team_logs["athlete_id"].isin(rot.index)]
                  .sort_values("date").groupby("athlete_id").tail(1).copy())
        # form including the most recent game (the stored columns exclude it)
        hist = full[full["athlete_id"].isin(latest["athlete_id"])].sort_values("date")
        g = hist.groupby("athlete_id")
        latest = latest.set_index("athlete_id")
        latest["p_appear"] = rot["p_appear"]
        latest["min_if_play"] = rot["min_if_play"]
        latest["minutes_basis"] = rot["basis"]
        for stat in STATS:
            num = g[stat].apply(lambda s: s.ewm(halflife=RATE_HALFLIFE).mean().iloc[-1])
            den = g["minutes"].apply(lambda s: s.ewm(halflife=RATE_HALFLIFE).mean().iloc[-1])
            latest[f"rate_ewm_{stat}"] = num / den
            cur = hist[hist["season"] == season_now].groupby("athlete_id")
            latest[f"rate_season_{stat}"] = cur[stat].sum() / cur["minutes"].sum()
        latest = latest.reset_index()

        # availability from the injury report
        p_play = np.ones(len(latest))
        if report is not None and len(report):
            listed = report[report["team"] == team]
            status = dict(zip(listed["player"].str.lower(), listed["status"]))
            p_play = np.array([rates.get(status.get(n.lower(), ""), 1.0)
                               if n.lower() in status else 1.0 for n in latest["athlete_name"]])
        latest["p_play"] = p_play
        # Scale so EXPECTED minutes (appearance x injury availability x
        # minutes-if-playing) fill the game; props are settled only if the
        # player plays, so the projection uses minutes-if-playing.
        expected = latest["p_appear"] * latest["p_play"] * latest["min_if_play"]
        scale = TEAM_MINUTES / expected.sum()
        latest["exp_min"] = (latest["min_if_play"] * scale).clip(upper=40.0)

        lg_pace, lg_rtg = game["league_pace"], game["league_rtg"]
        exp_poss = (game["home_pts"] + game["away_pts"]) / 2 / (lg_rtg / 100.0)
        opp_def_factor = (game["home_pts"] if not is_home else game["away_pts"]) / (exp_poss * lg_rtg / 100.0)
        latest["pace_factor"] = exp_poss / lg_pace
        latest["def_factor"] = opp_def_factor
        latest = raw_projections(latest)
        latest["team"], latest["opp"] = team, opp
        for stat in STATS:
            latest[f"proj_{stat}"] = models[stat].mean(latest)
        out.append(latest)

    res = pd.concat(out, ignore_index=True)
    res.attrs["models"] = models
    cols = ["team", "athlete_name", "p_play", "exp_min"] + [f"proj_{s}" for s in STATS]
    return res[cols + [c for c in res.columns if c not in cols]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Player prop projections.")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--auto-injuries", action="store_true")
    args = parser.parse_args()
    if args.backtest:
        backtest(args.season)
        return
    if not (args.home and args.away):
        parser.error("give --backtest, or --home and --away")

    report = rates = None
    if args.auto_injuries:
        from wnba.scraping.official_injuries import combined_report, measure_play_rates
        db = SessionLocal()
        try:
            report, rates = combined_report(db, args.date), measure_play_rates(db)
        finally:
            db.close()
    res = project_game(args.home.upper(), args.away.upper(), args.date, report, rates)
    res = res[res["exp_min"] >= 8].sort_values(["team", "proj_pts"], ascending=[True, False])
    print(f"{'team':4} {'player':24} {'plays':>5} {'min':>5} {'pts':>5} {'reb':>5} {'ast':>5} {'3pm':>5}")
    for r in res.itertuples(index=False):
        print(f"{r.team:4} {r.athlete_name[:24]:24} {r.p_play:5.0%} {r.exp_min:5.1f} "
              f"{r.proj_pts:5.1f} {r.proj_reb:5.1f} {r.proj_ast:5.1f} {r.proj_fg3m:5.1f}")


if __name__ == "__main__":
    main()
