"""Does the model beat the closing line? Benchmark against market spreads.

Compares the model's OUT-OF-SAMPLE predictions for a season (trained only on
earlier seasons, see train.walk_forward) with the closing point spreads in
`data/closing_lines.csv` (wnba.scraping.teamrankings), on the same games.

Spread -> probability: the final margin is roughly normal around the market's
expected margin (-spread), so P(home wins) = Phi(-spread / sigma). sigma is
estimated from the margins themselves, using two folds (fit on one half of the
season, score the other) so the market is not handed an in-sample advantage.

Betting test: the model's probability is converted back to an implied spread
with the same sigma. When it differs from the closing line by at least
`threshold` points, bet that side against the spread at standard -110
(win +0.909 units, lose -1, push 0). Break-even is 52.4%.

Caveat: the feature set and C were chosen by walk-forward log loss over
2022-2026, which includes this season, so the model's side carries a small
selection advantage. The ATS sample is also one season: treat ROI as a signal,
not a result.

Usage:
    python -m wnba.ml.market                  # 2026
    python -m wnba.ml.market --season 2026
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from wnba.db import SessionLocal
from wnba.ml import features, train
from wnba.paths import MODELS_DIR
from wnba.scraping.teamrankings import LINES_PATH

WIN, LOSE = 100 / 110, -1.0          # -110 pricing
BREAK_EVEN = 110 / 210               # 52.38%


def load_predictions(season: int) -> pd.DataFrame:
    meta = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    db = SessionLocal()
    try:
        games, cols = features.build_dataset(db, meta["feature_set"])
        teams = pd.read_sql("SELECT id, abbreviation FROM teams", db.bind)
    finally:
        db.close()
    preds = train.walk_forward(games, cols, [season], meta["C"])
    abbr = dict(zip(teams["id"], teams["abbreviation"]))
    ctx = games[["game_key", "date", "home_team_id", "away_team_id",
                 "home_pts", "away_pts"]]
    p = preds.merge(ctx, on="game_key")
    p["game_date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
    p["home"] = p["home_team_id"].map(abbr)
    p["away"] = p["away_team_id"].map(abbr)
    p["margin"] = p["home_pts"] - p["away_pts"]
    return p[["game_date", "home", "away", "prob", "target", "margin"]]


def attach_lines(preds: pd.DataFrame, path=LINES_PATH) -> pd.DataFrame:
    """Join closing spreads, re-orienting games whose home/away is swapped
    relative to basketball-reference (seen for four 2026 Toronto games)."""
    lines = pd.read_csv(path)
    key = ["game_date", "home", "away"]
    direct = preds.merge(lines[key + ["home_spread"]], on=key)
    flipped = preds.merge(
        lines.rename(columns={"home": "away", "away": "home"})[key + ["home_spread"]],
        on=key)
    flipped["home_spread"] = -flipped["home_spread"]
    out = pd.concat([direct, flipped], ignore_index=True)
    return out.drop_duplicates(key).sort_values("game_date").reset_index(drop=True)


def fit_sigma(margin: pd.Series, spread: pd.Series) -> float:
    """Std of the final margin around the market's expected margin (-spread)."""
    return float(np.std(margin + spread, ddof=1))


def market_probs(df: pd.DataFrame) -> np.ndarray:
    """Two-fold P(home wins) from spreads: sigma from the other half."""
    half = len(df) // 2
    a, b = df.iloc[:half], df.iloc[half:]
    sig_a, sig_b = fit_sigma(b.margin, b.home_spread), fit_sigma(a.margin, a.home_spread)
    return np.concatenate([norm.cdf(-a.home_spread / sig_a),
                           norm.cdf(-b.home_spread / sig_b)])


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)),
            "accuracy": float(np.mean((p >= 0.5) == y))}


def ats_backtest(df: pd.DataFrame, sigma: float, thresholds=(0, 1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Bet the side the model prefers against the closing spread."""
    p = np.clip(df["prob"].to_numpy(), 1e-4, 1 - 1e-4)
    model_margin = sigma * norm.ppf(p)                  # model's expected home margin
    edge = model_margin - (-df["home_spread"].to_numpy())
    cover = df["margin"].to_numpy() + df["home_spread"].to_numpy()   # >0 home covers

    rows = []
    for k in thresholds:
        bet_home, bet_away = edge >= k, edge <= -k
        if k == 0:
            bet_away = ~bet_home & (edge < 0)
        picked = bet_home | bet_away
        side_cover = np.where(bet_home, cover, -cover)[picked]
        wins, pushes = int((side_cover > 0).sum()), int((side_cover == 0).sum())
        losses = int((side_cover < 0).sum())
        n = wins + losses
        profit = wins * WIN + losses * LOSE
        rows.append({"min_edge_pts": k, "bets": int(picked.sum()), "won": wins,
                     "lost": losses, "push": pushes,
                     "win_rate": wins / n if n else float("nan"),
                     "units": round(profit, 2),
                     "roi": profit / (n + pushes) if n + pushes else float("nan")})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model vs closing spreads.")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()

    df = attach_lines(load_predictions(args.season))
    y = df["target"].to_numpy()
    df["market_prob"] = market_probs(df)
    sigma = fit_sigma(df["margin"], df["home_spread"])

    model, market = scores(y, df["prob"].to_numpy()), scores(y, df["market_prob"].to_numpy())
    blend = scores(y, (df["prob"].to_numpy() + df["market_prob"].to_numpy()) / 2)

    print(f"{args.season}: {len(df)} games with both an out-of-sample prediction "
          f"and a closing spread (sigma of margin around the line: {sigma:.1f} pts)\n")
    print(f"{'':22} {'log loss':>9} {'brier':>8} {'accuracy':>9}")
    for name, s in (("model", model), ("closing spread", market),
                    ("50/50 blend", blend)):
        print(f"  {name:20} {s['log_loss']:9.4f} {s['brier']:8.4f} {s['accuracy']:9.1%}")

    diff = df["prob"] - df["market_prob"]
    print(f"\nmodel vs market probability: mean abs gap {diff.abs().mean():.1%}, "
          f"correlation {np.corrcoef(df['prob'], df['market_prob'])[0, 1]:.3f}")

    print(f"\nAGAINST THE SPREAD at -110 (break-even {BREAK_EVEN:.1%})")
    print(ats_backtest(df, sigma).to_string(index=False,
          formatters={"win_rate": "{:.1%}".format, "roi": "{:+.1%}".format}))


if __name__ == "__main__":
    main()
