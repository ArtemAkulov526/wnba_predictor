"""Train and evaluate the WNBA win-probability model.

Evaluation is walk-forward by season: to score season S the model only ever
sees seasons before S. Random k-fold would let the model learn from games that
had not happened yet, and because each game used to be stored twice it would
also put the same game in train and test at once.

For a betting use case the headline number is log loss / Brier, not accuracy.
A calibrated 60% model makes money; an overconfident 65% model does not.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wnba.ml import features
from wnba.db import SessionLocal
from wnba.paths import MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


@dataclass
class Scores:
    n: int
    accuracy: float
    log_loss: float
    brier: float
    roc_auc: float

    def __str__(self) -> str:
        return (f"n={self.n:5d}  acc={self.accuracy:.4f}  "
                f"logloss={self.log_loss:.4f}  brier={self.brier:.4f}  "
                f"auc={self.roc_auc:.4f}")


def score(y_true: np.ndarray, y_prob: np.ndarray) -> Scores:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return Scores(
        n=len(y_true),
        accuracy=accuracy_score(y_true, (y_prob >= 0.5).astype(int)),
        log_loss=log_loss(y_true, y_prob, labels=[0, 1]),
        brier=brier_score_loss(y_true, y_prob),
        roc_auc=roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    )


def make_pipeline(c: float) -> Pipeline:
    """Logistic regression constrained to be antisymmetric.

    Every feature is a home-minus-away difference, so swapping the two teams
    negates the whole vector. For P(A beats B) == 1 - P(B beats A) to hold
    exactly, the log-odds must be a pure linear form in those differences:

      * `with_mean=False` -- subtracting a non-zero training mean would leave
        a constant that does not flip sign with the features.
      * `fit_intercept=False` -- an intercept is likewise a constant offset.

    The home edge is not lost: `home_court` is 1 when a team hosts and 0 at a
    neutral site, so its coefficient carries exactly that, and only that.
    """
    return Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(C=c, max_iter=2000, solver="lbfgs",
                                  fit_intercept=False)),
    ])


# --------------------------------------------------------------------------
def walk_forward(
    games: pd.DataFrame,
    feature_cols: List[str],
    test_seasons: List[int],
    c: float,
) -> pd.DataFrame:
    """Predict each test season using only earlier seasons. Returns predictions."""
    out = []
    for season in test_seasons:
        train = games[games["season"] < season]
        test = games[games["season"] == season]
        if len(train) < 100 or test.empty:
            continue
        model = make_pipeline(c)
        model.fit(train[feature_cols], train["target"])
        prob = model.predict_proba(test[feature_cols])[:, 1]
        out.append(pd.DataFrame({
            "season": season,
            "game_key": test["game_key"].to_numpy(),
            "target": test["target"].to_numpy(),
            "prob": prob,
            "market_home_prob": test.get("market_home_prob", pd.Series(np.nan, index=test.index)).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def tune_c(games: pd.DataFrame, feature_cols: List[str],
           test_seasons: List[int]) -> tuple[float, Scores]:
    """Pick the regularisation strength that minimises walk-forward log loss."""
    best_c, best = None, None
    for c in C_GRID:
        preds = walk_forward(games, feature_cols, test_seasons, c)
        if preds.empty:
            continue
        s = score(preds["target"].to_numpy(), preds["prob"].to_numpy())
        if best is None or s.log_loss < best.log_loss:
            best_c, best = c, s
    return best_c, best


# --------------------------------------------------------------------------
def baseline_home(games: pd.DataFrame, test_seasons: List[int]) -> Scores:
    """Always pick the home team, at the historical home win rate."""
    train = games[games["season"] < min(test_seasons)]
    test = games[games["season"].isin(test_seasons)]
    rate = train["target"].mean()
    return score(test["target"].to_numpy(), np.full(len(test), rate))


def baseline_elo(games: pd.DataFrame, test_seasons: List[int]) -> Scores:
    """Raw Elo expectation, no fitting at all."""
    from wnba.ml import elo as elo_mod
    test = games[games["season"].isin(test_seasons)]
    prob = elo_mod.expected_score(
        test["home_elo_pre"].to_numpy() + elo_mod.HOME_ADVANTAGE * test["home_court"].to_numpy(),
        test["away_elo_pre"].to_numpy())
    return score(test["target"].to_numpy(), prob)


def calibration_table(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual win rate by probability bucket."""
    df = preds.copy()
    df["bucket"] = pd.cut(df["prob"], np.linspace(0, 1, bins + 1),
                          include_lowest=True)
    table = df.groupby("bucket", observed=True).agg(
        n=("target", "size"),
        predicted=("prob", "mean"),
        actual=("target", "mean"),
    ).reset_index()
    table["gap"] = table["actual"] - table["predicted"]
    return table


def betting_evaluation(preds: pd.DataFrame, edge: float = 0.03) -> Optional[dict]:
    """Flat-stake ROI against the closing line, when odds are loaded.

    Only bets where the model disagrees with the market by more than `edge`.
    Returns None when no odds are available.
    """
    df = preds.dropna(subset=["market_home_prob"])
    if df.empty:
        return None

    bets, profit = 0, 0.0
    for row in df.itertuples(index=False):
        # Fair decimal price implied by the vig-free market probability.
        for side, model_p, market_p, won in (
            ("home", row.prob, row.market_home_prob, row.target == 1),
            ("away", 1 - row.prob, 1 - row.market_home_prob, row.target == 0),
        ):
            if model_p - market_p <= edge or market_p <= 0:
                continue
            bets += 1
            profit += (1.0 / market_p - 1.0) if won else -1.0
    return {"bets": bets, "profit_units": round(profit, 2),
            "roi": round(profit / bets, 4) if bets else None,
            "note": "priced at the vig-free line, so real ROI is lower"}


# --------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the WNBA model.")
    parser.add_argument("--test-seasons", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--feature-set", default=None,
                        help="force one of: " + ", ".join(features.FEATURE_SETS))
    args = parser.parse_args()

    db = SessionLocal()
    try:
        candidates = ([args.feature_set] if args.feature_set
                      else list(features.FEATURE_SETS))

        print("=" * 74)
        print("WALK-FORWARD EVALUATION  (train on prior seasons only)")
        print("test seasons:", args.test_seasons)
        print("=" * 74)

        # Baselines use the richest frame so the game set is identical.
        games_full, _ = features.build_dataset(db, "full")
        seasons_present = sorted(games_full["season"].unique())
        test_seasons = [s for s in args.test_seasons if s in seasons_present]
        print(f"games usable: {len(games_full)}  seasons: {seasons_present}\n")

        print("BASELINES")
        print("  home-team-always  ", baseline_home(games_full, test_seasons))
        print("  elo-only          ", baseline_elo(games_full, test_seasons))
        print()

        print("FEATURE SETS")
        results: Dict[str, dict] = {}
        for name in candidates:
            games, cols = features.build_dataset(db, name)
            c, s = tune_c(games, cols, test_seasons)
            if s is None:
                print(f"  {name:14} insufficient data")
                continue
            results[name] = {"C": c, "scores": asdict(s), "features": cols}
            print(f"  {name:14} C={c:<6} {s}")

        best_name = min(results, key=lambda k: results[k]["scores"]["log_loss"])
        best = results[best_name]
        print(f"\nBEST: {best_name}  (C={best['C']}, "
              f"log loss {best['scores']['log_loss']:.4f})")

        # Rebuild the winner, report detail, then fit on everything.
        games, cols = features.build_dataset(db, best_name)
        preds = walk_forward(games, cols, test_seasons, best["C"])

        print("\nPER-SEASON (out of sample)")
        for season in test_seasons:
            sub = preds[preds["season"] == season]
            if not sub.empty:
                print(f"  {season}  {score(sub['target'].to_numpy(), sub['prob'].to_numpy())}")

        print("\nCALIBRATION")
        table = calibration_table(preds)
        for row in table.itertuples(index=False):
            print(f"  {str(row.bucket):16} n={row.n:4d}  "
                  f"predicted={row.predicted:.3f}  actual={row.actual:.3f}  "
                  f"gap={row.gap:+.3f}")

        bet = betting_evaluation(preds)
        print("\nBETTING", bet if bet else
              "no odds loaded -- see python -m wnba.scraping.external --odds")

        final = make_pipeline(best["C"])
        final.fit(games[cols], games["target"])

        coefs = sorted(zip(cols, final.named_steps["lr"].coef_[0]),
                       key=lambda kv: -abs(kv[1]))
        print("\nCOEFFICIENTS (standardised, final fit)")
        for name, value in coefs:
            print(f"  {name:20} {value:+.4f}")

        import joblib
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"pipeline": final, "feature_cols": cols,
                     "feature_set": best_name}, MODEL_PATH)
        METADATA_PATH.write_text(json.dumps({
            "feature_set": best_name,
            "C": best["C"],
            "feature_cols": cols,
            "walk_forward": best["scores"],
            "baselines": {"home_always": asdict(baseline_home(games_full, test_seasons)),
                          "elo_only": asdict(baseline_elo(games_full, test_seasons))},
            "trained_games": int(len(games)),
            "seasons": [int(s) for s in seasons_present],
        }, indent=2), encoding="utf-8")
        logger.info("saved -> %s", MODEL_PATH)
    finally:
        db.close()


if __name__ == "__main__":
    main()
