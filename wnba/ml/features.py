"""Feature engineering for the WNBA model.

Design rules, all of which the previous version broke:

1. Every feature is computed from games played STRICTLY BEFORE the game being
   predicted. Season aggregates are never used as features -- a season average
   contains the result you are trying to predict.
2. Rolling windows are computed inside `groupby(team)` with shift-then-roll,
   so a team's window never absorbs another team's games.
3. Rates are pace-adjusted (per 100 possessions). Raw per-game totals conflate
   playing fast with playing well.
4. The model sees DIFFERENCES (home minus away). Swapping the two teams
   negates every feature, so P(A beats B) == 1 - P(B beats A) by construction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from wnba.ml import elo

# The 2020 season was played entirely at IMG Academy with no crowds, so no
# team actually had home court even though the schedule labels one.
NEUTRAL_SEASONS = {2020}

LONG_WINDOW = 20    # stable measure of team quality
SHORT_WINDOW = 5    # recent form
MIN_PERIODS = 3     # below this a rolling value is too noisy to trust

BOX = ["fg", "fga", "fg3", "fg3a", "ft", "fta",
       "orb", "trb", "ast", "stl", "blk", "tov", "pf"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_game_logs(db: Session) -> pd.DataFrame:
    """Per-team-game box scores, one row per team per game."""
    cols = ", ".join(
        ["id", "team_id", "opponent_id", "season", "date", "game_num",
         "is_home", "won", "pts", "opp_pts"]
        + BOX + [f"opp_{c}" for c in BOX]
    )
    df = pd.read_sql(f"SELECT {cols} FROM game_logs", db.bind)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["team_id", "date"]).reset_index(drop=True)


def load_teams(db: Session) -> pd.DataFrame:
    return pd.read_sql("SELECT id, name, abbreviation FROM teams", db.bind)


def load_odds(db: Session) -> pd.DataFrame:
    try:
        df = pd.read_sql(
            "SELECT date, home_team_id, away_team_id, home_moneyline, "
            "away_moneyline, spread, total FROM odds", db.bind)
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------
# per-game efficiency metrics  (improvement #2: pace adjustment)
# --------------------------------------------------------------------------
def add_efficiency_metrics(logs: pd.DataFrame) -> pd.DataFrame:
    """Turn raw box score counts into pace-independent rates.

    Possessions use the standard estimator
        POSS = FGA - ORB + TOV + 0.44 * FTA
    and the two teams' estimates are averaged, since both played the same
    number of possessions in reality.
    """
    df = logs.copy()

    team_poss = df["fga"] - df["orb"] + df["tov"] + 0.44 * df["fta"]
    opp_poss = df["opp_fga"] - df["opp_orb"] + df["opp_tov"] + 0.44 * df["opp_fta"]
    poss = (team_poss + opp_poss) / 2.0
    poss = poss.replace(0, np.nan)

    df["possessions"] = poss
    df["off_rtg"] = 100.0 * df["pts"] / poss
    df["def_rtg"] = 100.0 * df["opp_pts"] / poss
    df["net_rtg"] = df["off_rtg"] - df["def_rtg"]
    df["pace"] = poss

    # Four factors, for and against.
    df["efg"] = (df["fg"] + 0.5 * df["fg3"]) / df["fga"].replace(0, np.nan)
    df["opp_efg"] = (df["opp_fg"] + 0.5 * df["opp_fg3"]) / df["opp_fga"].replace(0, np.nan)
    df["tov_pct"] = df["tov"] / poss
    df["opp_tov_pct"] = df["opp_tov"] / poss

    team_drb = df["trb"] - df["orb"]
    opp_drb = df["opp_trb"] - df["opp_orb"]
    df["orb_pct"] = df["orb"] / (df["orb"] + opp_drb).replace(0, np.nan)
    df["drb_pct"] = team_drb / (team_drb + df["opp_orb"]).replace(0, np.nan)

    df["ft_rate"] = df["ft"] / df["fga"].replace(0, np.nan)
    df["opp_ft_rate"] = df["opp_ft"] / df["opp_fga"].replace(0, np.nan)

    df["margin"] = df["pts"] - df["opp_pts"]
    return df


# Metrics that get rolled into "form" features.
ROLL_METRICS = [
    "off_rtg", "def_rtg", "efg", "opp_efg", "tov_pct", "opp_tov_pct",
    "orb_pct", "drb_pct", "ft_rate", "opp_ft_rate", "pace",
]


def add_rolling_features(
    logs: pd.DataFrame,
    long_window: int = LONG_WINDOW,
    short_window: int = SHORT_WINDOW,
) -> pd.DataFrame:
    """Trailing averages per team, excluding the current game.

    shift(1) BEFORE rolling(): the current game must not be in its own
    feature window, and the shift has to happen inside the group or a team's
    first rows absorb the previous team's games.
    """
    df = logs.sort_values(["team_id", "date"]).copy()
    grouped = df.groupby("team_id", sort=False)

    for metric in ROLL_METRICS:
        df[f"r_{metric}"] = grouped[metric].transform(
            lambda s: s.shift(1).rolling(long_window, min_periods=MIN_PERIODS).mean())

    # Recent form: short window on net rating only, to limit dimensionality.
    df["r_net_rtg_short"] = grouped["net_rtg"].transform(
        lambda s: s.shift(1).rolling(short_window, min_periods=2).mean())

    # Experience so far, used to damp unreliable early-season windows.
    df["games_played"] = grouped.cumcount()
    return df


# --------------------------------------------------------------------------
# rest and schedule  (improvement #4)
# --------------------------------------------------------------------------
def add_rest_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Days off, back-to-backs and recent workload -- free from the dates."""
    df = logs.sort_values(["team_id", "date"]).copy()
    grouped = df.groupby("team_id", sort=False)

    prev_date = grouped["date"].shift(1)
    rest = (df["date"] - prev_date).dt.days
    # First game of a season: treat as fully rested rather than missing.
    df["rest_days"] = rest.clip(upper=7).fillna(7)
    df["is_b2b"] = (rest <= 1).fillna(False).astype(int)

    def games_last_7(group: pd.DataFrame) -> pd.Series:
        dates = group.set_index("date")["pts"]
        # Count prior games only: rolling on a shifted series of ones.
        ones = pd.Series(1, index=dates.index)
        return ones.shift(1).rolling("7D").sum()

    counts = []
    for _, group in df.groupby("team_id", sort=False):
        group = group.sort_values("date")
        ones = pd.Series(1.0, index=pd.DatetimeIndex(group["date"]))
        rolled = ones.rolling("7D").sum().to_numpy() - 1.0  # exclude current game
        counts.append(pd.Series(rolled, index=group.index))
    df["games_last_7"] = pd.concat(counts).reindex(df.index).fillna(0.0)
    return df


# --------------------------------------------------------------------------
# game-level table (one row per real game, home perspective)
# --------------------------------------------------------------------------
def build_game_table(logs: pd.DataFrame) -> pd.DataFrame:
    """Collapse the mirrored team-game rows into one row per game.

    Each real game is stored twice (once per team). Keeping both would put the
    same game in several CV folds, which silently inflates every score.
    """
    logs = logs.copy()
    home_id = np.where(logs["is_home"], logs["team_id"], logs["opponent_id"])
    away_id = np.where(logs["is_home"], logs["opponent_id"], logs["team_id"])
    logs["game_key"] = (
        logs["date"].dt.strftime("%Y%m%d") + "_"
        + pd.Series(home_id, index=logs.index).astype(str) + "_"
        + pd.Series(away_id, index=logs.index).astype(str)
    )

    feature_cols = ([f"r_{m}" for m in ROLL_METRICS]
                    + ["r_net_rtg_short", "games_played",
                       "rest_days", "is_b2b", "games_last_7"])

    home = logs[logs["is_home"]].copy()
    away = logs[~logs["is_home"]].copy()

    keep_home = ["game_key", "date", "season", "team_id", "opponent_id",
                 "pts", "opp_pts", "won"] + feature_cols
    home = home[keep_home].rename(
        columns={"team_id": "home_team_id", "opponent_id": "away_team_id",
                 "pts": "home_pts", "opp_pts": "away_pts", "won": "home_won",
                 **{c: f"home_{c}" for c in feature_cols}})

    away = away[["game_key"] + feature_cols].rename(
        columns={c: f"away_{c}" for c in feature_cols})

    games = home.merge(away, on="game_key", how="inner", validate="one_to_one")
    return games.sort_values(["date", "game_key"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# availability, reconstructed from player box scores
# --------------------------------------------------------------------------
AVAILABILITY_WINDOW = 10   # games of recent usage used to weight an absence


def load_player_logs(db: Session) -> pd.DataFrame:
    try:
        df = pd.read_sql(
            "SELECT team_id, date, athlete_id, minutes FROM player_game_logs",
            db.bind)
    except Exception:
        return pd.DataFrame(columns=["team_id", "date", "athlete_id", "minutes"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    return df


def compute_availability(
    player_logs: pd.DataFrame, window: int = AVAILABILITY_WINDOW
) -> pd.DataFrame:
    """Share of a team's normal minutes that was missing for each game.

    For every team-game we ask: which players would we have expected to play,
    based on their minutes share over the team's previous `window` games, and
    how much of that expected share actually did not take the floor?

    The expectation is built with shift-then-roll inside the team, so a game
    never contributes to its own availability figure. A player who leaves the
    roster for good decays out of the window over `window` games rather than
    counting as "injured" forever.

    Returns one row per (team_id, date) with `out_share` in 0..1.
    """
    if player_logs.empty:
        return pd.DataFrame(columns=["team_id", "date", "out_share"])

    records = []
    for team_id, group in player_logs.groupby("team_id", sort=False):
        # Dense games x players grid; absent from a box score means 0 minutes.
        panel = (group.pivot_table(index="date", columns="athlete_id",
                                   values="minutes", aggfunc="sum")
                      .sort_index().fillna(0.0))
        if panel.empty:
            continue

        team_minutes = panel.sum(axis=1).replace(0, np.nan)
        share = panel.div(team_minutes, axis=0).fillna(0.0)

        expected = share.shift(1).rolling(window, min_periods=1).mean()
        missing = (panel <= 0) & expected.notna()
        out_share = (expected.where(missing, 0.0)).sum(axis=1)

        records.append(pd.DataFrame({
            "team_id": team_id,
            "date": panel.index,
            "out_share": out_share.to_numpy(),
        }))

    return pd.concat(records, ignore_index=True)


def add_availability_features(
    games: pd.DataFrame, availability: pd.DataFrame
) -> pd.DataFrame:
    """Attach home/away missing-minutes share, defaulting to 0 when unknown."""
    df = games.copy()
    if availability is None or availability.empty:
        df["home_out_share"] = 0.0
        df["away_out_share"] = 0.0
        return df

    for side in ("home", "away"):
        df = df.merge(
            availability.rename(columns={"team_id": f"{side}_team_id",
                                         "out_share": f"{side}_out_share"}),
            on=[f"{side}_team_id", "date"], how="left")
        df[f"{side}_out_share"] = df[f"{side}_out_share"].fillna(0.0)
    return df


# --------------------------------------------------------------------------
# optional external signals  (improvements #6 and #7)
# --------------------------------------------------------------------------
def american_to_prob(odds: pd.Series) -> pd.Series:
    """American moneyline -> implied probability (with vig still in)."""
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def add_market_probability(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Vig-free market probability that the home team wins.

    Used as the benchmark to beat. Kept out of the feature matrix by default:
    a model fed the closing line mostly learns to copy it.
    """
    df = games.copy()
    if odds is None or odds.empty:
        df["market_home_prob"] = np.nan
        return df

    odds = odds.copy()
    home_p = american_to_prob(odds["home_moneyline"])
    away_p = american_to_prob(odds["away_moneyline"])
    total = home_p + away_p
    odds["market_home_prob"] = np.where(total > 0, home_p / total, np.nan)

    return df.merge(
        odds[["date", "home_team_id", "away_team_id", "market_home_prob"]],
        on=["date", "home_team_id", "away_team_id"], how="left")


# --------------------------------------------------------------------------
# assembling the model matrix  (improvement #3: differences)
# --------------------------------------------------------------------------
DIFF_METRICS = [
    ("off_rtg", "d_off_rtg"),
    ("def_rtg", "d_def_rtg"),
    ("net_rtg_short", "d_net_rtg_short"),
    ("efg", "d_efg"),
    ("opp_efg", "d_opp_efg"),
    ("tov_pct", "d_tov_pct"),
    ("opp_tov_pct", "d_opp_tov_pct"),
    ("orb_pct", "d_orb_pct"),
    ("drb_pct", "d_drb_pct"),
    ("ft_rate", "d_ft_rate"),
    ("opp_ft_rate", "d_opp_ft_rate"),
]

FEATURE_SETS: Dict[str, List[str]] = {
    # Cheapest baseline: rating gap plus home court.
    "elo_only": ["d_elo", "home_court"],
    # Elo plus schedule situation.
    "elo_rest": ["d_elo", "home_court", "d_rest_days", "d_is_b2b", "d_games_last_7"],
    # Efficiency without Elo.
    "ratings": ["d_off_rtg", "d_def_rtg", "d_net_rtg_short", "home_court",
                "d_rest_days", "d_is_b2b", "d_games_last_7"],
    # Four factors without Elo.
    "four_factors": ["d_efg", "d_opp_efg", "d_tov_pct", "d_opp_tov_pct",
                     "d_orb_pct", "d_drb_pct", "d_ft_rate", "d_opp_ft_rate",
                     "home_court", "d_rest_days", "d_is_b2b", "d_games_last_7"],
    # Elo + pace-adjusted efficiency + home court. Ablation on 2022-2026 showed
    # the four factors and the rest/schedule group are net noise on top of
    # these, so the smaller model both scores better and generalises better.
    "lean": ["d_elo", "d_off_rtg", "d_def_rtg", "d_net_rtg_short", "home_court"],
    # Lean plus the reconstructed availability signal.
    "lean_avail": ["d_elo", "d_off_rtg", "d_def_rtg", "d_net_rtg_short",
                   "home_court", "d_out_share"],
    # Everything.
    "full": ["d_elo", "d_off_rtg", "d_def_rtg", "d_net_rtg_short",
             "d_efg", "d_opp_efg", "d_tov_pct", "d_opp_tov_pct",
             "d_orb_pct", "d_drb_pct", "d_ft_rate", "d_opp_ft_rate",
             "home_court", "d_rest_days", "d_is_b2b", "d_games_last_7"],
}
# Availability difference (share of normal minutes missing, home minus away).
INJURY_FEATURE = "d_out_share"


def add_difference_features(games: pd.DataFrame) -> pd.DataFrame:
    """Home minus away for every paired feature, plus the home-court flag."""
    df = games.copy()

    for metric, name in DIFF_METRICS:
        df[name] = df[f"home_r_{metric}"] - df[f"away_r_{metric}"]

    df["d_rest_days"] = df["home_rest_days"] - df["away_rest_days"]
    df["d_is_b2b"] = df["home_is_b2b"] - df["away_is_b2b"]
    df["d_games_last_7"] = df["home_games_last_7"] - df["away_games_last_7"]
    df["d_elo"] = df["home_elo_pre"] - df["away_elo_pre"]

    if "home_out_share" in df.columns:
        df[INJURY_FEATURE] = df["home_out_share"] - df["away_out_share"]

    # No home court in a neutral-site season.
    df["home_court"] = (~df["season"].isin(NEUTRAL_SEASONS)).astype(float)
    df["target"] = df["home_won"].astype(int)
    return df


def build_dataset(
    db: Session,
    feature_set: str = "full",
    include_injuries: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """End-to-end: database -> model-ready game table.

    Returns the full frame (features plus context columns) and the list of
    feature column names to train on.
    """
    logs = load_game_logs(db)
    if logs.empty:
        raise SystemExit("game_logs is empty -- run: python -m wnba.scraping.basketball_reference")

    logs = add_efficiency_metrics(logs)
    logs = add_rolling_features(logs)
    logs = add_rest_features(logs)

    games = build_game_table(logs)

    ratings = elo.compute_elo(games[[
        "game_key", "date", "season", "home_team_id", "away_team_id",
        "home_pts", "away_pts", "home_won"]])
    games = games.merge(ratings, on="game_key", how="left")

    if include_injuries:
        # Training uses availability reconstructed from box scores: it is the
        # only source with history (all nine seasons). Injury reports carry
        # current status only, so they feed live predictions
        # (predict.missing_share), never training.
        games = add_availability_features(
            games, compute_availability(load_player_logs(db)))
    games = add_market_probability(games, load_odds(db))
    games = add_difference_features(games)

    # Feature-set membership is explicit. Auto-appending the availability
    # column to every set made "lean" and "lean_avail" the same model, which
    # silently destroyed the comparison between them.
    feature_cols = list(FEATURE_SETS[feature_set])
    if INJURY_FEATURE in feature_cols and INJURY_FEATURE not in games.columns:
        raise SystemExit(
            f"feature set {feature_set!r} needs {INJURY_FEATURE}, but no "
            "availability or injury data is loaded -- run: "
            "python -m wnba.scraping.player_logs")

    # A game is usable once both teams have enough history for the windows.
    games = games.dropna(subset=feature_cols).reset_index(drop=True)
    return games, feature_cols


def team_form_snapshot(db: Session) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """Each team's latest rolling form and Elo, for predicting upcoming games."""
    logs = load_game_logs(db)
    logs = add_efficiency_metrics(logs)
    logs = add_rolling_features(logs)
    logs = add_rest_features(logs)
    games = build_game_table(logs)

    ratings = elo.final_ratings(games[[
        "game_key", "date", "season", "home_team_id", "away_team_id",
        "home_pts", "away_pts", "home_won"]])

    # The rolling columns on a team's most recent row already exclude that
    # game, so roll them forward one game to include it.
    cols = [f"r_{m}" for m in ROLL_METRICS] + ["r_net_rtg_short"]
    latest = []
    for team_id, group in logs.groupby("team_id"):
        group = group.sort_values("date")
        row = {"team_id": team_id, "last_game": group["date"].iloc[-1]}
        for metric in ROLL_METRICS:
            row[f"r_{metric}"] = group[metric].tail(LONG_WINDOW).mean()
        row["r_net_rtg_short"] = group["net_rtg"].tail(SHORT_WINDOW).mean()
        latest.append(row)

    return pd.DataFrame(latest), ratings
