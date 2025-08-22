import pandas as pd
from db import SessionLocal
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

def load_data():
    db = SessionLocal()
    try:
        teams = pd.read_sql("SELECT * FROM teams", db.bind)
        games = pd.read_sql("SELECT * FROM games", db.bind)
        stats = pd.read_sql("SELECT * FROM stats", db.bind)
    finally:
        db.close()
    return teams, games, stats

def preprocess_data(teams, games, stats):
    """Merge tables and create features for modeling."""
    team_name_to_id = dict(zip(teams.name, teams.id))
    games["opponent_id"] = games["opponent"].map(team_name_to_id)

    games_subset = games[[
        "team_id", "season", "date", "game_location", "opponent", "opponent_id",
        "result", "points", "opp_points"
    ]]

    df = games_subset.merge(
        stats,
        left_on=["season", "team_id"],
        right_on=["season", "team_id"],
        suffixes=("", "_team_stats")
    )

    df = df.merge(
        stats,
        left_on=["season", "opponent_id"],
        right_on=["season", "team_id"],
        suffixes=("", "_opp_stats")
    )

    df["target"] = df["result"].apply(lambda r: 1 if r == "W" else 0)
    df["home"] = df["game_location"].apply(lambda x: 1 if x == "home" else 0)

    df = df.sort_values(["team_id", "date"])
    df["last_3_pts"] = df.groupby("team_id")["points"].shift(1).rolling(3, min_periods=1).mean()
    df["last_3_opp_pts"] = df.groupby("team_id")["opp_points"].shift(1).rolling(3, min_periods=1).mean()

    stat_cols = [
        "fg_per_g", "fga_per_g", "fg_pct", "fg3_per_g", "fg3a_per_g", "fg3_pct",
        "fg2_per_g", "fg2a_per_g", "fg2_pct", "ft_per_g", "fta_per_g", "ft_pct",
        "orb_per_g", "drb_per_g", "trb_per_g", "ast_per_g", "stl_per_g",
        "blk_per_g", "tov_per_g", "pf_per_g", "pts_per_g"
    ]
    opp_stat_cols = [c + "_opp_stats" for c in stat_cols]
    feature_cols = stat_cols + opp_stat_cols + ["home", "last_3_pts", "last_3_opp_pts"]

    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].fillna(0)
    y = df["target"]

    return df, X, y, feature_cols, stat_cols

def split_and_scale(df, X, y, feature_cols):
    train_df = df[df["season"] < 2024]
    test_df = df[df["season"] == 2024]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

    return model, acc, cv_acc, roc_auc, logloss

def predict_future_games(future_games_df, log_reg, scaler, feature_cols, past_games_df, stat_cols):
    """
    Predict WNBA game outcomes for future games.
    """
    past_games_df = past_games_df.sort_values(["team_id", "date"])

    for col in stat_cols:
        rolling_team = past_games_df.groupby("team_id")[col].rolling(3, min_periods=1).mean().shift(1)
        past_games_df[f"last_3_{col}"] = rolling_team.reset_index(level=0, drop=True)

    for col in stat_cols:
        opp_col = col + "_opp_stats"
        rolling_opp = past_games_df.groupby("team_id")[opp_col].rolling(3, min_periods=1).mean().shift(1)
        past_games_df[f"last_3_{opp_col}"] = rolling_opp.reset_index(level=0, drop=True)

    for col in stat_cols:
        future_games_df = future_games_df.merge(
            past_games_df[["team_id", f"last_3_{col}"]].drop_duplicates("team_id"),
            on="team_id", how="left"
        )
        future_games_df = future_games_df.merge(
            past_games_df[["team_id", f"last_3_{col}_opp_stats"]].drop_duplicates("team_id"),
            left_on="opponent_id",
            right_on="team_id",
            how="left",
            suffixes=("", "_opp")
        )

    for col in feature_cols:
        if col not in future_games_df.columns:
            future_games_df[col] = 0

    X_future = future_games_df[feature_cols].fillna(0)
    X_future_scaled = scaler.transform(X_future)

    future_games_df["predicted_win"] = log_reg.predict(X_future_scaled)
    future_games_df["win_probability"] = log_reg.predict_proba(X_future_scaled)[:, 1]

    return future_games_df

def main():
    teams, games, stats = load_data()
    df, X, y, feature_cols, stat_cols = preprocess_data(teams, games, stats)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df, X, y, feature_cols)
    log_reg, acc, cv_acc, roc_auc, logloss = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

    print("=== Logistic Regression ===")
    print("Accuracy:", acc)
    print("Cross-validation Accuracy:", cv_acc)
    print("ROC-AUC:", roc_auc)
    print("Log Loss:", logloss)

if __name__ == "__main__":
    main()

#=== Logistic Regression ===
# Accuracy: 0.7479166666666667
# Cross-validation Accuracy: 0.6603384410393071
# ROC-AUC: 0.8036458333333334
# Log Loss: 0.5497072508248722
