import pandas as pd
from dataframes2 import load_data, preprocess_data, split_and_scale, predict_future_games

teams, games, stats = load_data()
df, X, y, feature_cols, stat_cols = preprocess_data(teams, games, stats)
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df, X, y, feature_cols)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X[df["season"] < 2024])
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train_scaled, y[df["season"] < 2024])

# It must have at least ['team_id', 'opponent_id', 'home'] columns
future_games_df = pd.DataFrame([
    {"team_id": 1, "opponent_id": 2, "home": 0},
    {"team_id": 2, "opponent_id": 10, "home": 1},
])

predictions_df = predict_future_games(
    future_games_df=future_games_df,
    log_reg=log_reg,
    scaler=scaler,
    feature_cols=feature_cols,
    past_games_df=df,
    stat_cols=[c for c in feature_cols if not c.endswith("_opp_stats") and c not in ["home", "last_3_pts", "last_3_opp_pts"]]
)

id_to_name = dict(zip(teams.id, teams.name))
predictions_df["team_name"] = predictions_df["team_id"].map(id_to_name)
predictions_df["opponent_name"] = predictions_df["opponent_id"].map(id_to_name)

print(predictions_df[["team_name", "opponent_name", "predicted_win", "win_probability"]])
