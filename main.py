from fastapi import FastAPI, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from sqlalchemy.orm import Session
from db import Base, engine, SessionLocal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dataframes import load_data, preprocess_data, split_and_scale, predict_future_games
import pandas as pd
import models

app = FastAPI()

Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    teams = db.query(models.Team).all()
    return templates.TemplateResponse("index.html", {"request": request, "teams": teams})

@app.get("/team/{team_id}", response_class=HTMLResponse)
async def get_team_stats(request: Request, team_id: int, db: Session = Depends(get_db)):
    team = db.query(models.Team).filter(models.Team.id == team_id).first()
    if not team:
        return HTMLResponse(content=f"<h2>Team with id {team_id} not found</h2>", status_code=404)

    stats = db.query(models.Stat).filter(models.Stat.team_id == team_id).all()

    return templates.TemplateResponse(
        "team_stats.html",
        {"request": request, "team": team, "stats": stats}
    )

@app.get("/team/{team_id}/games", response_class=HTMLResponse)
async def get_team_games(request: Request, team_id: int, season: int, db: Session = Depends(get_db)):
    team = db.query(models.Team).filter(models.Team.id == team_id).first()
    if not team:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "message": f"Team with id {team_id} not found"},
            status_code=404,
        )

    games = db.query(models.Game).filter(
        models.Game.team_id == team_id,
        models.Game.season == season
    ).all()

    return templates.TemplateResponse(
        "team_games.html",
        {"request": request, "team": team, "season": season, "games": games}
    )

@app.post("/predict-game")
async def predict_game(
    team1: int = Form(...),
    team2: int = Form(...),
    home: int = Form(...),
    db: Session = Depends(get_db),
):
    if team1 == team2:
        return JSONResponse({"error": "You must select two different teams."}, status_code=400)

    teams, games, stats = load_data()
    df, X, y, feature_cols, stat_cols = preprocess_data(teams, games, stats)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df, X, y, feature_cols)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[df["season"] < 2024])
    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(X_train_scaled, y[df["season"] < 2024])

    future_games_df = pd.DataFrame([
        {"team_id": team1, "opponent_id": team2, "home": home}
    ])

    predictions_df = predict_future_games(
        future_games_df=future_games_df,
        log_reg=log_reg,
        scaler=scaler,
        feature_cols=feature_cols,
        past_games_df=df,
        stat_cols=[c for c in feature_cols if not c.endswith("_opp_stats")
                   and c not in ["home", "last_3_pts", "last_3_opp_pts"]]
    )

    id_to_name = dict(zip(teams.id, teams.name))
    predictions_df["team_name"] = predictions_df["team_id"].map(id_to_name)
    predictions_df["opponent_name"] = predictions_df["opponent_id"].map(id_to_name)

    result = predictions_df.iloc[0]
    predicted_winner_name = result["team_name"] if bool(result["predicted_win"]) else result["opponent_name"]
    

    return {
        "team_name": result["team_name"],
        "opponent_name": result["opponent_name"],
        "predicted_winner": predicted_winner_name,  
        "win_probability": float(result["win_probability"]),
    }
