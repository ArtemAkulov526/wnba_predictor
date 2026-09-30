"""FastAPI app: team browsing plus the matchup predictor.

The model is trained offline by `train.py` and loaded once at startup. The
previous version reloaded all three tables and refit a LogisticRegression on
every single request.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from wnba import models
from wnba.db import Base, SessionLocal, engine
from wnba.paths import TEMPLATES_DIR
from wnba.ml.predict import get_predictor
from wnba.ml.train import METADATA_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    try:
        get_predictor()          # warm the cache so the first request is fast
        app.state.model_ready = True
        app.state.model_error = None
    except Exception as exc:     # app still serves the browsing pages
        app.state.model_ready = False
        app.state.model_error = str(exc)
    yield


app = FastAPI(title="WNBA Predictor", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    teams = db.query(models.Team).order_by(models.Team.name).all()
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return templates.TemplateResponse(request, "index.html", {
        "teams": teams,
        "model_ready": getattr(app.state, "model_ready", False),
        "model_error": getattr(app.state, "model_error", None),
        "metadata": metadata,
    })


@app.get("/team/{team_id}", response_class=HTMLResponse)
async def get_team_stats(request: Request, team_id: int,
                         db: Session = Depends(get_db)):
    team = db.get(models.Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    stats = (db.query(models.Stat)
               .filter(models.Stat.team_id == team_id)
               .order_by(models.Stat.season.desc())
               .all())
    seasons = [s.season for s in stats]
    return templates.TemplateResponse(request, "team_stats.html", {
        "team": team, "stats": stats, "seasons": seasons,
    })


@app.get("/team/{team_id}/games", response_class=HTMLResponse)
async def get_team_games(request: Request, team_id: int,
                         season: Optional[int] = None,
                         db: Session = Depends(get_db)):
    team = db.get(models.Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    query = db.query(models.Game).filter(models.Game.team_id == team_id)
    available = sorted({s for (s,) in db.query(models.Game.season)
                                        .filter(models.Game.team_id == team_id)
                                        .distinct()}, reverse=True)
    # Default to the team's most recent season instead of erroring out.
    if season is None:
        season = available[0] if available else None
    if season is not None:
        query = query.filter(models.Game.season == season)

    return templates.TemplateResponse(request, "team_games.html", {
        "team": team, "season": season,
        "games": query.all(), "available_seasons": available,
    })


@app.post("/predict-game")
async def predict_game(
    team1: int = Form(...),
    team2: int = Form(...),
    home: int = Form(...),
):
    """Predict a matchup.

    `home` is 1 if team1 hosts, 0 if team2 hosts, -1 for a neutral court.
    """
    if team1 == team2:
        return JSONResponse({"error": "You must select two different teams."},
                            status_code=400)
    try:
        predictor = get_predictor()
    except SystemExit as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    neutral = home == -1
    # The model always predicts from the home side, so order the teams first.
    home_id, away_id = (team1, team2) if home != 0 else (team2, team1)

    try:
        result = predictor.predict(home_id, away_id, neutral=neutral)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return {
        "home_team": result.home_team,
        "away_team": result.away_team,
        "home_win_probability": result.home_win_probability,
        "away_win_probability": result.away_win_probability,
        "predicted_winner": result.predicted_winner,
        "neutral_site": result.neutral_site,
        "drivers": dict(list(result.drivers.items())[:5]),
    }


@app.get("/api/model")
async def model_info():
    """What the currently loaded model is and how it scored."""
    if not METADATA_PATH.exists():
        return JSONResponse({"error": "no trained model"}, status_code=404)
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
