from fastapi import FastAPI, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from sqlalchemy.orm import Session
from db import Base, engine, SessionLocal
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
