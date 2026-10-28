"""FastAPI app: team pages, the matchup predictor and the prediction log.

The model is trained offline by `train.py` and loaded once at startup. The
previous version reloaded all three tables and refit a LogisticRegression on
every single request.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from wnba import models
from wnba.db import SessionLocal
from wnba.paths import TEMPLATES_DIR
from wnba.ml import tracking
from wnba.ml.predict import get_predictor, injury_scenarios, missing_share
from wnba.ml.train import METADATA_PATH
from wnba.dates import us_today


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tables are created by scripts/migrate.py (setup step 3), not here.
    _model_status()              # warm the cache so the first request is fast
    yield


def _model_status() -> tuple:
    """(ready, error) -- checked per page load, so a model trained after the app
    started is picked up without a restart."""
    try:
        get_predictor()
        return True, None
    except (Exception, SystemExit) as exc:   # the browsing pages still work
        return False, str(exc)


app = FastAPI(title="WNBA Predictor", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, db: Session = Depends(get_db)):
    teams = db.query(models.Team).order_by(models.Team.name).all()
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    ready, error = _model_status()
    return templates.TemplateResponse(request, "index.html", {
        "teams": teams, "model_ready": ready, "model_error": error, "metadata": metadata,
    })


def _season_summary(db: Session, team_id: int) -> List[dict]:
    """Season-by-season line for one team, built from game_logs."""
    logs = pd.read_sql(
        "SELECT season, is_playoff, won, pts, opp_pts, fg, fga, fg3, fg3a, ft, fta, "
        "orb, trb, ast, tov, opp_fga, opp_orb, opp_tov, opp_fta "
        "FROM game_logs WHERE team_id = %(t)s", db.bind, params={"t": team_id})
    if logs.empty:
        return []
    logs["poss"] = logs.fga - logs.orb + logs.tov + 0.44 * logs.fta
    logs["opp_poss"] = logs.opp_fga - logs.opp_orb + logs.opp_tov + 0.44 * logs.opp_fta
    out = []
    for season, g in logs.groupby("season"):
        reg, po = g[~g.is_playoff], g[g.is_playoff]
        base = reg if len(reg) else g
        poss = (base.poss.sum() + base.opp_poss.sum()) / 2
        out.append({
            "season": int(season), "g": len(reg),
            "w": int(reg.won.sum()), "l": int((~reg.won).sum()),
            "po_g": len(po), "po_w": int(po.won.sum()), "po_l": int((~po.won).sum()),
            "pts": base.pts.mean(), "opp_pts": base.opp_pts.mean(),
            "margin": (base.pts - base.opp_pts).mean(),
            "off_rtg": 100 * base.pts.sum() / poss,
            "def_rtg": 100 * base.opp_pts.sum() / poss,
            "fg_pct": base.fg.sum() / base.fga.sum(),
            "fg3_pct": base.fg3.sum() / base.fg3a.sum(),
            "ft_pct": base.ft.sum() / base.fta.sum(),
            "trb": base.trb.mean(), "ast": base.ast.mean(), "tov": base.tov.mean(),
        })
    return sorted(out, key=lambda r: -r["season"])


@app.get("/team/{team_id}", response_class=HTMLResponse)
def get_team_stats(request: Request, team_id: int,
                         db: Session = Depends(get_db)):
    team = db.get(models.Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    return templates.TemplateResponse(request, "team_stats.html", {
        "team": team, "seasons": _season_summary(db, team_id),
    })


@app.get("/team/{team_id}/games", response_class=HTMLResponse)
def get_team_games(request: Request, team_id: int,
                         season: Optional[int] = None,
                         db: Session = Depends(get_db)):
    team = db.get(models.Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    available = sorted({s for (s,) in db.query(models.GameLog.season)
                                        .filter(models.GameLog.team_id == team_id)
                                        .distinct()}, reverse=True)
    # Default to the team's most recent season instead of erroring out.
    if season is None and available:
        season = available[0]

    games = []
    if season is not None:
        rows = (db.query(models.GameLog, models.Team.abbreviation)
                  .join(models.Team, models.Team.id == models.GameLog.opponent_id)
                  .filter(models.GameLog.team_id == team_id,
                          models.GameLog.season == season)
                  .order_by(models.GameLog.date).all())
        record = {False: [0, 0], True: [0, 0]}          # regular / playoff W-L
        for g, opp in rows:
            rec = record[bool(g.is_playoff)]
            rec[0 if g.won else 1] += 1
            games.append({"date": g.date, "is_playoff": g.is_playoff, "is_home": g.is_home,
                          "opponent": opp, "opponent_id": g.opponent_id, "won": g.won,
                          "pts": g.pts, "opp_pts": g.opp_pts,
                          "record": f"{rec[0]}-{rec[1]}"})

    return templates.TemplateResponse(request, "team_games.html", {
        "team": team, "season": season,
        "games": games, "available_seasons": available,
    })


def _names(raw: Optional[str]) -> List[str]:
    return [n.strip() for n in (raw or "").split(",") if n.strip()]


@app.post("/predict-game")
def predict_game(
    team1: int = Form(...),
    team2: int = Form(...),
    home: int = Form(...),
    team1_out: Optional[str] = Form(None),
    team2_out: Optional[str] = Form(None),
    auto_injuries: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Predict a matchup. (Plain `def`: FastAPI runs it in a worker thread, which
    the ESPN fallback needs -- it uses the synchronous Playwright API.)

    `home` is 1 if team1 hosts, 0 if team2 hosts, -1 for a neutral court.
    Missing players are typed per team, or taken from the latest injury report
    (`auto_injuries`), and turned into the share of normal minutes missing --
    the same availability feature the model was trained on.
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
    if home != 0:
        home_id, away_id, home_names, away_names = team1, team2, team1_out, team2_out
    else:
        home_id, away_id, home_names, away_names = team2, team1, team2_out, team1_out
    abbr = {v: k for k, v in predictor.by_abbr.items()}
    if home_id not in abbr or away_id not in abbr:
        return JSONResponse({"error": "Unknown team."}, status_code=400)
    home_abbr, away_abbr = abbr[home_id], abbr[away_id]

    extra, missing = {}, []
    if auto_injuries and (_names(home_names) or _names(away_names)):
        extra["note"] = ("The names you typed were not used: the latest injury report "
                         "replaces them when that box is ticked.")
    try:
        if auto_injuries:
            from wnba.scraping.official_injuries import combined_report, measure_play_rates
            report = combined_report(db, us_today())
            s = injury_scenarios(db, predictor, home_abbr, away_abbr, report, neutral,
                                 rates=measure_play_rates(db))
            result = predictor.predict(home_id, away_id, neutral=neutral,
                                       home_out=s["sides"]["home"]["expected"],
                                       away_out=s["sides"]["away"]["expected"])
            extra = {"healthy": s["healthy"], "low": s["low"], "high": s["high"]}
            for side in ("home", "away"):
                for p in s["sides"][side]["players"]:
                    missing.append({"team": s["sides"][side]["team"], "player": p["player"],
                                    "status": p["status"], "share": p["share"]})
        else:
            outs = {}
            for side, team, names in (("home", home_abbr, home_names),
                                      ("away", away_abbr, away_names)):
                players = _names(names)
                share, details = missing_share(db, team, players) if players else (0.0, [])
                outs[side] = share
                missing += [{"team": team, "player": d.get("player", ""), "status": "out",
                             "share": d.get("share", 0.0)} for d in details]
            result = predictor.predict(home_id, away_id, neutral=neutral,
                                       home_out=outs["home"], away_out=outs["away"])
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:     # injury sources: network, PDF, Playwright timeouts
        return JSONResponse({"error": f"Could not get the injury report: {type(exc).__name__}: {exc}. "
                                      "Try again, or untick the box and type the missing players."},
                            status_code=502)

    return {
        "home_team": result.home_team,
        "away_team": result.away_team,
        "home_win_probability": result.home_win_probability,
        "away_win_probability": result.away_win_probability,
        "predicted_winner": result.predicted_winner,
        "neutral_site": result.neutral_site,
        "drivers": dict(list(result.drivers.items())[:5]),
        "missing": missing,
        **extra,
    }


def _fmt(value, spec: str) -> str:
    return "" if value is None or pd.isna(value) else format(value, spec)


@app.get("/log", response_class=HTMLResponse)
def prediction_log(request: Request):
    """The prediction log (data/prediction_log.csv), newest first."""
    df = tracking.read_log()
    rows = []
    model_hits = user_hits = graded = user_graded = 0
    for r in df.sort_values("game_date", ascending=False).itertuples():
        done = not pd.isna(r.home_won)
        model_ok = user_ok = None
        if done:
            graded += 1
            model_ok = (r.model_home_prob >= 0.5) == bool(r.home_won)
            model_hits += model_ok
            if r.user_pick:
                user_graded += 1
                user_ok = (r.user_pick == r.home) == bool(r.home_won)
                user_hits += user_ok
        picked_home = r.user_pick == r.home if r.user_pick else None
        conf = None
        if r.user_pick and not pd.isna(r.user_home_prob):
            conf = r.user_home_prob if picked_home else 1 - r.user_home_prob
        odds = (r.home_odds if picked_home else r.away_odds) if r.user_pick else None
        score = ""
        if done and not pd.isna(r.home_pts):
            score = f"{int(r.away_pts)}–{int(r.home_pts)}"
        rows.append({
            "game_date": r.game_date, "home": r.home, "away": r.away,
            "model": _fmt(r.model_home_prob, ".1%"), "pick": r.user_pick or "",
            "conf": _fmt(conf, ".0%"), "odds": _fmt(odds, ".2f"),
            "close": _fmt(r.closing_home_prob, ".1%"), "score": score,
            "model_ok": "" if model_ok is None else ("✓" if model_ok else "✗"),
            "model_cls": "" if model_ok is None else ("win" if model_ok else "loss"),
            "user_ok": "" if user_ok is None else ("✓" if user_ok else "✗"),
            "user_cls": "" if user_ok is None else ("win" if user_ok else "loss"),
            "tags": r.tags or "",
        })
    summary = {"logged": len(df), "graded": graded,
               "model": f"{model_hits}/{graded}" if graded else "—",
               "user": f"{user_hits}/{user_graded}" if user_graded else "—"}
    return templates.TemplateResponse(request, "log.html",
                                      {"rows": rows, "summary": summary})


@app.get("/api/model")
def model_info():
    """What the currently loaded model is and how it scored."""
    if not METADATA_PATH.exists():
        return JSONResponse({"error": "no trained model"}, status_code=404)
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
