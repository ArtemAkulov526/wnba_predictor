from pydantic import BaseModel
from typing import Optional

class TeamBase(BaseModel):
    name: str
    abbreviation: str

class TeamCreate(TeamBase):
    pass

class TeamOut(TeamBase):
    id: int
    
    class Config:
        orm_mode = True


class GameBase(BaseModel):
    season: int
    date: Optional[str] = None
    game_location: Optional[str] = None
    opponent: Optional[str] = None
    result: Optional[str] = None
    points: Optional[int] = None
    opp_points: Optional[int] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    game_streak: Optional[str] = None

class GameCreate(GameBase):
    team_id: int

class GameOut(GameBase):
    id: int
    team_id: int
    
    class Config:
        orm_mode = True


class StatBase(BaseModel):
    season: int
    g: Optional[int] = None
    mp_per_g: Optional[float] = None
    fg_per_g: Optional[float] = None
    fga_per_g: Optional[float] = None
    fg_pct: Optional[float] = None
    fg3_per_g: Optional[float] = None
    fg3a_per_g: Optional[float] = None
    fg3_pct: Optional[float] = None
    fg2_per_g: Optional[float] = None
    fg2a_per_g: Optional[float] = None
    fg2_pct: Optional[float] = None
    ft_per_g: Optional[float] = None
    fta_per_g: Optional[float] = None
    ft_pct: Optional[float] = None
    orb_per_g: Optional[float] = None
    drb_per_g: Optional[float] = None
    trb_per_g: Optional[float] = None
    ast_per_g: Optional[float] = None
    stl_per_g: Optional[float] = None
    blk_per_g: Optional[float] = None
    tov_per_g: Optional[float] = None
    pf_per_g: Optional[float] = None
    pts_per_g: Optional[float] = None

class StatCreate(StatBase):
    team_id: int

class StatOut(StatBase):
    id: int
    team_id: int
    
    class Config:
        orm_mode = True
