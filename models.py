from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from db import Base

class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    abbreviation = Column(String, unique=True, nullable=False)

    games = relationship("Game", back_populates="team")
    stats = relationship("Stat", back_populates="team")


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    season = Column(Integer, nullable=False)
    date = Column(String, nullable=True)
    game_location = Column(String, nullable=True) 
    opponent = Column(String, nullable=True)
    result = Column(String, nullable=True)
    points = Column(Integer, nullable=True)
    opp_points = Column(Integer, nullable=True)
    wins = Column(Integer, nullable=True)
    losses = Column(Integer, nullable=True)
    game_streak = Column(String, nullable=True)
    
    team = relationship("Team", back_populates="games")


class Stat(Base):
    __tablename__ = "stats"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    season = Column(Integer, nullable=False)

    g = Column(Integer, nullable=True)
    mp_per_g = Column(Float, nullable=True)
    fg_per_g = Column(Float, nullable=True)
    fga_per_g = Column(Float, nullable=True)
    fg_pct = Column(Float, nullable=True)
    fg3_per_g = Column(Float, nullable=True)
    fg3a_per_g = Column(Float, nullable=True)
    fg3_pct = Column(Float, nullable=True)
    fg2_per_g = Column(Float, nullable=True)
    fg2a_per_g = Column(Float, nullable=True)
    fg2_pct = Column(Float, nullable=True)
    ft_per_g = Column(Float, nullable=True)
    fta_per_g = Column(Float, nullable=True)
    ft_pct = Column(Float, nullable=True)
    orb_per_g = Column(Float, nullable=True)
    drb_per_g = Column(Float, nullable=True)
    trb_per_g = Column(Float, nullable=True)
    ast_per_g = Column(Float, nullable=True)
    stl_per_g = Column(Float, nullable=True)
    blk_per_g = Column(Float, nullable=True)
    tov_per_g = Column(Float, nullable=True)
    pf_per_g = Column(Float, nullable=True)
    pts_per_g = Column(Float, nullable=True)

    team = relationship("Team", back_populates="stats")
