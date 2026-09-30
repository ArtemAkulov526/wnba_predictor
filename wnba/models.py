from sqlalchemy import (
    Column, Integer, String, Float, Date, Boolean, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from wnba.db import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    abbreviation = Column(String, unique=True, nullable=False)

    games = relationship("Game", back_populates="team")
    stats = relationship("Stat", back_populates="team")
    game_logs = relationship("GameLog", back_populates="team",
                             foreign_keys="GameLog.team_id")


class Game(Base):
    """Schedule-level row scraped from the team's /_games.html page.

    Kept for the existing UI; the model now trains off GameLog instead.
    """
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
    # The _games.html page has a separate "Playoffs Table"; see GameLog.is_playoff.
    is_playoff = Column(Boolean, nullable=False, server_default="false")

    team = relationship("Team", back_populates="games")

    __table_args__ = (
        UniqueConstraint("team_id", "season", "date", "opponent",
                         name="uq_game_team_date_opp"),
    )


class GameLog(Base):
    """One team's full box score for one game. Source of truth for modeling.

    Each real game appears twice, once per team. `date` is a real DATE so it
    sorts chronologically, and every column here is known at the final buzzer
    of THIS game -- never a season aggregate.
    """
    __tablename__ = "game_logs"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    opponent_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    season = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    game_num = Column(Integer, nullable=True)
    is_home = Column(Boolean, nullable=False)
    won = Column(Boolean, nullable=False)
    # True for postseason. basketball-reference's team gamelog page is regular
    # season only, so playoff rows are rebuilt from player box scores.
    is_playoff = Column(Boolean, nullable=False, server_default="false")

    pts = Column(Integer)
    opp_pts = Column(Integer)

    # team box score
    fg = Column(Integer); fga = Column(Integer)
    fg3 = Column(Integer); fg3a = Column(Integer)
    ft = Column(Integer); fta = Column(Integer)
    orb = Column(Integer); trb = Column(Integer)
    ast = Column(Integer); stl = Column(Integer)
    blk = Column(Integer); tov = Column(Integer); pf = Column(Integer)

    # opponent box score in the same game
    opp_fg = Column(Integer); opp_fga = Column(Integer)
    opp_fg3 = Column(Integer); opp_fg3a = Column(Integer)
    opp_ft = Column(Integer); opp_fta = Column(Integer)
    opp_orb = Column(Integer); opp_trb = Column(Integer)
    opp_ast = Column(Integer); opp_stl = Column(Integer)
    opp_blk = Column(Integer); opp_tov = Column(Integer); opp_pf = Column(Integer)

    team = relationship("Team", back_populates="game_logs", foreign_keys=[team_id])
    opponent = relationship("Team", foreign_keys=[opponent_id])

    __table_args__ = (
        UniqueConstraint("team_id", "date", "opponent_id", name="uq_gamelog"),
        Index("ix_gamelog_team_date", "team_id", "date"),
        Index("ix_gamelog_season", "season"),
    )


class PlayerGameLog(Base):
    """One player's line in one game, from the ESPN/sportsdataverse feed.

    This is what makes availability knowable: a player on the roster who did
    not appear has `minutes = NULL`, and players who were unavailable entirely
    are simply absent for that game. Comparing that against each player's
    recent minutes share is how `features.add_availability_features` measures
    who was missing, without needing an injury feed.
    """
    __tablename__ = "player_game_logs"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    season = Column(Integer, nullable=False)
    season_type = Column(Integer, nullable=True)   # 2 = regular season

    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    opponent_id = Column(Integer, ForeignKey("teams.id"), nullable=True)

    athlete_id = Column(String, nullable=False)
    athlete_name = Column(String, nullable=True)
    position = Column(String, nullable=True)

    minutes = Column(Float, nullable=True)         # NULL when the player sat
    did_not_play = Column(Boolean, nullable=True)
    reason = Column(String, nullable=True)
    starter = Column(Boolean, nullable=True)

    pts = Column(Integer); fgm = Column(Integer); fga = Column(Integer)
    fg3m = Column(Integer); fg3a = Column(Integer)
    ftm = Column(Integer); fta = Column(Integer)
    orb = Column(Integer); drb = Column(Integer); reb = Column(Integer)
    ast = Column(Integer); stl = Column(Integer); blk = Column(Integer)
    tov = Column(Integer); pf = Column(Integer)
    plus_minus = Column(Float)

    team = relationship("Team", foreign_keys=[team_id])

    __table_args__ = (
        UniqueConstraint("game_id", "team_id", "athlete_id", name="uq_player_game"),
        Index("ix_player_team_date", "team_id", "date"),
        Index("ix_player_athlete", "athlete_id"),
    )


class Stat(Base):
    """Season-aggregate team stats. Retained for the UI only.

    Deliberately NOT used as model features: a season average includes the
    game being predicted, which leaks the outcome backwards in time.
    """
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

    __table_args__ = (
        UniqueConstraint("team_id", "season", name="uq_stat_team_season"),
    )


class Injury(Base):
    """Player availability. Populated from CSV via `load_external.py`.

    Features read this when rows exist and fall back to neutral when empty,
    so the pipeline runs with or without it.
    """
    __tablename__ = "injuries"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    date = Column(Date, nullable=False)
    player = Column(String, nullable=False)
    status = Column(String, nullable=True)     # out / doubtful / questionable / probable
    # Share of the team's minutes this player normally takes (0-1). Drives the
    # "weight of who is missing" feature; 1 row per player per date.
    minutes_share = Column(Float, nullable=True)

    team = relationship("Team")

    __table_args__ = (
        UniqueConstraint("team_id", "date", "player", name="uq_injury"),
        Index("ix_injury_team_date", "team_id", "date"),
    )


class Odds(Base):
    """Closing market line for a game. Populated from CSV via `load_external.py`.

    Used as an evaluation benchmark and, optionally, as a feature.
    """
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    home_moneyline = Column(Integer, nullable=True)   # American odds, e.g. -150
    away_moneyline = Column(Integer, nullable=True)
    spread = Column(Float, nullable=True)             # home spread, e.g. -4.5
    total = Column(Float, nullable=True)
    book = Column(String, nullable=True)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

    __table_args__ = (
        UniqueConstraint("date", "home_team_id", "away_team_id", name="uq_odds"),
        Index("ix_odds_date", "date"),
    )
