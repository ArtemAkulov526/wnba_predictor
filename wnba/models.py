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

    game_logs = relationship("GameLog", back_populates="team",
                             foreign_keys="GameLog.team_id")


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


class Odds(Base):
    """Closing market line for a game. Populated from CSV via `wnba.scraping.external --odds`.

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
