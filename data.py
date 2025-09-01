import asyncio, logging
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page
from sqlalchemy.orm import Session
from db import SessionLocal
from models import Stat, Game, Team
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

async def collect_team_data(db: Session, page: Page) -> None:
    TEAMS = ["MIN", "NYL", "ATL", "PHO", "IND", "LVA", 
             "SEA", "LAS", "WAS", "CHI", "DAL", "CON"]
    for team in TEAMS:
        season = 2018  
        while season <= 2025:
            url_stats = f"https://www.basketball-reference.com/wnba/teams/{team}/{season}.html"
            url_games = f"https://www.basketball-reference.com/wnba/teams/{team}/{season}_games.html"
            await get_team_stats(db, page, url_stats, team, season)
            await get_team_games(db, page, url_games, team, season)
            season += 1

async def get_team_games(db: Session, page: Page, link: str, team: str, season: int) -> List[Dict[str, Optional[str]]]: 
            await page.goto(link)
            await asyncio.sleep(3) # crawl delay 
            await page.wait_for_selector('#teams_games')  

            tbody_html = await page.inner_html('#teams_games')

            soup = BeautifulSoup(tbody_html, 'html.parser')
            rows = soup.find_all('tr')

            games: List[Dict[str, Optional[str]]] = []
            exclude_columns = ['overtimes', 'notes']
            for row in rows:
                game: Dict[str, Optional[str]] = {}
                for td in row.find_all('td'):
                    key = td.get('data-stat')
                    if key and key not in exclude_columns:
                        text = td.text.strip()
                        if key == 'game_location':
                            if text == '@':
                                text = 'away'
                            else:
                                text = 'home'
                        game[key] = text
                if 'pts' in game and game['pts'] != '':
                    games.append(game)

            team_obj = db.query(Team).filter_by(abbreviation=team).first()
                
            db_games = []
            for game in games:
                mapped_game = {
                    "date": game.get("date_game"),
                    "game_location": game.get("game_location"),
                    "opponent": game.get("opp_name"),
                    "result": game.get("game_result"),
                    "points": int(game.get("pts")) if game.get("pts") else None,
                    "opp_points": int(game.get("opp_pts")) if game.get("opp_pts") else None,
                    "wins": int(game.get("wins")) if game.get("wins") else None,
                    "losses": int(game.get("losses")) if game.get("losses") else None,
                    "game_streak": game.get("game_streak")
                }

                db_games.append(Game(team_id=team_obj.id,season=season,**mapped_game))

                try:
                    with db.begin():
                        db.add_all(db_games)
                    logger.info(f"{team} games from {season} saved to DB")
                except Exception as e:
                    logger.error(f"Failed to save games for {team} {season}: {e}")

                return games

async def get_team_stats(db: Session, page: Page, link: str, team: str, season: int) -> None: 
            await page.goto(link)
            await asyncio.sleep(3)  # crawl delay 
            await page.wait_for_selector('#div_team-stats', state="visible")

            content = await page.inner_html('#div_team-stats')
            soup = BeautifulSoup(content, 'html.parser')
            table = soup.find('table', {'id': 'team-stats'})

            if not table or not table.tbody:
                logger.error(f"Table or tbody not found for {team} season {season}")
                return

            rows = table.tbody.find_all('tr')

            stats = {}
            # row[1] = totals for season, we want per-game averages
            first_row = rows[1]  
            cell_g = first_row.find("td", {"data-stat": "g"})
            stats["g"] = cell_g.text.strip() if cell_g else "None"

            stats_to_extract = [
                "mp_per_g", "fg_per_g", "fga_per_g", "fg_pct", "fg3_per_g",
                "fg3a_per_g", "fg3_pct", "fg2_per_g", "fg2a_per_g", "fg2_pct",
                "ft_per_g", "fta_per_g", "ft_pct", "orb_per_g", "drb_per_g",
                "trb_per_g", "ast_per_g", "stl_per_g", "blk_per_g", "tov_per_g",
                "pf_per_g", "pts_per_g"
            ]

            # row[2] = per-game averages
            target_row = rows[2]  
            for stat in stats_to_extract:
                cell = target_row.find("td", {"data-stat": stat})
                stats[stat] = cell.text.strip() if cell else "None"

            team_obj = db.query(Team).filter_by(abbreviation=team).first()
            db_stats = Stat(
                team_id=team_obj.id,
                season=season,
                **stats
            )
            try:
                with db.begin():
                    db.add(db_stats)
                logger.info(f"{team} stats from {season} saved to DB")
            except Exception as e:
                logger.error(f"Failed to save stats for {team} {season}: {e}")
        
async def main():
    db = SessionLocal()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                extra_http_headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/139.0.0.0 Safari/537.36"
                }
            )

            page = await context.new_page()
            await collect_team_data(db, page)
            await get_team_stats(db, page, "https://www.basketball-reference.com/wnba/teams/GSV/2025.html", "GSV", 2025)
            await get_team_games(db, page, "https://www.basketball-reference.com/wnba/teams/GSV/2025_games.html","GSV", 2025)
            await browser.close()
    finally:
        db.close()
       
if __name__ == "__main__":
    asyncio.run(main())