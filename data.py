from bs4 import BeautifulSoup
import asyncio
from playwright.async_api import async_playwright

filename1="test.txt"
filename2="text.txt"

async def url_builder():
    teams = ["MIN", "NYL", "ATL", "PHO", "IND", "LVA", 
             "SEA", "LAS", "WAS", "CHI", "DAL", "CON"]
    for team in teams:
        season = 2020  
        while season <= 2025:
            url_stats = f"https://www.basketball-reference.com/wnba/teams/{team}/{season}.html"
            url_team = f"https://www.basketball-reference.com/wnba/teams/{team}/{season}_games.html"
            await get_stats(url_stats, team, season)
            await get_info(url_team, team, season)
            season += 1

async def get_info(link, team, season):

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(link)
            await asyncio.sleep(3) # crawl delay 
            await page.wait_for_selector('#teams_games')  

            tbody_html = await page.inner_html('#teams_games')

            soup = BeautifulSoup(tbody_html, 'html.parser')
            rows = soup.find_all('tr')

            games = []
            exclude_columns = ['overtimes', 'notes']
            for row in rows:
                game = {}
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

            with open(filename2, "a", encoding="utf-8") as f:
                for game in games: 
                    f.write(f"team: {team}\n")
                    f.write(f"season: {season}\n")
                    for key, value in game.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
            
            print(f"{team} games from {season} saved to {filename2}")

            await browser.close()
            return games

async def get_stats(link, team, season):
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(link)
            await asyncio.sleep(3)  # crawl delay 
            await page.wait_for_selector('#div_team-stats')

            content = await page.inner_html('#div_team-stats')
            soup = BeautifulSoup(content, 'html.parser')
            table = soup.find('table')

            if not table or not table.tbody:
                print("Table or tbody not found")
                await browser.close()
                return

            rows = table.tbody.find_all('tr')

            stats = {}
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

            target_row = rows[2]  
            for stat in stats_to_extract:
                cell = target_row.find("td", {"data-stat": stat})
                stats[stat] = cell.text.strip() if cell else "None"

                    # Save to text file
            with open(filename1, "a", encoding="utf-8") as f:
                f.write(f"team: {team}\n")
                f.write(f"season: {season}\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

            print(f"{team} stats from {season} saved to {filename1}")
            await browser.close()
        
async def main():
    await url_builder()
    await get_stats("https://www.basketball-reference.com/wnba/teams/GSV/2025.html", "GSV", 2025)
    await get_info("https://www.basketball-reference.com/wnba/teams/GSV/2025_games.html", "GSV", 2025)

if __name__ == "__main__":
    asyncio.run(main())


