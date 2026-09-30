# wnba_predictor

Predicts WNBA games and player props, and tracks how those predictions — and
your own picks — hold up against the betting market.

- **Game winner model**: win probabilities from pace-adjusted team ratings,
  Elo and player availability (who is missing from the lineup), trained and
  tested walk-forward on 2018–2026.
- **Injury reports**: the official league PDF reports and ESPN, turned into
  the minutes a team is missing before tip-off.
- **Player props**: projected points, rebounds, assists, threes and
  points+rebounds+assists, with P(over) for a bookmaker's line.
- **Playoffs**: series and bracket odds from single-game probabilities.
- **Tracking**: logs of predictions, picks, odds and closing lines, graded
  automatically from box scores, with a report of model vs. you vs. market.
- **Web app**: team pages and a matchup predictor (FastAPI).

Data comes from basketball-reference, ESPN (via sportsdataverse), the WNBA's
official injury reports and TeamRankings, stored in PostgreSQL.

**Tech stack:** Python, PostgreSQL, SQLAlchemy, pandas, scikit-learn, FastAPI,
Playwright + BeautifulSoup.

How the models work, current performance and everyday commands:
[docs/GUIDE.md](docs/GUIDE.md).

---

## Setup

Requires Python 3.11+ and a running PostgreSQL server.

**1. Install**

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\playwright install chromium
.venv\Scripts\pip install -r requirements-ingest.txt   # player box scores (large, ~600 MB)
```

**2. Configure the database**

```bash
copy .env.example .env      # then set your Postgres password / WNBA_DB_URL
```

`.env` is gitignored.

**3. Create the tables**

```bash
python scripts/seed_teams.py     # the 15 franchises, incl. 2026 expansion teams
python scripts/migrate.py        # tables + constraints; safe to re-run
```

**4. Load the data** (first run takes a while; later runs only fill gaps)

```bash
python -m wnba.scraping.basketball_reference     # team box scores, schedules, stats (3 s crawl delay)
python -m wnba.scraping.player_logs              # per-player box scores 2018–now
python -m wnba.scraping.official_injuries        # latest official injury report
python -m wnba.scraping.teamrankings             # current-season closing spreads/totals
```

**5. Train**

```bash
python -m wnba.ml.train                          # walk-forward evaluation; saves models/
python -m wnba.ml.predict --home MIN --away LVA  # one prediction
```

**6. Run the web app**

```bash
uvicorn wnba.web.app:app --reload                # http://127.0.0.1:8000
```
