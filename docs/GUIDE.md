# Guide — how it works and how to use it

Setup is in the [README](../README.md).

---

## Everyday commands

**Game predictions and picks**

```bash
python -m wnba.ml.predict --home MIN --away LVA                 # one game
python -m wnba.ml.predict --home MIN --away LVA --neutral
python -m wnba.ml.predict --home PHO --away DAL --auto-injuries # uses the latest injury report

python -m wnba.ml.tracking log --home PHO --away DAL --auto-injuries \
       --home-odds 2.77 --away-odds 1.40 --pick DAL --confidence 0.55 --tags injury
python -m wnba.ml.tracking closing --home PHO --away DAL --date 2026-09-21 --home-odds 2.90 --away-odds 1.38
python -m wnba.ml.tracking result --home PHO --away DAL --date 2026-09-21 --home-pts 90 --away-pts 85
python -m wnba.ml.tracking grade                                # results from scraped box scores
python -m wnba.ml.tracking report                               # model vs you vs market
python -m wnba.ml.tracking amend --home PHO --away DAL --date 2026-09-21 --confidence 0.55   # backfilled rows only
```

**Player props**

Prop lines come from a bookmaker's page, collected in your own browser with
the in-browser collector in `wnba/scraping/` (it is written for one page
layout and must be adapted for another bookmaker), or typed in by hand.

```bash
# read collected lines (text file) and add them to the props log; --closing stores them as closing odds
python -m wnba.scraping.saved_lines data/vbet_2026-09-30_ATL@WAS.txt --home WAS --away ATL --date 2026-09-30 --log --auto-injuries
python -m wnba.scraping.saved_lines data/vbet_2026-09-30_ATL@WAS_close.txt --home WAS --away ATL --date 2026-09-30 --closing

python -m wnba.ml.props_log add --home WAS --away ATL --date 2026-09-30 --player "Lauren Betts" \
       --stat pts --line 4.5 --over-odds 1.74 --under-odds 1.96 --pick over
python -m wnba.ml.props_log rank        # pending props by model expected value
python -m wnba.ml.props_log grade       # results from player box scores (per game)
python -m wnba.ml.props_log grade --espn   # straight from ESPN, before player_game_logs catches up
python -m wnba.ml.props_log report

python -m wnba.ml.props --home NYL --away ATL --auto-injuries   # projections for one game
python -m wnba.ml.props --backtest
```

**Parlays and live bets** (parlays are built from logged picks and compared with the same legs as singles)

```bash
python -m wnba.ml.parlays add --date 2026-09-30                     # all your picks that day
python -m wnba.ml.parlays add --date 2026-09-30 --games-only
python -m wnba.ml.parlays add --date 2026-10-04 --name "ATL same game" --leg "NYL@ATL:ATL" --leg "Angel Reese:reb"
python -m wnba.ml.parlays report

python -m wnba.ml.live_bets add --home ATL --away NYL --date 2026-10-04 --pick NYL --odds 1.94 --state "HT NYL 40-37"
python -m wnba.ml.live_bets report
```

**Playoffs**

```bash
python -m wnba.ml.series --seeds MIN,GSV,LVA,ATL,WAS,IND,DAL,NYL \
       --state MIN-NYL:0-2 LVA-IND:1-1 --auto-injuries
```

**Data refresh**

```bash
python -m wnba.scraping.basketball_reference --from-season 2026   # team box scores incl. playoffs
python -m wnba.scraping.player_logs --from-season 2026
python -m wnba.scraping.official_injuries          # latest official report (PDF)
python -m wnba.scraping.injuries --team PHO        # ESPN report (fallback)
python -m wnba.scraping.espn_odds                  # game odds (ESPN/DraftKings): open, current, close
python -m wnba.scraping.teamrankings               # current-season closing spreads + totals
```

**Benchmarks**

```bash
python -m wnba.ml.train                            # walk-forward evaluation + model selection
python -m wnba.ml.market                           # model vs closing spread
python -m wnba.ml.market --moneyline               # model vs DraftKings closing moneylines
python -m wnba.ml.totals --backtest                # totals model vs closing totals
```

---

## Current performance

Walk-forward over 2022–2026 (1,398 games, retrained 2026-10-01 with the full
regular season and the first playoff games): to score a season the model only
ever sees the seasons before it.

| Model | Accuracy | Log loss | Brier | ROC-AUC |
|---|---|---|---|---|
| Always pick the home team | 0.547 | 0.6890 | 0.2479 | 0.500 |
| Elo + home court only | 0.664 | 0.6134 | 0.2122 | 0.726 |
| All box-score features | 0.673 | 0.6099 | 0.2109 | 0.726 |
| Lean (5 features) | 0.677 | 0.6069 | 0.2096 | 0.728 |
| **Lean + availability** | **0.677** | **0.6024** | **0.2077** | **0.734** |

**Log loss is the headline number, not accuracy.** A well-calibrated 60% model
is useful for betting; an overconfident 65% model is not. Calibration gaps are
within ±0.06 across probability buckets.

Most of the signal is Elo. All box-score feature engineering together is worth
about 0.005 log loss; the most valuable single addition was **player
availability**, which helps most in evenly matched games (2026-09-20 run):

| Elo gap | Games | Lean | + availability | Delta |
|---|---|---|---|---|
| Toss-up (<80) | 497 | 0.6757 | 0.6657 | **−0.0100** |
| Moderate (80–200) | 578 | 0.6025 | 0.6043 | +0.0019 |
| Mismatch (200+) | 209 | 0.4747 | 0.4615 | **−0.0131** |

Against the market (2026 closing spreads) the model does **not** win: log loss
0.598 vs 0.582. Neither the totals model nor the props model has shown an
edge over closing lines yet.

---

## How the winner model works

One row per game, from the home team's perspective. Every feature is a
**difference** (home minus away), computed only from games played *before* the
game being predicted.

- **Pace-adjusted efficiency** — offensive/defensive rating per 100
  possessions, `POSS = FGA - ORB + TOV + 0.44 * FTA`. Raw per-game totals
  conflate playing fast with playing well.
- **Four factors**, for and against — eFG%, turnover rate, rebound rates,
  free-throw rate.
- **Form** — 20-game rolling window for stable quality, 5-game for recent form.
- **Elo** — margin-of-victory weighted, regressed 25% toward the mean between
  seasons.
- **Schedule** — rest days, back-to-backs, games in the last 7 days.
- **Home court** — 0 for the 2020 bubble season (single neutral site).
- **Availability** — the share of a team's normal minutes missing from the
  lineup.

Not every feature is in the selected model: `train.py` compares feature sets on
walk-forward log loss and saves the winner (currently "lean + availability").
With ~2,000 games, the regularisation strength `C` is tuned the same way.

### Availability without an injury history

`player_game_logs` holds every player's line in every game. For each team-game
the pipeline asks: given each player's minutes share over the team's previous
10 games, how much expected court time did not appear? A starter sitting
costs far more than a bench player. The window is shift-then-roll inside the
team, so a game never feeds its own figure, and a player who leaves the roster
decays out over 10 games instead of counting as injured forever. This needs no
injury feed and covers all nine seasons.

**Training and live prediction use different sources for the same feature.**
Injury reports only show *current* status, so they cannot train anything; the
model learns what "X% of normal minutes missing" is worth from box scores.
Before a game, the injury report says *who* will be missing, and
`predict.missing_share()` turns those names into the same X%, weighting
Questionable/Probable players by how often such players actually played
(measured from the 2026 report archive). In the playoffs the minutes shares
follow the playoff rotation (`rotation.py`).

### Symmetry

Predictions are exactly symmetric: `P(A beats B) == 1 - P(B beats A)` at a
neutral site. The scaler runs with `with_mean=False` and the regression with
`fit_intercept=False`, so the log-odds are a pure linear form in the
differences and negate when the teams swap.

---

## How the props model works

projection = expected minutes × per-minute rate × pace/defence adjustment

- **Minutes** (`rotation.py`): averaged over games a player actually played,
  weighted by how often she appears; scaled so the available players fill 200
  team minutes. In the playoffs, the top 5 get more minutes, the 6th the same
  and the bench less
  (measured on 64 playoff runs), until a team has 3 playoff games of its own.
- **Rates**: minutes-weighted recent and season-to-date per-minute rates.
- **Pace/defence**: from the totals model's possessions and opponent defence.
- **P(over)**: negative binomial around the projection (its spread fitted from
  box scores per stat; PRA fitted directly, so the three stats' correlation is
  included).

Ladder markets ("7+ rebounds") are stored as over 6.5 with one price; their
market percentage still includes the bookmaker's margin.
