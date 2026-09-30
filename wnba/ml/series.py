"""Playoff series and bracket odds.

Format (official WNBA site):
  * top 8 by record, seeded 1-8; higher seed has home court in every round
  * first round  best-of-3, 1-1-1       (higher seed hosts games 1 and 3)
  * semifinals   best-of-5, 2-2-1       (hosts 1, 2, 5)
  * finals       best-of-7, 2-2-1-1-1   (hosts 1, 2, 5, 7)

Assumption: fixed bracket, no reseeding -- 1/8 vs 4/5 and 2/7 vs 3/6 meet in
the semifinals. Change BRACKET if the league reseeds.

Each game uses the winner model's probability for that venue (higher seed at
home, or away). Series odds are computed EXACTLY by walking every possible
sequence of results, not simulated, so they are deterministic and fast.
Mid-series, pass the current state (e.g. 1-0) to condition on games played.

Usage:
    python -m wnba.ml.series --seeds MIN,GSV,LVA,ATL,IND,WAS,DAL,NYL
    python -m wnba.ml.series --seeds ... --state MIN-NYL:1-0
    python -m wnba.ml.series --seeds ... --auto-injuries
"""
from __future__ import annotations

import argparse
from datetime import date
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# True = higher seed at home for that game number.
FORMATS = {
    "first_round": [True, False, True],
    "semifinals": [True, True, False, False, True],
    "finals": [True, True, False, False, True, False, True],
}
BRACKET = [((1, 8), (4, 5)), ((2, 7), (3, 6))]     # first-round pairs per semifinal


# --------------------------------------------------------------------------
# pure maths
# --------------------------------------------------------------------------
def series_outcomes(p_home: float, p_away: float, pattern: Sequence[bool],
                    state: Tuple[int, int] = (0, 0)) -> Dict[Tuple[str, int], float]:
    """P(each (winner, games played)) for a series from `state`.

    p_home / p_away: probability the HIGHER seed wins a game at home / away.
    state: (higher seed wins, lower seed wins) so far.
    """
    need = len(pattern) // 2 + 1
    pattern = tuple(pattern)

    @lru_cache(maxsize=None)
    def walk(w: int, l: int) -> Tuple[Tuple[Tuple[str, int], float], ...]:
        if w == need:
            return ((("high", w + l), 1.0),)
        if l == need:
            return ((("low", w + l), 1.0),)
        p = p_home if pattern[w + l] else p_away
        out: Dict[Tuple[str, int], float] = {}
        for key, v in walk(w + 1, l):
            out[key] = out.get(key, 0.0) + p * v
        for key, v in walk(w, l + 1):
            out[key] = out.get(key, 0.0) + (1 - p) * v
        return tuple(out.items())

    w, l = state
    if w > need or l > need or w + l > len(pattern):
        raise ValueError(f"impossible series state {state} for best-of-{len(pattern)}")
    return dict(walk(w, l))


def series_win(p_home: float, p_away: float, pattern: Sequence[bool],
               state: Tuple[int, int] = (0, 0)) -> float:
    """P(higher seed wins the series)."""
    return sum(v for (who, _), v in series_outcomes(p_home, p_away, pattern, state).items()
               if who == "high")


def bracket_odds(seeds: List[str], game_prob: Callable[[str, str], Tuple[float, float]],
                 states: Optional[Dict[frozenset, Tuple[str, int, int]]] = None
                 ) -> Dict[str, Dict[str, float]]:
    """P(each team reaches the semifinals, the finals, and wins the title).

    seeds: 8 team codes, seed 1 first.
    game_prob(high, low) -> (P(high wins at home), P(high wins away)).
    states: optional in-progress series, {frozenset({a, b}): (leader_code, wins_a_side...)}
            given as {frozenset: (team_x, x_wins, other_wins)}.
    """
    states = states or {}
    rank = {t: i for i, t in enumerate(seeds)}

    def win_prob(a: str, b: str, fmt: str) -> float:
        high, low = (a, b) if rank[a] < rank[b] else (b, a)
        ph, pa = game_prob(high, low)
        st = (0, 0)
        key = frozenset((a, b))
        if key in states:
            team, x, y = states[key]
            st = (x, y) if team == high else (y, x)
        p_high = series_win(ph, pa, FORMATS[fmt], st)
        return p_high if a == high else 1 - p_high

    odds = {t: {"semifinals": 0.0, "finals": 0.0, "title": 0.0} for t in seeds}
    finalists: List[Dict[str, float]] = []
    for pair_a, pair_b in BRACKET:
        # first round
        reach: Dict[str, float] = {}
        for s1, s2 in (pair_a, pair_b):
            a, b = seeds[s1 - 1], seeds[s2 - 1]
            p = win_prob(a, b, "first_round")
            reach[a], reach[b] = p, 1 - p
        for t, p in reach.items():
            odds[t]["semifinals"] = p
        # semifinal
        side_a = [seeds[pair_a[0] - 1], seeds[pair_a[1] - 1]]
        side_b = [seeds[pair_b[0] - 1], seeds[pair_b[1] - 1]]
        final: Dict[str, float] = {t: 0.0 for t in side_a + side_b}
        for x in side_a:
            for y in side_b:
                meet = reach[x] * reach[y]
                if meet == 0:
                    continue
                px = win_prob(x, y, "semifinals")
                final[x] += meet * px
                final[y] += meet * (1 - px)
        for t, p in final.items():
            odds[t]["finals"] = p
        finalists.append(final)

    for x, px in finalists[0].items():
        for y, py in finalists[1].items():
            meet = px * py
            if meet == 0:
                continue
            p = win_prob(x, y, "finals")
            odds[x]["title"] += meet * p
            odds[y]["title"] += meet * (1 - p)
    return odds


# --------------------------------------------------------------------------
# wiring to the winner model
# --------------------------------------------------------------------------
def model_game_prob(auto_injuries: bool = False, game_date: Optional[date] = None):
    """game_prob(high, low) backed by the trained model (+ current injuries)."""
    from wnba.db import SessionLocal
    from wnba.ml import predict

    db = SessionLocal()
    predictor = predict.build_predictor(db)
    report = rates = None
    if auto_injuries:
        from wnba.scraping.official_injuries import combined_report, measure_play_rates
        report, rates = combined_report(db, game_date or date.today()), measure_play_rates(db)

    @lru_cache(maxsize=None)
    def prob(high: str, low: str) -> Tuple[float, float]:
        if report is not None:
            at_home = predict.injury_scenarios(db, predictor, high, low, report, rates=rates)["headline"]
            away = 1 - predict.injury_scenarios(db, predictor, low, high, report, rates=rates)["headline"]
        else:
            at_home = predictor.predict(high, low).home_win_probability
            away = 1 - predictor.predict(low, high).home_win_probability
        return at_home, away

    return prob


def parse_states(raw: Optional[List[str]]) -> Dict[frozenset, Tuple[str, int, int]]:
    """'MIN-NYL:1-0' -> {frozenset({MIN, NYL}): ('MIN', 1, 0)}."""
    out = {}
    for item in raw or []:
        teams, score = item.upper().split(":")
        a, b = teams.split("-")
        x, y = (int(v) for v in score.split("-"))
        out[frozenset((a, b))] = (a, x, y)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Playoff series and bracket odds.")
    parser.add_argument("--seeds", required=True, help="8 team codes, seed 1 first")
    parser.add_argument("--state", nargs="*", help="in-progress series, e.g. MIN-NYL:1-0")
    parser.add_argument("--auto-injuries", action="store_true")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    seeds = [s.strip().upper() for s in args.seeds.split(",")]
    if len(seeds) != 8 or len(set(seeds)) != 8:
        parser.error("--seeds needs 8 different team codes")
    prob = model_game_prob(args.auto_injuries, args.date)
    states = parse_states(args.state)

    print("FIRST ROUND (best-of-3, 1-1-1)")
    for pair_a, pair_b in BRACKET:
        for s1, s2 in (pair_a, pair_b):
            high, low = seeds[s1 - 1], seeds[s2 - 1]
            ph, pa = prob(high, low)
            st = (0, 0)
            if frozenset((high, low)) in states:
                t, x, y = states[frozenset((high, low))]
                st = (x, y) if t == high else (y, x)
            out = series_outcomes(ph, pa, FORMATS["first_round"], st)
            p = sum(v for (w, _), v in out.items() if w == "high")
            lengths = "  ".join(f"{'%s' % (high if w == 'high' else low)} in {g}: {v:4.0%}"
                                for (w, g), v in sorted(out.items(), key=lambda kv: -kv[1]))
            state_txt = f"  [series {st[0]}-{st[1]}]" if st != (0, 0) else ""
            print(f"  ({s1}) {high} vs ({s2}) {low}:  {high} {p:5.1%}{state_txt}   "
                  f"per game: home {ph:.0%} / away {pa:.0%}   |  {lengths}")

    odds = bracket_odds(seeds, prob, states)
    print(f"\n{'seed team':10} {'semis':>7} {'finals':>7} {'title':>7}")
    for i, t in enumerate(seeds, 1):
        o = odds[t]
        print(f"  ({i}) {t:4}  {o['semifinals']:6.1%} {o['finals']:7.1%} {o['title']:7.1%}")
    print(f"\n  titles sum to {sum(o['title'] for o in odds.values()):.3f}  "
          "(bracket assumed fixed, no reseeding)")


if __name__ == "__main__":
    main()
