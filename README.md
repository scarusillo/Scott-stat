# The Scott Stat

A cross-sport impact stat that separates **how good a player is** from **the
context he plays in**: coach, teammates, system. Basketball (NBA) first.

## Layers

| Layer | Question | Status |
|---|---|---|
| 1. Impact | How good is he, on offense and defense, independent of team? | **RAPM built and validated (1996-97 → 2024-25)** |
| 2. Wins | How much did he add to the wins that actually happened? | **WAR + leverage-weighted cWAR built (1996-97 → 2024-25)** |
| 3. Breakdown | *Why* — scoring, creation, playmaking, turnovers, defense | planned |
| 4. Context | Coach / teammate / system effects, learned from players who switch teams (AKM-style) | planned |

## Pipeline

```bash
pip install pandas numpy scipy pyarrow
python src/fetch.py 1996 2024          # play-by-play mirror (github.com/shufinskiy/nba_data)
python src/build_stints.py 1996 2024   # lineups → stints (same 10 on the floor)
python src/rapm.py 2022 2024           # offense/defense RAPM, lambda by game-grouped CV
python src/validate.py 2010 2016 2023  # out-of-sample test
python src/box.py 1996 2024            # box-score totals from the same play-by-play
python src/wins.py 1996 2024           # Layer 2: season ratings -> WAR / cWAR
python src/validate_wins.py            # Layer 2 out-of-sample test
```

Season labels in files are the START year: `war_2024.csv` = 2024-25.

`out/rapm_<a>_<b>.csv` holds 3-season rolling RAPM for every window 1996-98 → 2022-24
(λ = 3000, the CV optimum). O and D are points per 100 possessions vs league
average; positive is good on both.

## Validation so far

**Lineup reconstruction** matches official league totals every season (e.g.
2023-24: 113.9 pts / 97.5 poss per team-game vs official 114.2 / 98.5;
1996-97: 96.5 / 90.6 vs 96.9 / 90.1). 0.3% of periods are unrecoverable and dropped.

**Sanity across eras** (3-season RAPM leaders): 1996-98 Jordan #1; 2007-09
LeBron #1, Garnett the top defender, Nash the top offense-only star; 2014-16
LeBron/Curry; 2022-24 Jokic, Gilgeous-Alexander.

**Out-of-sample**: fit on Y-3..Y-1, predict every game of season Y from the
lineups that played.

| Season | Pick winner: RAPM | Raw +/- | Home team |
|---|---|---|---|
| 2000-01 | 63.8% | 60.6% | 58.8% |
| 2010-11 | 66.2% | 61.8% | 59.6% |
| 2016-17 | 65.8% | 62.3% | 58.0% |
| 2023-24 | 66.2% | 60.4% | 53.7% |
| 2024-25 | 64.1% | 60.8% | 53.9% |

## Layer 2 — Wins

`out/war_<YYYY>.csv`, one row per player-season:

- **Season rating (O, D, net):** single-season RAPM shrunk toward the player's
  prior two seasons (rookies toward the newcomer average), then centred so the
  minutes-weighted average player = 0.
- **Replacement level:** fixed in wins, as baseball WAR does: a team of freely
  available players wins 20%. That works out to about −2.0 per 100 every season,
  which matches the convention Basketball-Reference's VORP uses (an independent
  agreement). League WAR = 0.3 × team-games every season (738 in an 82-game year).
- **Points → wins:** fit each season from actual results (29–37 points per win; the
  number rises as scoring rises).
- **WAR** = (rating − replacement) × possessions / 100 × wins per point.
- **cWAR** is the same, with each possession weighted by **leverage**: how much one
  point moved win probability at that score and clock (logistic win-probability
  model). Garbage time ≈ 0, crunch time > 1. This is the "contributed to wins that
  mattered" number. `leverage` > 1 means the player's minutes were above average
  in leverage.
- `team_margin_r` in `war_meta.json`: summed player ratings reproduce each team's
  season margin (r ≈ 0.98–0.996, in-sample).

**Out-of-sample:** rate players from season Y only, apply those ratings to the minutes
each team gave in Y+1, and predict Y+1 wins (831 team-seasons): error 8.2 wins
per 82 games, vs 9.8 for last year's record regressed to the mean (r 0.75 vs 0.62).

**Same points, different wins:**

| | PPG | TS% | Off | Def | Net/100 | WAR | Team W |
|---|---|---|---|---|---|---|---|
| Lillard 2022-23 | 32.2 | .65 | +5.3 | −0.9 | +4.4 | 8.0 | 33 |
| Gilgeous-Alexander 2022-23 | 31.4 | .63 | +4.2 | −0.6 | +3.5 | 8.3 | 40 |
| Gilgeous-Alexander 2024-25 | 32.7 | .64 | +6.6 | +3.2 | +9.8 | 17.3 | 68 |

At about 32 PPG on mediocre teams, the two were the same value (about 8 WAR). Shai's
2024-25 doubling came from his own defense (−0.6 → +3.2) and bigger offensive impact,
not from his points or his team. Across all 20+ PPG seasons, scoring explains
only about 15% of the variation in WAR (r = 0.38).

## Known limits

- Play-by-play starts 1996-97: most of Jordan's career is box-score only.
  Cross-era layers need a box-score model calibrated against RAPM on modern seasons.
- Pure RAPM is noisy for low-minute players. The standard fix (a box-score
  prior, as EPM / LEBRON do) is the next accuracy step.
- WAR scale runs below B-Ref's (≈ 2.7 × VORP): RAPM-based ratings are shrunk. It is
  internally consistent; compare within this system.
- The win-probability model ignores pregame team strength, so early-game leverage for
  heavy favourites is slightly overstated.
- Possessions are counted from possession-ending events; possessions still open
  at the end of a period are not counted (≈1% low).
