# The Scott Stat

A cross-sport impact stat that separates **how good a player is** from **the
context he plays in**: coach, teammates, system. Basketball (NBA) first.

## Layers

| Layer | Question | Status |
|---|---|---|
| 1. Impact | How good is he, on offense and defense, independent of team? | **RAPM built and validated (1996-97 → 2024-25)** |
| 2. Wins | How much did he add to the wins that actually happened? | next |
| 3. Breakdown | *Why* — scoring, creation, playmaking, turnovers, defense | planned |
| 4. Context | Coach / teammate / system effects, learned from players who switch teams (AKM-style) | planned |

## Pipeline

```bash
pip install pandas numpy scipy pyarrow
python src/fetch.py 1996 2024          # play-by-play mirror (github.com/shufinskiy/nba_data)
python src/build_stints.py 1996 2024   # lineups → stints (same 10 on the floor)
python src/rapm.py 2022 2024           # offense/defense RAPM, lambda by game-grouped CV
python src/validate.py 2010 2016 2023  # out-of-sample test
```

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

## Known limits

- Play-by-play starts 1996-97: most of Jordan's career is box-score only.
  Cross-era layers need a box-score model calibrated against RAPM on modern seasons.
- Pure RAPM is noisy for low-minute players. The standard fix (a box-score
  prior, as EPM / LEBRON do) is the next accuracy step.
- Possessions are counted from possession-ending events; possessions still open
  at the end of a period are not counted (≈1% low).
