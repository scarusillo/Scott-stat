# The Scott Stat

A cross-sport impact stat that separates **how good a player is** from **the
context he plays in**: coach, teammates, system. Basketball (NBA) first.

## Layers

| Layer | Question | Status |
|---|---|---|
| 1. Impact | How good is he, on offense and defense, independent of team? | **RAPM built and validated (1996-97 → 2024-25)** |
| 2. Wins | How much did he add to the wins that actually happened? | **WAR + leverage-weighted cWAR built (1996-97 → 2024-25)** |
| 3. Breakdown | *Why* — scoring, creation, playmaking, turnovers, defense | planned |
| 4. Context | Coach / teammate / system effects, learned from players who switch teams (AKM-style) | **coach effects v1 built; split-half validated** |

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
python src/coaches.py                  # head coach for every game (W-L-aligned)
python src/coach_effects.py            # Layer 4 v1: coach O/D effects + SEs
python src/validate_coaches.py         # split-half reliability of coach effects
python src/archetypes.py               # 8 player types per season
python src/coach_styles.py             # coach x player-type model, style fingerprints
python src/coach_cells.py              # flags cells that are really one star
python src/rotation.py                 # Rotation IQ + Closing IQ
python src/development.py              # young-player growth by coach
python src/firings.py                  # mid-season firings: out-of-sample test
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

## Layer 4 (v1) — Coaches

**Coach for every game:** `data/coaches/coach_games.parquet`, `out/coach_seasons.csv`.
Source: every NBA head-coaching tenure (Wikipedia's per-franchise lists, CC BY-SA),
compiled in `rptetzloff/are-they-sports` (PR #119). That gives W-L per tenure but no
firing dates, so `coaches.py` places each tenure in the franchise's game sequence with a
dynamic programme that must reproduce the tenure's W-L from our own results. All
68,626 team-games are assigned. Of 290 tenures fully inside 1996-2025, 274 match the
source W-L exactly and all 290 are within one game (mostly COVID 2019-20 games missing
from the play-by-play). Two source errors were found this way and fixed in
`CORRECTIONS` (Westphal / Fitzsimmons, PHX 1995-97). Check: `out/coach_check.csv`.

**Coach effects:** `out/coach_effects.csv`. One ridge regression over all 1M stints:
career-level player O/D + a league experience curve + coach O/D. Players are held at
their career level, so a coach only gets credit when players beat their own norm under
him. Players moving between coaches separate the two. λ_coach = 3000 by game-grouped CV.

**Is it real?** Estimated separately on even and odd seasons (no shared possessions),
coach effects agree: r = 0.42 net (105 coaches with 82+ games in both halves), and
**defense r = 0.52–0.63 vs offense r = 0.18–0.24**. Coaching is measurable, and it
shows up mostly on defense.

**Read with care:**
- Typical SE is ±0.7–1.1 per 100. Only the ends of the table are clearly separated.
- Coach + long-tenured star (Popovich/Duncan, Kerr/Curry, Sloan/Stockton-Malone) is
  hard to separate: the shared credit goes mostly to the player. Their ratings here are
  conservative.
- "Coach" still contains the front office, the system's assistants, and arena/era
  effects. A player's prime coinciding with one coach is partly absorbed by the
  experience curve, not fully.
- Players already active in 1996-97 have unknown true experience (separate curve).

## Layer 4b — Which players, which styles, and is it real?

**Player types** (`data/archetypes.parquet`): k-means on role stats z-scored within
season. The 8 types: Lead creator, Playmaking guard, Scoring wing, Spot-up shooter,
Defensive wing, Interior scorer, Frontcourt finisher, Rim runner.

**1. Coach x player type** (`out/coach_archetype.csv`): for each player type, the points
per 100 a player gains under this coach beyond his own career level. Split-half r =
0.33-0.47, rising with sample size: real. `coach_cells.py` flags cells that are really
one star; 462 of 513 well-sampled cells are broad. Strongest broad findings: Thibodeau
x lead creators +4.2 (8 players), Donovan x lead creators +3.6 (7), Doc Rivers x rim
runners +3.2 (11), Sloan x lead creators +3.4, D'Antoni x defensive wings +2.6.
Phil Jackson x lead creators (+5.6) is 84% Kobe: a Kobe finding, not a type finding.
**Style -> type** (`out/style_effects.csv`): only 5 of 64 trait x type links reach |t| >= 2,
roughly what chance would produce. Hints: three-heavy offenses get less from interior
scorers and frontcourt finishers (about -0.5 per SD). Treat as tentative.

**2. Rotation IQ / Closing IQ** (`out/coach_rotation.csv`). Rotation IQ = wins per 82 games
gained or lost vs league by how each game's minutes were split across the available
players, judged on coach-free talent, within bigs/perimeter. Closing IQ = how much
more talent is on the floor in high-leverage possessions. Both are stable: split-half
r = 0.73 and 0.71. Rotation IQ is not just rotation length (r with depth = -0.03).
Closing leaders: Kerr, Popovich, Budenholzer, Phil Jackson, Sloan. Caveats: tanking
seasons (e.g. OKC 2020-23) are front-office choices that score as poor rotation, and
injuries/DNPs aren't seen.

**3. Development** (`out/coach_development.csv`): NULL result. Same-coach young-player
growth shows no detectable coach differences (true-effect SD 0.06/100, split-half
r = 0.09). Season ratings are shrunk toward prior seasons, which damps year-to-year
change, so this is low power, not proof of absence.

**4. Mid-season firings** (`out/firings.csv`, 76 changes, both coaches 15+ games): with
those team-seasons held out of the fit, coach effects learned elsewhere predict the
before/after change: r = 0.30, slope 0.59 (47 changes where both coaches have other
data; p about 0.04). They do NOT beat a no-effect guess on squared error, so the
direction is right but individual predictions are noisy. Teams that fire their coach
were underplaying their talent by 1.4/100 and still were after (-1.4): **no new-coach
bump** once talent is controlled.

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
