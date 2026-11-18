# The Scott Stat

A cross-sport impact stat that separates **how good a player is** from **the
context he plays in**: coach, teammates, system. Basketball (NBA) first.

## Layers

| Layer | Question | Status |
|---|---|---|
| 1. Impact | How good is he, on offense and defense, independent of team? | **RAPM built and validated (1996-97 → 2024-25)** |
| 2. Wins | How much did he add to the wins that actually happened? | **WAR + leverage-weighted cWAR built (1996-97 → 2024-25)** |
| 3. Breakdown | *Why* — scoring, creation, playmaking, turnovers, defense | **skill prices + per-season breakdown built (1996-97 → 2024-25)** |
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
python src/breakdown.py                # Layer 3: skill prices + every player-season broken down
python src/validate_boxprior.py 2010 2016 2023 [--span 1]   # does a box-score prior help?
python src/shots.py 1996 2024          # every shot with the shooter's 5 on the floor
python src/teammates.py                # teammate shot-making with each star on vs off
python src/teammates_fit.py            # star x teammate-type lineup fit
python src/classic.py                  # CLASSIC tier: 1977-78+ from season box + team results
python src/greats.py                   # all-time comparison on one scale
python src/clutch.py                   # clutch impact + clutch shooting + is it real?
python src/clutch_breakdown.py         # where a clutch rise comes from (own play / teammates / era)
python src/build_stints.py 1996 2024 --playoffs; python src/shots.py 1996 2024 --playoffs
python src/playoffs.py                 # playoff clutch, all clutch, playoff elevation
python src/clutch_defense.py           # the defensive evidence behind clutch D numbers
python src/defense_attribution.py "LeBron James"   # audit: how a lineup's D gets split, luck, coaches
python src/luck.py                     # luck adjustment written into every stint (run before the models)
python src/player_coach.py             # each player under each coach (raw + own-trend removed)
python src/composite.py                # THE SCOTT STAT: careers (Plus + Classic), seasons
python src/hist_games.py               # game logs (Kaggle, 1946-47 on): with/without signal, tested
python src/classic.py                  # (again) Classic now game-informed where adopted
python src/progression.py              # risers/fallers, career arcs, coach arrivals
python src/scenarios.py                # coach style x player type matrix, best coaches per type
python src/build_site_data.py          # site/data.json for the site prototype (site/index.html)
```

**Two tiers.** *Scott Stat Plus* (1996-97 on) is the play-by-play impact model above:
the official number wherever it exists. *Scott Stat Classic* (1977-78 on) estimates the
same thing from season box scores + team results, calibrated on Plus, so players from
any era can be compared on one scale. Historical tables: Basketball-Reference via
`sumitrodatta/bball-reference-datasets` (data/hist/, not committed).

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

## Layer 3 — Why

`out/breakdown_<YYYY>.csv`, one row per player-season: each skill's contribution to the
season rating in points per 100, plus what's left over ("not in the box score").

**Skill prices** (`out/breakdown_coefs.json`): 3-season RAPM regressed on the same
seasons' box-score rates, relative to league, all 27 rolling windows. Per 100
possessions: +1 point above average efficiency on his own attempts = +0.54;
assist +0.33; offensive rebound +0.34; turnover -0.75; steal +0.96; block +0.54;
defensive rebound +0.13; extra 3PA +0.16 (spacing).

**How much the box score explains** (5-fold CV, grouped by player): **61% of offensive
impact, 25% of defensive impact.** Most defense is invisible in the box score.

**Is "not in the box score" real?** In non-overlapping 3-season windows (no shared data)
it repeats: r = 0.20 on offense, 0.36 on defense. Part noise, part real skill
(spacing, screens, positioning, off-ball defense), especially on defense.

**Shai vs Dame, broken down** (points per 100):

| | Lillard 22-23 | Shai 22-23 | Shai 24-25 |
|---|---|---|---|
| Scoring efficiency | +2.3 | +1.7 | +2.4 |
| Shot volume + self-creation | +2.3 | +2.5 | +3.0 |
| Shot profile (3s) | +1.2 | -0.6 | 0.0 |
| Playmaking | +1.5 | +0.7 | +1.1 |
| Ball security | -1.2 | -0.7 | -0.5 |
| Offense not in box | -0.5 | +1.0 | +0.8 |
| Steals + blocks + boards | -0.7 | +0.7 | +0.8 |
| **Defense not in box** | -0.2 | **-1.3** | **+2.4** |
| **Net** | **+4.4** | **+3.5** | **+9.8** |

Shai's +6.3 jump: about +2.4 on offense (efficiency, volume, fewer turnovers) and
+3.8 on defense, almost all of it **not in the box score**. That is off-ball and
positional defense, the part steals and blocks don't show. Caveat: it coincides with
OKC's elite defensive group and Daigneault (+2.1 D in Layer 4), so some of it may be
scheme the player model can't fully separate.

**Box-score prior as an accuracy upgrade** (`validate_boxprior.py`): shrinking ratings
toward the box estimate instead of 0. With 3 training seasons it changes nothing (RMSE
13.11 vs 13.11). With 1 training season it helps a little (13.04 vs 13.12; winners
63.7% vs 63.1%). Worth using where data is thin, not a big upgrade.

## Teammates — LeBron vs Jordan (and 19 other stars)

A star's impact rating already contains any lift he gives teammates: in lineup data,
"makes others better" and "is good" can't be told apart. So this measures the lift from
the teammates' side, with the shot-level data (`data/shots/`, 7.3M attempts, each with
the shooter's five on the floor).

**Teammate shot-making** (`out/teammates_stars.csv`): each teammate's true shooting with
the star on vs off the floor, same season. Every team's #1 player gets some lift for free
(he plays with starters, against starters), so each star is measured against that
season's league baseline, the same split for every team's #1 player. In TS percentage
points above baseline:

| | Seasons | Teammate TS lift | ± SE | Teammates take more 3s | More of their makes assisted |
|---|---|---|---|---|---|
| LeBron, career | 22 | **+1.4** | 0.4 | +3.6 pts | +6.2 pts |
| LeBron, Cavs 2014-18 | 4 | **+3.4** (best star-era measured) | 0.4 | +5.3 | +7.6 |
| LeBron, Heat 2010-14 | 4 | +1.1 | 1.1 | +2.8 | +8.5 |
| Jordan, all available | 4 | +0.8 | 0.8 | +2.0 | 0.0 |
| Jordan, Bulls 1996-98 | 2 | +0.5 | 1.8 | +2.5 | -2.0 |
| Curry | 15 | +2.9 | 0.4 | -1.4 | +3.2 |
| Jokić | 10 | +2.1 | 0.7 | -0.2 | +7.9 |

- LeBron makes teammates better shooters, consistently, for 22 years: 77th percentile
  of #1 players in a typical season, and 95th in 2014-18. He does it by creating: his
  teammates' shots become more threes and more assisted. Biggest beneficiaries:
  frontcourt finishers (+3.3) and spot-up shooters (+2.7).
- Jordan's teammates also shot better with him (+0.8), but with fewer of their makes
  assisted (-2.0 in Chicago): the triangle spread creation around rather than running it
  through him.
- **LeBron vs Jordan cannot be called from this data.** Jordan's 1996-98 estimate is
  +0.5 ± 1.8 (two seasons, both at age 33-35). Nothing from 1984-96 exists here. The
  honest verdict: LeBron's lift is measured and real; Jordan's is unmeasured, not
  smaller.
- Curry's lift (+2.9 over 15 seasons) is the strongest sustained number. It shows up as
  more rim attempts for teammates (+5.1): defenders chasing him leave the paint open.

**Star x teammate-type fit** (`out/teammates_fit.csv`): lineup residuals regressed on the
mix of teammate types around each star. Nothing clears noise: all within ~1.5 SE of 0.
LeBron's best (playmaking guards +0.8, defensive wings +0.8, ±0.5) is suggestive only.

## Classic tier — every season since 1977-78

`out/classic_<YYYY>.csv` (start-year labels). Two steps, fit where both tiers exist:
1. **Box estimate**: Plus O/D regressed on per-100 box rates vs league (efficiency, volume,
   assists, turnovers, boards, steals, blocks, fouls, 3PA), position and role.
2. **Team adjustment**: the gap between the team's actual offensive/defensive rating and the
   sum of its players' box estimates is shared out per possession played. This is how
   defense the box can't see reaches the players on the floor. Full share (alpha = 1) won
   on held-out seasons.

**Validation** (train 1997-2013, test 2014-2025 vs Plus, players 1000+ poss, n = 4,298):

| | r (net) | r (O) | r (D) |
|---|---|---|---|
| Box only | 0.61 | 0.69 | 0.48 |
| **Classic (box + team)** | **0.64** | **0.71** | **0.54** |
| Basketball-Reference BPM | 0.63 | 0.67 | 0.54 |

Error SD 1.8 per 100 per season. Overall calibration is right (Plus = 1.03 x Classic).
Small point-guard lean (+0.26). Like every box model, Classic **compresses the very top**:
the 50 best Plus seasons average +8.1 in Plus but +5.0 in Classic. Players whose impact is
mostly invisible to the box score get the biggest compression: Garnett -2.7, Dirk -2.5,
Duncan -1.9, Shaq -1.8, LeBron -1.5 (Classic minus Plus on the same seasons).

**All-time, one scale** (`out/greats.csv`, Classic, regular season, per 100 poss):

| | Best season | Best 3 | Best 5 | Career /100 | Career WAR |
|---|---|---|---|---|---|
| Jokić | 7.3 | 7.0 | 6.8 | 5.4 | 104 |
| Stockton | 7.0 | 6.8 | 6.8 | 5.4 | 221 |
| **Jordan** | **7.4** (90-91) | **6.7** | **6.70** (87-92) | **5.2** | **175** |
| **LeBron** | **7.5** (08-09) | **6.7** | **6.65** (08-13) | **4.6** | **242** |
| Curry | 8.1 | 6.8 | 6.3 | 4.2 | 134 |
| Magic | 6.4 | 6.2 | 5.8 | 5.1 | 156 |
| Bird | 5.9 | 5.8 | 5.5 | 4.6 | 150 |

Stockton's rank isn't a box-score artifact: Plus rates his 1996-2003 seasons even higher
than Classic does (5.97 vs 4.65).

## Clutch (1996-97 on)

NBA definition: last 5 minutes of the 4th quarter or overtime, within 5 points.
`out/clutch_players.csv`. 66,301 clutch stints.

- **Clutch delta**: how much better or worse a player's lineups did in clutch possessions
  than the ten players' *normal* ratings predict, fit by ridge so it is adjusted for
  teammates and opponents. **Clutch rating** = normal rating + clutch delta.
- **Clutch shooting**: true shooting in clutch vs his normal, relative to the league's own
  clutch drop (.542 normal -> .537 clutch).

**Is clutch real?** Re-estimated on even vs odd seasons (no shared possessions):
clutch *impact* delta r = 0.01-0.10 (149-620 players); clutch *shooting* r = 0.17-0.22.
League-wide, rising in the clutch is mostly noise, and the CV curve is nearly flat. A few
players rise in BOTH independent halves: LeBron (+1.2 / +1.6), Lillard (+1.2 / +1.0),
Dirk (+1.5 / +0.7), Kobe (+0.6 / +0.5). Jordan's 1996-2003 sample is small (-0.2 / +0.3).

**Who is best in the clutch** (normal + delta, 1500+ clutch possessions): LeBron +8.3,
Jokić +7.2, Dirk +7.0, Embiid +6.7, Garnett +6.2, Chris Paul / Shaq / Stockton +6.1,
Lillard +5.8, Curry +5.7. Mostly the best players overall. Being great is what makes
someone great in the clutch.

**LeBron's +2.3 clutch rise, broken down** (`out/clutch_breakdown.json`):
- Offense +1.2, defense +1.0.
- He takes over: 31% -> 39% of his lineup's shots, 36 -> 50 points per 100 possessions,
  with efficiency held (TS .590 -> .581; the league drops about .005).
- Not passing more (assists 10.2 -> 9.2 per 100), no extra turnovers. Steals + blocks up
  (3.0 -> 3.5).
- Teammates don't shoot better with him in clutch (.560 -> .558): the rise is him, not
  setting others up.
- By era (raw lineup over-performance vs normal ratings, per 100 clutch possessions):
  Cavs 2003-10 +11.9, Heat +6.0, Cavs 2014-18 +8.8, **Lakers -1.6** (clutch defense -7.7).
  The rise is a Cleveland/Miami phenomenon.
- Is he #1? P(biggest rise) = 36% (Lillard 11%, Marion 9%, Dirk 9%). P(best clutch rating)
  = 85%. The most likely #1, not a certain one.
- Contrast: Kobe also takes over (31% -> 42% of shots) but efficiency drops (.548 -> .538).
  Curry's teammates shoot worse in his clutch minutes (.582 -> .546).

## Playoffs (1996-97 on)

Every playoff game is loaded (`stints_po_*`, `shots_po_*`; the same lineup checks pass).
`out/playoffs_players.csv`, three runs against each player's normal regular-season rating:

| Run | Possessions | Split-half r | Top of the list |
|---|---|---|---|
| Playoff clutch | 4,766 stints | too few players to test | LeBron +2.1 ± 1.0 (level 8.2) |
| All clutch (regular season + playoffs) | 71,067 stints | 0.06-0.13 | LeBron +2.6 ± 0.7 (level 8.7), Marion, Dirk, Chandler, Lillard |
| **Playoff elevation** (every playoff possession) | 73,348 stints | **0.14-0.18** | **LeBron +2.1 ± 0.7** (level 8.1), P.J. Tucker, Kyrie, Draymond, Ray Allen |

Playoff elevation is a little more repeatable than clutch. Jordan's numbers here are
1996-2003 only (age 33-39, including Washington) and aren't representative of his career.

**Clutch defense, shown** (`out/clutch_defense.json`, regular + playoffs, him on the floor):

| | LeBron | Dirk | Kobe | Duncan |
|---|---|---|---|---|
| Opp. points/100, actual | 104.4 | 105.5 | 107.6 | 104.0 |
| Predicted from everyone's normal ratings | 108.0 | 110.2 | 109.7 | 104.5 |
| Lineup over-performance | **+3.6** | +4.7 | +2.1 | +0.5 |
| Credited to him (ridge) | **+1.5** | +1.2 | +0.4 | +0.1 |
| Opp. true shooting, clutch / normal | **.523 / .540** | .516 / .532 | .526 / .524 | .511 / .496 |
| His blocks per 100, clutch / normal | 1.6 / 1.0 | 1.6 / 1.3 | 1.0 / 0.7 | 3.0 / 3.4 |
| His rebounds per 100, clutch / normal | 12.7 / 10.4 | 14.5 / 11.7 | 8.7 / 7.4 | 17.9 / 17.0 |

League clutch opponent TS is .537 (normal .542). LeBron's all-game defensive rating is
+1.25 per 100 over his career (peaks +2.8 in 2010-11, +3.6 in 2019-20): above average, not
elite. Duncan +4.0 and Garnett +3.8 are elite.

**Audit: why LeBron gets 1.5 of his lineups' 3.6** (`out/defense_attribution_james.json`):
- Split of the 3.6: **LeBron +1.53 (±0.48)**, his four teammates +0.23 combined, and
  **1.88 credited to no one** (ridge treats it as noise). An even split would be 0.73 each.
- Why him: the over-performance follows him while teammates change. Their clutch D,
  with LeBron vs without: Wade +11.7 / +1.6, Bosh +9.4 / +0.2, J.R. Smith +12.1 / -2.4,
  Love +8.1 / -11.0, Gooden +10.2 / -4.7. Exceptions: Ilgauskas +4.1 / +6.8, Chalmers +3.8 / +4.4.
  Blocks and rebounds play no part in the calculation; they are supporting evidence only.
- Luck: opponents' FT% is normal (.769 vs .766 league clutch), but they shot 30.4% from
  three vs 31.5%, worth **about 0.9 of the 3.6**. Two-point % allowed is also lower (.449 vs .457).
- By coach (raw lineup clutch D over-performance): Silas +8.6, Brown +6.5, Spoelstra +5.7,
  Lue +5.6, Blatt +4.6; then every Lakers coach negative (Walton -4.7, Vogel -8.3, Ham -9.5,
  Redick -5.8). It's Cleveland/Miami LeBron, age 19-33.

## The Scott Stat (composite)

`out/scott_careers_plus.csv`, `out/scott_careers_classic.csv`, `out/scott_seasons.csv`.

**Scott Stat = all-game impact + 12% x clutch rise + playoff share x playoff rise.**
Clutch possessions are 4.0% of all possessions but 12.0% of the swings in win
probability, so clutch counts 12%: as much as it decides games, measured, not chosen.

Every impact number splits into **own production** (what his box score explains) and
**lift** (teammates playing better and the team defending better while he's on the floor).
Kidd +1.7 lift vs Iverson 0.0 and Carmelo -0.3. Marcus Smart: production -0.2, lift +1.4.

**Luck adjustment** (`luck.py`): every 3PA and FTA is replaced by the shooter's expected %
(his season %, shrunk to league). Out-of-sample error vs **real** final margins improves
13.11 -> 12.99 (winner accuracy unchanged at about 64%). Standard errors are now on every
player-season rating (`se_O`, `se_D`, `se_net` in `war_*.csv`).

**Player x coach** (`out/player_coach.csv`, 3,944 pairs with 2,000+ shared possessions):
how each player did under each coach, beyond his own career level, the league aging
curve and the coach's general effect. The *trend-adjusted* version also removes the
player's own career arc. The CV-chosen penalty is the largest tried, so pair effects are
heavily shrunk: player-specific coach fit is a small effect next to the coach's general
effect. Curry: Mark Jackson +0.0, Kerr +0.8 (+0.9 trend-adjusted). J.R. Smith: Karl +0.9,
Woodson +0.6, Lue -0.6.

**Results** (Plus, 20k+ possessions): career Scott Stat: Jokić 6.26, LeBron 5.91,
Stockton 5.74 (1996-2003 only), Duncan 5.53, Garnett 5.38. Peak (best 5 + clutch +
playoffs): LeBron 8.11, Duncan 7.57, Curry 7.21, Jokić 7.18, Garnett 7.17. Classic peak
(1977-78 on, era-matched): **Jordan 6.27**, Jokić 6.16, **LeBron 6.10**, Paul 5.93, Curry 5.79,
Robinson 5.65, Stockton 5.26, Magic 5.05, Hakeem 4.99, Bird 4.96.

**Classic fix (Stockton).** The first Classic ranked Stockton's 1987-92 peak #1. The cause
was two kinds of extrapolation, not a judgment call:
1. His assist rate (+12.9 per 100 over league) is above every one of the 15,006 training
   player-seasons (max +12.0, Nash). The linear model credited his assists +4.9 per 100
   (Jordan's +1.1). Fix: every box rate is capped at the training-era range.
2. Skill prices drift by era (assist 0.33 -> 0.42, efficiency 0.49 -> 0.43, defensive
   rebound 0.28 -> 0.09 from 1997-2004 to 2019-25), and the all-era model priced the 1980s
   at mostly three-point-era values. Fix: era-matched prices (exponential weight on the
   nearest play-by-play seasons, half-width 5 seasons chosen by backcasting 1996-2002
   from 2002-03 onward).
Holdout (2014-2025) r improves 0.653 -> 0.660 (BPM 0.644); backcast r 0.673 -> 0.676.
`src/classic_variants.py` keeps the comparison of capping variants. Plus-tier players whose
careers began before 1996-97 are labeled "late career" on the site.

**Site prototype** (`site/`): leaderboard (Plus / Classic), player pages with
production / lift / clutch / playoffs / eras / coaches, compare, coaches, GM hidden-value
finder, method, launch plan.

## All eras, progression, scenarios

**Classic back to 1951-52** (first season with individual minutes). Each era is rated with
only the stats it recorded, by its own model trained on the play-by-play seasons:

| Band | Stats | Holdout r vs Plus |
|---|---|---|
| 1977-78 on | full box score | 0.66 |
| 1973-74 to 1976-77 | no turnovers | 0.65 |
| 1951-52 to 1972-73 | no steals, blocks, turnovers; total rebounds only | 0.60 |

Before 1951-52 there are no individual minutes, so no rating. Also covers the completed
2025-26 season (Basketball-Reference tables). Classic peak (best 5): Jokić 6.41,
Jordan 6.27, LeBron 6.10, Paul, Curry, Robinson, Kareem 5.42, Stockton, Magic, Wilt 5.03.
Bill Russell's peak is 3.60: with no blocks recorded, Boston's team defense is shared
evenly per minute, which undercounts him. Game-by-game with/without data would fix it.

**Progression** (`out/progression_changes.csv`, `out/progression_arcs.csv`): one-year changes
listed only when larger than 2x their own SE. Biggest climbs to a peak: Nash, Durant, Shai,
Hayward, Giannis. **Coach arrivals** (`out/coach_arrivals.csv`, 3,163 moves): a player's
change the year he arrives at a new coach, vs expectation. Not reliable: even/odd-season
halves agree at r = 0.02. The lineup-based coach effects are the measure to use.

**Scenarios** (`out/scenario_matrix.csv`, `out/scenario_best_coaches.csv`): coach groups
(defense-first, offense-first, pace, three-point heavy, inside-out, ball movement,
isolation, rotation depth) x eight player types. 5 of 80 cells clear |z| >= 2 (about 4 by
chance). Leads: rim runners under defense-first coaches +0.5 (z 2.5), scoring wings in
three-point-heavy systems +0.3, interior scorers under isolation coaches -0.6, scoring
wings in deep rotations -0.4. Per-type coach rankings (broad cells only) are the reliable
part: e.g. rim runners under Doc Rivers +3.5 across 11 players.

## Game logs: the with/without signal for every era

Source: every NBA box score since 1946-47 (Kaggle `eoinamoore/historical-nba-data-and-player-box-scores`,
downloaded with the owner's Kaggle token into `data/kaggle_dl/`, not committed). Pre-1975
minutes are mostly blank; who played is recorded, and blanks get the player's season
minutes per game. `hist_games.py` explains every game's score (each team's points per 100)
by who played, minutes-weighted, adjusted for opponents and home court, per three-season
window, as a ridge regression **toward each player's Classic rating**. Game results move a
player off his box-score estimate only as far as the with/without evidence supports.

**Tested honestly:** Classic fit on 1996-97..2012-13 only, then 2013-14..2024-25 (3,912
player-seasons) rated with and without game logs vs Plus: r 0.658 -> **0.674**, RMSE
1.62 -> 1.59 (penalty 300, chosen by RMSE). Adopted. About 92% of Classic player-seasons
are game-informed.

**Bill Russell, raw:** Boston +7.0 per game in his 963 games, +1.5 in the 52 he missed;
+0.7 the season before he arrived, +5.8 his first full season; +6.0 -> -1.9 after he
retired (with Sam Jones). Against a backup for ~44 minutes, that's about +3.5 per 100
over average, close to Classic's box + team estimate. Game logs confirm him; they don't
lift him further. Peak 3.68 (1960-65).

**New all-time peaks (best 5):** Jokić 6.50, LeBron 6.37, Curry 6.10, Jordan 6.06,
Paul 5.98, Robinson 5.97, Kareem 5.40, Stockton 5.18, Magic 5.16. The top four sit within
0.45; a five-season peak is good to about ±0.7, so they're a statistical tie.
Jordan's 1987-92 seasons moved -0.2 on average, along with Pippen (-0.1) and Grant (-0.2):
game by game, those Bulls were slightly less dominant than season totals suggest. LeBron
rose +0.3, most in 2009-11, windows that include Cleveland's 61 -> 19 win collapse after he left.
Jokić's 6.41 -> 6.50 and earlier jump past Jordan come from adding his 2025-26 season (+6.9,
Classic only so far; his best five moved from 2020-25 to 2021-26).

## Era audit (`src/era_audit.py`, `out/era_audit.json`)

**Pace** is removed by construction: every stat is per 100 possessions vs that season's league.
**Threes**: 3PA enters relative to the league that season; Jordan's three-point term is ~0
(Curry's +1.25). Era-matched pricing *helps* Jordan: production 6.02 vs 5.87 under one yardstick.

| Peak 5 | Production | + team results | + game logs |
|---|---|---|---|
| Jokić | 6.18 | 6.41 | 6.50 |
| Jordan | **6.02** | 6.27 | 6.06 |
| LeBron | 5.37 | 6.10 | 6.37 |
| Curry | 5.29 | 5.79 | 6.09 |
| Wilt | 4.80 | 4.88 | 4.35 |
| Russell | 2.50 | 3.53 | 3.68 |

LeBron passes Jordan only on the team-results steps (team share +0.73 vs +0.25; game logs
+0.27 vs -0.21). **Fewer recorded stats** (1951-73 set) lower top players' estimates by about
0.4 (top-50 2014-25 seasons: 3.71 lite vs 4.10 full) and widen their error. Russell's
scoring efficiency was below his era's average; Wilt's game logs move him -0.5.

**Three Classic views** (`scott_careers_classic.csv`, site leaderboard): impact peak (`best5`),
production peak (`production_peak5`, own box only), era dominance (`dominance_peak5`: SDs above
that season's league, plus `seasons_ranked_1` / `seasons_top3`). Most #1 seasons:
Kareem 11, Wilt 8, Jokić 6, Jordan 6, Robinson 6, LeBron 5.

## Checked against Basketball-Reference's own advanced stats (through 2025-26)

| B-Ref stat | #1 | #2 | #3 |
|---|---|---|---|
| Career BPM | Jokić 10.6 | Jordan 9.3 | LeBron 8.4 |
| Best-5 BPM | Jokić 13.5 | Jordan 11.8 | LeBron 11.1 |
| Career WS/48 | Jokić .265 | Jordan .253 | Wilt .248 |
| Career PER | Jokić 28.8 | Jordan 28.0 | Shaq 27.0 |
| Career VORP (total) | LeBron 159.5 | Jordan 113.7 | Paul 98.8 |

Scott Stat **production peak** reproduces the rate-stat order (Jokić 6.18, Jordan 6.02,
LeBron 5.37). Jordan trails LeBron and Curry only in the **impact** view, entirely through
team results (team share and game logs). B-Ref's total-value stat (VORP) also has LeBron #1.

**Classic playoffs: tested, not adopted** (`src/classic_playoffs.py`). Playoff box scores
since 1951-52 (Kaggle), rates relative to that season's playoff league, the same era prices,
team playoff margin vs opponents' regular-season strength, rise centered on the typical
playoff player that season. Career playoff rise agrees with Plus's play-by-play playoff
elevation at only r = 0.17 (201 players), too weak to move rankings. What it shows:
Jordan's playoffs were about as dominant as his regular seasons (rise -0.6, within noise),
LeBron +0.2, Russell +0.4, Hakeem +0.4, Curry -1.3, Jokić -1.5.

## Championships (`src/championships.py`)

Rings depend on teammates and luck, so they aren't counted raw. Two measures, every season
since 1951-52 (75 champions, read from each season's last playoff game; playoff game ids
give the season, so the Oct-2020 bubble Finals land in 2019-20):
- **Title odds added**: conditional logit of the champion on regular-season SRS (b = 0.60);
  a player's value = his team's title odds minus its odds with a replacement player in his
  minutes. Regular-season strength only, so it misses teams that won from modest regular
  seasons (Hakeem's Rockets ~0). Shown, not used in Legacy.
- **Title shares**: each real title (1.0) split among the champion's players by their
  contribution in those playoffs ((rating - replacement) x playoff possessions). League-size
  version: a title among N teams counts N/30. Leaders (adjusted): **Jordan 1.70**, Russell 1.33,
  Pippen 1.31, Duncan 1.28, Magic 1.19, Shaq 1.16, Kobe 1.04, LeBron 1.04, Kareem 0.93.

**Legacy view (site)**: peak (Classic best 5) + career wins added + title shares (adjusted),
each standardized over players with 8+ seasons, weighted by the reader. After the audit fixes
(LeBron's 2020 title credited): equal weights are a dead heat (6.26 each); championships x2 or
peak + championships -> Jordan; peak x2, career-heavy, or no championships -> LeBron.

## Audit (adversarial pass, two loops)

Bugs found and fixed:
1. **Pace double-counted in `hist_games.py`**: the target was per 100 possessions and each player's
   share was also multiplied by pace/100. Inflated stars in slow eras (~8% at pace 92), deflated
   them in fast ones (~20% at pace 125). The 2013-25 holdout couldn't see it (pace ~100).
   LeBron peak 6.37 -> 6.23, Curry 6.09 -> 6.03, Jordan 6.06 -> 6.04, Wilt 4.35 -> 4.63, Russell 3.68 -> 3.92.
2. **Seasons from dates** misfiled the 2020 bubble (regular season Aug 2020, playoffs Aug-Oct).
   Now `season_from_id` (game id TYY#####). LeBron title shares 1.04 -> 1.34.
3. **Kaggle `playerteamId` blank for all of 2021-22 and half of 2000-01**; rebuilt from
   Games.csv + the home flag (`fill_team_ids`). Holdout r 0.674 -> 0.677; 2022 title now credited.
4. **Clutch on luck-adjusted points** replaced made clutch threes/FTs with expectations and picked a
   looser penalty. Clutch now uses raw points (LeBron +3.4 -> +2.8); all-game and playoff ratings
   stay luck-adjusted.
Invariants now checked: 75/75 titles sum to 1.0; league possession-weighted Classic mean within
±0.08 each season; league WAR/games 0.60-0.64; team sum of player ratings vs team net rating r 0.97-0.99.

**Individual vs team credit** is now reported separately everywhere in Classic
(`peak_individual`, `peak_team_credit`): Jordan 6.02 + 0.02, LeBron 5.37 + 0.86, Curry 5.30 + 0.74,
Jokić 6.18 + 0.33, Russell 2.50 + 1.43, Duncan 3.40 + 1.18.

**Aging** (`src/aging.py`): delta method on Classic, 14,253 season pairs. Peak 25-26 for every
position and type; prime ~24-28; spot-up shooters hold to ~30. **Schemes** (`src/schemes.py`,
111 coaches): defense effect vs opponent rim attempts allowed -0.39/SD (t -3.7), substitution churn
+0.43 (t 4.7); offense vs rotation depth -0.31 (t -3.2). Style explains ~33% of coach defense
effects, ~18% of offense. Associations, not experiments. Site: Careers and Coach lab tabs.

## Six skeptic objections, each tested

1. **Even split of team credit** (`src/team_credit_test.py`). Stars were not shortchanged (holdout bias
   +0.09 for box net >= 4). Scaling the share with minutes per game vs team average helped
   (r 0.660 -> 0.666, RMSE 1.600 -> 1.592) and is adopted (`fit_credit` in classic.py:
   aO 0.72, bO 0.47, aD 0.73, bD 0.31). With game logs on top: holdout r 0.685.
2. **Top-end compression**: mostly selection on a noisy target. Binned by Classic rating, seasons at
   +4 and above read ~0.4 low vs Plus (top 50 by Classic: 5.83 vs 6.21), not 2.
3. **Playoffs** (`classic_playoffs.py`): small playoff samples now shrink toward the player's own
   regular-season rates (shrinking toward league had dragged every star down and faked
   repeatability). Playoff rise even/odd r 0.21-0.42; r 0.25 with Plus elevation. Reported as
   `best5_with_playoffs` and `playoff_rise`, not folded into the headline. Hakeem +1.1, LeBron +0.4,
   Jordan -0.2, Curry -0.5, Bird -0.8.
4. **Availability**: `peak_wins_per_season` (best five consecutive seasons of WAR / 5).
5. **Contract value** (`src/value.py`, Kaggle salaries 2009-10..2024-25): price of a win per season
   ($6.5M in 2024-25), worth, surplus. Salary vs PPG r 0.61, vs wins added r 0.42.
6. **Legacy weights**: rank range across 27 weightings on the site. LeBron 1-2, Jordan 1-3.

Current Classic peaks: Jokić 6.61 (6.18 + 0.43), LeBron 6.60 (5.37 + 1.23), Curry 6.21 (5.29 + 0.92),
Jordan 6.20 (5.79 + 0.40, 1988-93), Robinson 6.10, Paul 5.99, Kareem 5.54, Russell 4.45 (2.50 + 1.96).

## Recency-bias checks (`src/recency_test.py`, `out/recency_test.json`)

- **Like-for-like team credit** (top-10 individual players on +4 to +9 teams): 1985-95 stars get more
  individual (3.99 vs ~3.5) and less team credit (0.48 vs ~0.70) than later stars, and their total
  impact (4.47) is as high or higher than 1996-2014 (4.2). The split differs by era; the total is not tilted.
- **Scale**: 1985-95 holds 31% of the 100 best seasons on 18% of player-seasons, the most of any era
  (2015-25: 27% on 19%). Pre-1977 eras are under-represented (fewer recorded stats).
- **Jordan-type players**: for star seasons with play-by-play, Plus minus Classic is +0.12 for shooting
  guards and +0.12 across high-scoring guards/wings; Jordan's own 1996-98 seasons: Classic 4.6, Plus 4.7.
- **Certainty**: a five-season peak has SE ~0.69, so the top four (0.41 apart) can't be ordered.

## The study: situation, coaches, style fit (`src/fit.py`, `src/situations.py`)

Everything measured against the player's own norm (aging-adjusted, other seasons), so talent is differenced out.

- **Coach split** (`out/coach_split.csv`): players' own production vs lift under each coach, empirical-Bayes
  shrunk. Coaches differ about twice as much on production (true SD 0.14) as on lift (0.06); even/odd
  reliability r ~0.3. Production: Kerr +0.40 raw, Karl +0.31, D'Antoni +0.22, Budenholzer +0.21.
  Lift: Skiles +0.38, Fratello +0.39, Van Gundy +0.37, Stotts +0.28, Thibodeau +0.25.
- **Style fit** (`out/fit_style_effects.csv`): player type x coach style. With same-season style the result
  looked strong (16 of 64 cells) but that double-counts the players' own results. With style from the
  coach's OTHER seasons: 7 of 64 cells (about 3 by chance), out-of-time r 0.02 (fit 1996-2011, test
  2012-24), top-vs-bottom fifth 0.06. Broad style fit is not detectable. Movers: change in fit does
  not predict change in performance (r 0.01).
- **Carryover** (`situations.py`, season-independent Classic ratings, 8,919 pairs since 1977): 61% of an
  up or down year persists (r 0.55). Down years of -2.0 become -1.2; up years +2.0 become +1.3.
  After a down year, stayers -1.22 next season vs movers -1.14: no change-of-scenery effect.
  (Plus season ratings are shrunk toward prior seasons and would overstate carryover: 0.75.)
- **Situation board** (`out/situation_board.csv`): each player's norm at his age, this season's gap, and
  expected next season = norm + 0.61 x gap, one year along the aging curve. Latest season 2025-26.
- Site: Study tab (tier chart with error bars, findings, coach scatter, honest style heatmap, situation board).

## Legacy fix: second options were over-ranked (Pippen 7th, Ginobili 19th, Parish)

Checked first whether team credit over-rates No. 2s on great teams: it does not (Classic minus Plus
by within-team rank on +4 teams: best -0.13, 2nd -0.08, 3rd +0.06, 4th-5th +0.03). Two real causes:
1. **Legacy used z-scores.** Most players have zero title shares, so the championship ingredient had a
   tiny SD and counted about 3x the others at "equal" weights. Now each ingredient is scored as a
   percentage of the all-time top-10 level.
2. **Titles were split by value above replacement**, giving a No. 2 about 80% of the star's share. Now split
   by each player's effect on the team's title odds, exp(b x points per game added in those playoffs) - 1,
   from the fitted title model (b = 0.60). 1995-96 Bulls: Jordan 0.60, Pippen 0.25, Rodman 0.05.
Title shares (league-size adjusted): Jordan 3.22, Russell 2.74, LeBron 2.59, Duncan 2.25, Magic 2.08,
Shaq 2.03, Curry 1.88, Pippen 1.45, Bird 1.40, Hakeem 1.38, Kareem 1.18, Kobe 1.08.
Legacy at equal weights: LeBron 119, Jordan 115, Russell 101, Duncan 93, Kareem 90, Curry 88, Magic 88,
Shaq 86, Hakeem 77, Wilt 76, Bird 75, Pippen 71 (12th, range 11-19), ... Wade 25th, Ginobili 26th.
Jordan is first when championships are weighted up or career length dropped.

## Peak-list audit: Doc Rivers above Julius Erving, Kirilenko 31st, Kobe below Pippen

Four fixes, each tested before adoption:
1. **Usage interaction (Classic box model).** Out of sample, high-usage stars were under-rated by the box
   model (+0.2 to +0.6 points per 100 vs play-by-play) relative to low-usage stars. Added two features:
   efficiency x shot volume (`pa_x_tsa`) and efficiency x minutes (`pa_x_mpg`). Forward test r 0.667 -> 0.679,
   RMSE 1.592 -> 1.573; backcast RMSE 1.586 -> 1.579; the high- vs low-usage star gap closed.
2. **ABA seasons (1967-68 to 1975-76) were missing**, so Julius Erving's three ABA MVP seasons (ages 23-25)
   didn't count. ABA seasons are rated against the ABA, then moved down to NBA level by a translation from
   players who switched leagues in consecutive seasons, aging-adjusted: late ABA (1973-74 on, 78 movers,
   mostly the 1976 merger) 0.27 per 100 per player (SE 0.15); early ABA 0.87 (SE 0.23). ABA titles count,
   split by the same title-odds rule on regular-season minutes, at league-size weight. Game-log ratings
   (NBA-only priors) are not applied to ABA seasons. Erving: 1973-76 ABA seasons +5.2, +5.2, +5.4.
3. **Play-by-play blended in from 1996-97 on.** Classic is a box-score estimate; where play-by-play exists it
   is now blended in by precision (about 55-60% weight on a full season's play-by-play), then
   quantile-matched to Classic's own distribution of ratings within each season and playing-time band, so
   the top tail is unchanged (season top-5 average 5.06 before, 5.07 after). Play-by-play decides who ranks
   where inside a season; it cannot stretch the modern scale. Fixes e.g. 2008-09/2009-10 Lakers, where the
   box model had Gasol above Kobe and play-by-play has Kobe clearly ahead.
4. **Mid-season trades to a champion were dropped from title shares** (Drexler 1995, Rasheed Wallace 2004,
   Marc Gasol 2019 and others). Each champion stint now gets its own row.

Peak ranks now: Erving 26th, Doc Rivers 39th, Drexler 41st, Wade 42nd, Kidd 43rd, Ewing 45th,
Kirilenko 46th, Kobe 54th, Marion 68th. Legacy (equal weights): Pippen 15th, Erving 18th, Kobe 21st.
Doc Rivers' peak is passing (+2.45 per 100 from assist rate 7.4 per 100 above league) and steals (+1.42);
scoring adds nothing (league-average efficiency, below-average volume). Drexler's high volume came at
near-league-average efficiency (+0.9 points per 100 shots above league), so it adds +1.27. The two are
statistically tied (peak SE about 0.7). Peak is a per-possession rate; Drexler's minutes and longevity show
up in career wins (119 vs 69) and Legacy (33rd vs 77th).

## Missed-games test of skill prices (`src/absence_test.py`, `out/absence_test.json`)

Question: does the box model over-price assists and steals relative to scoring? Independent evidence: the
team's opponent- and home-adjusted margin per 100 with vs without each player, 1977-78 on (7,747
player-seasons, 159,839 missed games). Observed with-minus-without regressed on each skill's contribution x
minutes share; a correctly priced skill has multiplier 1. Starters only (28+ mpg, absences mostly injury,
1,705 player-seasons): scoring 1.08 (SE 0.14), passing 0.96 (0.20), steals 1.00 (0.29), blocks 1.45 (0.37),
turnovers 1.09 (0.27), rebounding 0.48 (0.27). No evidence assists/steals are over-priced vs scoring.
Usage quintiles (starters): the top quintile (35% usage) hurts its team 2.85 per 100 when out vs 2.48
predicted; the other quintiles match within 0.3. Player checks: Iverson's teams +2.07 predicted, -0.23
observed (77 missed games); Doc Rivers +2.95 predicted, +3.85 observed; Kobe +2.15 / +2.74; Carmelo
+2.08 / +2.39. Rotation players' absences are often coaching decisions, so the all-player version is biased.

## Known limits

- Play-by-play starts 1996-97: most of Jordan's career is box-score only.
  Cross-era layers need a box-score model calibrated against RAPM on modern seasons.
- Pure RAPM is noisy for low-minute players. A box-score prior (EPM / LEBRON style) was
  tested in Layer 3: small gain with 1 season of data, none with 3.
- WAR scale runs below B-Ref's (≈ 2.7 × VORP): RAPM-based ratings are shrunk. It is
  internally consistent; compare within this system.
- The win-probability model ignores pregame team strength, so early-game leverage for
  heavy favourites is slightly overstated.
- Possessions are counted from possession-ending events; possessions still open
  at the end of a period are not counted (≈1% low).
