"""Scenarios — coach style x player type, for front offices.

Coach groups (a coach can be in several), from what his teams DO (style
fingerprints, coach_styles.py) and what he PRODUCES (coach effects):
  Defense-first      coach defensive effect minus offensive effect, top third
  Offense-first      offensive minus defensive effect, top third
  Fast pace / Slow pace                         pace, top / bottom third
  Three-point heavy / Inside-out                3PA rate top third / rim rate top third
  Ball movement / Isolation                     assisted-make rate top / bottom third
  Deep rotation / Tight rotation                rotation depth top / bottom third
Player types are the eight archetypes (pass-first point guard = Playmaking
guard, Gobert-type rim protector = Rim runner, and so on).

Cell = how much a player of that type plays above his own career level under
coaches of that group, relative to the same type under all coaches (coach x
type model, coach_archetype.csv: coach general effect shared per player + the
type-specific term), weighted by exposure, with a standard error and a z-score.
Per type, also the coaches who get the most out of it (broad cells only, so a
single star can't make the list).

Usage: python src/scenarios.py   -> out/scenario_matrix.csv, out/scenario_best_coaches.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MIN_EXPOSURE = 20000


def groups():
    st = pd.read_csv(OUT / "coach_style.csv", index_col=0)
    ce = pd.read_csv(OUT / "coach_effects.csv").set_index("coach_id")
    st = st.join(ce[["O", "D", "games"]], rsuffix="_ce")
    st = st[st.games >= 164]
    lean = st.D - st.O
    q = lambda s, hi: s >= s.quantile(2 / 3) if hi else s <= s.quantile(1 / 3)
    return {
        "Defense-first": q(lean, True), "Offense-first": q(-lean, True),
        "Fast pace": q(st.pace, True), "Slow pace": q(st.pace, False),
        "Three-point heavy": q(st.three, True), "Inside-out": q(st.rim, True),
        "Ball movement": q(st.astd, True), "Isolation": q(st.astd, False),
        "Deep rotation": q(st.depth, True), "Tight rotation": q(st.depth, False),
    }, st


def main():
    g, st = groups()
    ca = pd.read_csv(OUT / "coach_archetype.csv")
    ca = ca[(ca.exposure >= MIN_EXPOSURE) & ca.broad]
    ca["se"] = np.sqrt(ca.se_type_O ** 2 + ca.se_type_D ** 2 + ca.se_main ** 2)
    base = ca.groupby("archetype").apply(lambda x: np.average(x.uplift, weights=x.exposure), include_groups=False)
    rows = []
    for gname, mask in g.items():
        members = set(mask[mask].index)
        for t, x in ca[ca.coach_id.isin(members)].groupby("archetype"):
            if len(x) < 4:
                continue
            w = x.exposure / x.exposure.sum()
            v = float(np.sum(w * x.uplift) - base[t])
            se = float(np.sqrt(np.sum(w ** 2 * x.se ** 2)))
            rows.append({"coach_group": gname, "archetype": t, "relative_uplift": v, "se": se, "z": v / se,
                         "coaches": len(x), "coach_list": ", ".join(x.sort_values("uplift", ascending=False).coach.head(4))})
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "scenario_matrix.csv", index=False, float_format="%.3f")
    best = ca.sort_values("uplift", ascending=False).groupby("archetype").head(8)
    best[["archetype", "coach", "uplift", "se", "exposure", "n_players", "top_player", "top_share"]] \
        .to_csv(OUT / "scenario_best_coaches.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print(m.pivot(index="coach_group", columns="archetype", values="relative_uplift").round(2).to_string())
    print(m.pivot(index="coach_group", columns="archetype", values="z").round(1).to_string())
    print("cells |z|>=2:", int((m.z.abs() >= 2).sum()), "of", len(m))
    print(best.groupby("archetype").head(3)[["archetype", "coach", "uplift", "se", "n_players"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
