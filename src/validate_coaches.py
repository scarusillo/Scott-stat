"""Is the coach effect signal or noise? Split-half reliability: estimate every
coach on even-numbered seasons and on odd-numbered seasons separately (no
shared possessions), then correlate the two estimates for coaches with
enough games in both halves. Pure noise -> r ~ 0.

Usage: python src/validate_coaches.py
"""
import numpy as np
import pandas as pd

from coach_effects import build, load, solve

s, cg = load()
est = {}
for half in (0, 1):
    sub = s[s.season % 2 == half]
    X, y, w, pen, meta = build(sub, 3000)
    beta, _ = solve(X, y, w, pen)
    P, C = meta["P"], meta["C"]
    est[half] = pd.DataFrame({"O": beta[2 * P:2 * P + C], "D": beta[2 * P + C:2 * P + 2 * C]},
                             index=meta["coaches"])
games = cg.assign(h=cg.season % 2).groupby(["coach_id", "h"]).size().unstack(fill_value=0)
for min_g in (82, 164, 246):
    ok = games[(games[0] >= min_g) & (games[1] >= min_g)].index
    a, b = est[0].reindex(ok), est[1].reindex(ok)
    print(f"coaches with {min_g}+ games in BOTH halves: {len(ok):3d}   "
          f"r(net) = {np.corrcoef(a.O + a.D, b.O + b.D)[0, 1]:.2f}   "
          f"r(O) = {np.corrcoef(a.O, b.O)[0, 1]:.2f}   r(D) = {np.corrcoef(a.D, b.D)[0, 1]:.2f}")
