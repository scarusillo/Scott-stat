"""Star x teammate-type FIT — which kinds of teammates make the star's lineups
better than the sum of their parts (both ends of the floor, not just shooting).

Fit the full coach_effects model (career-level players + experience + coach),
take each possession-row's residual (how much better the offense did than
those ten players and two coaches predict), and for each star regress the
residuals of his on-floor rows on HOW MANY teammates of each type were out
there with him. Teammate counts always sum to ~4, so only differences between
types are identified: each coefficient is points per 100 for swapping one
average teammate for one of that type, centred so the star's teammate mix
averages 0. Positive = that type fits him.

Usage: python src/teammates_fit.py   -> out/teammates_fit.csv
"""
import numpy as np
import pandas as pd

from archetypes import NAMES
from coach_effects import A, H, OUT, ROOT, build, load, solve
from teammates import STARS, ids_by_name

LAM = 2000.0


def main():
    s, _ = load()
    X, y, w, pen, meta = build(s, 3000)
    beta, _ = solve(X, y, w, pen)
    resid = y - X @ beta
    keep = np.r_[s.poss_h, s.poss_a] > 0
    n = len(s)
    off_cols = [np.r_[s[h].astype("int64"), s[a].astype("int64")][keep] for h, a in zip(H, A)]
    def_cols = [np.r_[s[a].astype("int64"), s[h].astype("int64")][keep] for h, a in zip(H, A)]
    season = np.r_[s.season, s.season][keep]
    off = np.column_stack(off_cols)
    dfn = np.column_stack(def_cols)

    arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet")
    code = pd.Series(arch.archetype.map({a: i for i, a in enumerate(NAMES)}).to_numpy(),
                     index=arch.season.astype("int64") * 10 ** 7 + arch.player_id.astype("int64"))
    K = len(NAMES)

    def type_counts(lineups, star):
        cnt = np.zeros((len(lineups), K))
        for j in range(5):
            pid = lineups[:, j]
            c = pd.Series(season[rows_idx] * 10 ** 7 + pid).map(code).to_numpy()
            ok = (pid != star) & ~np.isnan(c)
            np.add.at(cnt, (np.flatnonzero(ok), c[ok].astype(int)), 1)
        return cnt

    ids = ids_by_name()
    out = []
    for name in STARS:
        star = int(ids[name])
        on_o = (off == star).any(axis=1)
        on_d = (dfn == star).any(axis=1)
        parts = []
        for mask, lu, sign in ((on_o, off, 1.0), (on_d, dfn, -1.0)):
            rows_idx = np.flatnonzero(mask)
            if len(rows_idx) == 0:
                continue
            cnt = type_counts(lu[rows_idx], star)
            parts.append((cnt, sign * resid[rows_idx], w[rows_idx]))
        if not parts:
            continue
        C = np.vstack([p[0] for p in parts])
        r = np.concatenate([p[1] for p in parts])
        ww = np.concatenate([p[2] for p in parts])
        mean_mix = np.average(C, axis=0, weights=ww)
        Z = C - mean_mix
        G = Z.T @ (Z * ww[:, None])
        M = G + LAM * np.eye(K)
        b = np.linalg.solve(M, Z.T @ (ww * (r - np.average(r, weights=ww))))
        sig2 = np.sum(ww * (r - np.average(r, weights=ww) - Z @ b) ** 2) / len(r)
        Minv = np.linalg.inv(M)
        se = np.sqrt(sig2 * np.diag(Minv @ G @ Minv))
        b -= np.average(b, weights=mean_mix)     # centre on his typical teammate
        for k, t in enumerate(NAMES):
            out.append({"star": name, "archetype": t, "fit": b[k], "se": se[k],
                        "avg_teammates_of_type": mean_mix[k], "possessions": ww.sum() / 2})
        print(name, end=" | ", flush=True)
    print()
    d = pd.DataFrame(out)
    d.to_csv(OUT / "teammates_fit.csv", index=False, float_format="%.3f")
    piv = d.pivot(index="star", columns="archetype", values="fit").round(1)
    print(piv.loc[["LeBron James", "Michael Jordan", "Stephen Curry", "Nikola Jokić", "Kobe Bryant", "Tim Duncan"]].to_string())
    print(d[d.star.isin(["LeBron James", "Michael Jordan"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
