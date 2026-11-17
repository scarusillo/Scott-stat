"""Test Classic model variants on the held-out seasons (train 1997-2013, test
2014-2025 against Plus) before changing the published model:

  linear       the current model
  capped       every box feature clipped to the training era's 0.5-99.5 percentile
               range: the model never extrapolates past what it was fit on
  capped_max   clipped only to the training era's min/max (nothing in range changes)
  capped+dr    capped, plus diminishing returns: assists and steals enter through
               a hinge (full slope up to the training 90th percentile, a second,
               fitted slope above it)

Prints holdout accuracy for each, and what each says about the cases that
prompted the test.

Usage: python src/classic_variants.py
"""
import numpy as np
import pandas as pd

import classic as K

CHECK = [("John Stockton", 1988, 1992), ("Michael Jordan", 1988, 1992), ("LeBron James", 2009, 2013),
         ("Magic Johnson", 1987, 1991), ("Steve Nash", 2005, 2009), ("Chris Paul", 2008, 2012),
         ("Nikola Jokić", 2021, 2025), ("Larry Bird", 1984, 1988)]


def transform(d, ref, variant):
    d = d.copy()
    feats = list(K.FEATS)
    if variant in ("capped", "capped+dr", "capped_max"):
        q = (0.0, 1.0) if variant == "capped_max" else (0.005, 0.995)
        for f in K.BOX:
            lo, hi = ref["f_" + f].quantile(list(q))
            d["f_" + f] = d["f_" + f].clip(lo, hi)
    if variant == "capped+dr":
        for f in ("ast", "stl"):
            knot = ref["f_" + f].quantile(0.90)
            d[f"f_{f}_hi"] = (d["f_" + f] - knot).clip(lower=0)
            feats.append(f"{f}_hi")
    return d, feats


def run(variant):
    b, per_team, ts = K.load_box()
    d = K.features(b)
    d["key"] = d.player.map(K.norm)
    p = K.plus_targets()
    dup = p.duplicated(["season", "key"], keep=False)
    d = d.merge(p[~dup].drop(columns="name"), on=["season", "key"], how="left")
    ref = d[d.nba_id.notna() & (d.season <= 2013) & (d.plus_poss >= 1000)]
    d, feats = transform(d, ref, variant)
    cols = ["f_" + f for f in feats]
    tr = d.nba_id.notna() & (d.season <= 2013) & (d.plus_poss >= 250)
    te = d.nba_id.notna() & (d.season >= 2014)
    bO = K.wls(d.loc[tr, cols].to_numpy(), d.loc[tr, "O"].to_numpy(), d.loc[tr, "plus_poss"].to_numpy())
    bD = K.wls(d.loc[tr, cols].to_numpy(), d.loc[tr, "D"].to_numpy(), d.loc[tr, "plus_poss"].to_numpy())
    X = np.c_[np.ones(len(d)), d[cols].to_numpy()]
    d["box_O"], d["box_D"] = X @ bO, X @ bD
    dt = K.team_adjust(d, per_team, ts, 1.0)
    test = K.score(dt, te, "O_c", "D_c")
    # top-end check inside the test seasons: the 40 best Plus seasons
    x = dt[te & (dt.plus_poss >= 1000)]
    top = x.nlargest(40, "net")
    test["top40_plus_mean"] = float(top.net.mean())
    test["top40_classic_mean"] = float((top.O_c + top.D_c).mean())
    # extreme passers inside the test seasons
    hi = x[x.f_ast >= ref.f_ast.quantile(0.95)] if "f_ast" in x else x.iloc[:0]
    test["high_assist_resid"] = float(((hi.O_c + hi.D_c) - hi.net).mean()) if len(hi) else None
    out = {}
    dt["net_c"] = dt.O_c + dt.D_c
    for name, a, z in CHECK:
        s = dt[(dt.player == name) & dt.season.between(a, z)]
        out[f"{name} {a - 1}-{z % 100:02d}"] = round(float(np.average(s.net_c, weights=s.poss)), 2) if len(s) else None
    return test, out


if __name__ == "__main__":
    rows, peaks = {}, {}
    for v in ("linear", "capped", "capped+dr", "capped_max"):
        rows[v], peaks[v] = run(v)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).T.round(3).to_string())
    print(pd.DataFrame(peaks).round(2).to_string())
