"""Luck adjustment: take shooting variance the players on the floor didn't control
out of every stint's points.

For every three-pointer and free throw, replace what happened with what that
SHOOTER makes on average that season (his percentage, shrunk toward league):
  luck = sum over 3PA of 3*(made - p3)  +  sum over FTA of (made - pft)
A defense no longer gets credit for opponents' cold shooting nights, and a
lineup no longer gets credit for a hot one; shooters keep credit for being good
shooters (their own % is the expectation). Two-point shots are left alone.
Playoff shots use the shooter's regular-season percentages.

Writes luck-adjusted points INTO the stint files (pts_h / pts_a) and keeps the
originals as pts_h_raw / pts_a_raw, so every model downstream uses adjusted
points. Game results (final_margin) and score states (margin0) stay raw.
Idempotent: always recomputes from the _raw columns.

Usage: python src/luck.py      (after build_stints.py and shots.py, both seasons + playoffs)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STINTS = ROOT / "data" / "stints"
SHOTS = ROOT / "data" / "shots"
K3, KFT = 100, 50     # pseudo-attempts shrinking each shooter's % toward league
H = [f"h{i}" for i in range(5)]
A = [f"a{i}" for i in range(5)]
ON = [f"on{i}" for i in range(5)]
OPP = [f"opp{i}" for i in range(5)]


def shooter_pcts(y):
    s = pd.read_parquet(SHOTS / f"shots_{y}.parquet", columns=["shooter", "kind", "three", "made"])
    th = s[(s.kind == "fg") & s.three].groupby("shooter").made.agg(["sum", "count"])
    ft = s[s.kind == "ft"].groupby("shooter").made.agg(["sum", "count"])
    lg3 = th["sum"].sum() / th["count"].sum()
    lgft = ft["sum"].sum() / ft["count"].sum()
    p3 = (th["sum"] + lg3 * K3) / (th["count"] + K3)
    pft = (ft["sum"] + lgft * KFT) / (ft["count"] + KFT)
    return p3, pft, lg3, lgft


def season(y, tag=""):
    p3, pft, lg3, lgft = shooter_pcts(y)
    s = pd.read_parquet(SHOTS / f"shots_{tag}{y}.parquet")
    three = (s.kind == "fg") & s.three
    ft = s.kind == "ft"
    exp3 = s.shooter.map(p3).fillna(lg3)
    expft = s.shooter.map(pft).fillna(lgft)
    s["luck"] = np.where(three, 3 * (s.made - exp3), 0.0) + np.where(ft, s.made - expft, 0.0)
    key_off = s[ON].astype("int64").astype(str).agg("-".join, axis=1)
    key_def = s[OPP].astype("int64").astype(str).agg("-".join, axis=1)
    lk = s.groupby([s.game_id, key_off, key_def]).luck.sum()
    lk.index.names = ["game_id", "off", "def"]

    path = STINTS / f"stints_{tag}{y}.parquet"
    st = pd.read_parquet(path)
    for side in ("h", "a"):
        if f"pts_{side}_raw" not in st:
            st[f"pts_{side}_raw"] = st[f"pts_{side}"]
    ok = st[H + A].notna().all(axis=1)
    hk = st[H].fillna(0).astype("int64").astype(str).agg("-".join, axis=1)
    ak = st[A].fillna(0).astype("int64").astype(str).agg("-".join, axis=1)
    for side, off, dfn, poss in (("h", hk, ak, "poss_h"), ("a", ak, hk, "poss_a")):
        key = pd.MultiIndex.from_arrays([st.game_id, off, dfn])
        total = pd.Series(lk.reindex(key).fillna(0).to_numpy(), index=st.index)
        # several stints can share one lineup pairing in a game: split by possessions
        grp = pd.Series(list(zip(st.game_id, off, dfn)), index=st.index)
        w = st[poss].astype(float).where(st[poss] > 0, 0.0)
        wsum = w.groupby(grp).transform("sum")
        n = grp.groupby(grp).transform("size")
        share = np.where(wsum > 0, w / wsum.where(wsum > 0, 1), 1 / n)
        st[f"luck_{side}"] = np.where(ok, total * share, 0.0)
        st[f"pts_{side}"] = st[f"pts_{side}_raw"] - st[f"luck_{side}"]
    st.to_parquet(path, index=False)
    matched = float((lk.abs().sum() and (st.luck_h.abs().sum() + st.luck_a.abs().sum()) / lk.abs().sum()))
    return y, tag, matched, float(st.luck_h.sum() + st.luck_a.sum())


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(y, t) for t in ("", "po_") for y in range(1996, 2025)]
    with Pool(4) as p:
        for y, tag, matched, net in p.starmap(season, jobs):
            print(f"{tag}{y}: luck matched to stints {matched:.3f}, net luck {net:+.1f} pts", flush=True)
