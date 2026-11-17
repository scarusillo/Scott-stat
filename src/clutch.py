"""Clutch — who plays BETTER (or worse) than his normal self when it counts?

Clutch = the NBA's definition: last 5 minutes of the 4th quarter or overtime,
score within 5 points. Play-by-play era only (1996-97 on).

1. Clutch impact (both ends): every clutch possession's result minus what the
   ten players' NORMAL season ratings (wins.py) predict. Each player's career
   clutch delta (offense, defense) is then fit by ridge regression on those
   residuals — so it is adjusted for teammates and opponents, like Plus — with
   the penalty chosen by game-grouped CV. "Clutch rating" = normal rating +
   clutch delta: how good he actually was in those minutes.
2. Clutch shooting: true shooting in clutch vs the rest of the game, relative
   to the league's own clutch drop (everyone shoots worse late in close games),
   and how much more of the team's shots he takes.
3. Is clutch REAL? Every number is re-estimated on even and on odd seasons
   separately (no shared possessions) and the halves correlated. If clutch
   were pure luck, r would be ~0.

Usage: python src/clutch.py   -> out/clutch_players.csv, out/clutch_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
STINTS = ROOT / "data" / "stints"
SHOTS = ROOT / "data" / "shots"
OUT = ROOT / "out"
H = [f"h{i}" for i in range(5)]
A = [f"a{i}" for i in range(5)]
LAMS = (1000, 3000, 10000, 30000, 100000)
PRIOR_TSA = 100


def clutch_stints(tags=("",), clutch_only=True):
    """tags: "" = regular season, "po_" = playoffs. Expectations always use the
    player's REGULAR-season rating for that season (his normal level).
    clutch_only=False keeps every possession (used for playoff elevation)."""
    rows = []
    for tag in tags:
      for y in range(1996, 2025):
        s = pd.read_parquet(STINTS / f"stints_{tag}{y}.parquet").dropna(subset=H + A)
        if clutch_only:
            end = s.clock0 - s.seconds
            frac = ((np.minimum(s.clock0, 300) - np.maximum(end, 0)).clip(lower=0) / s.seconds.clip(lower=1))
            frac = np.where(s.seconds == 0, (s.clock0 <= 300).astype(float), frac)
            keep = (s.period >= 4) & (frac > 0) & (s.margin0.abs() <= 5)
        else:
            frac = np.ones(len(s))
            keep = pd.Series(True, index=s.index)
        s = s[keep].copy()
        s["frac"] = np.asarray(frac)[keep.to_numpy()]
        s["playoffs"] = tag == "po_"
        w = pd.read_csv(OUT / f"war_{y}.csv").set_index("player_id")
        for c in H + A:
            s[c + "_O"] = s[c].astype("int64").map(w.O).fillna(0).to_numpy()
            s[c + "_D"] = s[c].astype("int64").map(w.D).fillna(0).to_numpy()
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def design(s, pids):
    """Rows: home offense then away offense. Columns: delta_O (offense players), delta_D (defenders)."""
    P = len(pids)
    n = len(s)
    rr, cc, vv = [], [], []
    expected = []
    for off, dfn, r0, side, opp in ((H, A, 0, "h", "a"), (A, H, n, "a", "h")):
        r = np.arange(n) + r0
        for c in off:
            rr.append(r); cc.append(np.searchsorted(pids, s[c].astype("int64"))); vv.append(np.ones(n))
        for c in dfn:
            rr.append(r); cc.append(P + np.searchsorted(pids, s[c].astype("int64"))); vv.append(-np.ones(n))
        expected.append(sum(s[c + "_O"] for c in off).to_numpy() - sum(s[c + "_D"] for c in dfn).to_numpy())
    X = sp.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(2 * n, 2 * P))
    poss = np.r_[s.poss_h, s.poss_a].astype(float)
    pts = np.r_[s.pts_h, s.pts_a].astype(float)
    w = poss * np.r_[s.frac, s.frac]
    keep = poss > 0
    y = np.where(keep, 100 * pts / np.where(keep, poss, 1), 0)
    home = np.r_[np.ones(n), np.zeros(n)]
    exp = np.concatenate(expected)
    return X[keep], y[keep], w[keep], exp[keep], home[keep], np.r_[s.game_id, s.game_id][keep], \
        np.r_[s.season, s.season][keep]


def ridge(X, r, w, lam, se=False):
    XtW = X.T.multiply(w).tocsr()
    G = (XtW @ X).toarray()
    M = G + lam * np.eye(G.shape[0])
    b = np.linalg.solve(M, XtW @ r)
    if not se:
        return b, None
    res = r - X @ b
    s2 = np.sum(w * res ** 2) / len(r)
    Minv = np.linalg.inv(M)
    return b, np.sqrt(s2 * np.sum((Minv @ G) * Minv, axis=1))


def residuals(y, w, exp, home):
    """Clutch possession result minus normal-rating expectation, after fitting the
    clutch scoring environment (intercept + home) by weighted least squares."""
    Z = np.c_[np.ones(len(y)), home]
    beta = np.linalg.lstsq(Z * np.sqrt(w)[:, None], (y - exp) * np.sqrt(w), rcond=None)[0]
    return y - exp - Z @ beta, beta


def clutch_impact(s):
    pids = np.unique(s[H + A].to_numpy("int64"))
    X, y, w, exp, home, games, seasons = design(s, pids)
    r, env = residuals(y, w, exp, home)
    rng = np.random.default_rng(0)
    ug = np.unique(games)
    f = pd.Series(rng.integers(0, 5, len(ug)), index=ug).loc[games].to_numpy()
    cv = {}
    for lam in LAMS:
        e = 0.0
        for k in range(5):
            tr, te = f != k, f == k
            b, _ = ridge(X[tr], r[tr], w[tr], lam)
            e += np.sum(w[te] * (r[te] - X[te] @ b) ** 2)
        cv[lam] = e / w.sum()
    lam = min(cv, key=cv.get)
    b, se = ridge(X, r, w, lam, se=True)
    P = len(pids)
    out = pd.DataFrame({"player_id": pids, "clutch_d_O": b[:P], "clutch_d_D": b[P:],
                        "se_O": se[:P], "se_D": se[P:]})
    # clutch possessions: offense + defense, per player
    on_o = np.asarray(X[:, :P].T @ w).ravel()
    on_d = -np.asarray(X[:, P:].T @ w).ravel()
    out["clutch_poss"] = (on_o + on_d) / 2
    out["clutch_delta"] = out.clutch_d_O + out.clutch_d_D
    out["se_delta"] = np.hypot(out.se_O, out.se_D)
    # split-half reliability with the chosen lambda
    halves = {}
    for h in (0, 1):
        m = seasons % 2 == h
        bh, _ = ridge(X[m], r[m], w[m], lam)
        po = np.asarray(X[m][:, :P].T @ w[m]).ravel()
        pd_ = -np.asarray(X[m][:, P:].T @ w[m]).ravel()
        halves[h] = pd.DataFrame({"d": bh[:P] + bh[P:], "poss": (po + pd_) / 2}, index=pids)
    rel = {}
    for mn in (300, 600, 1000):
        ok = (halves[0].poss >= mn) & (halves[1].poss >= mn)
        rel[mn] = {"players": int(ok.sum()),
                   "r": float(np.corrcoef(halves[0].d[ok], halves[1].d[ok])[0, 1]) if ok.sum() > 5 else None}
    return out, {"lambda": lam, "cv": cv, "env_intercept": float(env[0]), "env_home": float(env[1]),
                 "split_half": rel}


def clutch_shooting():
    rows = []
    for y in range(1996, 2025):
        s = pd.read_parquet(SHOTS / f"shots_{y}.parquet",
                            columns=["season", "game_id", "team_id", "shooter", "kind", "pts", "period", "clock",
                                     "margin", "on0", "on1", "on2", "on3", "on4"])
        s["clutch"] = (s.period >= 4) & (s.clock <= 300) & (s.margin.abs() <= 5)
        s["tsa"] = np.where(s.kind == "fg", 1.0, 0.44)
        rows.append(s.groupby(["season", "shooter", "clutch"])[["tsa", "pts"]].sum().reset_index())
    d = pd.concat(rows)
    lg = d.groupby("clutch")[["tsa", "pts"]].sum()
    lg_ts = lg.pts / (2 * lg.tsa)
    p = d.pivot_table(index=["season", "shooter"], columns="clutch", values=["tsa", "pts"], aggfunc="sum",
                      fill_value=0)
    p.columns = [f"{a}_{'c' if b else 'n'}" for a, b in p.columns]
    p = p.reset_index()

    def ts_diff(x):
        # shrink clutch TS toward the player's own normal TS minus the league clutch drop
        base = x.pts_n.sum() / (2 * x.tsa_n.sum()) - (lg_ts[False] - lg_ts[True])
        tc = (x.pts_c.sum() + base * 2 * PRIOR_TSA) / (2 * (x.tsa_c.sum() + PRIOR_TSA))
        return pd.Series({"clutch_tsa": x.tsa_c.sum(), "ts_normal": x.pts_n.sum() / (2 * x.tsa_n.sum()),
                          "ts_clutch_raw": x.pts_c.sum() / (2 * max(x.tsa_c.sum(), 1)),
                          "ts_vs_expected": tc - base})
    car = p.groupby("shooter").apply(ts_diff, include_groups=False)
    halves = {h: p[p.season % 2 == h].groupby("shooter").apply(ts_diff, include_groups=False) for h in (0, 1)}
    j = halves[0].join(halves[1], lsuffix="_0", rsuffix="_1", how="inner")
    rel = {}
    for mn in (100, 200, 400):
        ok = (j.clutch_tsa_0 >= mn) & (j.clutch_tsa_1 >= mn)
        rel[mn] = {"players": int(ok.sum()), "r": float(np.corrcoef(j.ts_vs_expected_0[ok], j.ts_vs_expected_1[ok])[0, 1])}
    return car, {"league_ts_normal": float(lg_ts[False]), "league_ts_clutch": float(lg_ts[True]), "split_half": rel}


def main():
    s = clutch_stints()
    print(f"clutch stints: {len(s)}", flush=True)
    imp, m1 = clutch_impact(s)
    shoot, m2 = clutch_shooting()
    # normal level: career season ratings weighted by clutch possessions per season
    war = pd.concat([pd.read_csv(OUT / f"war_{y}.csv").assign(season=y) for y in range(1996, 2025)])
    normal = war.groupby("player_id").apply(lambda x: np.average(x.net, weights=x.poss.clip(lower=1)),
                                            include_groups=False).rename("normal_net")
    names = war.dropna(subset=["name"]).groupby("player_id").name.last()
    d = imp.set_index("player_id").join(normal).join(shoot)
    d["clutch_net"] = d.normal_net + d.clutch_delta
    d.insert(0, "name", names.reindex(d.index))
    d = d.sort_values("clutch_net", ascending=False)
    d.to_csv(OUT / "clutch_players.csv", float_format="%.3f")
    meta = {"impact": m1, "shooting": m2}
    (OUT / "clutch_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(meta, indent=1, default=float))
    big = d[d.clutch_poss >= 1500]
    cols = ["name", "clutch_poss", "normal_net", "clutch_delta", "se_delta", "clutch_net", "ts_vs_expected", "clutch_tsa"]
    pd.set_option("display.width", 220)
    print("\nBEST IN THE CLUTCH (normal + clutch delta), 1500+ clutch possessions")
    print(big.head(15)[cols].round(2).to_string(index=False))
    print("\nRISES MOST")
    print(big.sort_values("clutch_delta", ascending=False).head(10)[cols].round(2).to_string(index=False))
    print("\nSHRINKS MOST")
    print(big.sort_values("clutch_delta").head(8)[cols].round(2).to_string(index=False))
    famous = ["Kobe Bryant", "Michael Jordan", "LeBron James", "Damian Lillard", "Kyrie Irving", "Dirk Nowitzki",
              "Kevin Durant", "Stephen Curry", "Paul Pierce", "Chris Paul", "DeMar DeRozan", "Jimmy Butler",
              "Reggie Miller", "Tim Duncan", "Nikola Jokić", "Shai Gilgeous-Alexander", "Russell Westbrook",
              "James Harden", "Carmelo Anthony", "Ray Allen"]
    print("\nREPUTATION CHECK")
    print(d[d.name.isin(famous)][cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
