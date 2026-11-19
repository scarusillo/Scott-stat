"""Coach or star? Who gets the credit when players grow?

Player growth under a coach (coach_legacy.py) could be the coach, or the star those players played next to (Curry's
gravity, Jordan's attention). Two ways to separate them:

1. ONE MODEL WITH BOTH. Each player-season's age-adjusted box production (box_O + box_D, no team results) is explained by
       his own career level (player fixed effect)
     + the coach he played for that season
     + every teammate he shared the floor with, weighted by how much that teammate played ("teammate lift": how much
       a player raises the production of the people around him)
   Ridge on the coach and teammate terms (penalties by 5-fold cross-validation on held-out player-seasons). Coaches and
   stars are told apart wherever they came apart: a star who changed coaches, a coach who coached without his star,
   a star who missed a season.
2. NATURAL EXPERIMENTS. Players' growth (production vs their own record elsewhere) in named splits: Kerr's teams with and
   without Curry, Curry's teammates under Mark Jackson vs Kerr, Phil Jackson with and without Jordan, Jordan's teammates
   under Collins vs Jackson.

Usage: python src/coach_vs_star.py -> out/coach_vs_star.json, out/teammate_lift.csv, out/coach_growth_adjusted.csv
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

import coach_legacy as CL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def load():
    ts = pd.read_csv(ROOT / "data" / "hist" / "Team Summaries.csv")
    ts = ts[(ts.lg == "NBA") & (ts.team != "League Average")].dropna(subset=["srs", "pace"])
    cs, recs = CL.coach_seasons(ts)
    cs = cs[cs.season <= 2026]
    c = pd.concat([pd.read_csv(f) for f in glob.glob(str(OUT / "classic_*[0-9].csv"))], ignore_index=True)
    c["season"] = c.season_start + 1
    c = c[(c.lg.fillna("NBA") == "NBA") & (c.poss > 0) & (c.season <= 2026) & c.age.notna()].copy()
    c["prod"] = c.box_O + c.box_D
    x = c[["season", "player_id", "age", "prod", "poss"]]
    pr = x.merge(x.assign(season=x.season - 1), on=["season", "player_id"], suffixes=("", "_n"))
    pr = pr[(pr.poss >= 500) & (pr.poss_n >= 500)]
    ch = pr.groupby(pr.age.clip(19, 39).round()).apply(lambda g: (g.prod_n - g["prod"]).mean(), include_groups=False)
    ages = np.arange(19, 41)
    lev = pd.Series(np.r_[0, np.cumsum(ch.reindex(ages).fillna(0).to_numpy()[:-1])], index=ages)
    c["padj"] = c["prod"] - c.age.clip(19, 40).round().map(lev).fillna(0)
    c["share"] = c.poss / (c.groupby(["season", "team"]).poss.transform("sum") / 5)
    main_coach = cs.sort_values("weight").drop_duplicates(["season", "team"], keep="last").set_index(["season", "team"]).coach_id
    c["coach_id"] = pd.MultiIndex.from_arrays([c.season, c.team]).map(main_coach)
    names = cs.drop_duplicates("coach_id").set_index("coach_id").coach
    return c, names


def growth(c, rows_mask, label):
    """Growth of the players in rows_mask: their production there vs their production everywhere else (age-adjusted)."""
    base = c[c.poss >= 300].assign(wp=lambda d: d.poss.clip(upper=6000))
    tot = base.assign(s_=base.wp * base.padj).groupby("player_id")[["wp", "s_"]].sum()
    sub = base[rows_mask.reindex(base.index).fillna(False).to_numpy()]
    g = sub.assign(s_=sub.wp * sub.padj).groupby("player_id")[["wp", "s_"]].sum().join(tot, rsuffix="_all")
    g["away"] = g.wp_all - g.wp
    g = g[g.away >= 1500]
    if not len(g):
        return {"label": label, "players": 0}
    d = g.s_ / g.wp - (g.s__all - g.s_) / g.away
    v = float(np.sum(g.wp * d) / g.wp.sum())
    se = float(np.sqrt(np.sum(g.wp ** 2 * (d - v) ** 2)) / g.wp.sum())
    return {"label": label, "growth": v, "se": se, "players": int(len(g)), "poss": float(g.wp.sum())}


def main():
    c, cnames = load()
    d = c[(c.poss >= 500) & c.coach_id.notna()].copy().reset_index(drop=True)
    # teammate exposure: everyone with 15,000+ career possessions can carry a "teammate lift" term
    career = c.groupby("player_id").poss.sum()
    lifters = pd.Index(career[career >= 15000].index)
    tm = c[c.player_id.isin(lifters)][["season", "team", "player_id", "share"]]
    pairs = d[["season", "team", "player_id"]].reset_index().merge(tm, on=["season", "team"], suffixes=("", "_t"))
    pairs = pairs[pairs.player_id != pairs.player_id_t]
    players = pd.Index(d.player_id.unique())
    coaches = pd.Index(d.coach_id.unique())
    n = len(d)
    Xp = sp.csr_matrix((np.ones(n), (np.arange(n), players.get_indexer(d.player_id))), shape=(n, len(players)))
    Xc = sp.csr_matrix((np.ones(n), (np.arange(n), coaches.get_indexer(d.coach_id))), shape=(n, len(coaches)))
    Xt = sp.csr_matrix(((pairs.share / 5).to_numpy(), (pairs["index"].to_numpy(), lifters.get_indexer(pairs.player_id_t))), shape=(n, len(lifters)))
    y = d.padj.to_numpy()
    w = d.poss.clip(upper=6000).to_numpy()

    def fit(blocks, pens, rows):
        X = sp.hstack(blocks).tocsr()[rows]
        pen = np.concatenate(pens)
        sw = np.sqrt(w[rows] / w[rows].mean())
        A = sp.diags(sw) @ X @ sp.diags(1 / np.sqrt(pen))
        b = lsqr(A, sw * y[rows], damp=1.0, atol=1e-10, btol=1e-10, iter_lim=20000)[0] / np.sqrt(pen)
        return b, X
    rng = np.random.default_rng(3)
    fold = rng.integers(0, 5, n)

    def cv(lc, lt):
        err = 0.0
        blocks = [Xp, Xc] + ([Xt] if lt else [])
        for k in range(5):
            tr, te = fold != k, fold == k
            pens = [np.full(Xp.shape[1], 1e-3), np.full(Xc.shape[1], lc)] + ([np.full(Xt.shape[1], lt)] if lt else [])
            b, _ = fit(blocks, pens, tr)
            Xte = sp.hstack(blocks).tocsr()[te]
            err += float(np.sum(w[te] * (y[te] - Xte @ b) ** 2))
        return err
    grid = {}
    for lc in (0.5, 2, 8):
        grid[f"coach_only_lc{lc}"] = cv(lc, None)
        for lt in (0.5, 2, 8):
            grid[f"lc{lc}_lt{lt}"] = cv(lc, lt)
    base_err = min(v for k, v in grid.items() if k.startswith("coach_only"))
    best = min((k for k in grid if k.startswith("lc")), key=grid.get)
    lc, lt = float(best.split("_")[0][2:]), float(best.split("_")[1][2:])
    lc0 = float(min((k for k in grid if k.startswith("coach_only")), key=grid.get).split("lc")[1])
    print("CV error, coach only:", round(base_err, 1), " with teammates:", round(grid[best], 1), "(", best, ")", flush=True)
    b0, _ = fit([Xp, Xc], [np.full(Xp.shape[1], 1e-3), np.full(Xc.shape[1], lc0)], np.ones(n, bool))
    b1, _ = fit([Xp, Xc, Xt], [np.full(Xp.shape[1], 1e-3), np.full(Xc.shape[1], lc), np.full(Xt.shape[1], lt)], np.ones(n, bool))
    P, C = Xp.shape[1], Xc.shape[1]
    coach_alone = pd.Series(b0[P:P + C], index=coaches)
    coach_net = pd.Series(b1[P:P + C], index=coaches)
    lift = pd.Series(b1[P + C:], index=lifters)
    pname = c.drop_duplicates("player_id", keep="last").set_index("player_id").player
    # minutes-weighted average presence: how much of that lift his teammates actually received per season
    lt_df = pd.DataFrame({"player_id": lifters, "player": pname.reindex(lifters).to_numpy(), "lift": lift.to_numpy(),
                          "career_poss": career.reindex(lifters).to_numpy()}).sort_values("lift", ascending=False)
    lt_df.to_csv(OUT / "teammate_lift.csv", index=False, float_format="%.3f")
    nseas = d.groupby("coach_id").season.nunique()
    cg = pd.DataFrame({"coach": cnames.reindex(coaches).to_numpy(), "growth_alone": coach_alone.to_numpy(),
                       "growth_net_of_teammates": coach_net.to_numpy(), "seasons": nseas.reindex(coaches).to_numpy()}, index=coaches)
    cg["star_share"] = 1 - cg.growth_net_of_teammates / cg.growth_alone.where(cg.growth_alone.abs() > 0.05)
    cg.sort_values("growth_net_of_teammates", ascending=False).to_csv(OUT / "coach_growth_adjusted.csv", float_format="%.3f")

    # reliability: refit the final model on odd and even seasons separately
    halves = []
    for h_ in (0, 1):
        m_ = (d.season.to_numpy() % 2) == h_
        bh, _ = fit([Xp, Xc, Xt], [np.full(Xp.shape[1], 1e-3), np.full(Xc.shape[1], lc), np.full(Xt.shape[1], lt)], m_)
        halves.append((pd.Series(bh[P:P + C], index=coaches), pd.Series(bh[P + C:], index=lifters)))
    big_c = nseas[nseas >= 8].index
    big_p = career[career >= 30000].index.intersection(lifters)
    rel = {"coach_growth_net": float(np.corrcoef(halves[0][0].reindex(big_c), halves[1][0].reindex(big_c))[0, 1]),
           "teammate_lift": float(np.corrcoef(halves[0][1].reindex(big_p), halves[1][1].reindex(big_p))[0, 1])}
    print("reliability odd/even:", rel, flush=True)
    # natural experiments
    def rows(**kw):
        m = pd.Series(True, index=c.index)
        for k, v in kw.items():
            m &= v
        return m
    cur = c.player_id == "curryst01"
    gsw = c.team == "GSW"
    curry_seasons = c[cur & (c.poss >= 2500)].season
    curry_poss = c[cur].set_index("season").poss
    kerr = c.coach_id == "kerr-steve"
    mj = c.coach_id == "jackson-mark"
    jor = c[(c.player_id == "jordami01") & (c.poss >= 2500)][["season", "team"]]
    jor_key = set(zip(jor.season, jor.team))
    with_jordan = pd.Series([(s, t) in jor_key for s, t in zip(c.season, c.team)], index=c.index)
    pj = c.coach_id == "jackson-phil"
    chi = c.team == "CHI"
    not_star = lambda pid: c.player_id != pid
    exps = {
        "kerr_with_curry": growth(c, kerr & c.season.isin(curry_seasons) & not_star("curryst01"), "Kerr, Curry healthy"),
        "kerr_without_curry": growth(c, kerr & ~c.season.isin(curry_seasons), "Kerr, Curry out (2019-20)"),
        "curry_under_mark_jackson": growth(c, mj & gsw & c.season.isin(curry_seasons) & not_star("curryst01"), "Mark Jackson with Curry (2012-14)"),
        "phil_with_jordan": growth(c, pj & with_jordan & not_star("jordami01"), "Phil Jackson with Jordan"),
        "phil_bulls_without_jordan": growth(c, pj & chi & ~with_jordan, "Phil Jackson, Bulls without Jordan (1994-95)"),
        "phil_lakers": growth(c, pj & (c.team == "LAL"), "Phil Jackson, Lakers"),
        "jordan_under_others": growth(c, with_jordan & ~pj & not_star("jordami01"), "Jordan's teammates under other coaches"),
    }
    out = {"reliability": rel, "cv": grid, "best": best, "coach_only_lambda": lc0, "teammates_improve_fit": bool(grid[best] < base_err),
           "fit_gain_pct": float(100 * (base_err - grid[best]) / base_err), "experiments": exps,
           "coaches": cg.loc[[k for k in ["kerr-steve", "jackson-phil", "sloan-jerry", "popovich-gregg", "spoelstra-erik", "daigneault-mark",
                                          "fitzsimmons-cotton", "antoni-d-mike", "dantoni-mike", "moe-doug", "riley-pat"] if k in cg.index]].round(3).to_dict("index")}
    (OUT / "coach_vs_star.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(json.dumps({k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} for k, v in exps.items()}, indent=1))
    print(cg[cg.seasons >= 6].sort_values("growth_alone", ascending=False).head(15).round(3).to_string())
    print("TEAMMATE LIFT, top (career 30,000+ possessions):")
    print(lt_df[lt_df.career_poss >= 30000].head(20).round(3).to_string(index=False))
    print(lt_df[lt_df.player.isin(["Stephen Curry", "Michael Jordan", "LeBron James", "Steve Nash", "Tim Duncan", "Magic Johnson", "Kobe Bryant", "Shaquille O'Neal", "Chris Paul", "Nikola Jokić"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
