"""Fit study — is a player in the right situation?

Everything is measured against the player's OWN norm: for each player-season
(Plus tier, 1,000+ possessions) the rating minus the league aging curve at his
age, minus the same thing averaged over his OTHER seasons (leave-one-out, 3,000+
possessions elsewhere). That deviation ("situation") is what a coach, a style
or a team can be credited with; talent is differenced out.

1. Coach split: average deviation of a coach's players in own PRODUCTION (what
   their box scores are worth), LIFT (impact beyond their box scores), offense,
   defense and total, shrunk by empirical Bayes, with even/odd-season reliability.
2. Style fit: per player type, deviation regressed on the team's style that
   season (eight traits, z-scored within season). Validated out of time: fit on
   1996-2011, predict 2012-2024.
3. Movers: when a player changes teams, does the change in predicted fit predict
   the change in his deviation? And do underperformers who move rebound more
   than underperformers who stay?
4. Situations: every player-season with its deviation and fit score; for the
   latest season, each player's best-fitting team styles.

Usage: python src/fit.py  -> out/coach_split.csv, out/fit_style_effects.csv,
                             out/situations.csv, out/fit_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coach_effects import load
from development import coach_of_player_seasons

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
STYLE = ["pace", "three", "rim", "astd", "depth", "churn", "opp_three", "opp_rim"]
TARGETS = ["net", "production", "lift", "O", "D"]
MIN_POSS, MIN_OTHER = 1000, 3000
SPLIT_YEAR = 2011


def aging_curve(d, col):
    nxt = d[["player_id", "season", col, "poss"]].assign(season=d.season - 1)
    p = d.merge(nxt, on=["player_id", "season"], suffixes=("", "_n"))
    p["delta"], p["w"] = p[col + "_n"] - p[col], np.minimum(p.poss, p.poss_n)
    g = p.groupby("age").apply(lambda x: pd.Series({"d": np.average(x.delta, weights=x.w), "n": len(x)}), include_groups=False)
    g = g[g.n >= 30].sort_index()
    vals = pd.Series(np.r_[0, np.cumsum(g.d.to_numpy())], index=[g.index[0]] + [a + 1 for a in g.index])
    return vals - vals.max()


def wls(X, y, w, lam=1.0):
    Xi = np.c_[np.ones(len(X)), X]
    W = w / w.mean()
    G = Xi.T @ (Xi * W[:, None])
    G[np.diag_indices_from(G)] += np.r_[0, np.full(X.shape[1], lam)]
    cov = np.linalg.inv(G)
    b = cov @ Xi.T @ (W * y)
    res = y - Xi @ b
    s2 = np.sum(W * res ** 2) / max(1, len(y) - Xi.shape[1])
    return b, np.sqrt(np.diag(cov) * s2)


def main():
    S = pd.read_csv(OUT / "scott_seasons.csv")
    S = S[S.poss >= MIN_POSS].copy()
    ages = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv", usecols=["season_start", "nba_id", "age"]) for y in range(1996, 2025)])
    ages = ages.dropna().drop_duplicates(["season_start", "nba_id"]).rename(columns={"season_start": "season", "nba_id": "player_id"})
    S = S.merge(ages, on=["season", "player_id"], how="left")
    S["age"] = S.age.fillna(S.groupby("player_id").age.transform("median") + (S.season - S.groupby("player_id").season.transform("median")))
    S = S.dropna(subset=["age"])
    S["age"] = S.age.round().astype(int)
    for col in TARGETS:
        curve = aging_curve(S, col)
        adj = S[col] - S.age.clip(curve.index.min(), curve.index.max()).map(curve)
        tot_w = S.groupby("player_id").poss.transform("sum")
        tot_v = (adj * S.poss).groupby(S.player_id).transform("sum")
        other_w = tot_w - S.poss
        S["dev_" + col] = np.where(other_w >= MIN_OTHER, adj - (tot_v - adj * S.poss) / other_w.where(other_w > 0), np.nan)
    S = S.dropna(subset=["dev_net"])

    s, cg = load()
    ch = coach_of_player_seasons(s).reset_index()[["season", "player_id", "coach_id", "share"]]
    S = S.merge(ch, on=["season", "player_id"], how="left")
    sty = pd.read_csv(OUT / "team_style_seasons.csv").sort_values("n").drop_duplicates(["season", "coach_id"], keep="last")
    # Style = how this coach's teams play in his OTHER seasons (leave-one-season-out). Using the same
    # season would count the players' own results twice (a team that defends badly "allows rim shots").
    tw = sty.groupby("coach_id").n.transform("sum")
    loo = sty[["season", "coach_id", "team_id", "n"]].copy()
    for c in STYLE:
        tv = (sty[c] * sty.n).groupby(sty.coach_id).transform("sum")
        loo[c] = np.where(tw - sty.n >= 60, (tv - sty[c] * sty.n) / (tw - sty.n).where(tw > sty.n), np.nan)
    S = S.merge(loo[["season", "coach_id"] + STYLE], on=["season", "coach_id"], how="left")
    sty = loo
    arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet")[["season", "player_id", "archetype"]]
    S = S.merge(arch, on=["season", "player_id"], how="left")
    names = cg.groupby("coach_id").coach.first()

    # ---- 1. coach split
    rows, rel = [], {}
    C = S[S.share >= 0.7]
    for col in TARGETS:
        v = "dev_" + col
        g = C.groupby("coach_id")[v].agg(["mean", "count"])
        sig2 = C[v].var()
        tau2 = max(1e-6, np.average((g["mean"] - C[v].mean()) ** 2 - sig2 / g["count"], weights=g["count"]))
        g[col] = g["mean"] * g["count"] / (g["count"] + sig2 / tau2)
        g[col + "_raw"] = g["mean"]
        g[col + "_se"] = np.sqrt(1 / (g["count"] / sig2 + 1 / tau2))
        h = C.assign(h=C.season % 2).groupby(["coach_id", "h"])[v].agg(["mean", "count"]).unstack()
        ok = (h["count"][0] >= 12) & (h["count"][1] >= 12)
        rel[col] = {"split_half_r": float(np.corrcoef(h["mean"][0][ok], h["mean"][1][ok])[0, 1]), "coaches": int(ok.sum()),
                    "sd_true": float(np.sqrt(tau2)), "sd_noise": float(np.sqrt(sig2))}
        rows.append(g[[col, col + "_raw", col + "_se"]])
    cs = pd.concat(rows, axis=1)
    cs["player_seasons"] = C.groupby("coach_id").size()
    cs.insert(0, "coach", cs.index.map(names))
    cs.sort_values("net", ascending=False).to_csv(OUT / "coach_split.csv", float_format="%.3f")

    # ---- 2. style fit by player type
    F = S.dropna(subset=STYLE + ["archetype"]).copy()
    eff, coefs, coefs_early = [], {}, {}
    for t, x in F.groupby("archetype"):
        for col in ("net", "O", "D"):
            b, se = wls(x[STYLE].to_numpy(), x["dev_" + col].to_numpy(), x.poss.to_numpy())
            if col == "net":
                coefs[t] = b
                e = x[x.season <= SPLIT_YEAR]
                coefs_early[t] = wls(e[STYLE].to_numpy(), e.dev_net.to_numpy(), e.poss.to_numpy())[0]
            for i, tr in enumerate(STYLE):
                eff.append({"archetype": t, "outcome": col, "trait": tr, "coef": b[i + 1], "se": se[i + 1], "t": b[i + 1] / se[i + 1], "n": len(x)})
    eff = pd.DataFrame(eff)
    eff.to_csv(OUT / "fit_style_effects.csv", index=False, float_format="%.3f")

    def score(df, cf):
        out = np.full(len(df), np.nan)
        for t, b in cf.items():
            m = (df.archetype == t).to_numpy()
            out[m] = df.loc[m, STYLE].to_numpy() @ b[1:]
        return out
    F["fit"] = score(F, coefs)
    F["fit_oos"] = score(F, coefs_early)
    late = F[F.season > SPLIT_YEAR]
    val = {"oos_r_fit_vs_deviation": float(np.corrcoef(late.fit_oos, late.dev_net)[0, 1]), "oos_n": int(len(late)),
           "top_vs_bottom_fifth": float(late[late.fit_oos >= late.fit_oos.quantile(.8)].dev_net.mean()
                                        - late[late.fit_oos <= late.fit_oos.quantile(.2)].dev_net.mean()),
           "sig_cells_net": int((eff[eff.outcome == "net"].t.abs() >= 2).sum()), "cells_net": int((eff.outcome == "net").sum())}

    # ---- 3. movers
    nxt = F[["player_id", "season", "dev_net", "fit_oos", "fit", "coach_id", "team_id"]].assign(season=F.season - 1)
    P = F.merge(nxt, on=["player_id", "season"], suffixes=("", "_n"))
    P["moved"] = P.team_id != P.team_id_n
    P["d_dev"], P["d_fit"] = P.dev_net_n - P.dev_net, P.fit_oos_n - P.fit_oos
    mv = P[P.moved & (P.season >= SPLIT_YEAR)]
    under = P[P.dev_net <= -1]
    val["movers"] = {"n": int(len(mv)), "r_change_in_fit_vs_change_in_deviation": float(np.corrcoef(mv.d_fit, mv.d_dev)[0, 1]),
                     "better_fit_minus_worse_fit": float(mv[mv.d_fit > 0].d_dev.mean() - mv[mv.d_fit <= 0].d_dev.mean())}
    val["underperformers"] = {"moved_n": int(under.moved.sum()), "moved_rebound": float(under[under.moved].d_dev.mean()),
                              "stayed_n": int((~under.moved).sum()), "stayed_rebound": float(under[~under.moved].d_dev.mean()),
                              "moved_to_better_fit_rebound": float(under[under.moved & (under.fit_n > under.fit)].d_dev.mean()),
                              "moved_to_worse_fit_rebound": float(under[under.moved & (under.fit_n <= under.fit)].d_dev.mean())}

    # ---- 3b. how much of an up or down year carries over? Norm from PRIOR seasons only (a real forecast).
    curve = aging_curve(S, "net")
    cl = lambda a_: curve.reindex(np.clip(a_, curve.index.min(), curve.index.max())).to_numpy()
    F = F.sort_values(["player_id", "season"])
    F["adj"] = F.net - cl(F.age.to_numpy())
    g = F.groupby("player_id")
    cw = g.poss.cumsum() - F.poss
    cv = (F.adj * F.poss).groupby(F.player_id).cumsum() - F.adj * F.poss
    F["prior_norm"] = np.where(cw >= MIN_OTHER, cv / cw.where(cw > 0), np.nan)
    F["dev_prior"] = F.adj - F.prior_norm
    nx = F[["player_id", "season", "adj", "team_id"]].assign(season=F.season - 1).rename(columns={"adj": "adj_n", "team_id": "team_n"})
    Q = F.dropna(subset=["dev_prior"]).merge(nx, on=["player_id", "season"])
    Q["next_dev"] = Q.adj_n - Q.prior_norm
    Q["moved"] = Q.team_id != Q.team_n
    slope = lambda d: float(np.polyfit(d.dev_prior, d.next_dev, 1)[0])
    down, up = Q[Q.dev_prior <= -1.5], Q[Q.dev_prior >= 1.5]
    carry = {"pairs": int(len(Q)), "carryover_all": slope(Q), "carryover_stayed": slope(Q[~Q.moved]), "carryover_moved": slope(Q[Q.moved]),
             "carryover_age_under_27": slope(Q[Q.age < 27]), "carryover_age_27_plus": slope(Q[Q.age >= 27]),
             "down_years": {"n": int(len(down)), "this_year": float(down.dev_prior.mean()), "next_year": float(down.next_dev.mean()),
                            "stayed_next": float(down[~down.moved].next_dev.mean()), "moved_next": float(down[down.moved].next_dev.mean())},
             "up_years": {"n": int(len(up)), "this_year": float(up.dev_prior.mean()), "next_year": float(up.next_dev.mean())}}
    val["carryover_note"] = ("not reported from here: Plus season ratings are shrunk toward prior seasons, "
                             "which inflates carryover. See situations.py (season-independent ratings).")
    b_carry = carry["carryover_all"]
    F["norm_now"] = F.prior_norm + cl(F.age.to_numpy())                      # what his history says he is at this age
    F["expected_next"] = F.prior_norm + b_carry * F.dev_prior.fillna(0) + cl(F.age.to_numpy() + 1)

    # ---- 4. situations
    teams = cg[["team_id", "franchise"]].drop_duplicates().set_index("team_id").franchise
    F["team"] = F.team_id.map(teams)
    last = F.season.max()
    ts_last = sty[sty.season == last].sort_values("n").drop_duplicates("team_id", keep="last")
    best = {}
    for t, b in coefs.items():
        sc = pd.Series(ts_last[STYLE].to_numpy() @ b[1:], index=ts_last.team_id.map(teams).to_numpy()).sort_values(ascending=False)
        best[t] = sc
    prev = F[["player_id", "season", "dev_net"]].assign(season=F.season + 1).rename(columns={"dev_net": "dev_prev"})
    F = F.merge(prev, on=["player_id", "season"], how="left")
    F["best_fits"] = [", ".join(f"{k} {v:+.1f}" for k, v in best[t].head(3).items()) if (y == last and t in best) else ""
                      for t, y in zip(F.archetype, F.season)]
    F["coach"] = F.coach_id.map(names)
    cols = ["season", "player_id", "name", "team", "coach", "archetype", "age", "poss", "net", "se_net", "dev_net", "dev_prev",
            "dev_production", "dev_lift", "dev_O", "dev_D", "fit", "best_fits"]
    F[cols].to_csv(OUT / "situations.csv", index=False, float_format="%.3f")
    (OUT / "fit_meta.json").write_text(json.dumps({"coach_split": rel, "fit": val,
                                                   "team_fit_latest": {t: {k: round(float(v), 2) for k, v in sc.items()} for t, sc in best.items()}}, indent=1))
    pd.set_option("display.width", 220)
    print(json.dumps({"coach_split": rel, "fit": val}, indent=1))
    big = cs[cs.player_seasons >= 40]
    print("INDIVIDUAL (production)\n", big.sort_values("production", ascending=False).head(8)[["coach", "player_seasons", "production", "production_raw", "production_se"]].round(2).to_string(index=False))
    print("LIFT\n", big.sort_values("lift", ascending=False).head(8)[["coach", "player_seasons", "lift", "lift_raw", "lift_se"]].round(2).to_string(index=False))
    e = eff[(eff.outcome == "net") & (eff.t.abs() >= 2)].sort_values("t")
    print(e.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
