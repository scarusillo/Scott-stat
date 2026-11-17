"""Scott Stat CLASSIC — impact for every season since 1977-78 from season box
scores + team results, calibrated on Scott Stat PLUS (the play-by-play tier).

Source: Basketball-Reference season tables mirrored at
github.com/sumitrodatta/bball-reference-datasets (data/hist/). Classic starts
in 1977-78, the first season with turnovers recorded (steals/blocks 1973-74).

Model, fit on every 1996-97+ player-season where both tiers exist:
  0. No extrapolation: every box rate is clipped to the range seen in the
     training era (John Stockton's 1987-92 assist rate exceeds every training
     season). Skill prices are ERA-MATCHED: each season is rated with prices
     fit mostly on the play-by-play seasons nearest to it (exponential weight,
     half-width TAU chosen by a backcast test), because prices drift (an assist
     was worth 0.33/100 in 1997-2004 and 0.42 in 2019-25).
  1. Box estimate: Plus offense and defense (season ratings, wins.py) regressed
     on per-100 box rates relative to league (efficiency, volume, assists,
     turnovers, boards, steals, blocks, fouls, 3PA), position and role.
  2. Team adjustment: the team's actual offensive/defensive rating vs the sum
     of its players' box estimates; a fraction ALPHA of the gap is shared out
     per possession played — this is how defense the box can't see (and team
     effects generally) reaches the players who were on the floor. ALPHA is
     chosen on held-out seasons.
Validation (printed): train on 1996-2012, test 2013-2024 against Plus, next to
Basketball-Reference's own BPM as a benchmark.

Usage: python src/classic.py   -> out/classic_<YYYY>.csv (START-year convention),
                                  out/classic_meta.json
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "hist"
OUT = ROOT / "out"
FIRST_END = 1952             # 1951-52: first season with individual minutes
BOX = ["pts_added", "tsa", "ast", "tov", "oreb", "dreb", "stl", "blk", "pf", "fg3a"]
EXTRA = ["guard", "big", "mpg"]
FEATS = BOX + EXTRA
# What the box score recorded, by era (season END years). Each band gets its own
# model, trained on the play-by-play seasons using only the stats that band has.
BANDS = {
    "full": (1978, 2100, BOX),                                                   # 1977-78 on
    "no_turnovers": (1974, 1977, [f for f in BOX if f != "tov"]),                # 1973-74 to 1976-77
    "lite": (1952, 1973, ["pts_added", "tsa", "ast", "reb", "pf"]),             # 1951-52 to 1972-73
}


def band_of(season):
    for name, (a, z, _) in BANDS.items():
        if a <= season <= z:
            return name
    return None
PRIOR_POSS = 300
LAM = 50.0
REPL_WINPCT = 0.20
TAUS = (3.0, 5.0, 8.0, 12.0, float("inf"))   # era-weighting half-widths (seasons) tried in the backcast


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_box():
    t = pd.read_csv(HIST / "Player Totals.csv")
    t = t[(t.lg == "NBA") & (t.season >= FIRST_END)]
    ts = pd.read_csv(HIST / "Team Summaries.csv")
    ts = ts[(ts.lg == "NBA") & (ts.team != "League Average")]
    pace = ts.set_index(["season", "abbreviation"]).pace
    multi = t.team.str.match(r"^\dTM$|^TOT$")
    per_team = t[~multi].copy()
    per_team["poss"] = per_team.mp * per_team.set_index(["season", "team"]).index.map(pace).to_numpy() / 48
    stats = ["g", "mp", "fga", "fta", "pts", "x3pa", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "poss"]
    agg = per_team.groupby(["season", "player_id"])[stats].sum()
    info = t.sort_values("g").groupby(["season", "player_id"]).agg(player=("player", "last"), pos=("pos", "last"),
                                                                    age=("age", "last"))
    team_main = per_team.sort_values("mp").groupby(["season", "player_id"]).team.last()
    b = agg.join(info).join(team_main.rename("team")).reset_index()
    b["x3pa"] = b.x3pa.fillna(0)
    return b, per_team, ts


def features(b):
    out = []
    for season, d in b.groupby("season"):
        d = d[d.poss > 0].copy()
        tsa = d.fga + 0.44 * d.fta
        raw = pd.DataFrame({"pts": d.pts, "tsa": tsa, "ast": d.ast, "tov": d.tov, "oreb": d.orb, "dreb": d.drb,
                            "stl": d.stl, "blk": d.blk, "pf": d.pf, "fg3a": d.x3pa, "reb": d.trb}, index=d.index)
        lg = raw.sum() / d.poss.sum() * 100
        pps = raw.pts.sum() / raw.tsa.sum()
        r = raw.add(lg * PRIOR_POSS / 100).div(d.poss + PRIOR_POSS, axis=0) * 100
        r["pts_added"] = r.pts - pps * r.tsa
        rel = r.drop(columns="pts") - lg.drop("pts")
        rel["pts_added"] = r.pts_added
        pos = d.pos.astype(str).str.split("-").str[0]
        rel["guard"] = pos.isin(["PG", "SG"]).astype(float)
        rel["big"] = pos.isin(["PF", "C"]).astype(float)
        rel["mpg"] = d.mp / d.g.clip(lower=1) / 48 - 0.5
        out.append(pd.concat([d, rel.add_prefix("f_")], axis=1))
    return pd.concat(out)


def plus_targets():
    rows = []
    for y in range(1996, 2025):
        w = pd.read_csv(OUT / f"war_{y}.csv")
        w["season"] = y + 1                         # Basketball-Reference labels seasons by END year
        rows.append(w[["season", "player_id", "name", "O", "D", "net", "poss"]])
    p = pd.concat(rows).rename(columns={"player_id": "nba_id", "poss": "plus_poss"})
    p["key"] = p.name.map(norm)
    return p


def wls(X, y, w, lam=LAM):
    Xi = np.c_[np.ones(len(X)), X]
    W = w / w.mean()
    G = Xi.T @ (Xi * W[:, None])
    G[np.diag_indices_from(G)] += np.r_[0, np.full(X.shape[1], lam / 1000)]
    return np.linalg.solve(G, Xi.T @ (W * y))


def predict(d, bO, bD):
    X = np.c_[np.ones(len(d)), d[["f_" + f for f in FEATS]].to_numpy()]
    return X @ bO, X @ bD


def era_fit(d, train, tau, force_band=None):
    """Box estimate for every row, with each season's skill prices fit on the
    training seasons weighted by exp(-|training season - anchor| / tau), where
    anchor is the target season clipped into the training range. tau = inf is
    one model for every era. Each season uses only the stats its era recorded
    (BANDS); force_band applies one band's stat set everywhere (for testing)."""
    d = d.copy()
    tr = d[train]
    lo, hi = tr.season.min(), tr.season.max()
    yO, yD, w0 = tr.O.to_numpy(), tr.D.to_numpy(), tr.plus_poss.to_numpy()
    d["box_O"], d["box_D"] = np.nan, np.nan
    d["band"] = None
    cache = {}
    for season in sorted(d.season.unique()):
        band = force_band or band_of(season)
        if band is None:
            continue
        feats = ["f_" + f for f in BANDS[band][2] + EXTRA]
        anchor = min(max(season, lo), hi)
        if (anchor, band) not in cache:
            w = w0 if np.isinf(tau) else w0 * np.exp(-np.abs(tr.season.to_numpy() - anchor) / tau)
            X_tr = tr[feats].to_numpy()
            cache[(anchor, band)] = (wls(X_tr, yO, w), wls(X_tr, yD, w))
        bO, bD = cache[(anchor, band)]
        m = d.season == season
        d.loc[m, "band"] = band
        X = np.c_[np.ones(m.sum()), d.loc[m, feats].to_numpy()]
        d.loc[m, "box_O"], d.loc[m, "box_D"] = X @ bO, X @ bD
        if season == 1990:
            d.attrs["coef_O_1990"], d.attrs["coef_D_1990"] = bO, bD
    return d


def team_adjust(d, per_team, ts, alpha):
    """Share alpha x (team actual - sum of box estimates) per possession played."""
    d = d.copy()
    pt = per_team[["season", "player_id", "team", "poss"]].merge(d[["season", "player_id", "box_O", "box_D"]],
                                                                 on=["season", "player_id"])
    tm = pt.assign(o=pt.box_O * pt.poss, dd=pt.box_D * pt.poss).groupby(["season", "team"])[["o", "dd", "poss"]].sum()
    tsi = ts.set_index(["season", "abbreviation"])
    lg_ortg = ts.groupby("season").o_rtg.mean()
    team_O = tsi.o_rtg.reindex(tm.index) - lg_ortg.reindex(tm.index.get_level_values(0)).to_numpy()
    team_D = -(tsi.d_rtg.reindex(tm.index) - lg_ortg.reindex(tm.index.get_level_values(0)).to_numpy())
    # sum of five players' estimates per 100 team possessions = 5 * possession-weighted mean
    tm["gap_O"] = team_O.to_numpy() - 5 * tm.o / tm.poss
    tm["gap_D"] = team_D.to_numpy() - 5 * tm.dd / tm.poss
    pt = pt.join(tm[["gap_O", "gap_D"]], on=["season", "team"])
    adj = pt.assign(aO=pt.gap_O * pt.poss, aD=pt.gap_D * pt.poss).groupby(["season", "player_id"])[["aO", "aD", "poss"]].sum()
    adj = (adj[["aO", "aD"]].div(adj.poss, axis=0) / 5).rename(columns={"aO": "gap_O", "aD": "gap_D"})
    d = d.join(adj, on=["season", "player_id"])
    d["O_c"] = d.box_O + alpha * d.gap_O.fillna(0)
    d["D_c"] = d.box_D + alpha * d.gap_D.fillna(0)
    return d


def score(d, mask, col_O, col_D):
    x = d[mask & (d.plus_poss >= 1000)]
    w = x.plus_poss
    net = x[col_O] + x[col_D]
    r = lambda a, b: np.corrcoef(a, b)[0, 1]
    return {"n": int(len(x)), "r_net": float(r(net, x.net)), "r_O": float(r(x[col_O], x.O)),
            "r_D": float(r(x[col_D], x.D)), "rmse_net": float(np.sqrt(np.average((net - x.net) ** 2, weights=w)))}


def main():
    b, per_team, ts = load_box()
    d = features(b)
    d["key"] = d.player.map(norm)
    p = plus_targets()
    dup = p.duplicated(["season", "key"], keep=False)
    d = d.merge(p[~dup].drop(columns="name"), on=["season", "key"], how="left")
    overlap = d[d.season >= 1997]
    matched = overlap.nba_id.notna()
    print(f"matched to Plus: {matched.mean():.3f} of player-seasons, "
          f"{np.average(matched, weights=overlap.poss):.3f} of possessions", flush=True)

    # 1) never extrapolate: clip every box rate to the range seen in the training era
    ref = d[d.nba_id.notna() & (d.plus_poss >= 1000)]
    for f_ in BOX + ["reb"]:
        lo, hi = ref["f_" + f_].min(), ref["f_" + f_].max()
        d["f_" + f_] = d["f_" + f_].clip(lo, hi)
    base = d.nba_id.notna() & (d.plus_poss >= 250)

    # 2) era-matched skill prices: choose TAU by a BACKCAST (train on 2002-03 on,
    #    predict 1996-97..2001-02 as if it were the unseen past), the situation
    #    Classic is in for every pre-1996 season
    back_tr = base & (d.season >= 2003)
    back_te = d.nba_id.notna() & (d.season <= 2002)
    backcast = {}
    for tau in TAUS:
        x = team_adjust(era_fit(d, back_tr, tau), per_team, ts, 1.0)
        backcast[str(tau)] = score(x, back_te, "O_c", "D_c")
    tau = min(TAUS, key=lambda t: backcast[str(t)]["rmse_net"])
    print("backcast by tau:", {k: round(v["r_net"], 3) for k, v in backcast.items()}, "-> tau", tau, flush=True)

    # 3) forward test (train 1997-2013, test 2014-2025), team-adjustment share alpha
    fwd_tr = base & (d.season <= 2013)
    test = d.nba_id.notna() & (d.season >= 2014)
    dfit = era_fit(d, fwd_tr, tau)
    best = None
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        sc = score(team_adjust(dfit, per_team, ts, a), fwd_tr, "O_c", "D_c")
        if best is None or sc["rmse_net"] < best[1]["rmse_net"]:
            best = (a, sc)
    alpha = best[0]
    dt = team_adjust(dfit, per_team, ts, alpha)
    du = team_adjust(era_fit(d, fwd_tr, float("inf")), per_team, ts, alpha)
    adv = pd.read_csv(HIST / "Advanced.csv")
    adv = adv[(adv.lg == "NBA") & ~adv.team.str.match(r"^\dTM$|^TOT$")]
    bpm = adv.assign(w=adv.mp).groupby(["season", "player_id"]).apply(
        lambda x: pd.Series({"obpm": np.average(x.obpm, weights=x.w.clip(lower=1)),
                             "dbpm": np.average(x.dbpm, weights=x.w.clip(lower=1))}), include_groups=False)
    dt = dt.join(bpm, on=["season", "player_id"])
    val = {"alpha": alpha, "tau": tau, "backcast": backcast,
           "test_box_only": score(dt, test, "box_O", "box_D"),
           "test_classic": score(dt, test, "O_c", "D_c"),
           "test_classic_uniform_era": score(du, test, "O_c", "D_c"),
           "test_bref_bpm": score(dt, test & dt.obpm.notna(), "obpm", "dbpm")}
    # how much accuracy each older era's stat set costs: same test, that band's stats only
    for band in ("no_turnovers", "lite"):
        xb = team_adjust(era_fit(d, fwd_tr, tau, force_band=band), per_team, ts, alpha)
        val[f"test_{band}_stats_only"] = score(xb, test, "O_c", "D_c")
    print(json.dumps({k: v for k, v in val.items() if k != "backcast"}, indent=1), flush=True)

    # final: every season rated with era-matched prices learned from ALL overlap seasons
    f = team_adjust(era_fit(d, base, tau), per_team, ts, alpha)
    f["net_c"] = f.O_c + f.D_c
    m = f.nba_id.notna() & (f.plus_poss >= 1000)
    resid_sd = float(np.sqrt(np.average((f.net_c[m] - f.net[m]) ** 2, weights=f.plus_poss[m])))
    bO, bD = f.attrs.get("coef_O_1990", np.zeros(len(FEATS) + 1)), f.attrs.get("coef_D_1990", np.zeros(len(FEATS) + 1))

    # wins: same replacement-level logic as wins.py, from each season's team results
    tsn = ts[ts.season >= FIRST_END]
    for season, x in f.groupby("season"):
        t = tsn[(tsn.season == season)].dropna(subset=["mov", "w", "l", "pace"])
        t = t[(t.w + t.l) > 0]
        if len(t) < 5:
            continue
        k = np.polyfit(t.mov, t.w / (t.w + t.l), 1)[0]
        pace = np.average(t.pace, weights=t.w + t.l)
        repl = (REPL_WINPCT - 0.5) / k / (pace / 100) / 5
        idx = x.index
        f.loc[idx, "WAR_c"] = (f.loc[idx, "net_c"] - repl) * f.loc[idx, "poss"] / 100 * k
    f["se_net"] = resid_sd
    f["season_start"] = f.season - 1
    cols = ["season_start", "band", "player_id", "player", "team", "age", "pos", "g", "mp", "poss", "O_c", "D_c", "net_c",
            "se_net", "WAR_c", "box_O", "box_D", "gap_O", "gap_D", "nba_id", "O", "D", "net"]
    for y, x in f.groupby("season_start"):
        x[cols].rename(columns={"O_c": "O_classic", "D_c": "D_classic", "net_c": "net_classic", "WAR_c": "WAR_classic",
                                "O": "O_plus", "D": "D_plus", "net": "net_plus"}).sort_values(
            "WAR_classic", ascending=False).to_csv(OUT / f"classic_{y}.csv", index=False, float_format="%.3f")
    meta = {"validation": val, "coef_O_1989_90": dict(zip(["intercept"] + FEATS, bO)),
            "coef_D_1989_90": dict(zip(["intercept"] + FEATS, bD)),
            "alpha": alpha, "resid_sd_net": resid_sd, "first_season": FIRST_END - 1}
    (OUT / "classic_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print("resid sd", round(resid_sd, 2))


if __name__ == "__main__":
    main()
