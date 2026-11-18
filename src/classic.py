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
# scoring efficiency x load: a +3 efficiency on 30 shots per 100 is worth more than on 12, and on 38 minutes more
# than on 24 (without these, held-out high-usage stars were under-rated by ~0.3-0.6; see README "usage fix")
INTER = ["pa_x_tsa", "pa_x_mpg"]
EXTRA = ["guard", "big", "mpg"] + INTER
CLIP = BOX + ["reb"] + INTER   # every rate clipped to the training range before pricing
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


def load_box(aba=False):
    """Season box totals. aba=True also loads the ABA (1967-68 to 1975-76), rated against its own league
    and translated to NBA level afterwards (aba_translation). A player who played in both leagues in one
    season keeps only the league where he played more minutes."""
    lgs = ["NBA", "ABA"] if aba else ["NBA"]
    t = pd.read_csv(HIST / "Player Totals.csv")
    t = t[t.lg.isin(lgs) & (t.season >= FIRST_END)]
    if aba:
        single = ~t.team.str.match(r"^\dTM$|^TOT$")
        mp_lg = t[single].groupby(["season", "player_id", "lg"]).mp.sum().reset_index()
        main_lg = mp_lg.sort_values("mp").groupby(["season", "player_id"]).lg.last()
        t = t[t.lg.to_numpy() == main_lg.reindex(pd.MultiIndex.from_frame(t[["season", "player_id"]])).to_numpy()]
    ts = pd.read_csv(HIST / "Team Summaries.csv")
    ts = ts[ts.lg.isin(lgs) & (ts.team != "League Average")]
    pace = ts.set_index(["season", "abbreviation"]).pace
    multi = t.team.str.match(r"^\dTM$|^TOT$")
    per_team = t[~multi].copy()
    per_team["poss"] = per_team.mp * per_team.set_index(["season", "team"]).index.map(pace).to_numpy() / 48
    stats = ["g", "mp", "fga", "fta", "pts", "x3pa", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "poss"]
    agg = per_team.groupby(["season", "player_id"])[stats].sum()
    info = t.sort_values("g").groupby(["season", "player_id"]).agg(player=("player", "last"), pos=("pos", "last"),
                                                                    age=("age", "last"), lg=("lg", "last"))
    team_main = per_team.sort_values("mp").groupby(["season", "player_id"]).team.last()
    b = agg.join(info).join(team_main.rename("team")).reset_index()
    b["x3pa"] = b.x3pa.fillna(0)
    return b, per_team, ts


def features(b):
    out = []
    for _, d in b.groupby(["season", "lg"] if "lg" in b else "season"):
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
        rel["pa_x_tsa"] = rel.pts_added * rel.tsa / 10
        # shot-creation load: his possession-ending plays per 100 vs the league's (ratio; tov 0 where unrecorded)
        use = d.fga + 0.44 * d.fta + d.tov.fillna(0)
        lg_use = use.sum() / d.poss.sum()
        d["usg_rel"] = ((use + lg_use * PRIOR_POSS) / (d.poss + PRIOR_POSS)) / lg_use   # shrunk like every rate
        rel["pa_x_mpg"] = rel.pts_added * rel.mpg
        out.append(pd.concat([d, rel.add_prefix("f_")], axis=1))
    return pd.concat(out)


def plus_targets():
    rows = []
    for y in range(1996, 2025):
        w = pd.read_csv(OUT / f"war_{y}.csv")
        w["season"] = y + 1                         # Basketball-Reference labels seasons by END year
        rows.append(w[["season", "player_id", "name", "O", "D", "net", "poss", "se_O", "se_D"]])
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


def team_adjust(d, per_team, ts, alpha, credit=None):
    """Team credit: the gap between a team's actual offense/defense and the sum of its players'
    box estimates, shared out per possession played.
    credit=None: alpha x an even share (the original).
    credit={"aO","aD","bO","bD"}: share = a x even share + b x even share x (his minutes per game
    minus his team's average, per 10 minutes): players who carry more minutes carry more of the
    gap. Chosen on held-out seasons (team_credit_test.py)."""
    d = d.copy()
    pt = per_team[["season", "player_id", "team", "poss", "mp", "g"]].merge(
        d[["season", "player_id", "box_O", "box_D"]], on=["season", "player_id"])
    pt["mpg"] = pt.mp / pt.g.clip(lower=1)
    tm = pt.assign(o=pt.box_O * pt.poss, dd=pt.box_D * pt.poss, mm=pt.mpg * pt.poss).groupby(
        ["season", "team"])[["o", "dd", "mm", "poss"]].sum()
    tsi = ts.set_index(["season", "abbreviation"])
    # each team against its OWN league's average (the ABA had its own)
    lg_ortg = ts.groupby(["season", "lg"]).o_rtg.transform("mean").set_axis(tsi.index).reindex(tm.index).to_numpy()
    team_O = tsi.o_rtg.reindex(tm.index) - lg_ortg
    team_D = -(tsi.d_rtg.reindex(tm.index) - lg_ortg)
    # sum of five players' estimates per 100 team possessions = 5 * possession-weighted mean
    tm["gap_O"] = team_O.to_numpy() - 5 * tm.o / tm.poss
    tm["gap_D"] = team_D.to_numpy() - 5 * tm.dd / tm.poss
    tm["m_mpg"] = tm.mm / tm.poss
    pt = pt.join(tm[["gap_O", "gap_D", "m_mpg"]], on=["season", "team"])
    rel = (pt.mpg - pt.m_mpg) / 10
    adj = pt.assign(aO=pt.gap_O * pt.poss, aD=pt.gap_D * pt.poss, mO=pt.gap_O * rel * pt.poss,
                    mD=pt.gap_D * rel * pt.poss).groupby(["season", "player_id"])[["aO", "aD", "mO", "mD", "poss"]].sum()
    adj = (adj[["aO", "aD", "mO", "mD"]].div(adj.poss, axis=0) / 5).rename(
        columns={"aO": "gap_O", "aD": "gap_D", "mO": "gapm_O", "mD": "gapm_D"})
    d = d.join(adj, on=["season", "player_id"])
    if credit is None:
        d["O_c"] = d.box_O + alpha * d.gap_O.fillna(0)
        d["D_c"] = d.box_D + alpha * d.gap_D.fillna(0)
    else:
        d["O_c"] = d.box_O + credit["aO"] * d.gap_O.fillna(0) + credit["bO"] * d.gapm_O.fillna(0)
        d["D_c"] = d.box_D + credit["aD"] * d.gap_D.fillna(0) + credit["bD"] * d.gapm_D.fillna(0)
    return d


def fit_credit(d, per_team, ts, train):
    """Least-squares shares (even + minutes-scaled), offense and defense separately, on training rows."""
    f = team_adjust(d, per_team, ts, 1.0)
    out = {}
    for s_ in ("O", "D"):
        m = train & f[f"gap_{s_}"].notna()
        X = f.loc[m, [f"gap_{s_}", f"gapm_{s_}"]].to_numpy()
        y = (f.loc[m, s_] - f.loc[m, f"box_{s_}"]).to_numpy()
        w = np.sqrt(f.loc[m, "plus_poss"].to_numpy())
        b = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        out["a" + s_], out["b" + s_] = float(b[0]), float(b[1])
    return out


def score(d, mask, col_O, col_D):
    x = d[mask & (d.plus_poss >= 1000)]
    w = x.plus_poss
    net = x[col_O] + x[col_D]
    r = lambda a, b: np.corrcoef(a, b)[0, 1]
    return {"n": int(len(x)), "r_net": float(r(net, x.net)), "r_O": float(r(x[col_O], x.O)),
            "r_D": float(r(x[col_D], x.D)), "rmse_net": float(np.sqrt(np.average((net - x.net) ** 2, weights=w)))}


def win_curve(ts):
    """Wins per point of margin as a smooth function of pace, pooled over every NBA season:
        win% = 0.5 + k0 x MOV x (100 / league pace)^gamma
    Per-season fits from 8-30 teams are noisy (8-10 teams in the 1950s-60s); the pooled curve isn't.
    Fitted gamma ~0.34: a point per game is worth somewhat less in a faster game (era_wins_test.py).
    Returns a function pace -> k."""
    from scipy.optimize import least_squares
    t = ts[(ts.lg == "NBA") & (ts.team != "League Average")].dropna(subset=["mov", "w", "l", "pace"])
    t = t[(t.w + t.l) > 0]
    wp = t.w / (t.w + t.l)
    lp = t.groupby("season").pace.transform("mean")
    fit = least_squares(lambda p: wp - 0.5 - p[0] * t.mov * (100 / lp) ** p[1], [0.03, 0.5])
    k0, gamma = float(fit.x[0]), float(fit.x[1])
    return lambda pace: k0 * (100 / pace) ** gamma, {"k0": k0, "gamma": gamma}


def aba_translation(f):
    """ABA -> NBA translation, from players who changed leagues between consecutive seasons (both
    directions; mostly the 1976 merger). A flat shift is NOT enough: the ABA stretched its stars (an ABA
    +5 was about +2.8 the next season in the NBA, an NBA +5 about +4.4). So the translation is
        NBA-equivalent = a + s x ABA rating
    one compression s for the whole ABA, separate shifts a for the early (to 1972-73) and late ABA. Fitted so that
    each mover's next season matches what the NBA's own year-to-year model (rating, age, age^2;
    1965-1990 NBA pairs) predicts from his translated rating:
        ABA -> NBA: next NBA rating  ~ P(a + s x ABA rating, age)
        NBA -> ABA: a + s x next ABA rating ~ P(NBA rating, age)
    Returns a function (season, rating) -> NBA-equivalent rating, and the evidence."""
    from scipy.optimize import least_squares
    x = f[f.poss >= 800][["season", "player_id", "lg", "age", "net_c", "poss"]]
    nxt = x.assign(season=x.season - 1).rename(columns={"lg": "lg2", "net_c": "net2", "poss": "poss2"})[
        ["season", "player_id", "lg2", "net2", "poss2"]]
    pr = x.merge(nxt, on=["season", "player_id"]).dropna(subset=["age"])
    pr["w"] = 1 / (1 / pr.poss + 1 / pr.poss2)
    pr["a1"], pr["a2"] = pr.age - 27, (pr.age - 27) ** 2
    nba = pr[(pr.lg == "NBA") & (pr.lg2 == "NBA") & pr.season.between(1965, 1990)]
    X = np.c_[np.ones(len(nba)), nba.net_c, nba.a1, nba.a2]
    W = nba.w.to_numpy()
    bn = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * nba.net2.to_numpy()))
    P = lambda r, g: bn[0] + bn[1] * r + bn[2] * g.a1.to_numpy() + bn[3] * g.a2.to_numpy()
    an = pr[(pr.lg == "ABA") & (pr.lg2 == "NBA")]
    na = pr[(pr.lg == "NBA") & (pr.lg2 == "ABA")]
    late = lambda g, col: (g[col] >= 1974).to_numpy()
    def res(p):
        a_l, a_e, s_ = p
        a_an = np.where(late(an, "season"), a_l, a_e)
        a_na = np.where(late(na.assign(s2=na.season + 1), "s2"), a_l, a_e)
        r1 = (an.net2.to_numpy() - P(a_an + s_ * an.net_c.to_numpy(), an)) * np.sqrt(an.w.to_numpy())
        r2 = (a_na + s_ * na.net2.to_numpy() - P(na.net_c.to_numpy(), na)) * np.sqrt(na.w.to_numpy())
        return np.r_[r1, r2]
    fit = least_squares(res, [0.0, 0.0, 1.0])
    J = fit.jac
    cov = np.linalg.inv(J.T @ J) * (fit.fun ** 2).sum() / max(len(fit.fun) - 3, 1)
    a_l, a_e, s_ = fit.x
    se = np.sqrt(np.diag(cov))
    ev = {"shift_late": float(a_l), "shift_early": float(a_e), "compression": float(s_),
          "se": {"shift_late": float(se[0]), "shift_early": float(se[1]), "compression": float(se[2])},
          "n_aba_to_nba": int(len(an)), "n_nba_to_aba": int(len(na)),
          "nba_year_to_year_slope": float(bn[1]),
          "examples": {str(r): float(a_l + s_ * r) for r in (0, 2, 4, 6)}}
    # the early ABA can only be weaker than the late ABA
    a_e_used = min(a_e, a_l)
    ev["shift_early_used"] = float(a_e_used)
    tr = lambda season, r: np.where(season >= 1974, a_l, a_e_used) + s_ * r
    return tr, ev


def main():
    b, per_team, ts = load_box(aba=True)
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
    for f_ in CLIP:
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
    alpha = 0.75                                   # even-share weight kept for reference comparisons
    credit_fwd = fit_credit(dfit, per_team, ts, fwd_tr)
    dt = team_adjust(dfit, per_team, ts, alpha, credit_fwd)
    d_even = team_adjust(dfit, per_team, ts, alpha)
    du = team_adjust(era_fit(d, fwd_tr, float("inf")), per_team, ts, alpha, credit_fwd)
    adv = pd.read_csv(HIST / "Advanced.csv")
    adv = adv[(adv.lg == "NBA") & ~adv.team.str.match(r"^\dTM$|^TOT$")]
    bpm = adv.assign(w=adv.mp).groupby(["season", "player_id"]).apply(
        lambda x: pd.Series({"obpm": np.average(x.obpm, weights=x.w.clip(lower=1)),
                             "dbpm": np.average(x.dbpm, weights=x.w.clip(lower=1))}), include_groups=False)
    dt = dt.join(bpm, on=["season", "player_id"])
    val = {"alpha": alpha, "tau": tau, "backcast": backcast, "credit_fwd": credit_fwd,
           "test_classic_even_split": score(d_even, test, "O_c", "D_c"),
           "test_box_only": score(dt, test, "box_O", "box_D"),
           "test_classic": score(dt, test, "O_c", "D_c"),
           "test_classic_uniform_era": score(du, test, "O_c", "D_c"),
           "test_bref_bpm": score(dt, test & dt.obpm.notna(), "obpm", "dbpm")}
    # how much accuracy each older era's stat set costs: same test, that band's stats only
    for band in ("no_turnovers", "lite"):
        xb = team_adjust(era_fit(d, fwd_tr, tau, force_band=band), per_team, ts, alpha, credit_fwd)
        val[f"test_{band}_stats_only"] = score(xb, test, "O_c", "D_c")
    print(json.dumps({k: v for k, v in val.items() if k != "backcast"}, indent=1), flush=True)

    # final: every season rated with era-matched prices learned from ALL overlap seasons
    dfin = era_fit(d, base, tau)
    credit = fit_credit(dfin, per_team, ts, base)
    val["credit"] = credit
    f = team_adjust(dfin, per_team, ts, alpha, credit)
    # SPREAD RESTORATION: with fewer recorded stats the model knows less, so its ratings sit closer to
    # average. Measured on the same modern players (1996-97 on, 2,000+ possessions): rated with only the
    # stats an older era recorded, the spread of ratings shrinks. That is missing information, not
    # missing talent, so older bands are widened back by the same ratio (below, before the ABA step).
    mm_ = (f.season >= 1997) & (f.poss >= 2000)
    full_sd = float((f.O_c + f.D_c)[mm_].std())
    spread = {}
    for band_ in ("no_turnovers", "lite"):
        xb = team_adjust(era_fit(d, base, tau, force_band=band_), per_team, ts, alpha, credit)
        spread[band_] = full_sd / float((xb.O_c + xb.D_c)[mm_].std())
    val["spread_restoration"] = spread
    print("spread restoration:", spread, flush=True)
    f["O_box"], f["D_box"] = f.O_c, f.D_c
    # game logs (hist_games.py): the with/without signal from every game, used only if it
    # beat box-only Classic on the held-out seasons
    gm_path, gm_meta = OUT / "games_ratings.csv", OUT / "hist_games_meta.json"
    games_used = False
    if gm_path.exists() and gm_meta.exists() and json.load(open(gm_meta)).get("adopted"):
        gr = pd.read_csv(gm_path)[["season", "player_id", "O_g", "D_g"]].drop_duplicates(["season", "player_id"])
        f = f.merge(gr, on=["season", "player_id"], how="left")
        # game-log ratings were fit with NBA-only priors (hist_games.py), so ABA seasons keep box + team
        f.loc[f.lg != "NBA", ["O_g", "D_g"]] = np.nan
        f["O_c"] = f.O_g.fillna(f.O_c)
        f["D_c"] = f.D_g.fillna(f.D_c)
        f["games_informed"] = f.O_g.notna()
        games_used = True
    f["net_c"] = f.O_c + f.D_c
    for band_, fac in spread.items():
        mb = f.band == band_
        if not mb.any():
            continue
        g_ = f[mb].groupby(["season", "lg"])
        for c in ("O_c", "box_O", "O_box", "D_c", "box_D", "D_box"):
            mu = g_.apply(lambda z: np.average(z[c], weights=z.poss), include_groups=False)
            base_ = pd.MultiIndex.from_arrays([f.loc[mb, "season"], f.loc[mb, "lg"]]).map(mu).to_numpy()
            f.loc[mb, c] = base_ + fac * (f.loc[mb, c] - base_)
    f["net_c"] = f.O_c + f.D_c
    # ABA seasons: rated against their own league above, then moved down to NBA level
    aba = f.lg == "ABA"
    tr_aba, aba_ev = aba_translation(f)
    sA = aba_ev["compression"]
    shift = pd.Series(np.where(f.season >= 1974, aba_ev["shift_late"], aba_ev["shift_early_used"]), index=f.index)
    # offense and defense each compressed by s, the shift split evenly: net -> shift + s x net
    for c in ("O_c", "box_O", "O_box", "D_c", "box_D", "D_box"):
        f.loc[aba, c] = sA * f.loc[aba, c] + shift[aba] / 2
    f["net_c"] = f.O_c + f.D_c
    val["aba_translation"] = aba_ev
    print("ABA translation:", aba_ev, flush=True)
    f["net_c_box_team"] = f.net_c
    m = f.nba_id.notna() & (f.plus_poss >= 1000)
    resid_sd = float(np.sqrt(np.average((f.net_c[m] - f.net[m]) ** 2, weights=f.plus_poss[m])))
    # BEST AVAILABLE EVIDENCE: from 1996-97 on, blend in the play-by-play rating, weighted by how precise
    # each is (Plus: its own standard error; Classic: its error against Plus minus Plus's own noise).
    # Then, within each season, map the blend back onto Classic's own distribution of values (quantile
    # matching), so a season with play-by-play has exactly the spread and top tail it had without it: the
    # play-by-play changes WHO ranks where inside a season, never how far the best stand above the field. (Without this, the
    # all-era model rated Pau Gasol above Kobe Bryant in 2008-09 and 2009-10; play-by-play has Kobe
    # +4.2 / +3.8 vs Gasol +1.6 / +3.0.)
    pb = f.nba_id.notna() & f.O.notna() & f.se_O.notna() & (f.season >= 1997)
    blend_ev, bl = {}, {}
    for s_ in ("O", "D"):
        mm = pb & (f.plus_poss >= 1000)
        var_c = np.average((f.loc[mm, f"{s_}_c"] - f.loc[mm, s_]) ** 2, weights=f.plus_poss[mm]) \
            - np.average(f.loc[mm, f"se_{s_}"] ** 2, weights=f.plus_poss[mm])
        var_c = max(var_c, 0.25)
        w = (1 / f.loc[pb, f"se_{s_}"] ** 2) / (1 / f.loc[pb, f"se_{s_}"] ** 2 + 1 / var_c)
        bl[s_] = w * f.loc[pb, s_] + (1 - w) * f.loc[pb, f"{s_}_c"]
        blend_ev[s_] = {"classic_error_sd": float(np.sqrt(var_c)), "mean_weight_on_plus": float(w.mean())}
    # quantile matching on the OVERALL rating (possession-weighted, within playing-time bands): a player at the
    # q-th percentile of the blend gets the q-th percentile Classic rating of his band and season, so the
    # distribution (top tail included) is exactly Classic's. Offense/defense keep the blend's split.
    net_bl = bl["O"] + bl["D"]
    net_new = net_bl.copy()
    band = pd.cut(f.poss, [-1, 1500, 3000, np.inf], labels=False)
    for _, idx in f[pb].groupby([f.season[pb], band[pb]]).groups.items():
        ww = f.loc[idx, "poss"].to_numpy()
        cv = f.loc[idx, "net_c"].to_numpy()
        oc = np.argsort(cv)
        qc = (np.cumsum(ww[oc]) - ww[oc] / 2) / ww.sum()
        bv = net_bl[idx].to_numpy()
        ob = np.argsort(bv)
        qb = np.empty(len(bv))
        qb[ob] = (np.cumsum(ww[ob]) - ww[ob] / 2) / ww.sum()
        net_new[idx] = np.interp(qb, qc, cv[oc])
    diff = bl["O"] - bl["D"]
    f.loc[pb, "O_c"] = (net_new + diff) / 2
    f.loc[pb, "D_c"] = (net_new - diff) / 2
    f["net_c"] = f.O_c + f.D_c
    val["plus_blend"] = blend_ev
    # CREATION PREMIUM (absence_test.py): the highest-usage players' teams lose more when they sit than
    # their rating predicts. Fitted there on starters' missed games; applied here only if adopted.
    f["creation"] = 0.0
    cm = OUT / "creation_meta.json"
    if cm.exists():
        c_ = json.load(open(cm))
        if c_.get("adopted"):
            f["creation"] = c_["k"] * (f.usg_rel.clip(upper=c_.get("r_max", np.inf)) - c_["r0"]).clip(lower=0)  # no extrapolation
            f["O_c"] = f.O_c + f.creation
            f["net_c"] = f.O_c + f.D_c
            val["creation"] = c_
    # WINNING (winning.py): each player's career-long effect on team results beyond the box score, fitted
    # jointly over every NBA team-season (players changing teams identify it). Added at the weight that
    # improves held-out play-by-play prediction (cross-fit on odd/even seasons), split evenly O/D.
    f["winning_adj"] = 0.0
    wm, wc = OUT / "winning_meta.json", OUT / "winning_careers.csv"
    if wm.exists() and wc.exists():
        atr = json.load(open(wm))["validation_vs_plus_leave_season_out"]["add_to_rating"]
        wv = pd.read_csv(wc).set_index("player_id").winning
        if "by_minutes" in atr:     # weight rises with minutes: a reserve's team-results credit is confirmed less
            bm = atr["by_minutes"]
            beta_w = (bm["b0"] + bm["b1_per_10_mpg"] * ((f.mp / f.g.clip(lower=1)).clip(15, 40) - 30) / 10).clip(lower=0)
        else:
            beta_w = atr["beta"]
        f["winning_adj"] = beta_w * f.player_id.map(wv).fillna(0)
        f["O_c"] = f.O_c + f.winning_adj / 2
        f["D_c"] = f.D_c + f.winning_adj / 2
        f["net_c"] = f.O_c + f.D_c
        val["winning_beta"] = atr
    print("play-by-play blend:", blend_ev, flush=True)
    bO, bD = f.attrs.get("coef_O_1990", np.zeros(len(FEATS) + 1)), f.attrs.get("coef_D_1990", np.zeros(len(FEATS) + 1))

    # wins: same replacement-level logic as wins.py, from each season's team results
    tsn = ts[ts.season >= FIRST_END]
    kfun, val["win_curve"] = win_curve(ts)
    for (season, lg), x in f.groupby(["season", "lg"]):
        # replacement level from the NBA's win curve that season (ABA ratings are already at NBA level)
        t = tsn[(tsn.season == season) & (tsn.lg == "NBA")].dropna(subset=["mov", "w", "l", "pace"])
        t = t[(t.w + t.l) > 0]
        if len(t) < 5:
            continue
        pace = np.average(t.pace, weights=t.w + t.l)
        k = kfun(pace)
        repl = (REPL_WINPCT - 0.5) / k / (pace / 100) / 5
        idx = x.index
        f.loc[idx, "WAR_c"] = (f.loc[idx, "net_c"] - repl) * f.loc[idx, "poss"] / 100 * k
    f["se_net"] = resid_sd
    f["season_start"] = f.season - 1
    cols = ["season_start", "band", "lg", "player_id", "player", "team", "age", "pos", "g", "mp", "poss", "O_c", "D_c", "net_c",
            "se_net", "WAR_c", "net_c_box_team", "creation", "winning_adj", "usg_rel", "box_O", "box_D", "gap_O", "gap_D", "O_box", "D_box", "nba_id", "O", "D", "net"]
    if games_used:
        cols.append("games_informed")
    for y, x in f.groupby("season_start"):
        x[cols].rename(columns={"O_c": "O_classic", "D_c": "D_classic", "net_c": "net_classic", "WAR_c": "WAR_classic", "net_c_box_team": "net_classic_box_team",
                                "O": "O_plus", "D": "D_plus", "net": "net_plus"}).sort_values(
            "WAR_classic", ascending=False).to_csv(OUT / f"classic_{y}.csv", index=False, float_format="%.3f")
    meta = {"validation": val, "coef_O_1989_90": dict(zip(["intercept"] + FEATS, bO)),
            "coef_D_1989_90": dict(zip(["intercept"] + FEATS, bD)),
            "alpha": alpha, "resid_sd_net": resid_sd, "first_season": FIRST_END - 1, "games_informed": games_used}
    (OUT / "classic_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print("resid sd", round(resid_sd, 2))


if __name__ == "__main__":
    main()
