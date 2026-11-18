"""Who played winning basketball? Each player's career-long effect on TEAM RESULTS beyond his box score.

Fitted jointly over every NBA team-season since 1951-52:
    team strength (SRS per 100 possessions) - sum over its players of (box-score production x share)
        = sum over its players of (WINNING_i x share)  + noise
share = his possessions / the team's (five per team). WINNING_i is one number per career: how much
more (or less) his teams won, per 100 possessions he played, than his and his teammates' box scores
say. It is the part of winning the box score can't see -- defense, spacing, screening, effort, lifting
or dragging the players around him -- and it is identified by players changing teams: a player whose
teams win beyond their box scores on every roster he joins gets the credit; a player whose teams
underperform everywhere gets the blame. Ridge penalty (toward 0) chosen by cross-validation on held-out
team-seasons, so a core that never split up (Duncan-Parker-Ginobili) shares its winning instead of
each member getting all of it.
ABA team-seasons are left out of the fit (team strength there is relative to a different league).

Tested against play-by-play (1996-97 on), leaving each season out of the fit in turn: does
box + WINNING predict a player's play-by-play impact better than box + the current team credit?

Also reported, descriptive: teammates' box production with him vs. their production elsewhere
(age-adjusted): did the players around him play better or worse?

Usage: python src/winning.py -> out/winning_careers.csv, out/winning_meta.json
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
LAMS = (0.1, 0.3, 1, 3, 10)


def load():
    c = pd.concat([pd.read_csv(f) for f in glob.glob(str(OUT / "classic_*[0-9].csv"))], ignore_index=True)
    c["season"] = c.season_start + 1
    c["lg"] = c.lg.fillna("NBA")
    c = c[(c.poss > 0) & (c.season <= 2026)].copy()
    c["prod"] = c.box_O + c.box_D
    tp = c.groupby(["season", "lg", "team"]).poss.transform("sum") / 5
    c["share"] = c.poss / tp
    ts = pd.read_csv(ROOT / "data" / "hist" / "Team Summaries.csv")
    ts = ts[ts.team != "League Average"].dropna(subset=["srs", "pace"])
    tsi = ts.set_index(["season", "lg", "abbreviation"])
    idx = pd.MultiIndex.from_arrays([c.season, c.lg, c.team])
    c["team_net"] = idx.map(tsi.srs * 100 / tsi.pace)
    c["games"] = idx.map(tsi.w + tsi.l)
    c["team_wpct"] = idx.map(tsi.w / (tsi.w + tsi.l))
    return c


def design(c):
    nba = c[(c.lg == "NBA") & c.team_net.notna()]
    tkey = nba.season.astype(str) + "|" + nba.team
    teams = pd.Index(tkey.unique())
    players = pd.Index(nba.player_id.unique())
    X = sp.csr_matrix((nba.share.to_numpy(), (teams.get_indexer(tkey), players.get_indexer(nba.player_id))),
                      shape=(len(teams), len(players)))
    tm = nba.assign(e=nba["prod"] * nba.share, k=tkey).groupby("k").agg(e=("e", "sum"), y=("team_net", "first"),
                                                                     g=("games", "first"), s=("season", "first"))
    tm = tm.reindex(teams)
    return X, (tm.y - tm.e).to_numpy(), tm.g.to_numpy(), tm.s.to_numpy(), players, teams


def fit(X, y, w, lam):
    sw = np.sqrt(w / w.mean())
    A = sp.diags(sw) @ X
    # intercept by centering
    mu = np.average(y, weights=w)
    return lsqr(A, sw * (y - mu), damp=np.sqrt(lam))[0], mu


def main():
    c = load()
    X, y, w, seas, players, teams = design(c)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(y))
    cv = {}
    for lam in LAMS:
        err = 0.0
        for k in range(5):
            tr = fold != k
            b, mu = fit(X[tr], y[tr], w[tr], lam)
            err += float(np.sum(w[~tr] * (y[~tr] - mu - X[~tr] @ b) ** 2))
        base = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        cv[lam] = 1 - err / base
    lam = max(cv, key=cv.get)
    print("CV share of unexplained team strength recovered:", {k: round(v, 3) for k, v in cv.items()}, "-> lambda", lam)
    b, mu = fit(X, y, w, lam)
    win = pd.Series(b, index=players, name="winning")

    # validation vs play-by-play, each season left out of the fit
    rows = []
    for s in range(1997, 2026):
        keep = seas != s
        bs, _ = fit(X[keep], y[keep], w[keep], lam)
        ws = pd.Series(bs, index=players)
        x = c[(c.season == s) & c.net_plus.notna() & (c.poss >= 1000)]
        rows.append(pd.DataFrame({"plus": x.net_plus, "box": x["prod"], "model": x.net_classic_box_team,
                                  "winning": x["prod"] + x.player_id.map(ws).fillna(0).to_numpy(),
                                  "team_credit": x.O_box + x.D_box, "w": x.poss}))
    v = pd.concat(rows)
    r = lambda col: float(np.corrcoef(v[col], v.plus)[0, 1])
    rmse = lambda col: float(np.sqrt(np.average((v[col] - v.plus) ** 2, weights=v.w)))
    val = {col: {"r": r(col), "rmse": rmse(col)} for col in ("box", "team_credit", "winning", "model")}
    # does WINNING add to the full model rating?
    Z = np.c_[np.ones(len(v)), v.model, v.winning - v.box]
    bz = np.linalg.lstsq(Z * np.sqrt(v.w.to_numpy())[:, None], v.plus * np.sqrt(v.w), rcond=None)[0]
    val["plus_on_model_and_winning"] = {"model": float(bz[1]), "winning_beyond_box": float(bz[2])}
    # weight for adding WINNING to the rating: model + beta x WINNING, beta fit on odd seasons and tested on
    # even ones and vice versa (WINNING itself is already leave-season-out)
    v["season"] = np.concatenate([np.full(len(r_), s_) for r_, s_ in zip(rows, range(1997, 2026))])
    v["wn"] = v.winning - v.box
    def beta_on(m):
        e = (v.plus - v.model)[m]
        return float(np.sum(v.w[m] * e * v.wn[m]) / np.sum(v.w[m] * v.wn[m] ** 2))
    odd = (v.season % 2 == 1).to_numpy()
    pred = v.model + np.where(odd, beta_on(~odd), beta_on(odd)) * v.wn
    val["add_to_rating"] = {"beta": beta_on(np.ones(len(v), bool)), "beta_odd": beta_on(odd), "beta_even": beta_on(~odd),
                            "rmse_model": rmse("model"),
                            "rmse_model_plus_winning_cross_fit": float(np.sqrt(np.average((pred - v.plus) ** 2, weights=v.w)))}
    print(json.dumps(val, indent=1))

    # teammate elevation (descriptive): teammates' box production with him vs their age-adjusted production elsewhere
    x = c[(c.poss >= 500) & c.age.notna()][["season", "player_id", "age", "prod"]]
    pr = x.merge(x.assign(season=x.season - 1), on=["season", "player_id"], suffixes=("", "_n"))
    ch = pr.groupby(pr.age.clip(19, 39).round()).apply(lambda g: (g.prod_n - g["prod"]).mean(), include_groups=False)
    ages = np.arange(19, 41)
    lev = pd.Series(np.r_[0, np.cumsum(ch.reindex(ages).fillna(0).to_numpy()[:-1])], index=ages)
    c["padj"] = c["prod"] - c.age.clip(19, 40).round().map(lev).fillna(0)
    c["w_"] = c.poss.clip(upper=6000)
    key = ["season", "lg", "team"]
    pairs = c[key + ["player_id"]].merge(c[key + ["player_id", "w_", "padj", "share"]], on=key, suffixes=("", "_tm"))
    pairs = pairs[pairs.player_id != pairs.player_id_tm]
    tot = c.assign(pw=c.w_ * c.padj).groupby("player_id")[["w_", "pw"]].sum()
    shr = pairs.assign(pw=pairs.w_ * pairs.padj).groupby(["player_id", "player_id_tm"])[["w_", "pw"]].sum()
    pairs = pairs.join(shr.rename(columns=lambda s: s + "_sh"), on=["player_id", "player_id_tm"]).join(
        tot.rename(columns=lambda s: s + "_tot"), on="player_id_tm")
    away_w = pairs.w__tot - pairs.w__sh
    pairs = pairs[away_w >= 1500]
    pairs["el"] = pairs.padj - (pairs.pw_tot - pairs.pw_sh) / (pairs.w__tot - pairs.w__sh)
    elev = pairs.assign(ew=pairs.el * pairs.w_).groupby("player_id").apply(
        lambda g: pd.Series({"elevation": g.ew.sum() / g.w_.sum(), "teammates_measured": g.player_id_tm.nunique()}),
        include_groups=False)

    c["pace_"] = c.season.map(pd.read_csv(ROOT / "data" / "hist" / "Team Summaries.csv").query("lg == 'NBA'").groupby("season").pace.mean())
    car = c.groupby("player_id").apply(lambda g: pd.Series({
        "player": g.sort_values("season").player.iloc[-1], "seasons": len(g), "poss": g.poss.sum(),
        "team_wpct": np.average(g.team_wpct.fillna(0.5), weights=g.poss),
        "team_credit_career": np.average(g.O_box + g.D_box - g["prod"], weights=g.poss),
        "pace": np.average(g.pace_.fillna(100), weights=g.poss),
        "prime_poss": g.poss.nlargest(10).sum()}), include_groups=False)
    car = car.join(win).join(elev)
    car["winning"] = car.winning.fillna(0)
    # career wins his teams won beyond what the box scores predicted, credited to him (win curve at his pace)
    import classic as K
    kfun, _ = K.win_curve(pd.read_csv(ROOT / "data" / "hist" / "Team Summaries.csv"))
    car["winning_wins"] = car.winning * car.poss / 100 * kfun(car.pace)
    # the Legacy ingredient: the same, over his ten heaviest seasons (career length is already counted
    # by career value, so it isn't counted twice)
    car["winning_prime_wins"] = car.winning * car.prime_poss / 100 * kfun(car.pace)
    car.to_csv(OUT / "winning_careers.csv", float_format="%.3f")
    meta = {"lambda": lam, "cv": cv, "validation_vs_plus_leave_season_out": val, "n_team_seasons": int(len(y))}
    (OUT / "winning_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(car[car.poss >= 30000].sort_values("winning", ascending=False).head(25).round(2).to_string())
    print(car[car.poss >= 30000].sort_values("winning").head(15).round(2).to_string())


if __name__ == "__main__":
    main()
