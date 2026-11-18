"""Are the box-score skill prices right? Test against what happens when a player MISSES games.

The Classic box model prices each skill (scoring, passing, steals, ...) by how well it predicts
play-by-play impact. Missed games are independent evidence the prices were never fit to: if scoring
is under-priced, teams should collapse more without their scorers than the model predicts, and
less without their passers/defenders.

For every player-season 1977-78 on with 15+ games played and 8+ games missed for one team (only
games between his first and last appearance for that team count as missed):
    observed = team's opponent- and home-adjusted margin per 100 possessions WITH him minus WITHOUT him
    predicted parts = each skill's contribution to his rating x his share of the game (mpg / 48)
Regression (weighted by games with/without): observed = sum_k beta_k x part_k + c x share.
If the prices are right, every beta is 1. beta_scoring > 1 means scoring is worth more than priced.

Usage: python src/absence_test.py -> out/absence_test.json, out/absence_players.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import classic as K
import hist_games as HG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
GROUPS = {"scoring": ["pts_added", "tsa", "pa_x_tsa", "pa_x_mpg", "fg3a"], "passing": ["ast"], "turnovers": ["tov"],
          "rebounding": ["oreb", "dreb"], "steals": ["stl"], "blocks": ["blk"], "fouls": ["pf"],
          "role": ["guard", "big", "mpg"]}


def contributions():
    """Per player-season skill contributions (price x rate), full-stat band (1977-78 on)."""
    b, per_team, ts = K.load_box()
    d = K.features(b)
    d["key"] = d.player.map(K.norm)
    p = K.plus_targets()
    dup = p.duplicated(["season", "key"], keep=False)
    d = d.merge(p[~dup].drop(columns="name"), on=["season", "key"], how="left")
    ref = d[d.nba_id.notna() & (d.plus_poss >= 1000)]
    for f_ in K.CLIP:
        d["f_" + f_] = d["f_" + f_].clip(ref["f_" + f_].min(), ref["f_" + f_].max())
    tr = d[d.nba_id.notna() & (d.plus_poss >= 250)]
    lo, hi = tr.season.min(), tr.season.max()
    feats = K.BANDS["full"][2] + K.EXTRA
    out = []
    for season, x in d[d.season >= 1978].groupby("season"):
        anchor = min(max(season, lo), hi)
        w = tr.plus_poss.to_numpy() * np.exp(-np.abs(tr.season.to_numpy() - anchor) / 5.0)
        X = tr[["f_" + f for f in feats]].to_numpy()
        beta = K.wls(X, tr.O.to_numpy(), w) + K.wls(X, tr.D.to_numpy(), w)
        c = pd.DataFrame(x[["f_" + f for f in feats]].to_numpy() * beta[1:], columns=feats, index=x.index)
        g = pd.DataFrame({k: c[v].sum(axis=1) for k, v in GROUPS.items()})
        g["role"] += beta[0]
        g["season"], g["key"], g["player_id"], g["player"] = season, x.key, x.player_id, x.player
        g["usg"] = (x.fga + 0.44 * x.fta + x.tov) / x.poss * 100
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    comp = contributions()
    fin = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1977, 2026)], ignore_index=True)
    fin["season"] = fin.season_start + 1
    comp = comp.merge(fin[["season", "player_id", "net_classic", "box_O", "box_D", "creation", "usg_rel"]], on=["season", "player_id"])
    comp["creation"] = comp.creation.fillna(0)
    comp["net_classic"] = comp.net_classic - comp.creation             # test the rating WITHOUT the premium
    comp["rest"] = comp.net_classic - comp[list(GROUPS)].sum(axis=1)     # team credit, game logs, play-by-play
    comp = comp[~comp.duplicated(["season", "key"], keep=False)]

    p, g = HG.load_games()
    g = g[g.season >= 1978].dropna(subset=["homeScore", "awayScore"])
    g = g[(g.homeScore > 0) & (g.awayScore > 0)]
    _, _, ts = K.load_box()
    pace = ts.groupby("season").pace.mean()
    tg = pd.concat([pd.DataFrame({"gameId": g.gameId, "season": g.season, "team": g.hometeamId, "opp": g.awayteamId,
                                  "margin": g.homeScore - g.awayScore, "home": 1}),
                    pd.DataFrame({"gameId": g.gameId, "season": g.season, "team": g.awayteamId, "opp": g.hometeamId,
                                  "margin": g.awayScore - g.homeScore, "home": 0})], ignore_index=True)
    tg["m100"] = 100 * tg.margin / tg.season.map(pace)
    strength = tg.groupby(["season", "team"]).m100.mean()
    hca = tg.groupby("season").apply(lambda x: x.m100[x.home == 1].mean(), include_groups=False)
    tg["adj"] = tg.m100 + pd.MultiIndex.from_arrays([tg.season, tg.opp]).map(strength) \
        - np.where(tg.home == 1, 1, -1) * tg.season.map(hca)
    tg["order"] = tg.groupby(["season", "team"]).gameId.rank(method="first")
    tgi = tg.set_index(["gameId", "team"])

    pp = p[p.season >= 1978][["gameId", "season", "playerteamId", "key", "numMinutes"]].rename(columns={"playerteamId": "team"})
    pp = pp.merge(comp[["season", "key"]].drop_duplicates(), on=["season", "key"])
    pp["order"] = pd.MultiIndex.from_arrays([pp.gameId, pp.team]).map(tgi.order)
    pp = pp.dropna(subset=["order"])
    span = pp.groupby(["season", "team", "key"]).agg(first=("order", "min"), last=("order", "max"), gp=("gameId", "nunique"),
                                                      mpg=("numMinutes", "mean")).reset_index()
    span = span[span.gp >= 15]
    rows = []
    played = pp.groupby(["season", "team", "key"]).gameId.apply(set)
    for (season, team), games in tg.groupby(["season", "team"]):
        sp_ = span[(span.season == season) & (span.team == team)]
        if not len(sp_):
            continue
        for r in sp_.itertuples():
            on = played[(season, team, r.key)]
            window = games[games.order.between(r.first, r.last)]
            w_in = window.gameId.isin(on)
            n_out = int((~w_in).sum())
            if n_out < 8:
                continue
            rows.append({"season": season, "team": team, "key": r.key, "gp": int(w_in.sum()), "missed": n_out,
                         "mpg": r.mpg, "observed": window.adj[w_in.to_numpy()].mean() - window.adj[~w_in.to_numpy()].mean()})
    a = pd.DataFrame(rows).merge(comp, on=["season", "key"])
    a = a[a.mpg >= 10]
    a["share"] = a.mpg / 48
    a["w"] = 1 / (1 / a.gp + 1 / a.missed)
    parts = list(GROUPS) + ["rest"]
    X = np.c_[a[parts].to_numpy() * a.share.to_numpy()[:, None], a.share.to_numpy()]
    W = a.w.to_numpy()
    y = a.observed.to_numpy()
    G = X.T @ (X * W[:, None])
    beta = np.linalg.solve(G, X.T @ (W * y))
    res = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(G) @ (X.T @ (X * (W ** 2 * res ** 2)[:, None])) @ np.linalg.inv(G)))
    out = {"n_player_seasons": int(len(a)), "games_missed": int(a.missed.sum()),
           "beta": {k: {"beta": float(b_), "se": float(s_)} for k, b_, s_ in zip(parts + ["share"], beta, se)}}
    # usage check: given the model's full rating, do high-usage players' absences hurt more than predicted?
    a["pred"] = (a.net_classic + 2.0) * a.share
    a["resid"] = a.observed - a.pred
    a["usg_q"] = pd.qcut(a.usg, 5, labels=False)
    out["by_usage_quintile"] = a.groupby("usg_q").apply(lambda x: {
        "usg": float(x.usg.mean()), "observed": float(np.average(x.observed, weights=x.w)),
        "predicted": float(np.average(x.pred, weights=x.w)), "n": int(len(x))}, include_groups=False).to_dict()
    # CREATION PREMIUM: on starters (28+ mpg, absences mostly injury), regress the miss (observed - predicted)
    # on share x max(0, relative usage - r0), with a share term for any overall level miss. r0 by best fit.
    st = a[a.mpg >= 28].dropna(subset=["usg_rel"])
    fits = {}
    for r0 in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        Xc = np.c_[st.share * (st.usg_rel - r0).clip(lower=0), st.share]
        Wc, yc = st.w.to_numpy(), st.resid.to_numpy()
        Gi = np.linalg.inv(Xc.T @ (Xc * Wc[:, None]))
        bc = Gi @ Xc.T @ (Wc * yc)
        rc = yc - Xc @ bc
        sec = np.sqrt(np.diag(Gi @ (Xc.T @ (Xc * (Wc ** 2 * rc ** 2)[:, None])) @ Gi))
        fits[r0] = {"k": float(bc[0]), "se": float(sec[0]), "level": float(bc[1]), "sse": float(np.sum(Wc * rc ** 2))}
    r0 = min(fits, key=lambda r: fits[r]["sse"])
    cm = {"r0": r0, "r_max": float(st.usg_rel.max()), **fits[r0], "t": fits[r0]["k"] / fits[r0]["se"], "n": int(len(st)), "grid": fits,
          "note": "points per 100 added to offense per unit of relative usage above r0 (usage ratio to league)"}
    # directional hypothesis (scorers worth MORE than priced, from the play-by-play usage gap and the owner's
    # claim), so a one-sided 5% test: t >= 1.645. Borderline: the premium is reported with its SE.
    cm["adopted"] = bool(cm["t"] >= 1.645 and cm["k"] > 0)
    (OUT / "creation_meta.json").write_text(json.dumps(cm, indent=1))
    out["creation_premium"] = cm
    (OUT / "absence_test.json").write_text(json.dumps(out, indent=1))
    a.to_csv(OUT / "absence_players.csv", index=False, float_format="%.3f")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
