"""Game-level impact for every era — the with/without signal from game logs.

Source: every NBA box score since 1946-47 (Kaggle: eoinamoore/historical-nba-
data-and-player-box-scores, data/kaggle_dl/; downloaded with the owner's
Kaggle token). Minutes are blank for most pre-1975 games, but who played is
recorded; a blank is filled with that player's season minutes per game
(Basketball-Reference), then each team-game is scaled to 240 minutes.

Model, per three-season window (rating assigned to the middle season):
  points scored by a team in a game  =  league + home
      + sum(O_i * share_i)  over its players   - sum(D_j * share_j) over opponents
  share = minutes/48 x pace/100   (so O and D are points per 100 possessions)
Ridge regression toward each player's Classic rating (his box-score + team
estimate) instead of toward 0: game results move a player off his Classic
rating only as far as the with/without evidence supports. A star whose team
collapses when he sits, and stays great through roster changes around him,
gets credit moved to him from his teammates.

Validation: with Classic trained on 1996-97..2012-13 only, rate 2013-14..
2024-25 with and without the game logs and compare both to Plus. The
penalty is chosen there; the game logs are used only if they help.

Usage: python src/hist_games.py   -> out/games_ratings.csv, out/hist_games_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import classic as K

ROOT = Path(__file__).resolve().parents[1]
KAG = ROOT / "data" / "kaggle_dl"
OUT = ROOT / "out"
LAMS = (30, 100, 300, 1000, 3000)


def season_end(dt):
    return np.where(dt.dt.month >= 8, dt.dt.year + 1, dt.dt.year)


def load_games():
    cols = ["gameId", "gameDateTimeEst", "firstName", "lastName", "personId", "playerteamId", "home",
            "numMinutes", "points", "fieldGoalsAttempted", "freeThrowsAttempted", "reboundsTotal", "assists",
            "foulsPersonal", "gameType", "comment"]
    p = pd.read_csv(KAG / "PlayerStatistics.csv", usecols=cols, low_memory=False)
    p = p[p.gameType == "Regular Season"]
    dt = pd.to_datetime(p.gameDateTimeEst, errors="coerce")
    p["season"] = season_end(dt)
    for c in ("numMinutes", "points", "fieldGoalsAttempted", "freeThrowsAttempted", "reboundsTotal", "assists", "foulsPersonal"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    dnp = p.comment.astype(str).str.contains("DNP|DND|NWT|Did not|Not With", case=False, na=False)
    stats = p[["points", "fieldGoalsAttempted", "freeThrowsAttempted", "reboundsTotal", "assists", "foulsPersonal"]].sum(axis=1)
    p = p[~dnp & ((p.numMinutes > 0) | (stats > 0))].copy()
    p["key"] = (p.firstName.fillna("") + " " + p.lastName.fillna("")).map(K.norm)
    g = pd.read_csv(KAG / "Games.csv", usecols=["gameId", "gameDateTimeEst", "hometeamId", "awayteamId",
                                                   "homeScore", "awayScore", "gameType"], low_memory=False)
    g = g[g.gameType == "Regular Season"].copy()
    g["season"] = season_end(pd.to_datetime(g.gameDateTimeEst, errors="coerce"))
    return p, g


def classic_priors(oos=False):
    """Classic season ratings keyed (season END year, normalized name). oos=True rebuilds Classic
    with skill prices fit on 1996-97..2012-13 only (for the honest test)."""
    b, per_team, ts = K.load_box()
    d = K.features(b)
    d["key"] = d.player.map(K.norm)
    p = K.plus_targets()
    dup = p.duplicated(["season", "key"], keep=False)
    d = d.merge(p[~dup].drop(columns="name"), on=["season", "key"], how="left")
    ref = d[d.nba_id.notna() & (d.plus_poss >= 1000)]
    for f_ in K.BOX + ["reb"]:
        d["f_" + f_] = d["f_" + f_].clip(ref["f_" + f_].min(), ref["f_" + f_].max())
    base = d.nba_id.notna() & (d.plus_poss >= 250)
    meta = json.load(open(OUT / "classic_meta.json"))
    tr = base & (d.season <= 2013) if oos else base
    f = K.team_adjust(K.era_fit(d, tr, meta["validation"]["tau"]), per_team, ts, meta["validation"]["alpha"])
    f["O_c"], f["D_c"] = f.O_c.astype(float), f.D_c.astype(float)
    f["mpg"] = f.mp / f.g.clip(lower=1)
    lg_pace = ts.groupby("season").pace.mean()
    return f[["season", "key", "player_id", "player", "O_c", "D_c", "mpg", "poss", "O", "D", "net", "plus_poss"]], lg_pace


def build_rows(p, g, pri, pace, seasons):
    """Rows: each game twice (each team on offense). Returns X (sparse, players O then D), y, prior arrays."""
    pp = p[p.season.isin(seasons)].merge(pri[["season", "key", "player_id", "mpg"]], on=["season", "key"], how="left")
    pp = pp.dropna(subset=["player_id"])
    pp["min"] = np.where(pp.numMinutes > 0, pp.numMinutes, pp.mpg)
    tot = pp.groupby(["gameId", "playerteamId"])["min"].transform("sum")
    pp["share"] = (pp["min"] / tot.clip(lower=1) * 5 * pp.season.map(pace).fillna(100) / 100).fillna(0)
    gg = g[g.season.isin(seasons) & g.gameId.isin(pp.gameId)].dropna(subset=["homeScore", "awayScore"])
    gg = gg[(gg.homeScore > 0) & (gg.awayScore > 0)]
    players = np.unique(pp.player_id)
    pix = {p_: i for i, p_ in enumerate(players)}
    P = len(players)
    gi = {gid: i for i, gid in enumerate(gg.gameId)}
    home_team = dict(zip(gg.gameId, gg.hometeamId))
    pp = pp[pp.gameId.isin(gi)]
    is_home = pp.playerteamId.to_numpy() == pp.gameId.map(home_team).to_numpy()
    r = pp.gameId.map(gi).to_numpy()
    col = pp.player_id.map(pix).to_numpy()
    n = len(gg)
    # row k: home offense (k), away offense (n + k)
    rows_off = np.where(is_home, r, n + r)
    rows_def = np.where(is_home, n + r, r)
    X = sp.csr_matrix((np.r_[pp.share, -pp.share], (np.r_[rows_off, rows_def], np.r_[col, P + col])),
                      shape=(2 * n, 2 * P))
    y = np.r_[gg.homeScore, gg.awayScore].astype(float)
    pace_g = np.nan_to_num(np.r_[gg.season.map(pace), gg.season.map(pace)].astype(float), nan=100.0)
    y = 100 * y / pace_g                                     # points per 100 possessions
    home = np.r_[np.ones(n), np.zeros(n)]
    seas = np.r_[gg.season, gg.season]
    return X, y, home, seas, players


def fit_window(X, y, home, seas, players, pri, lam, mid):
    pr = pri.drop_duplicates(["season", "player_id"]).set_index(["season", "player_id"])
    o0 = pr.O_c.reindex(pd.MultiIndex.from_arrays([np.full(len(players), mid), players])).to_numpy()
    d0 = pr.D_c.reindex(pd.MultiIndex.from_arrays([np.full(len(players), mid), players])).to_numpy()
    # players absent from the middle season: their own average Classic in the window
    avg = pri.groupby("player_id")[["O_c", "D_c"]].mean()
    o0 = np.where(np.isnan(o0), avg.O_c.reindex(players).to_numpy(), o0)
    d0 = np.where(np.isnan(d0), avg.D_c.reindex(players).to_numpy(), d0)
    beta0 = np.nan_to_num(np.r_[o0, d0])
    # season means + home handled as unpenalized nuisance terms
    S = pd.get_dummies(seas).to_numpy(float)
    Z = sp.hstack([X, sp.csr_matrix(S), sp.csr_matrix(home[:, None])]).tocsr()
    k = Z.shape[1]
    b0 = np.r_[beta0, np.zeros(k - len(beta0))]
    G = (Z.T @ Z).toarray()
    pen = np.r_[np.full(len(beta0), lam), np.full(k - len(beta0), 1e-6)]
    G[np.diag_indices_from(G)] += pen
    rhs = Z.T @ y + pen * b0
    b = np.linalg.solve(G, rhs)
    P = len(players)
    return pd.DataFrame({"player_id": players, "O_g": b[:P], "D_g": b[P:2 * P], "O_0": o0, "D_0": d0})


def rate(p, g, pri, pace, mids, lam):
    out = []
    for mid in mids:
        X, y, home, seas, players = build_rows(p, g, pri, pace, [mid - 1, mid, mid + 1])
        if len(players) == 0:
            continue
        w = pri[pri.season.between(mid - 1, mid + 1)]
        r = fit_window(X, y, home, seas, players, w, lam, mid)
        r["season"] = mid
        out.append(r)
    return pd.concat(out, ignore_index=True)


def main():
    p, g = load_games()
    # honest test: Classic fit on <=2013, game logs layered on, judged against Plus 2014-2025
    pri_oos, pace = classic_priors(oos=True)
    test_mids = list(range(2014, 2025))
    truth = pri_oos[(pri_oos.season.isin(test_mids)) & (pri_oos.plus_poss >= 1000)][["season", "player_id", "net", "plus_poss",
                                                                                     "O_c", "D_c"]]
    res = {}
    for lam in LAMS:
        r = rate(p, g, pri_oos, pace, test_mids, lam).merge(truth, on=["season", "player_id"]).dropna(
            subset=["O_g", "D_g", "net", "O_c", "D_c"])
        # both methods judged on exactly the same player-seasons
        res["classic_only"] = float(np.corrcoef(r.O_c + r.D_c, r.net)[0, 1])
        res[f"with_games_lam{lam}"] = float(np.corrcoef(r.O_g + r.D_g, r.net)[0, 1])
        res[f"rmse_classic_lam{lam}"] = float(np.sqrt(np.mean((r.O_c + r.D_c - r.net) ** 2)))
        res[f"rmse_games_lam{lam}"] = float(np.sqrt(np.mean((r.O_g + r.D_g - r.net) ** 2)))
        res["n"] = int(len(r))
        print(lam, round(res[f"with_games_lam{lam}"], 4), "vs classic", round(res["classic_only"], 4), "n", len(r), flush=True)
    # penalty by lowest error (the more conservative choice), adopted only if it beats Classic alone
    best = min(LAMS, key=lambda l: res[f"rmse_games_lam{l}"])
    use = res[f"rmse_games_lam{best}"] < res[f"rmse_classic_lam{best}"] and res[f"with_games_lam{best}"] > res["classic_only"]
    # final: real Classic priors, every season with game logs (1951-52 .. 2025-26)
    pri, pace = classic_priors(oos=False)
    allr = rate(p, g, pri, pace, list(range(1952, 2027)), best)
    allr.to_csv(OUT / "games_ratings.csv", index=False, float_format="%.3f")
    meta = {"test_r": res, "lambda": best, "adopted": bool(use),
            "test": "Classic fit on 1996-97..2012-13; seasons 2013-14..2024-25 rated with and without game logs; r vs Plus"}
    (OUT / "hist_games_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
