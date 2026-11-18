"""Is the Scott Stat better than the stats people already use? Head-to-head tests on neutral benchmarks.

Metrics: Scott Stat (Classic rating, every era), BPM, PER, Win Shares per 48 (WS/48), VORP per 100 possessions,
and, from 1996-97 on, raw plus-minus per 100 possessions on the floor and on/off (on minus off).
Every metric is used as-is; no metric is refit to the benchmark.

1. NEXT SEASON FROM ROSTER CHANGES. For every team-season t+1, each player on the roster is rated with his
   season-t value of the metric (players with no season t get the metric's own average for newcomers), weighted
   by his share of the team's possessions in t+1. Correlation of that sum with the team's actual strength in t+1
   (SRS per 100 possessions). Nothing from season t+1 except who played and how much.
2. MISSED GAMES. Starters (28+ mpg) who missed 8+ games: correlation of his PREVIOUS season's metric x minutes
   share with the team's actual drop when he sat (opponent- and home-adjusted margin per 100), from absence_test.py.
3. CONSISTENCY. Year-to-year correlation of each metric for the same player (1,500+ possessions both seasons).

Usage: python src/compare_metrics.py -> out/compare_metrics.json
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
HIST = ROOT / "data" / "hist"
NAMES = {"scott": "Scott Stat", "bpm": "BPM", "per": "PER", "ws48": "Win Shares / 48", "vorp": "VORP",
         "pm": "Raw plus-minus", "onoff": "On/off"}


def plus_minus():
    """Per player-season (NBA ids, season END year): raw on-court margin per 100 and on/off, from stints."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "stints" / "stints_*.parquet"))):
        s = pd.read_parquet(f)
        season = int(s.season.iloc[0]) + 1
        poss = (s.poss_h + s.poss_a).to_numpy() / 2
        mh = (s.pts_h_raw - s.pts_a_raw).to_numpy()
        parts = []
        for side, cols, team, sign in (("h", [f"h{i}" for i in range(5)], s.home_team, 1), ("a", [f"a{i}" for i in range(5)], s.away_team, -1)):
            for c in cols:
                parts.append(pd.DataFrame({"pid": s[c].to_numpy(), "team": team.to_numpy(), "poss": poss, "m": sign * mh}))
        d = pd.concat(parts)
        on = d.groupby(["pid", "team"])[["poss", "m"]].sum()
        tm = pd.DataFrame({"poss": np.r_[poss, poss], "m": np.r_[mh, -mh], "team": np.r_[s.home_team, s.away_team]}).groupby("team")[["poss", "m"]].sum()
        on = on[on.poss > 0].join(tm, on="team", rsuffix="_t")
        on["pm"] = 100 * on.m / on.poss
        off_p = on.poss_t - on.poss
        on["onoff"] = on.pm - 100 * (on.m_t - on.m) / off_p.where(off_p > 200)
        on = on.reset_index()
        g = on.groupby("pid").apply(lambda x: pd.Series({"pm": np.average(x.pm, weights=x.poss),
                                                          "onoff": np.average(x.onoff.fillna(0), weights=x.poss)}), include_groups=False)
        g["season"] = season
        rows.append(g.reset_index())
    return pd.concat(rows)


def main():
    c = pd.concat([pd.read_csv(f) for f in glob.glob(str(OUT / "classic_*[0-9].csv"))], ignore_index=True)
    c["season"] = c.season_start + 1
    c = c[(c.lg.fillna("NBA") == "NBA") & (c.season >= 1978)]
    adv = pd.read_csv(HIST / "Advanced.csv")
    adv = adv[(adv.lg == "NBA") & (adv.season >= 1978)]
    multi = adv.team.str.match(r"^\dTM$|^TOT$")
    one = adv[multi | ~adv.duplicated(["season", "player_id"], keep=False)].drop_duplicates(["season", "player_id"])
    m = c.merge(one[["season", "player_id", "per", "ws_48", "bpm", "vorp", "mp"]].rename(columns={"mp": "mp_adv"}),
                on=["season", "player_id"], how="left")
    m["scott"], m["ws48"] = m.net_classic, m.ws_48
    m["vorp"] = m.vorp / m.poss.clip(lower=1) * 100 * 100     # per 100 possessions (x100 to a readable scale)
    cache = OUT / "plus_minus_seasons.csv"
    pm = pd.read_csv(cache) if cache.exists() else plus_minus().rename(columns={"pid": "nba_id"})
    pm.to_csv(cache, index=False, float_format="%.3f")
    pm = pm.drop_duplicates(["season", "nba_id"])
    m = m.merge(pm, on=["season", "nba_id"], how="left")
    m.loc[m.nba_id.isna(), ["pm", "onoff"]] = np.nan
    m = m.drop_duplicates(["season", "player_id"])
    metrics = ["scott", "bpm", "per", "ws48", "vorp", "pm", "onoff"]
    out = {"names": NAMES}

    # 1. next season from roster changes
    t = pd.read_csv(HIST / "Player Totals.csv")
    t = t[(t.lg == "NBA") & ~t.team.str.match(r"^\dTM$|^TOT$") & (t.season >= 1979)]
    ts = pd.read_csv(HIST / "Team Summaries.csv")
    ts = ts[(ts.lg == "NBA") & (ts.team != "League Average")].dropna(subset=["srs", "pace"])
    tsi = ts.set_index(["season", "abbreviation"])
    t["poss"] = t.mp * pd.MultiIndex.from_arrays([t.season, t.team]).map(tsi.pace).to_numpy() / 48
    t = t.dropna(subset=["poss"])
    t["share"] = t.poss / (t.groupby(["season", "team"]).poss.transform("sum") / 5)
    prev = m.set_index(["season", "player_id"])
    first = m.season == m.groupby("player_id").season.transform("min")
    res = {}
    for era, lo in (("1979-2026", 1979), ("1998-2026", 1998)):
        tt = t[t.season >= lo]
        actual = pd.MultiIndex.from_arrays([tt.season, tt.team]).map(tsi.srs * 100 / tsi.pace)
        r_ = {}
        for k in metrics:
            if lo < 1998 and k in ("pm", "onoff"):
                continue
            newbie = float(np.average(m.loc[first & m[k].notna(), k], weights=m.loc[first & m[k].notna(), "poss"]))
            val = pd.MultiIndex.from_arrays([tt.season - 1, tt.player_id]).map(prev[k]).to_numpy(dtype=float)
            val = np.where(np.isnan(val), newbie, val)
            team = pd.DataFrame({"s": tt.season.to_numpy(), "tm": tt.team.to_numpy(), "v": val * tt.share.to_numpy(), "a": actual})
            team = team.groupby(["s", "tm"]).agg(v=("v", "sum"), a=("a", "first")).dropna()
            r_[k] = {"r": float(np.corrcoef(team.v, team.a)[0, 1]), "n": int(len(team))}
        res[era] = r_
    out["next_season"] = res

    # 1b. CHANGE from season to season: change in team strength vs change in the roster-weighted metric (season t
    # roster at season t values vs season t+1 roster at season t values). Removes anything that just says "good
    # teams stay good".
    chg = {}
    tt = t[t.season >= 1980]
    for k in metrics:
        newbie = float(np.average(m.loc[first & m[k].notna(), k], weights=m.loc[first & m[k].notna(), "poss"]))
        cur = pd.MultiIndex.from_arrays([tt.season, tt.player_id]).map(prev[k]).to_numpy(dtype=float)
        nxt = pd.MultiIndex.from_arrays([tt.season - 1, tt.player_id]).map(prev[k]).to_numpy(dtype=float)
        d0 = pd.DataFrame({"s": tt.season.to_numpy(), "tm": tt.team.to_numpy(), "v": np.where(np.isnan(cur), newbie, cur) * tt.share.to_numpy()})
        d1 = pd.DataFrame({"s": tt.season.to_numpy(), "tm": tt.team.to_numpy(), "v": np.where(np.isnan(nxt), newbie, nxt) * tt.share.to_numpy()})
        now = d0.groupby(["s", "tm"]).v.sum()                       # season s roster, season s values
        pred = d1.groupby(["s", "tm"]).v.sum()                      # season s roster, season s-1 values
        before = now.rename(lambda i: i, level=0)
        g = pd.DataFrame({"pred_next": pred})
        g["base"] = [now.get((s_ - 1, tm_), np.nan) for s_, tm_ in g.index]
        g["act"] = [tsi.srs.get((s_, tm_), np.nan) * 100 / tsi.pace.get((s_, tm_), np.nan) for s_, tm_ in g.index]
        g["act0"] = [tsi.srs.get((s_ - 1, tm_), np.nan) * 100 / tsi.pace.get((s_ - 1, tm_), np.nan) for s_, tm_ in g.index]
        g = g.dropna()
        if k in ("pm", "onoff"):
            g = g[g.index.get_level_values(0) >= 1999]
        if len(g) < 100:
            continue
        chg[k] = {"r": float(np.corrcoef(g.pred_next - g.base, g.act - g.act0)[0, 1]), "n": int(len(g))}
    out["change"] = chg

    # 2. missed games
    a = pd.read_csv(OUT / "absence_players.csv")
    # each metric from the PREVIOUS season, so no metric has seen the games being tested (same-season on/off
    # counts the missed games as "off" time, and the Scott Stat's game logs include them too)
    a = a[a.mpg >= 28].merge(m[["season", "player_id"] + metrics].assign(season=m.season + 1), on=["season", "player_id"], how="left")
    mg = {}
    for k in metrics:
        x = a.dropna(subset=[k])
        if len(x) < 200:
            continue
        w = x.w
        cw = np.cov(x[k] * x.share, x.observed, aweights=w)
        mg[k] = {"r": float(cw[0, 1] / np.sqrt(cw[0, 0] * cw[1, 1])), "n": int(len(x))}
    out["missed_games"] = mg

    # 3. consistency
    q = m[m.poss >= 1500][["season", "player_id"] + metrics]
    pr = q.merge(q.assign(season=q.season - 1), on=["season", "player_id"], suffixes=("", "_n"))
    out["consistency"] = {k: {"r": float(pr[[k, k + "_n"]].dropna().corr().iloc[0, 1]), "n": int(pr[[k, k + "_n"]].dropna().shape[0])} for k in metrics}
    (OUT / "compare_metrics.json").write_text(json.dumps(out, indent=1))
    for sec in ("next_season", "change", "missed_games", "consistency"):
        print(sec)
        if sec == "next_season":
            for era, r_ in out[sec].items():
                print(" ", era, {NAMES[k]: round(v["r"], 3) for k, v in sorted(r_.items(), key=lambda kv: -kv[1]["r"])})
        else:
            print(" ", {NAMES[k]: round(v["r"], 3) for k, v in sorted(out[sec].items(), key=lambda kv: -kv[1]["r"])})


if __name__ == "__main__":
    main()
