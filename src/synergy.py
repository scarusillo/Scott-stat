"""Synergy: which teammates are better together than their individual ratings say? (Play-by-play era, 1996-97 on.)

Every stint (the same ten players on the floor) gives an actual margin per 100 possessions (luck-adjusted:
each three-pointer and free throw valued at the shooter's expected percentage). Expected margin = the sum of
the five players' season ratings (offense + defense, wins.py) minus the opponents', plus home court. For
every pair (and trio) of teammates on the floor together:
    synergy = possession-weighted average of (actual - expected) while they share the floor,
              shrunk toward 0 by K extra possessions (a lucky 300 possessions can't top the list)
Players' individual ratings already come from these same stints, so a pair that always plays together
shows ~0: synergy appears only when their time TOGETHER beats what their time APART implies. That's the
point: two players who make each other better than either is alone.
Also per pair: possessions together, actual net together, and each player's team net with the other OFF
the floor (the raw on/off picture).

Usage: python src/synergy.py -> out/synergy_pairs.csv, out/synergy_trios.csv, out/synergy_meta.json
"""
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
K_PAIR, K_TRIO = 3000.0, 3000.0
MIN_PAIR, MIN_TRIO = 1500, 1500


def main():
    global K_PAIR, K_TRIO
    pair_parts, trio_parts, names = [], [], {}
    for f in sorted(glob.glob(str(ROOT / "data" / "stints" / "stints_*.parquet"))):
        s = pd.read_parquet(f)
        season = int(s.season.iloc[0])
        wpath = OUT / f"war_{season}.csv"
        if not wpath.exists():
            continue
        w = pd.read_csv(wpath)
        names.update(dict(zip(w.player_id, w.name)))
        rat = pd.Series((w.O + w.D).to_numpy(), index=w.player_id)
        H = s[[f"h{i}" for i in range(5)]].to_numpy().astype(np.int64)
        A = s[[f"a{i}" for i in range(5)]].to_numpy().astype(np.int64)
        look = lambda M: pd.Series(M.ravel()).map(rat).fillna(-2.0).to_numpy().reshape(M.shape)
        exp_h = look(H).sum(axis=1) - look(A).sum(axis=1)
        poss = (s.poss_h + s.poss_a).to_numpy() / 2
        ok = poss > 0
        act_h = np.where(ok, 100 * (s.pts_h - s.pts_a).to_numpy() / np.where(ok, poss, 1), 0)
        res = act_h - exp_h
        hca = np.average(res[ok], weights=poss[ok])
        res, act_h = res - hca, act_h - hca
        for L, r_, act, tm in ((H, res, act_h, s.home_team.to_numpy()), (A, -res, -act_h, s.away_team.to_numpy())):
            L, r_, act, tm, p_ = np.sort(L[ok], axis=1), r_[ok], act[ok], tm[ok], poss[ok]
            for n, parts in ((2, pair_parts), (3, trio_parts)):
                for combo in itertools.combinations(range(5), n):
                    key = np.zeros(len(L), dtype=np.int64)
                    for c_ in combo:
                        key = key * (1 << 21) + L[:, c_]
                    parts.append(pd.DataFrame({"key": key, "poss": p_, "r": p_ * r_, "a": p_ * act,
                                               "season": season, "team": tm}).groupby(
                        ["key", "season", "team"], as_index=False)[["poss", "r", "a"]].sum())
        print(season, flush=True)

    def collect(parts):
        d = pd.concat(parts).groupby(["key", "season", "team"], as_index=False)[["poss", "r", "a"]].sum()
        g = d.groupby("key")
        return pd.DataFrame({"poss": g.poss.sum(), "r": g.r.sum(), "a": g.a.sum(), "first": g.season.min(),
                             "last": g.season.max(), "seasons": g.season.nunique(),
                             "team_ids": g.team.apply(lambda x: sorted(set(x)))})
    pair_acc, trio_acc = collect(pair_parts), collect(trio_parts)
    # reliability: does a pair's synergy repeat? odd vs even seasons, pairs with 2,000+ possessions in each half
    d = pd.concat(pair_parts).groupby(["key", "season"], as_index=False)[["poss", "r"]].sum()
    d["half"] = d.season % 2
    h = d.groupby(["key", "half"])[["poss", "r"]].sum().unstack()
    h = h[(h["poss"][0] >= 2000) & (h["poss"][1] >= 2000)]
    s0, s1 = h["r"][0] / (h["poss"][0] + K_PAIR), h["r"][1] / (h["poss"][1] + K_PAIR)
    rel = {"split_half_r": float(np.corrcoef(s0, s1)[0, 1]), "n_pairs": int(len(h))}
    # shrinkage chosen by prediction: K that best predicts each pair's OTHER half (raw), both directions
    best = None
    for K in (3000, 10000, 20000, 40000, 80000, 160000):
        err = 0.0
        for a_, b_ in ((0, 1), (1, 0)):
            pred = h["r"][a_] / (h["poss"][a_] + K)
            raw = h["r"][b_] / h["poss"][b_]
            err += float(np.sum(h["poss"][b_] * (raw - pred) ** 2))
        rel[f"err_K{K}"] = err
        if best is None or err < best[1]:
            best = (K, err)
    rel["K_chosen"] = best[0]
    K_PAIR = K_TRIO = float(best[0])

    print("reliability:", rel, flush=True)

    teams = pd.read_parquet(ROOT / "data" / "coaches" / "coach_games.parquet")[["team_id", "franchise"]].drop_duplicates()
    tname = dict(zip(teams.team_id, teams.franchise))

    def frame(acc, k, min_poss, n):
        acc = acc[acc.poss >= min_poss]
        rows = []
        for key, r in acc.iterrows():
            ids, kk = [], int(key)
            for _ in range(n):
                ids.append(kk % (1 << 21))
                kk //= (1 << 21)
            ids = ids[::-1]
            rows.append({**{f"p{i + 1}": names.get(ids[i], str(ids[i])) for i in range(n)},
                         **{f"id{i + 1}": ids[i] for i in range(n)},
                         "poss": r.poss, "synergy": r.r / (r.poss + k), "synergy_raw": r.r / r.poss,
                         "net_together": r.a / r.poss, "first": int(r["first"]), "last": int(r["last"]),
                         "seasons": int(r.seasons), "teams": "/".join(sorted({tname.get(t, str(t)) for t in r.team_ids}))})
        df = pd.DataFrame(rows)
        df["se"] = 13.0 / np.sqrt(df.poss)     # ~13 points per 100 of stint-level noise per sqrt(possession)
        # total effect: per-100 synergy x possessions together / 100 = extra points produced together
        df["extra_points"] = df.synergy * df.poss / 100
        return df.sort_values("synergy", ascending=False)

    pairs = frame(pair_acc, K_PAIR, MIN_PAIR, 2)
    trios = frame(trio_acc, K_TRIO, MIN_TRIO, 3)
    pairs.to_csv(OUT / "synergy_pairs.csv", index=False, float_format="%.3f")
    trios.to_csv(OUT / "synergy_trios.csv", index=False, float_format="%.3f")
    # reliability: does a pair's synergy repeat? split each pair's seasons odd/even
    meta = {"pairs": int(len(pairs)), "trios": int(len(trios)), "k_pair": K_PAIR, "min_poss": MIN_PAIR, "reliability": rel}
    (OUT / "synergy_meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 220)
    show = ["p1", "p2", "teams", "first", "last", "poss", "net_together", "synergy", "extra_points"]
    print(pairs[pairs.poss >= 5000].head(30)[show].round(2).to_string(index=False))
    print(pairs.sort_values("extra_points", ascending=False).head(20)[show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
