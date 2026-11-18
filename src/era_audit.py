"""Era audit — is Classic fair to players from slower, pre-three-point eras, and to
players from eras with fewer recorded stats?

  1. Skill-by-skill breakdown of each great's box estimate over his peak
     (pace is already removed: every stat is per 100 possessions vs that
     season's league average).
  2. The same peaks under ONE yardstick for every era (skill prices fit on all
     play-by-play seasons equally) vs the era-matched prices in use.
  3. Information compression: on 2014-2025 seasons, how much narrower are the
     predictions when only the 1951-73 stat set is available, and does the
     narrower model still put the right players on top?

Usage: python src/era_audit.py   -> out/era_audit.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import classic as K

OUT = Path(__file__).resolve().parents[1] / "out"
PEAKS = {"Michael Jordan": (1988, 1992), "LeBron James": (2009, 2013), "Stephen Curry": (2015, 2019),
         "Nikola Jokić": (2022, 2026), "Magic Johnson": (1987, 1991), "Larry Bird": (1984, 1988),
         "Kareem Abdul-Jabbar": (1974, 1978), "Wilt Chamberlain": (1963, 1967), "Bill Russell": (1961, 1965)}


def prepared():
    b, per_team, ts = K.load_box()
    d = K.features(b)
    d["key"] = d.player.map(K.norm)
    p = K.plus_targets()
    dup = p.duplicated(["season", "key"], keep=False)
    d = d.merge(p[~dup].drop(columns="name"), on=["season", "key"], how="left")
    ref = d[d.nba_id.notna() & (d.plus_poss >= 1000)]
    for f_ in K.BOX + ["reb"]:
        d["f_" + f_] = d["f_" + f_].clip(ref["f_" + f_].min(), ref["f_" + f_].max())
    return d, per_team, ts


def main():
    d, per_team, ts = prepared()
    meta = json.load(open(OUT / "classic_meta.json"))
    tau, alpha = meta["validation"]["tau"], meta["validation"]["alpha"]
    credit = meta["validation"].get("credit")
    base = d.nba_id.notna() & (d.plus_poss >= 250)
    out = {}
    versions = {"era_matched": K.team_adjust(K.era_fit(d, base, tau), per_team, ts, alpha, credit),
                "one_yardstick": K.team_adjust(K.era_fit(d, base, float("inf")), per_team, ts, alpha, credit)}
    games = pd.read_csv(OUT / "games_ratings.csv")
    games["shift"] = (games.O_g + games.D_g) - (games.O_0 + games.D_0)
    shift = games.set_index(["season", "player_id"])["shift"]

    # 1 + 2: peaks and breakdowns
    tr = d[base]
    rows = {}
    for name, (a, z) in PEAKS.items():
        row = {}
        for v, f in versions.items():
            x = f[(f.player == name) & f.season.between(a, z)]
            w = x.poss
            row[f"{v}_box"] = float(np.average(x.box_O + x.box_D, weights=w))
            row[f"{v}_box_plus_team"] = float(np.average(x.O_c + x.D_c, weights=w))
            sh = shift.reindex(pd.MultiIndex.from_arrays([x.season, x.player_id])).fillna(0).to_numpy()
            row[f"{v}_final"] = float(np.average(x.O_c + x.D_c + sh, weights=w))
        x = d[(d.player == name) & d.season.between(a, z)]
        band = K.band_of(int(x.season.iloc[0]))
        feats = K.BANDS[band][2] + K.EXTRA
        # skill contributions at era-matched prices for the peak's middle season
        anchor = min(max(int(np.median(x.season)), int(tr.season.min())), int(tr.season.max()))
        wts = tr.plus_poss * np.exp(-np.abs(tr.season - anchor) / tau)
        X = tr[["f_" + c for c in feats]].to_numpy()
        bO, bD = K.wls(X, tr.O.to_numpy(), wts.to_numpy()), K.wls(X, tr.D.to_numpy(), wts.to_numpy())
        coef = bO[1:] + bD[1:]
        vals = np.average(x[["f_" + c for c in feats]], axis=0, weights=x.poss)
        row["skills"] = {c: round(float(k * v), 2) for c, k, v in zip(feats, coef, vals)}
        row["league_relative_3pa_rate"] = round(float(np.average(x.f_fg3a, weights=x.poss)), 2)
        row["band"] = band
        rows[name] = row
    out["peaks"] = rows

    # 3: compression with fewer recorded stats (2014-2025 seasons, honest: prices fit <= 2013)
    fwd = base & (d.season <= 2013)
    test = d.nba_id.notna() & (d.season >= 2014) & (d.plus_poss >= 1000)
    comp = {}
    for band in ("full", "lite"):
        f = K.team_adjust(K.era_fit(d, fwd, tau, force_band=band), per_team, ts, alpha)
        x = f[test]
        pred = x.O_c + x.D_c
        top = x.nlargest(50, "net")
        comp[band] = {"sd_pred": float(pred.std()), "sd_plus": float(x.net.std()), "slope_plus_on_pred": float(np.polyfit(pred, x.net, 1)[0]),
                      "top50_plus_mean": float(top.net.mean()), "top50_pred_mean": float((top.O_c + top.D_c).mean()),
                      "top50_by_pred_plus_mean": float(x.loc[pred.nlargest(50).index, "net"].mean())}
    out["compression"] = comp
    (OUT / "era_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
