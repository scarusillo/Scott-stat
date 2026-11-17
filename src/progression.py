"""Progression — every player's year-by-year arc, the biggest risers and fallers,
and the coaches players rise (or fall) under.

1. Year-over-year change in impact for every player with 1,000+ possessions in
   both seasons. Plus tier (1996-97 on, season ratings with standard errors) and
   Classic tier (1951-52 on). A change counts as "big" only if it clears its own
   uncertainty (|change| > 2 x combined SE where SEs exist).
2. Career arcs: start (first two qualifying seasons), peak (best 3 consecutive),
   end (last two) -> rise to peak, fall from peak.
3. Coach arrivals (Plus era): every time a player starts playing for a NEW coach
   (trade, free agency or a coaching change), his change in impact versus what
   is expected for his experience and starting level. A coach's arrival lift is
   the shrunk average over everyone who arrived (empirical Bayes), overall and by
   player type, with an even/odd-season reliability check.

Usage: python src/progression.py   -> out/progression_changes.csv, out/progression_arcs.csv,
                                      out/coach_arrivals.csv, out/coach_arrivals_by_type.csv,
                                      out/progression_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coach_effects import load
from development import coach_of_player_seasons
from talent import first_seen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MIN_POSS = 1000


def changes_plus():
    s = pd.read_csv(OUT / "scott_seasons.csv")
    s = s[s.poss >= MIN_POSS][["season", "player_id", "name", "team", "net", "se_net", "O", "D", "poss"]]
    nxt = s.assign(season=s.season - 1)
    j = s.merge(nxt, on=["player_id", "season"], suffixes=("_0", "_1"))
    j["change"] = j.net_1 - j.net_0
    j["se"] = np.hypot(j.se_net_0, j.se_net_1)
    j["clear"] = j.change.abs() > 2 * j.se
    j["tier"] = "plus"
    j["to_season"] = j.season + 1
    return j.rename(columns={"name_0": "player", "team_0": "team_from", "team_1": "team_to"})


def changes_classic():
    a = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1951, 2026)], ignore_index=True)
    a = a[a.poss >= MIN_POSS][["season_start", "player_id", "player", "team", "net_classic", "poss", "band"]]
    a = a.rename(columns={"season_start": "season", "net_classic": "net"})
    nxt = a.assign(season=a.season - 1)
    j = a.merge(nxt, on=["player_id", "season"], suffixes=("_0", "_1"))
    j["change"] = j.net_1 - j.net_0
    j["se"] = np.sqrt(2) * 1.59        # Classic's season error SD (classic_meta resid_sd)
    j["clear"] = j.change.abs() > 2 * j.se
    j["tier"] = "classic"
    j["to_season"] = j.season + 1
    return j.rename(columns={"player_0": "player", "team_0": "team_from", "team_1": "team_to"})


def arcs(df, tier, id_col="player_id", name_col="name"):
    rows = []
    for pid, x in df.groupby(id_col):
        x = x.sort_values("season")
        if len(x) < 5:
            continue
        roll = x.net.rolling(3).mean()
        i = int(np.nanargmax(roll.to_numpy()))
        peak = float(roll.iloc[i])
        rows.append({"tier": tier, "player_id": pid, "player": x[name_col].iloc[-1],
                     "first": int(x.season.min()), "last": int(x.season.max()), "seasons": len(x),
                     "start": float(x.net.iloc[:2].mean()), "peak3": peak,
                     "peak_years": f"{int(x.season.iloc[i - 2])}-{(int(x.season.iloc[i]) + 1) % 100:02d}",
                     "end": float(x.net.iloc[-2:].mean())})
    a = pd.DataFrame(rows)
    a["rise_to_peak"] = a.peak3 - a.start
    a["fall_from_peak"] = a.end - a.peak3
    return a


def coach_arrivals():
    s, cg = load()
    ch = coach_of_player_seasons(s)                     # (season, player) -> main coach, share
    fs = first_seen(s)
    war = pd.concat([pd.read_csv(OUT / f"war_{y}.csv").assign(season=y) for y in range(1996, 2025)])
    war = war[war.poss >= MIN_POSS].set_index(["season", "player_id"])[["name", "net", "poss"]]
    a = war.join(ch[["coach_id", "share"]]).reset_index()
    b = a.assign(season=a.season - 1)
    p = a.merge(b, on=["season", "player_id"], suffixes=("", "_next"))
    p = p[(p.coach_id != p.coach_id_next) & (p.share_next >= 0.7)]          # arrived under a new coach
    p["exp_next"] = (p.season + 1 - p.player_id.map(fs)).clip(upper=15)
    p["censored"] = (p.player_id.map(fs) == 1996).astype(int)
    p["delta"] = p.net_next - p.net
    X = pd.get_dummies(p.exp_next.astype(int).astype(str) + "_" + p.censored.astype(str), prefix="e").astype(float)
    X["start"] = p.net
    beta, *_ = np.linalg.lstsq(X.to_numpy(), p.delta.to_numpy(), rcond=None)
    p["resid"] = p.delta - X.to_numpy() @ beta
    arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet").set_index(["season", "player_id"]).archetype
    p["type"] = arch.reindex(pd.MultiIndex.from_arrays([p.season + 1, p.player_id])).to_numpy()

    def eb(g, sig2, tau2):
        n = g["count"]
        g["arrival_lift"] = g["mean"] * n / (n + sig2 / tau2)
        g["se"] = np.sqrt(1 / (n / sig2 + 1 / tau2))
        return g

    g = p.groupby("coach_id_next").resid.agg(["mean", "count"])
    sig2 = p.resid.var()
    tau2 = max(1e-6, np.average((g["mean"] - p.resid.mean()) ** 2 - sig2 / g["count"], weights=g["count"]))
    g = eb(g, sig2, tau2)
    names = cg.groupby("coach_id").coach.first()
    g.insert(0, "coach", g.index.map(names))
    best = p.sort_values("resid", ascending=False).groupby("coach_id_next").head(1).set_index("coach_id_next")
    g["best_arrival"] = best.name + " (" + (best.season + 1).astype(str) + ", " + best.delta.round(1).map("{:+.1f}".format) + ")"
    g = g.rename(columns={"mean": "raw_mean", "count": "arrivals"}).sort_values("arrival_lift", ascending=False)
    # reliability
    h = p.assign(h=p.season % 2).groupby(["coach_id_next", "h"]).resid.agg(["mean", "count"]).unstack()
    ok = (h["count"][0] >= 5) & (h["count"][1] >= 5)
    rel = float(np.corrcoef(h["mean"][0][ok], h["mean"][1][ok])[0, 1])
    # by player type
    gt = p.dropna(subset=["type"]).groupby(["coach_id_next", "type"]).resid.agg(["mean", "count"]).reset_index()
    gt["arrival_lift"] = gt["mean"] * gt["count"] / (gt["count"] + sig2 / tau2)
    gt.insert(1, "coach", gt.coach_id_next.map(names))
    p_out = p[["season", "player_id", "name", "coach_id", "coach_id_next", "net", "net_next", "delta", "resid", "type"]]
    p_out = p_out.assign(coach_from=p_out.coach_id.map(names), coach_to=p_out.coach_id_next.map(names),
                         to_season=p_out.season + 1)
    p_out.to_csv(OUT / "coach_arrival_events.csv", index=False, float_format="%.3f")
    meta = {"arrivals": int(len(p)), "coaches": int(len(g)), "sd_true_coach_effect": float(np.sqrt(tau2)),
            "sd_noise_per_arrival": float(np.sqrt(sig2)), "split_half_r": rel, "split_half_coaches": int(ok.sum()),
            "expected_change_from_start_level": float(beta[-1])}
    return g, gt, meta


def main():
    ch = pd.concat([changes_plus(), changes_classic()], ignore_index=True)
    cols = ["tier", "player_id", "player", "season", "to_season", "team_from", "team_to", "net_0", "net_1",
            "change", "se", "clear"]
    ch[cols].to_csv(OUT / "progression_changes.csv", index=False, float_format="%.3f")
    sp = pd.read_csv(OUT / "scott_seasons.csv")
    sp = sp[sp.poss >= MIN_POSS]
    ca = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1951, 2026)], ignore_index=True)
    ca = ca[ca.poss >= MIN_POSS].rename(columns={"season_start": "season", "net_classic": "net"})
    ar = pd.concat([arcs(sp, "plus"), arcs(ca, "classic", name_col="player")], ignore_index=True)
    ar.to_csv(OUT / "progression_arcs.csv", index=False, float_format="%.3f")
    g, gt, meta = coach_arrivals()
    g.to_csv(OUT / "coach_arrivals.csv", float_format="%.3f")
    gt.to_csv(OUT / "coach_arrivals_by_type.csv", index=False, float_format="%.3f")
    (OUT / "progression_meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 220)
    pl = ch[(ch.tier == "plus") & ch.clear]
    show = ["player", "season", "team_from", "team_to", "net_0", "net_1", "change", "se"]
    print("PLUS biggest one-year rises\n", pl.nlargest(10, "change")[show].round(2).to_string(index=False))
    print("PLUS biggest one-year falls\n", pl.nsmallest(10, "change")[show].round(2).to_string(index=False))
    print("CAREER rise to peak (Plus)\n", ar[ar.tier == "plus"].nlargest(10, "rise_to_peak").round(2).to_string(index=False))
    print(json.dumps(meta))
    print("COACH arrival lift\n", g[g.arrivals >= 15].head(12).round(2).to_string())
    print(g[g.arrivals >= 15].tail(6).round(2).to_string())


if __name__ == "__main__":
    main()
