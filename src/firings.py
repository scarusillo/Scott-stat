"""Mid-season firings — the natural experiment that tests the coach model.

Same roster, same season, new coach. For every mid-season change (both coaches
15+ games), measure the team's "coaching residual" before and after: how much
better/worse it played than its on-court players' coach-free talent predicts
(lineup by lineup, so injuries and trades are controlled for).

Then the out-of-sample test: refit the coach model with every one of those
team-seasons REMOVED, and ask whether (new coach effect − old coach effect),
learned only from other seasons, predicts the residual change.
Pure noise -> slope ~0, r ~0. A perfect model -> slope ~1.

Usage: python src/firings.py   -> out/firings.csv, out/firings_meta.json
"""
import json

import numpy as np
import pandas as pd

from coach_effects import OUT, build, load, solve

MIN_GAMES = 15


def events(cg):
    ev = []
    for (season, team), d in cg.sort_values("game_id").groupby(["season", "team_id"]):
        seq = d.coach_id.to_numpy()
        runs = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1]])
        bounds = list(runs) + [len(seq)]
        blocks = [(seq[bounds[i]], bounds[i + 1] - bounds[i], d.game_id.iloc[bounds[i]]) for i in range(len(runs))]
        for (c0, n0, _), (c1, n1, g1) in zip(blocks, blocks[1:]):
            if n0 >= MIN_GAMES and n1 >= MIN_GAMES and c0 != c1:
                ev.append({"season": season, "team_id": team, "old": c0, "new": c1,
                           "old_games": n0, "new_games": n1, "switch_game": g1})
    return pd.DataFrame(ev)


def main():
    s, cg = load()
    ev = events(cg)
    held = set(zip(ev.season, ev.team_id))
    mask = ~(pd.Series(list(zip(s.season, s.home_team))).isin(held).to_numpy()
             | pd.Series(list(zip(s.season, s.away_team))).isin(held).to_numpy())
    print(f"{len(ev)} mid-season changes; refitting without {(~mask).sum()} stints", flush=True)

    # 1) coach effects learned WITHOUT the held-out team-seasons
    Xo, yo, wo, peno, mo = build(s[mask].reset_index(drop=True), 3000)
    bo, _ = solve(Xo, yo, wo, peno)
    P, C = mo["P"], mo["C"]
    ce = pd.Series(bo[2 * P:2 * P + C] + bo[2 * P + C:2 * P + 2 * C], index=mo["coaches"])
    pl = pd.DataFrame({"O": bo[:P], "D": bo[P:2 * P]}, index=mo["pids"])

    # 2) residuals in the held-out seasons, using the SAME refit's players + experience
    #    curve (coach terms zeroed): build the full design and plug the refit's
    #    coefficients in by name, so player/coach column orders can differ.
    X, y, w, pen, m = build(s, 3000)
    Pf, Cf, E = m["P"], m["C"], m["E"]
    beta = np.zeros(X.shape[1])
    po = pl.reindex(m["pids"])
    beta[:Pf] = po.O.fillna(0).to_numpy()
    beta[Pf:2 * Pf] = po.D.fillna(0).to_numpy()
    oE = 2 * Pf + 2 * Cf
    beta[oE:oE + 2 * E] = bo[2 * P + 2 * C:2 * P + 2 * C + 2 * E]
    beta[-1] = bo[-1]
    resid = y - X @ beta
    n = len(s)
    keep = np.r_[s.poss_h, s.poss_a] > 0
    off_team = np.r_[s.home_team, s.away_team][keep]
    def_team = np.r_[s.away_team, s.home_team][keep]
    game = np.r_[s.game_id, s.game_id][keep]
    season = np.r_[s.season, s.season][keep]
    R = pd.DataFrame({"season": season, "game_id": game, "off": off_team, "def": def_team, "r": resid, "w": w})

    rows = []
    for e in ev.itertuples():
        d = R[(R.season == e.season) & ((R.off == e.team_id) | (R["def"] == e.team_id))]
        sign = np.where(d.off == e.team_id, 1, -1)   # defense rows: opponent over-performing is bad
        g = cg[(cg.season == e.season) & (cg.team_id == e.team_id)].set_index("game_id").coach_id
        who = d.game_id.map(g).to_numpy()
        res = {}
        for tag, c in (("before", e.old), ("after", e.new)):
            k = who == c
            res[tag] = np.sum(sign[k] * d.r.to_numpy()[k] * d.w.to_numpy()[k]) / (np.sum(d.w.to_numpy()[k]) / 2)
        rows.append({**e._asdict(), "resid_before": res["before"], "resid_after": res["after"],
                     "actual_change": res["after"] - res["before"],
                     "pred_change": ce.get(e.new, 0.0) - ce.get(e.old, 0.0),
                     "new_has_other_data": e.new in ce.index and abs(ce.get(e.new, 0)) > 0})
    f = pd.DataFrame(rows).drop(columns=["Index"])
    names = cg.groupby("coach_id").coach.first()
    f.insert(2, "old_coach", f.old.map(names))
    f.insert(3, "new_coach", f.new.map(names))
    f.to_csv(OUT / "firings.csv", index=False, float_format="%.3f")

    def stats(d):
        wts = np.minimum(d.old_games, d.new_games)
        b = np.polyfit(d.pred_change, d.actual_change, 1, w=np.sqrt(wts))
        return {"events": int(len(d)), "r": float(np.corrcoef(d.pred_change, d.actual_change)[0, 1]),
                "slope": float(b[0]), "mean_actual_change": float(d.actual_change.mean()),
                "mse_model": float(np.mean((d.actual_change - d.pred_change) ** 2)),
                "mse_no_effect": float(np.mean((d.actual_change - d.actual_change.mean()) ** 2))}
    # both coaches need data from other seasons for a real prediction
    both = f[(f.pred_change != 0) & f.old.isin(ce.index) & f.new.isin(ce.index)]
    known = both[(ce.reindex(both.old).abs().to_numpy() > 0.2) | (ce.reindex(both.new).abs().to_numpy() > 0.2)]
    meta = {"all": stats(f), "both_known": stats(both), "clear_prediction": stats(known),
            "resid_before_mean": float(f.resid_before.mean()), "resid_after_mean": float(f.resid_after.mean())}
    (OUT / "firings_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    show = f.reindex(f.actual_change.abs().sort_values(ascending=False).index).head(10)
    print(show[["season", "old_coach", "new_coach", "old_games", "new_games", "resid_before",
                "resid_after", "pred_change"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
