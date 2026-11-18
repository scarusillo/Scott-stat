"""Schemes — which coaching styles go with better results?

Per coach (164+ games): style fingerprint (z-scores within season: pace, 3PA
rate, rim rate, assisted-make rate, rotation depth, substitution churn,
opponent 3PA rate allowed, opponent rim rate allowed) against his measured
effects: offense, defense, net (points per 100 his players play above their
own career levels), plus Rotation IQ and Closing IQ.
For each outcome: a weighted regression on all eight traits together
(weights = 1/SE^2), coefficient per 1 SD of the trait.
These are associations across coaches, not experiments: a style can follow
from the roster as much as cause the result.

Usage: python src/schemes.py   -> out/scheme_effects.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "out"
TRAITS = ["pace", "three", "rim", "astd", "depth", "churn", "opp_three", "opp_rim"]


def main():
    st = pd.read_csv(OUT / "coach_style.csv", index_col=0)
    ce = pd.read_csv(OUT / "coach_effects.csv").set_index("coach_id")
    rot = pd.read_csv(OUT / "coach_rotation.csv", index_col=0)
    d = st[TRAITS].join(ce[["O", "D", "net", "se_O", "se_D", "se_net", "games"]]).join(rot[["rotation_wins_82", "closing"]])
    d = d[d.games >= 164].dropna()
    Z = (d[TRAITS] - d[TRAITS].mean()) / d[TRAITS].std()
    rows = []
    for outcome, se in (("O", "se_O"), ("D", "se_D"), ("net", "se_net"), ("rotation_wins_82", None), ("closing", None)):
        w = 1 / d[se] ** 2 if se else np.sqrt(d.games)
        X = np.c_[np.ones(len(d)), Z.to_numpy()]
        W = np.diag(w / w.mean())
        cov = np.linalg.inv(X.T @ W @ X)
        b = cov @ X.T @ W @ d[outcome].to_numpy()
        res = d[outcome].to_numpy() - X @ b
        s2 = np.sum(np.diag(W) * res ** 2) / (len(d) - X.shape[1])
        seb = np.sqrt(np.diag(cov) * s2)
        r2 = 1 - np.sum(np.diag(W) * res ** 2) / np.sum(np.diag(W) * (d[outcome] - np.average(d[outcome], weights=np.diag(W))) ** 2)
        for i, t in enumerate(TRAITS):
            rows.append({"outcome": outcome, "trait": t, "coef_per_sd": b[i + 1], "se": seb[i + 1], "t": b[i + 1] / seb[i + 1],
                         "r_simple": float(np.corrcoef(Z[t], d[outcome])[0, 1]), "coaches": len(d), "r2_all_traits": r2})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "scheme_effects.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 200)
    print(r.pivot(index="trait", columns="outcome", values="coef_per_sd").round(2).to_string())
    print(r.pivot(index="trait", columns="outcome", values="t").round(1).to_string())
    print(r.groupby("outcome").r2_all_traits.first().round(2).to_dict(), "coaches", len(d))


if __name__ == "__main__":
    main()
