"""Bundle everything the Scott Stat site needs into site/data.json.

Usage: python src/build_site_data.py   (after composite.py, player_coach.py, rotation.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SITE = ROOT / "site"
MIN_CAREER_POSS = 0          # every player with at least one 1,000-possession season


def r(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def main():
    meta = json.load(open(OUT / "scott_meta.json"))
    plus = pd.read_csv(OUT / "scott_careers_plus.csv")
    seasons = pd.read_csv(OUT / "scott_seasons.csv")
    classic = pd.read_csv(OUT / "scott_careers_classic.csv")
    ca = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1951, 2026)], ignore_index=True)
    pcoach = pd.read_csv(OUT / "player_coach.csv")
    coaches = pd.read_csv(OUT / "coach_effects.csv")
    rot = pd.read_csv(OUT / "coach_rotation.csv")
    carch = pd.read_csv(OUT / "coach_archetype.csv")
    arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet")

    keep = plus[plus.poss >= MIN_CAREER_POSS]
    # careers that began before play-by-play: Plus measures only their later seasons
    cfirst = classic.assign(k=classic.player.str.lower()).groupby("k")["first"].min()
    S = seasons[seasons.player_id.isin(keep.player_id) & (seasons.poss >= 500)]
    main_type = arch.groupby("player_id").archetype.agg(lambda x: x.mode().iat[0])
    players = []
    for p in keep.itertuples():
        x = S[S.player_id == p.player_id].sort_values("season")
        eras = []
        for part in str(p.eras).split(" | "):
            if ": " in part:
                head, v = part.rsplit(": ", 1)
                eras.append([head, float(v)])
        players.append({
            "id": int(p.player_id), "name": p.player, "first": int(p.first), "last": int(p.last),
            "partial_from": int(cfirst.get(p.player.lower(), p.first)) if cfirst.get(p.player.lower(), p.first) < p.first else None,
            "seasons": int(p.seasons), "poss": int(p.poss), "type": main_type.get(p.player_id),
            "scott": r(p.scott_stat), "impact": r(p.impact), "se": r(p.impact_se),
            "off": r(p.offense), "def": r(p.defense), "prod": r(p.production), "lift": r(p.lift),
            "clutch": r(p.clutch_rise), "clutch_se": r(p.clutch_se),
            "po": r(p.playoff_elevation), "po_se": r(p.playoff_elevation_se), "po_share": r(p.playoff_share, 3),
            "best5": r(p.best5), "best5_years": p.best5_years if isinstance(p.best5_years, str) else None,
            "peak5": r(p.scott_peak5), "best": r(p.best_season), "best_year": p.best_season_year,
            "war": r(p.career_WAR, 1), "cwar": r(p.career_cWAR, 1), "eras": eras,
            "s": [[int(q.season), q.team if isinstance(q.team, str) else "", r(q.net), r(q.se_net), r(q.O), r(q.D),
                   r(q.production), r(q.lift), r(q.WAR, 1), r(q.ppg, 1)] for q in x.itertuples()],
        })
    ch = pd.read_csv(OUT / "championships_careers.csv", index_col=0)
    sc = pd.read_csv(OUT / "scoring_careers.csv").set_index("player_id")
    wn = pd.read_csv(OUT / "winning_careers.csv").set_index("player_id")
    ckeep = classic[classic.poss >= MIN_CAREER_POSS]
    classic_rows = []
    for p in ckeep.itertuples():
        x = ca[(ca.player_id == p.player_id) & (ca.poss >= 500)].sort_values("season_start")
        classic_rows.append({
            "id": p.player_id, "name": p.player, "first": int(p.first), "last": int(p.last), "seasons": int(p.seasons),
            "impact": r(p.impact), "off": r(p.offense), "def": r(p.defense), "prod": r(p.production), "lift": r(p.lift),
            "best": r(p.best_season), "best_year": p.best_season_year, "best5": r(p.best5),
            "best5_years": p.best5_years if isinstance(p.best5_years, str) else None, "war": r(p.career_WAR, 1),
            "lite": r(p.lite_share, 2),
            "prod5": r(p.production_peak5), "prod5_years": p.production_peak5_years if isinstance(p.production_peak5_years, str) else None,
            "ind": r(p.peak_individual), "team": r(p.peak_team_credit),
            "best5po": r(p.best5_with_playoffs), "porise": r(p.playoff_rise), "pwar5": r(p.peak_wins_per_season, 1), "pwar5_era": r(p.peak_wins_in_era, 1), "pwar5_years": p.peak_wins_years if isinstance(p.peak_wins_years, str) else None,
            "pos": (lambda z: "G" if z in ("PG", "SG") else "W" if z == "SF" else "B")(str(x.pos.mode().iat[0]).split("-")[0]) if x.pos.notna().any() else None,
            "dom5": r(p.dominance_peak5), "dom5_years": p.dominance_peak5_years if isinstance(p.dominance_peak5_years, str) else None,
            "no1": int(p.seasons_ranked_1), "top3": int(p.seasons_top3),
            "titles": int(ch.titles.get(p.player, 0)) if p.player in ch.index else 0,
            "tshare": r(ch.title_shares_league_adj.get(p.player)) if p.player in ch.index else 0,
            "score5": r(sc.scoring5.get(p.player_id)) if p.player_id in sc.index else None,
            "score5_years": sc.scoring5_years.get(p.player_id) if p.player_id in sc.index and isinstance(sc.scoring5_years.get(p.player_id), str) else None,
            "stitles": int(sc.scoring_titles.get(p.player_id)) if p.player_id in sc.index else 0,
            "create5": r(sc.create5.get(p.player_id)) if p.player_id in sc.index else None,
            "winwins": r(wn.winning_wins_per_season.get(p.player_id), 2) if p.player_id in wn.index else None,
            "winwins_career": r(wn.winning_wins.get(p.player_id), 1) if p.player_id in wn.index else None,
            "winning": r(wn.winning.get(p.player_id)) if p.player_id in wn.index else None,
            "elev": r(wn.elevation.get(p.player_id)) if p.player_id in wn.index and pd.notna(wn.elevation.get(p.player_id)) else None,
            "twpct": r(wn.team_wpct.get(p.player_id), 3) if p.player_id in wn.index else None,
            "tshare_raw": r(ch.title_shares.get(p.player)) if p.player in ch.index else 0,
            "codds": r(ch.championships_added.get(p.player)) if p.player in ch.index else 0,
            "s": [[int(q.season_start), q.team, r(q.net_classic), r(q.O_classic), r(q.D_classic), r(q.WAR_classic, 1),
                   r(q.net_plus), {"full": "F", "no_turnovers": "T", "lite": "L"}.get(q.band, ""), (int(q.age) if q.age == q.age else None), r(q.box_O + q.box_D)] for q in x.itertuples()],
        })
    rot = rot.set_index("coach_id")
    style = pd.read_csv(OUT / "coach_style.csv", index_col=0)
    STYLE = ["pace", "three", "rim", "astd", "depth", "churn", "opp_three", "opp_rim"]
    csplit = pd.read_csv(OUT / "coach_split.csv", index_col=0)
    coach_rows = []
    for c in coaches[coaches.games >= 164].itertuples():
        cells = carch[(carch.coach_id == c.coach_id) & (carch.exposure >= 20000) & carch.broad]
        best = cells.sort_values("type_net", ascending=False)
        coach_rows.append({
            "id": c.coach_id, "name": c.name, "teams": c.teams, "first": int(c.first), "last": int(c.last),
            "seasons": int(c.seasons), "games": int(c.games), "win_pct": r(c.win_pct, 3),
            "off": r(c.O), "def": r(c.D), "net": r(c.net), "se": r(c.se_net),
            "rotation": r(rot.rotation_wins_82.get(c.coach_id)), "closing": r(rot.closing.get(c.coach_id)),
            "best_types": [[t.archetype, r(t.type_net)] for t in best.head(2).itertuples()],
            "worst_types": [[t.archetype, r(t.type_net)] for t in best.tail(1).itertuples()] if len(best) > 2 else [],
            "types": [[t.archetype, r(t.uplift), r(float(np.sqrt(t.se_type_O ** 2 + t.se_type_D ** 2 + t.se_main ** 2))),
                       int(t.exposure), int(t.n_players) if t.n_players == t.n_players else 0, t.top_player if isinstance(t.top_player, str) else "",
                       r(t.top_share), bool(t.broad)]
                      for t in carch[(carch.coach_id == c.coach_id) & (carch.exposure >= 8000)].itertuples()],
            "style": {k: r(style.at[c.coach_id, k]) for k in STYLE} if c.coach_id in style.index else None,
            "ind": r(csplit.production_raw.get(c.coach_id)), "indlift": r(csplit.lift_raw.get(c.coach_id)),
            "ind_s": r(csplit.production.get(c.coach_id)), "indlift_s": r(csplit.lift.get(c.coach_id)),
            "dev_n": int(csplit.player_seasons.get(c.coach_id, 0) or 0),
        })
    pairs = []
    pc = pcoach[pcoach.player_id.isin(keep.player_id)]
    # play style under that coach (most common archetype in those seasons) and position (G / W / B)
    ar_ = arch[["season", "player_id", "archetype", "poss"]]
    cls_ = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv", usecols=["nba_id", "pos", "poss"]) for y in range(1996, 2026)])
    cls_ = cls_.dropna(subset=["nba_id"])
    pos_ = cls_.assign(p=cls_.pos.astype(str).str.split("-").str[0]).groupby(["nba_id", "p"]).poss.sum().reset_index() \
        .sort_values("poss").groupby("nba_id").p.last()
    gwb = lambda z: "G" if z in ("PG", "SG") else "W" if z == "SF" else "B" if z in ("PF", "C") else ""
    for q in pc.itertuples():
        a_ = ar_[(ar_.player_id == q.player_id) & ar_.season.between(q.first_season, q.last_season)]
        style_ = a_.groupby("archetype").poss.sum().idxmax() if len(a_) else ""
        pairs.append([int(q.player_id), q.coach, int(q.first_season), int(q.last_season), int(q.shared_poss),
                      r(q.total_uplift_raw), r(q.raw_se), r(q.total_uplift_trend_adjusted), r(q.trend_adjusted_se),
                      style_, gwb(pos_.get(float(q.player_id), ""))])
    # synergy (synergy.py): duos and trios
    sp = pd.read_csv(OUT / "synergy_pairs.csv")
    st = pd.read_csv(OUT / "synergy_trios.csv")
    st["extra"] = st.synergy * st.poss / 100
    st = st[st.poss >= 3000].sort_values("extra", ascending=False).head(800)
    smeta = json.load(open(OUT / "synergy_meta.json"))
    syn = {"pairs": [[int(q.id1), int(q.id2), q.p1, q.p2, q.teams, int(q.first), int(q.last), int(q.poss), r(q.synergy), r(q.net_together)]
                     for q in sp.itertuples()],
           "trios": [[q.p1, q.p2, q.p3, q.teams, int(q.first), int(q.last), int(q.poss), r(q.synergy), r(q.net_together)]
                     for q in st.itertuples()],
           "reliability": r(smeta["reliability"]["split_half_r"]), "k": int(smeta["reliability"]["K_chosen"])}
    # every player-season of the last three seasons, for the GM finder
    recent = []
    val = pd.read_csv(OUT / "value_seasons.csv").set_index(["season", "player_id"])
    rs = seasons[(seasons.season >= seasons.season.max() - 2) & (seasons.poss >= 1500)]
    arch_ix = arch.set_index(["season", "player_id"]).archetype
    for q in rs.itertuples():
        recent.append([int(q.player_id), q.name, int(q.season), q.team if isinstance(q.team, str) else "", r(q.net),
                       r(q.se_net), r(q.O), r(q.D), r(q.production), r(q.lift), r(q.ppg, 1), int(q.poss),
                       arch_ix.get((q.season, q.player_id)),
                       r(val.salary.get((q.season, q.player_id), np.nan) / 1e6, 1), r(val.surplus.get((q.season, q.player_id), np.nan) / 1e6, 1),
                       r(q.WAR, 1)])
    # progression, coach arrivals, scenarios
    chg = pd.read_csv(OUT / "progression_changes.csv")
    arcs = pd.read_csv(OUT / "progression_arcs.csv")
    def lst(df, cols, n=40):
        return [[(None if pd.isna(v) else (round(float(v), 2) if isinstance(v, (float, np.floating)) else
                 (int(v) if isinstance(v, (np.integer,)) else v))) for v in row] for row in df[cols].head(n).itertuples(index=False)]
    ccols = ["player_id", "player", "season", "team_from", "team_to", "net_0", "net_1", "change", "se"]
    acols = ["player_id", "player", "first", "last", "start", "peak3", "peak_years", "end", "rise_to_peak", "fall_from_peak"]
    prog = {}
    for tier in ("plus", "classic"):
        c = chg[(chg.tier == tier) & chg.clear]
        a = arcs[arcs.tier == tier]
        prog[tier] = {"rises": lst(c.sort_values("change", ascending=False), ccols),
                      "falls": lst(c.sort_values("change"), ccols),
                      "arc_rise": lst(a.sort_values("rise_to_peak", ascending=False), acols),
                      "arc_fall": lst(a.sort_values("fall_from_peak"), acols)}
    arr = pd.read_csv(OUT / "coach_arrivals.csv")
    arr = arr[arr.arrivals >= 15]
    prog["arrivals"] = lst(arr, ["coach", "arrivals", "raw_mean", "arrival_lift", "se", "best_arrival"], 200)
    prog["arrivals_meta"] = json.load(open(OUT / "progression_meta.json"))
    sm = pd.read_csv(OUT / "scenario_matrix.csv")
    sb = pd.read_csv(OUT / "scenario_best_coaches.csv")
    scen = {"matrix": lst(sm, ["coach_group", "archetype", "relative_uplift", "se", "z", "coaches", "coach_list"], 500),
            "best": lst(sb, ["archetype", "coach", "uplift", "se", "exposure", "n_players", "top_player", "top_share"], 500)}
    val = {}
    for name in ("classic_meta", "clutch_meta", "playoffs_meta", "coach_rotation_meta", "breakdown_coefs", "firings_meta",
                 "player_coach_meta"):
        f = OUT / f"{name}.json"
        if f.exists():
            val[name] = json.load(open(f))
    ag = pd.read_csv(OUT / "aging_curves.csv")
    aging = {"curves": {g: [[int(q.age), r(q.value)] for q in x.itertuples()] for g, x in ag.groupby("group")},
             "meta": json.load(open(OUT / "aging_meta.json"))}
    se_ = pd.read_csv(OUT / "scheme_effects.csv")
    schemes = [[q.outcome, q.trait, r(q.coef_per_sd), r(q.se), r(q.t), r(q.r_simple), r(q.r2_all_traits)] for q in se_.itertuples()]
    vmeta = json.load(open(OUT / "value_meta.json"))
    fm = json.load(open(OUT / "fit_meta.json"))
    sm_ = json.load(open(OUT / "situations_meta.json"))
    sb = pd.read_csv(OUT / "situation_board.csv")
    sb = sb[(sb.season == sm_["latest_season"]) & (sb.poss >= 2000)]
    fe = pd.read_csv(OUT / "fit_style_effects.csv")
    fe = fe[fe.outcome == "net"]
    study = {"coach_split": fm["coach_split"], "fit": {k: v for k, v in fm["fit"].items() if k != "carryover_note"},
             "carry": sm_, "style_cells": [[q.archetype, q.trait, r(q.coef), r(q.t, 1)] for q in fe.itertuples()],
             "board": [[q.player_id, q.player, q.team, int(q.age), q.pos, r(q.rating, 1), r(q.norm_now, 1), r(q.gap, 1),
                        r(q.expected_next, 1)] for q in sb.itertuples()]}
    data = {"study": study, "value": {"r_wins": r(vmeta["r_salary_vs_wins"]), "r_ppg": r(vmeta["r_salary_vs_ppg"]),
                      "price": {k: r(v / 1e6, 1) for k, v in vmeta["price_per_win_by_season"].items()}}, "aging": aging, "schemes": schemes, "meta": meta, "players": players, "recent": recent, "progression": prog, "scenarios": scen, "classic": classic_rows, "coaches": coach_rows, "pairs": pairs, "syn": syn,
            "validation": {
                "classic": val.get("classic_meta", {}).get("validation"),
                "clutch_split_half": val.get("playoffs_meta", {}).get("all_clutch", {}).get("split_half"),
                "elevation_split_half": val.get("playoffs_meta", {}).get("elevation", {}).get("split_half"),
                "rotation": val.get("coach_rotation_meta"),
                "box_fit": val.get("breakdown_coefs", {}).get("fit"),
                "firings": val.get("firings_meta", {}).get("both_known"),
            }}
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":"), default=lambda o: None))
    print(f"players {len(players)}, classic {len(classic_rows)}, coaches {len(coach_rows)}, pairs {len(pairs)}, "
          f"{(SITE / 'data.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
