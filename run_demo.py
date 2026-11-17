#!/usr/bin/env python
"""Integrated demo:  python run_demo.py [--seed 0] [--n-stops 6] [--budget 300] [--must Museum]

Prints the request, the CSP patterns, the search trace and the final itinerary, compares the
system with the NN on and off, and writes figures/demo.png showing how CSP, search and NN cooperate.
"""
import argparse
import time

import numpy as np

from tourplan import csp as C
from tourplan import data as D
from tourplan import integration as I
from tourplan import nn
from tourplan import search as S


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-stops", type=int, default=6)
    ap.add_argument("--budget", type=float, default=300, help="minutes")
    ap.add_argument("--must", default="Museum")
    ap.add_argument("--start-id", type=int, default=7, help="poiID of the start place (default: a park)")
    ap.add_argument("--end-id", type=int, default=10, help="poiID of the end place (default: another park)")
    ap.add_argument("--hour", type=float, default=10.0)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    cd = D.get_city("Edin")
    id2idx = dict(zip(cd.pois.poi_id, cd.pois.poi_idx))
    models = nn.train_all(cd, a.seed)
    prior = C.category_prior(cd, models["data"]["tr_t"])
    nnb = models["bundles"]["NN"]
    req = I.Request(int(id2idx[a.start_id]), int(id2idx[a.end_id]), a.budget, a.must, a.n_stops, a.hour)

    banner("1. INPUT")
    print(f"city Edinburgh | start place {a.start_id} [{cd.pois.category[req.start]}] | end place {a.end_id} "
          f"[{cd.pois.category[req.end]}]\nbudget {a.budget:.0f} min | must-see: {a.must} | {a.n_stops} stops | "
          f"start {a.hour:02.0f}:00")

    banner("2. NN  (trained on real Flickr trajectories, user-split)")
    te = models["data"]["te_t"]
    for k, lab in (("nn_next", "NN next-place"), ("markov", "Markov counts")):
        m = nn.next_metrics(models[k], te)
        print(f"{lab:15s} test NLL {m['nll']:.3f}  top-1 {m['top1']:.3f}  top-3 {m['top3']:.3f}")

    banner("3. CSP: category patterns (BT vs BT+FC vs BT+FC+AC-3)")
    prob = C.build_csp(cd, nnb, req.start, req.end, req.n_stops, req.must_cat, req.budget_min, prior=prior)
    csp_stats = {}
    for name, kw in [("BT", {}), ("BT+FC", dict(fc=True)), ("BT+FC+AC3", dict(fc=True, ac3=True))]:
        t = time.perf_counter()
        sols, st = C.solve(prob, **kw)
        csp_stats[name] = st
        print(f"{name:10s} patterns {len(sols):4d} | assignments {st.assignments:6d} | backtracks {st.backtracks:6d} "
              f"| checks {st.checks:6d} | revisions {st.revisions:5d} | {1000 * (time.perf_counter() - t):6.1f} ms")
    doms0, _ = C.root_domains(prob)
    doms1, _ = C.root_domains(prob, fc=True, ac3=True)
    print("domains before propagation:", [[cd.categories[c] for c in sorted(d)] if len(d) < 6 else "all" for d in doms0])
    print("domains after FC + AC-3   :", [[cd.categories[c] for c in sorted(d)] for d in doms1])
    top = C.rank_patterns(sols, prior, 10)
    print("\ntop-10 patterns (by category-transition likelihood):")
    for k, pt in enumerate(top, 1):
        print(f"  {k:2d}. {' -> '.join(I.pattern_names(cd, pt))}")

    banner("4. SEARCH on the best pattern (UCS vs A* vs Greedy)")
    pb = S.SearchProblem(cd, nnb, req.start, req.end, top[0], req.start_hour, req.budget_min)
    srch = {}
    for algo in ("ucs", "astar", "greedy"):
        tr = []
        it, st = S.search(pb, algo, trace=tr)
        srch[algo] = (it, st, tr)
        print(f"{algo:7s} cost {it.cost:7.3f} | expanded {st.expanded:5d} | generated {st.generated:5d} "
              f"| max open {st.max_open:4d}")
    print("\nA* search trace (first 12 expansions):  n  pos  place  elapsed   g      h      f=g+h")
    for r in srch["astar"][2][:12]:
        print(f"   {r['n']:3d}  {r['pos']:3d}  {int(cd.pois.poi_id[r['place']]):5d}  {r['elapsed']:6d}  "
              f"{r['g']:6.2f} {r['h']:6.2f} {r['g'] + r['h']:7.2f}")

    banner("5. FULL PLAN (CSP -> search -> duration check)")
    plan = I.plan(cd, nnb, req, prior)
    print(I.format_plan(cd, plan))
    print(f"patterns tried: {len(plan.patterns_tried)}, rejected for budget: "
          f"{sum(1 for _, c, _ in plan.patterns_tried if c is None)}")

    banner("6. NN ON vs OFF")
    plan_off = I.plan(cd, models["bundles"]["Off"], req, prior)
    print("NN ON :", " -> ".join(f"{int(cd.pois.poi_id[p])}" for p in plan.itinerary.places),
          f"| avg NN log-prob/step {-plan.itinerary.nll / (req.n_stops - 1):.2f}")
    # evaluate both itineraries under the SAME (NN) model
    pbc = S.SearchProblem(cd, nnb, req.start, req.end, plan_off.pattern, req.start_hour, None)
    g_off, _ = S.path_cost(pbc, plan_off.itinerary.places)
    print("NN OFF:", " -> ".join(f"{int(cd.pois.poi_id[p])}" for p in plan_off.itinerary.places),
          f"| walking {plan_off.itinerary.travel_min:.0f} min (ON: {plan.itinerary.travel_min:.0f} min)")
    print(f"plausibility cost under the NN (lower = more typical): ON {plan.itinerary.cost:.2f}  vs  OFF {g_off:.2f}")

    if not a.no_plots:
        make_plot(cd, req, plan, plan_off, csp_stats, srch, top, prior)
        print("\nfigure written to figures/demo.png")


def make_plot(cd, req, plan, plan_off, csp_stats, srch, top, prior):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    pal = dict(zip(cd.categories, ["#4C78A8", "#B279A2", "#E45756", "#F58518", "#54A24B", "#9D9D9D"]))
    # map
    ax = fig.add_subplot(gs[0, 0])
    for c in cd.categories:
        m = cd.pois.category == c
        ax.scatter(cd.pois.lon[m], cd.pois.lat[m], s=22, color=pal[c], label=c, zorder=3)
    for it, ls, lab in ((plan.itinerary, "-", "NN on"), (plan_off.itinerary, "--", "NN off")):
        xs = [cd.pois.lon[p] for p in it.places]; ys = [cd.pois.lat[p] for p in it.places]
        ax.plot(xs, ys, ls, color="k" if ls == "-" else "gray", lw=1.2, zorder=2, label=lab)
    for k, p in enumerate(plan.itinerary.places):
        ax.annotate(str(k + 1), (cd.pois.lon[p], cd.pois.lat[p]), xytext=(3, 3), textcoords="offset points", fontsize=8, weight="bold")
    ax.set_title("Itinerary on the map"); ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(fontsize=6, frameon=False, ncol=2)
    # timeline
    ax = fig.add_subplot(gs[0, 1:])
    it = plan.itinerary
    for k, (p, ar, sy) in enumerate(zip(it.places, it.arrive, it.stay)):
        ax.barh(0, sy, left=ar / 60 * 60, color=pal[cd.pois.category[p]], height=0.5)
        ax.text(ar + sy / 2, 0, str(k + 1), ha="center", va="center", fontsize=8, color="white", weight="bold")
    t_end = it.arrive[-1] + it.stay[-1]
    t_bud = req.start_hour * 60 + req.budget_min
    ax.set_xlim(it.arrive[0] - 10, max(t_end, min(t_bud, t_end + 60)) + 15)
    if t_bud <= ax.get_xlim()[1]:
        ax.axvline(t_bud, color="r", ls=":", lw=1)
        ax.text(t_bud, 0.3, " budget", color="r", fontsize=7)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks); ax.set_xticklabels([f"{int(t // 60) % 24:02d}:{int(t % 60):02d}" for t in ticks])
    ax.set_yticks([]); ax.set_xlabel("clock time  (bars = NN-predicted stay, gaps = walking)")
    ax.set_title("Timeline with NN-predicted visit durations"); ax.spines["left"].set_visible(False)
    # CSP counters
    ax = fig.add_subplot(gs[1, 0])
    names = list(csp_stats); x = np.arange(len(names))
    ax.bar(x - 0.2, [csp_stats[n].assignments for n in names], 0.4, label="assignments", color="#4C78A8")
    ax.bar(x + 0.2, [csp_stats[n].backtracks for n in names], 0.4, label="backtracks", color="#E45756")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_yscale("log"); ax.legend(frameon=False)
    ax.set_title("CSP effort for this request")
    # search nodes
    ax = fig.add_subplot(gs[1, 1])
    al = ["ucs", "astar", "greedy"]
    ax.bar(np.arange(3) - 0.2, [srch[a][1].expanded for a in al], 0.4, label="expanded", color="#4C78A8")
    ax.bar(np.arange(3) + 0.2, [srch[a][1].generated for a in al], 0.4, label="generated", color="#F58518")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{n}\ncost {srch[a][0].cost:.1f}" for n, a in zip(["UCS", "A*", "Greedy"], al)])
    ax.legend(frameon=False)
    ax.set_title("Search effort (top pattern)")
    # cooperation: pattern costs
    ax = fig.add_subplot(gs[1, 2])
    tried = plan.patterns_tried
    costs = [c if c is not None else np.nan for _, c, _ in tried]
    ax.bar(range(len(tried)), [0 if np.isnan(c) else c for c in costs],
           color=["#54A24B" if (not np.isnan(c)) else "#BBBBBB" for c in costs])
    for i, c in enumerate(costs):
        if np.isnan(c):
            ax.text(i, 0.3, "rej.", rotation=90, ha="center", fontsize=7)
    best = np.nanargmin(costs) if not all(np.isnan(costs)) else None
    if best is not None:
        ax.bar(best, costs[best], color="#F58518")
    ax.set_xlabel("pattern rank (CSP)"); ax.set_ylabel("search cost")
    ax.set_title("CSP patterns -> search cost (orange = chosen)")
    fig.savefig(D.ROOT / "figures" / "demo.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
