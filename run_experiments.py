#!/usr/bin/env python
"""Runs the NN comparison and the three project experiments.  Usage:  python run_experiments.py [--seeds 10]

Writes results/*.csv, results/*.json and figures/*.png.  Everything is seeded (seed = 0 .. seeds-1;
each seed changes the user split, the NN initialisation and the generated requests).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from tourplan import csp as C
from tourplan import data as D
from tourplan import evaluation as E
from tourplan import integration as I
from tourplan import nn
from tourplan import search as S

RES, FIG = D.ROOT / "results", D.ROOT / "figures"


def mean_std(df, by, cols):
    g = df.groupby(by)[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


# --------------------------------------------------------------- NN vs baselines ---
def exp_nn(cd, models):
    rows = []
    for seed, r in models.items():
        te, tr = r["data"]["te_t"], r["data"]["tr_t"]
        for name, m in [("NN", r["nn_next"]), ("Markov (counts)", r["markov"]), ("Popularity", r["pop"])]:
            rows.append(dict(seed=seed, task="next place", model=name, **nn.next_metrics(m, te)))
        ted = r["data"]["te_d"]
        for name, m in [("NN", r["nn_dur"]), ("Per-place mean", r["mean_dur"]), ("Global mean", r["glob_dur"])]:
            rows.append(dict(seed=seed, task="duration", model=name, **nn.duration_metrics(m, ted)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "nn_results_per_seed.csv", index=False)
    a = mean_std(df[df.task == "next place"], "model", ["nll", "top1", "top3"])
    b = mean_std(df[df.task == "duration"], "model", ["mae_log", "rmse_log", "mae_min"])
    a.to_csv(RES / "nn_next_summary.csv", index=False)
    b.to_csv(RES / "nn_duration_summary.csv", index=False)
    return a, b


# -------------------------------------------------- exp 1: UCS vs A* vs Greedy -----
def exp_search(cd, models, seeds, per_n=15, ns=(4, 5, 6, 7, 8)):
    rows = []
    for seed in seeds:
        r = models[seed]
        b = r["bundles"]["NN"]
        prior = C.category_prior(cd, r["data"]["tr_t"])
        rng = np.random.default_rng(1000 + seed)
        for n in ns:
            for req in E.random_requests(cd, rng, n, per_n * 3):
                p = C.build_csp(cd, b, req.start, req.end, n, req.must_cat, None, prior=prior)
                pats, _ = C.solve(p, fc=True, ac3=True, max_solutions=50_000, order_values=True)
                pats = C.rank_patterns(pats, prior, 1)
                if not pats:
                    continue
                pb = S.SearchProblem(cd, b, req.start, req.end, pats[0], req.start_hour, 10_000.0)
                res = {}
                for algo in ("ucs", "astar", "greedy"):
                    t = time.perf_counter()
                    it, st = S.search(pb, algo)
                    res[algo] = (it, st, time.perf_counter() - t)
                opt = res["ucs"][0].cost
                for algo, (it, st, dt) in res.items():
                    rows.append(dict(seed=seed, n=n, algo=algo, expanded=st.expanded,
                                     generated=st.generated, max_open=st.max_open, ms=1000 * dt,
                                     cost=it.cost, gap=(it.cost - opt), optimal=int(it.cost <= opt + 1e-9)))
                if sum(1 for x in rows if x["seed"] == seed and x["n"] == n and x["algo"] == "ucs") >= per_n:
                    break
    df = pd.DataFrame(rows)
    df.to_csv(RES / "exp1_search_raw.csv", index=False)
    per_seed = df.groupby(["seed", "n", "algo"], as_index=False)[
        ["expanded", "generated", "max_open", "ms", "cost", "gap", "optimal"]].mean()
    summ = mean_std(per_seed, ["n", "algo"], ["expanded", "generated", "ms", "gap", "optimal"])
    summ.to_csv(RES / "exp1_search_summary.csv", index=False)
    return per_seed, summ


# ------------------------------------------------------ exp 2: BT / FC / AC-3 -----
CSP_VARIANTS = [("BT", dict()), ("BT+FC", dict(fc=True)), ("BT+FC+AC3", dict(fc=True, ac3=True))]


def exp_csp(cd, models, seeds, per_n=25, ns=(4, 5, 6, 7, 8, 9, 10)):
    rows = []
    for seed in seeds:
        b = models[seed]["bundles"]["NN"]
        rng = np.random.default_rng(2000 + seed)
        for n in ns:
            for req in E.random_requests(cd, rng, n, per_n, budget_range=(60, 400)):
                p = C.build_csp(cd, b, req.start, req.end, n, req.must_cat, req.budget_min)
                sols = {}
                for name, kw in CSP_VARIANTS:
                    t = time.perf_counter()
                    s, st = C.solve(p, **kw)
                    dt = time.perf_counter() - t
                    sols[name] = set(s)
                    rows.append(dict(seed=seed, n=n, variant=name, ms=1000 * dt, n_solutions=len(s),
                                     **{k: v for k, v in st.as_dict().items() if k != "solutions"}))
                assert sols["BT"] == sols["BT+FC"] == sols["BT+FC+AC3"], "solvers disagree!"
    df = pd.DataFrame(rows)
    df.to_csv(RES / "exp2_csp_raw.csv", index=False)
    per_seed = df.groupby(["seed", "n", "variant"], as_index=False)[
        ["assignments", "backtracks", "checks", "revisions", "prunings", "ms", "n_solutions"]].mean()
    summ = mean_std(per_seed, ["n", "variant"], ["assignments", "backtracks", "checks", "ms", "n_solutions"])
    summ.to_csv(RES / "exp2_csp_summary.csv", index=False)
    return per_seed, summ


# ----------------------------------------- exp 3: NN costs vs Markov vs popularity --
def exp_f1(cd, models, seeds):
    rows = []
    for seed in seeds:
        r = models[seed]
        prior = C.category_prior(cd, r["data"]["tr_t"])
        trajs = D.trajectories(cd.visits, r["split"]["test"], min_len=3)
        reqs = E.requests_from_trajectories(cd, trajs, np.random.default_rng(3000 + seed))
        for name in ("NN", "Markov", "Popularity", "Off"):
            b = r["bundles"][name]
            for t, req in reqs:
                pl = I.plan(cd, b, req, prior)
                if pl.itinerary is None:
                    rows.append(dict(seed=seed, model=name, traj=t["traj"], n=req.n_stops, status=pl.status,
                                     f1_mid=0.0, f1_all=0.0, prec=0.0, rec=0.0))
                    continue
                pred, true = pl.itinerary.places, t["places"]
                p, rc, f = E.f1_sets(pred[1:-1], true[1:-1])
                _, _, fa = E.f1_sets(pred, true)
                rows.append(dict(seed=seed, model=name, traj=t["traj"], n=req.n_stops, status=pl.status,
                                 f1_mid=f, f1_all=fa, prec=p, rec=rc))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "exp3_f1_raw.csv", index=False)
    per_seed = df.groupby(["seed", "model"], as_index=False).agg(
        f1_mid=("f1_mid", "mean"), f1_all=("f1_all", "mean"), prec=("prec", "mean"), rec=("rec", "mean"),
        n_requests=("traj", "size"), over_budget=("status", lambda s: (s != "ok").mean()))
    summ = mean_std(per_seed, "model", ["f1_mid", "f1_all", "prec", "rec", "over_budget"])
    summ["n_requests_per_seed"] = per_seed.groupby("model")["n_requests"].mean().values
    summ.to_csv(RES / "exp3_f1_summary.csv", index=False)
    return per_seed, summ


# ------------------------------------------------------------------- plots --------
def plots(cd, models, e1, e2, e3, nn_per_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"ucs": "#4C78A8", "astar": "#F58518", "greedy": "#54A24B",
           "BT": "#4C78A8", "BT+FC": "#F58518", "BT+FC+AC3": "#54A24B"}
    # exp1
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    for a, (m, lab) in zip(ax, [("expanded", "nodes expanded"), ("generated", "nodes generated"),
                                 ("gap", "cost above optimum (nats)")]):
        for algo in ("ucs", "astar", "greedy"):
            s = e1[e1.algo == algo].groupby("n")[m].agg(["mean", "std"])
            a.errorbar(s.index, s["mean"], s["std"], marker="o", ms=4, capsize=2, label=algo.upper() if algo != "astar" else "A*", color=col[algo])
        a.set_xlabel("stops"); a.set_ylabel(lab)
        if m != "gap":
            a.set_yscale("log")
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "exp1_search.png", dpi=160); plt.close(fig)
    # exp2
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    for a, (m, lab) in zip(ax, [("assignments", "assignments"), ("backtracks", "backtracks"), ("ms", "time (ms)")]):
        for v in ("BT", "BT+FC", "BT+FC+AC3"):
            s = e2[e2.variant == v].groupby("n")[m].agg(["mean", "std"])
            a.errorbar(s.index, s["mean"], s["std"], marker="o", ms=4, capsize=2, label=v, color=col[v])
        a.set_xlabel("stops"); a.set_ylabel(lab); a.set_yscale("log")
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "exp2_csp.png", dpi=160); plt.close(fig)
    # exp3
    order = ["NN", "Markov", "Popularity", "Off"]
    lab = {"NN": "NN costs", "Markov": "Markov counts", "Popularity": "Popularity", "Off": "No model (travel only)"}
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    m = e3.groupby("model")["f1_mid"].agg(["mean", "std"]).loc[order]
    ax.bar([lab[o] for o in order], m["mean"], yerr=m["std"], capsize=3, color=["#F58518", "#4C78A8", "#54A24B", "#B279A2"])
    ax.set_ylabel("F1 on intermediate stops"); plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout(); fig.savefig(FIG / "exp3_f1.png", dpi=160); plt.close(fig)
    # training curves (seed 0)
    r = models[0]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    h = r["nn_next"].history
    ax[0].plot(h["train_nll"], label="train"); ax[0].plot(h["val_nll"], label="validation")
    ax[0].axhline(r["markov"].nll(*r["data"]["va_t"]), ls="--", c="gray", label="Markov (val)")
    ax[0].set_title("next-place head"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("NLL"); ax[0].legend(frameon=False)
    h = r["nn_dur"].history
    ax[1].plot(h["train_mse"], label="train"); ax[1].plot(h["val_mse"], label="validation")
    ax[1].set_title("duration head"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("MSE (log s)"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "nn_training.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--city", default="Edin")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    cd = D.get_city(a.city)
    seeds = list(range(a.seeds))
    t0 = time.time()
    models = {s: nn.train_all(cd, s) for s in seeds}
    print(f"trained {len(seeds)} model sets in {time.time() - t0:.1f}s")
    na, nb = exp_nn(cd, models)
    print("\n== NN vs baselines: next place (test, mean/std over seeds)\n", na.to_string(index=False))
    print("\n== NN vs baselines: duration\n", nb.to_string(index=False))
    e1, s1 = exp_search(cd, models, seeds); print("\n== Exp 1 search\n", s1.to_string(index=False))
    e2, s2 = exp_csp(cd, models, seeds); print("\n== Exp 2 CSP\n", s2.to_string(index=False))
    e3, s3 = exp_f1(cd, models, seeds); print("\n== Exp 3 F1\n", s3.to_string(index=False))
    plots(cd, models, e1, e2, e3, None)
    json.dump({"seeds": seeds, "city": a.city, "seconds": time.time() - t0}, open(RES / "run_info.json", "w"))
    print(f"\nall done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
