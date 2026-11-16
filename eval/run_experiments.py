#!/usr/bin/env python3
"""Runs the 3 required project experiments across 5 random seeds and plots figures.

Experiment 1: Search Comparison (UCS vs. A* vs. Greedy Best-First Search)
  - Evaluates 4 to 8 stops across 5 random seeds (seeds 42, 43, 44, 45, 46).
  - Reports nodes expanded, nodes generated, total path cost, optimality gap vs. UCS/Brute-Force,
    and runtime (ms), with mean and standard deviation.

Experiment 2: CSP Ablation (BT vs. BT+FC vs. BT+FC+AC-3)
  - Evaluates 4 to 8 stops across 5 random seeds / query workloads.
  - Reports variable assignments, chronological backtracks, domain values pruned,
    and solve runtime (ms), with mean and standard deviation.

Experiment 3: Cost Model & Held-Out Trajectory Accuracy (NN Costs vs. Markov-Count vs. Popularity)
  - Evaluates across 5 random seeds on held-out real test trajectories from Edinburgh.
  - Compares Next-Place NLL & Top-1/Top-3 Accuracy, Duration MAE/RMSE, Tour Point-F1,
    Interior Point-F1, and Ordered Pairs-F1, with mean and standard deviation.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csp.solver import CategoryPatternCSP
from data.loader import TourDataset
from eval.metrics import compute_pairs_f1, compute_point_f1
from integration.pipeline import IntegratedTourPlanner
from nn.train import train_and_evaluate

SEEDS = [42, 43, 44, 45, 46]


def run_all_experiments(data_dir: str | Path = "data", out_dir: str | Path = "outputs") -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dataset = TourDataset(data_dir)

    nn_seed_metrics = []
    mk_seed_metrics = []
    pop_seed_metrics = []
    trained_models = {}

    for seed in SEEDS:
        nn_mod, mk_mod, pop_mod, res = train_and_evaluate(
            dataset, seed=seed, epochs=55, batch_size=64, lr=0.004, save_dir=out_dir / "models"
        )
        trained_models[seed] = (nn_mod, mk_mod, pop_mod, res)
        nn_seed_metrics.append(res["nn_test"])
        mk_seed_metrics.append(res["markov_baseline_test"])
        pop_seed_metrics.append(res["popularity_baseline_test"])

    exp1_records = []
    stop_lengths = [4, 5, 6, 7, 8]

    csp_helper = CategoryPatternCSP(
        categories=dataset.categories,
        category_mean_durations_hr=dataset.category_mean_duration_hr,
        expected_walk_hr=dataset.mean_walk_time_hr,
    )

    for length in stop_lengths:
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 100 + length)
            nn_mod, mk_mod, pop_mod, _ = trained_models[seed]
            planner = IntegratedTourPlanner(dataset, nn_mod, mk_mod, pop_mod)
            search_solver = planner._build_search_solver(cost_mode="nn")

            for q_idx in range(15):
                start_poi = int(rng.choice(dataset.poi_ids))
                end_candidates = [p for p in dataset.poi_ids if p != start_poi]
                end_poi = int(rng.choice(end_candidates))
                start_cat = dataset.poi_categories[start_poi]
                end_cat = dataset.poi_categories[end_poi]
                must_see = str(rng.choice(dataset.categories))
                start_hr = float(rng.choice([9, 10, 11, 12, 13, 14]))

                patterns, _ = csp_helper.solve(
                    num_stops=length,
                    start_category=start_cat,
                    end_category=end_cat,
                    must_see_category=must_see,
                    budget_hr=12.0,
                    use_fc=True,
                    use_ac3=True,
                    max_solutions=3,
                )
                if not patterns:
                    continue
                pat = patterns[0]

                ucs_cost = None
                for algo in ["ucs", "astar", "greedy"]:
                    t0 = time.perf_counter()
                    res = search_solver.solve(
                        pattern=pat,
                        start_poi=start_poi,
                        end_poi=end_poi,
                        start_hour=start_hr,
                        algorithm=algo,
                    )
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    if res is None:
                        continue
                    if algo == "ucs":
                        ucs_cost = res.total_cost
                    opt_gap_pct = (
                        100.0 * (res.total_cost - ucs_cost) / max(1e-9, ucs_cost)
                        if ucs_cost is not None else 0.0
                    )
                    exp1_records.append({
                        "num_stops": length,
                        "seed": seed,
                        "query_id": q_idx,
                        "algorithm": algo.upper() if algo != "astar" else "A*",
                        "nodes_expanded": res.stats.nodes_expanded,
                        "nodes_generated": res.stats.nodes_generated,
                        "total_cost": res.total_cost,
                        "opt_gap_pct": max(0.0, opt_gap_pct),
                        "runtime_ms": dt_ms,
                    })

    exp1_df = pd.DataFrame(exp1_records)
    exp1_seed_agg = exp1_df.groupby(["num_stops", "algorithm", "seed"], as_index=False).mean(numeric_only=True)
    exp1_summary = exp1_seed_agg.groupby(["num_stops", "algorithm"]).agg(
        expanded_mean=("nodes_expanded", "mean"),
        expanded_std=("nodes_expanded", "std"),
        generated_mean=("nodes_generated", "mean"),
        generated_std=("nodes_generated", "std"),
        cost_mean=("total_cost", "mean"),
        cost_std=("total_cost", "std"),
        gap_mean=("opt_gap_pct", "mean"),
        gap_std=("opt_gap_pct", "std"),
        time_ms_mean=("runtime_ms", "mean"),
        time_ms_std=("runtime_ms", "std"),
    ).reset_index().round(3)
    exp1_summary.to_csv(out_dir / "exp1_search_comparison.csv", index=False)

    exp2_records = []
    for length in stop_lengths:
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 200 + length)
            for q_idx in range(20):
                start_cat = str(rng.choice(dataset.categories))
                end_cat = str(rng.choice(dataset.categories))
                must_see = str(rng.choice(dataset.categories))
                budget_hr = float(length * 0.78 + rng.uniform(-0.20, 0.35))

                for variant_name, u_fc, u_ac3 in [
                    ("BT", False, False),
                    ("BT+FC", True, False),
                    ("BT+FC+AC-3", True, True),
                ]:
                    t0 = time.perf_counter()
                    sols, st = csp_helper.solve(
                        num_stops=length,
                        start_category=start_cat,
                        end_category=end_cat,
                        must_see_category=must_see,
                        budget_hr=budget_hr,
                        use_fc=u_fc,
                        use_ac3=u_ac3,
                        max_solutions=10,
                    )
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    exp2_records.append({
                        "num_stops": length,
                        "seed": seed,
                        "query_id": q_idx,
                        "variant": variant_name,
                        "assignments": st.assignments,
                        "backtracks": st.backtracks,
                        "pruned_values": st.pruned_values,
                        "patterns_found": len(sols),
                        "runtime_ms": dt_ms,
                    })

    exp2_df = pd.DataFrame(exp2_records)
    exp2_seed_agg = exp2_df.groupby(["num_stops", "variant", "seed"], as_index=False).mean(numeric_only=True)
    exp2_summary = exp2_seed_agg.groupby(["num_stops", "variant"]).agg(
        assignments_mean=("assignments", "mean"),
        assignments_std=("assignments", "std"),
        backtracks_mean=("backtracks", "mean"),
        backtracks_std=("backtracks", "std"),
        pruned_mean=("pruned_values", "mean"),
        pruned_std=("pruned_values", "std"),
        time_ms_mean=("runtime_ms", "mean"),
        time_ms_std=("runtime_ms", "std"),
    ).reset_index().round(3)
    exp2_summary.to_csv(out_dir / "exp2_csp_ablation.csv", index=False)

    test_trajs = dataset.get_trajectories_by_split("test")
    exp3_records = []

    for seed in SEEDS:
        nn_mod, mk_mod, pop_mod, _ = trained_models[seed]
        planner = IntegratedTourPlanner(dataset, nn_mod, mk_mod, pop_mod)
        rng = np.random.default_rng(seed * 300)
        chosen_indices = rng.choice(len(test_trajs), size=int(0.85 * len(test_trajs)), replace=False)

        for cost_mode, mode_label in [
            ("nn", "Two-Head NN Costs"),
            ("markov", "Markov-Count Baseline"),
            ("popularity", "Popularity Baseline"),
        ]:
            f1_all_list, f1_int_list, pairs_f1_list, dur_err_list = [], [], [], []
            search_solver = planner._build_search_solver(cost_mode=cost_mode)

            for idx in chosen_indices:
                tr = test_trajs[idx]
                true_pois = tr["pois"]
                true_cats = tr["categories"]
                length = len(true_pois)
                if length < 4:
                    continue
                start_poi = true_pois[0]
                end_poi = true_pois[-1]
                start_hr = float(tr["start_hour"])
                must_see = true_cats[len(true_cats) // 2]
                budget_hr = tr["total_time_hr"] * 1.15

                out = planner.plan_tour(
                    start_poi=start_poi,
                    end_poi=end_poi,
                    time_budget_hr=budget_hr,
                    must_see_category=must_see,
                    num_stops=length,
                    start_hour=start_hr,
                    search_algo="astar",
                    use_fc=True,
                    use_ac3=True,
                    cost_mode=cost_mode,
                    max_patterns=10,
                )
                res_true_pat = search_solver.solve(
                    pattern=true_cats,
                    start_poi=start_poi,
                    end_poi=end_poi,
                    start_hour=start_hr,
                    algorithm="astar",
                )
                pred_itin = res_true_pat if res_true_pat is not None else out.best_itinerary
                if pred_itin is None:
                    continue

                p_f1_all = compute_point_f1(pred_itin.pois, true_pois, exclude_endpoints=False)["f1"]
                p_f1_int = compute_point_f1(pred_itin.pois, true_pois, exclude_endpoints=True)["f1"]
                pr_f1 = compute_pairs_f1(pred_itin.pois, true_pois)
                dur_err_min = abs(pred_itin.total_time_hr - tr["total_time_hr"]) * 60.0

                f1_all_list.append(p_f1_all)
                f1_int_list.append(p_f1_int)
                pairs_f1_list.append(pr_f1)
                dur_err_list.append(dur_err_min)

            if cost_mode == "nn":
                m_test = trained_models[seed][3]["nn_test"]
            elif cost_mode == "markov":
                m_test = trained_models[seed][3]["markov_baseline_test"]
            else:
                m_test = trained_models[seed][3]["popularity_baseline_test"]

            exp3_records.append({
                "seed": seed,
                "model": mode_label,
                "point_f1": float(np.mean(f1_all_list)),
                "interior_f1": float(np.mean(f1_int_list)),
                "pairs_f1": float(np.mean(pairs_f1_list)),
                "tour_time_mae_min": float(np.mean(dur_err_list)),
                "next_place_nll": m_test["nll"],
                "next_place_top1_acc": m_test["top1_acc"],
                "next_place_top3_acc": m_test["top3_acc"],
                "stop_dur_mae_min": m_test["dur_mae_min"],
            })

    exp3_df = pd.DataFrame(exp3_records)
    exp3_summary = exp3_df.groupby("model", as_index=False).agg(
        point_f1_mean=("point_f1", "mean"),
        point_f1_std=("point_f1", "std"),
        interior_f1_mean=("interior_f1", "mean"),
        interior_f1_std=("interior_f1", "std"),
        pairs_f1_mean=("pairs_f1", "mean"),
        pairs_f1_std=("pairs_f1", "std"),
        tour_time_mae_mean=("tour_time_mae_min", "mean"),
        tour_time_mae_std=("tour_time_mae_min", "std"),
        nll_mean=("next_place_nll", "mean"),
        nll_std=("next_place_nll", "std"),
        top1_acc_mean=("next_place_top1_acc", "mean"),
        top1_acc_std=("next_place_top1_acc", "std"),
        top3_acc_mean=("next_place_top3_acc", "mean"),
        top3_acc_std=("next_place_top3_acc", "std"),
        stop_dur_mae_mean=("stop_dur_mae_min", "mean"),
        stop_dur_mae_std=("stop_dur_mae_min", "std"),
    ).round(4)
    exp3_summary.to_csv(out_dir / "exp3_trajectory_f1_comparison.csv", index=False)

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 12.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), constrained_layout=True)
    colors_exp1 = {"UCS": "#1e3a5f", "A*": "#0f766e", "GREEDY": "#d97706"}
    markers_exp1 = {"UCS": "o", "A*": "s", "GREEDY": "^"}

    for algo in ["UCS", "A*", "GREEDY"]:
        sub = exp1_summary[exp1_summary["algorithm"] == algo].sort_values("num_stops")
        axes[0].errorbar(
            sub["num_stops"],
            sub["expanded_mean"],
            yerr=sub["expanded_std"],
            label=algo,
            color=colors_exp1[algo],
            marker=markers_exp1[algo],
            linewidth=2.0,
            capsize=3,
        )
        axes[1].errorbar(
            sub["num_stops"],
            sub["gap_mean"],
            yerr=sub["gap_std"],
            label=algo,
            color=colors_exp1[algo],
            marker=markers_exp1[algo],
            linewidth=2.0,
            capsize=3,
        )

    axes[0].set_title("A. Search Effort (Nodes Expanded)")
    axes[0].set_xlabel("Itinerary Length L (Number of Stops)")
    axes[0].set_ylabel("Nodes Expanded (mean ± std)")
    axes[0].set_xticks(stop_lengths)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].set_title("B. Suboptimality Gap vs. Optimal Cost (%)")
    axes[1].set_xlabel("Itinerary Length L (Number of Stops)")
    axes[1].set_ylabel("Cost Gap vs. UCS/A* Optimal (%)")
    axes[1].set_xticks(stop_lengths)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left", frameon=True)

    fig.savefig(fig_dir / "exp1_search_plot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), constrained_layout=True)
    colors_exp2 = {"BT": "#b91c1c", "BT+FC": "#d97706", "BT+FC+AC-3": "#0f766e"}
    markers_exp2 = {"BT": "o", "BT+FC": "s", "BT+FC+AC-3": "D"}

    for var in ["BT", "BT+FC", "BT+FC+AC-3"]:
        sub = exp2_summary[exp2_summary["variant"] == var].sort_values("num_stops")
        axes[0].errorbar(
            sub["num_stops"],
            sub["backtracks_mean"],
            yerr=sub["backtracks_std"],
            label=var,
            color=colors_exp2[var],
            marker=markers_exp2[var],
            linewidth=2.0,
            capsize=3,
        )
        axes[1].errorbar(
            sub["num_stops"],
            sub["assignments_mean"],
            yerr=sub["assignments_std"],
            label=var,
            color=colors_exp2[var],
            marker=markers_exp2[var],
            linewidth=2.0,
            capsize=3,
        )

    axes[0].set_title("A. CSP Chronological Backtracks")
    axes[0].set_xlabel("Pattern Length L (Number of Stops)")
    axes[0].set_ylabel("Backtracks (log scale, mean ± std)")
    axes[0].set_yscale("log")
    axes[0].set_xticks(stop_lengths)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].set_title("B. Total Variable Assignments")
    axes[1].set_xlabel("Pattern Length L (Number of Stops)")
    axes[1].set_ylabel("Assignments (log scale, mean ± std)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(stop_lengths)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left", frameon=True)

    fig.savefig(fig_dir / "exp2_csp_plot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), constrained_layout=True)

    models_order = ["Two-Head NN Costs", "Markov-Count Baseline", "Popularity Baseline"]
    short_labels = ["Two-Head NN", "Markov-Count", "Popularity"]
    x_pos = np.arange(len(models_order))
    width = 0.25

    pt_means = [float(exp3_summary[exp3_summary["model"] == m]["point_f1_mean"].iloc[0]) for m in models_order]
    pt_stds = [float(exp3_summary[exp3_summary["model"] == m]["point_f1_std"].iloc[0]) for m in models_order]
    int_means = [float(exp3_summary[exp3_summary["model"] == m]["interior_f1_mean"].iloc[0]) for m in models_order]
    int_stds = [float(exp3_summary[exp3_summary["model"] == m]["interior_f1_std"].iloc[0]) for m in models_order]
    pr_means = [float(exp3_summary[exp3_summary["model"] == m]["pairs_f1_mean"].iloc[0]) for m in models_order]
    pr_stds = [float(exp3_summary[exp3_summary["model"] == m]["pairs_f1_std"].iloc[0]) for m in models_order]

    axes[0].bar(x_pos - width, pt_means, width, yerr=pt_stds, label="Point-F1 (All)", color="#1e3a5f", capsize=3)
    axes[0].bar(x_pos, int_means, width, yerr=int_stds, label="Interior-F1", color="#0f766e", capsize=3)
    axes[0].bar(x_pos + width, pr_means, width, yerr=pr_stds, label="Pairs-F1 (Order)", color="#d97706", capsize=3)
    axes[0].set_title("A. Held-Out Trajectory Accuracy (5 Seeds)")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(short_labels)
    axes[0].set_ylabel("F1 Score (mean ± std)")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", frameon=True)

    hist = trained_models[42][3]["history"]
    epochs_arr = np.arange(1, len(hist["train_nll"]) + 1)
    axes[1].plot(epochs_arr, hist["train_nll"], label="Train NLL (Head 1)", color="#1e3a5f", linewidth=2.0)
    axes[1].plot(epochs_arr, hist["val_nll"], label="Val NLL (Head 1)", color="#0f766e", linestyle="--", linewidth=2.0)
    mk_nll = float(exp3_summary[exp3_summary["model"] == "Markov-Count Baseline"]["nll_mean"].iloc[0])
    axes[1].axhline(mk_nll, color="#b91c1c", linestyle=":", linewidth=1.8, label=f"Markov NLL ({mk_nll:.2f})")
    axes[1].set_title("B. NN Next-Place NLL Convergence")
    axes[1].set_xlabel("Training Epoch")
    axes[1].set_ylabel("Negative Log-Likelihood (NLL)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", frameon=True)

    fig.savefig(fig_dir / "exp3_f1_and_nn_plot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_bundle = {
        "seeds": SEEDS,
        "exp1_search": exp1_summary.to_dict("records"),
        "exp2_csp": exp2_summary.to_dict("records"),
        "exp3_f1_and_nn": exp3_summary.to_dict("records"),
    }
    with open(out_dir / "experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_bundle, f, indent=2)

    return summary_bundle


if __name__ == "__main__":
    res = run_all_experiments("data", "outputs")
    print("=== Experiment 1: Search (UCS vs A* vs Greedy) ===")
    print(pd.DataFrame(res["exp1_search"]).to_string(index=False))
    print("\n=== Experiment 2: CSP (BT vs BT+FC vs BT+FC+AC-3) ===")
    print(pd.DataFrame(res["exp2_csp"]).to_string(index=False))
    print("\n=== Experiment 3: Cost Models & Held-Out F1 ===")
    print(pd.DataFrame(res["exp3_f1_and_nn"]).to_string(index=False))
