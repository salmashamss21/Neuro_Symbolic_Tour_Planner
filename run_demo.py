#!/usr/bin/env python3
"""Integrated Demo Script (`python run_demo.py`).

Demonstrates:
1. Input query (Edinburgh, 5.0 hours budget, 6 stops: Park -> ... -> Park, must include a Museum,
   starting at Princes Street Gardens [POI 4] and ending at Calton Hill [POI 5] at 10:00 AM).
2. How the three modules cooperate:
   - Part 1 (CSP): Generates up to 10 valid category patterns with BT+FC+AC-3 and compares
     pruning counts with and without FC/AC-3.
   - Part 2 (Heuristic Search): Runs A* (and compares against UCS and Greedy) to ground each
     category pattern into the most plausible sequence of real Edinburgh landmarks.
   - Part 3 (Two-Head Neural Network): Supplies time-aware transition probabilities P(next | cur, hour)
     and predicts visit durations (with comparison of NN ON vs. NN OFF / Popularity fallback).
3. Fallback demonstration when budget is too tight (e.g., 2.4 hours for 6 stops), reporting the
   exact minimum working budget and returning the cheapest itinerary with a warning.
4. Saves a 3-panel visual cooperation figure (`outputs/figures/demo_cooperation.png`).
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from data.loader import TourDataset
from integration.pipeline import IntegratedTourPlanner
from nn.train import train_and_evaluate


def format_clock(hour_float: float) -> str:
    total_min = int(round(hour_float * 60)) % (24 * 60)
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"


def main() -> None:
    print("=" * 78)
    print("P6: PLAUSIBLE TOUR ITINERARIES — INTEGRATED DEMO (CSP + SEARCH + TWO-HEAD NN)")
    print("=" * 78)

    dataset = TourDataset("data")
    nn_model, markov_base, pop_base, nn_eval = train_and_evaluate(
        dataset, seed=42, epochs=55, batch_size=64, lr=0.004, save_dir="outputs/models"
    )
    planner = IntegratedTourPlanner(dataset, nn_model, markov_base, pop_base)

    start_poi = 4   # Princes Street Gardens (Park)
    end_poi = 5     # Calton Hill (Park)
    budget_hr = 5.0
    must_see = "Museum"
    num_stops = 6
    start_hour = 10.0

    print("\n[1] INPUT QUERY")
    print(f"  City             : Edinburgh ({len(dataset.poi_ids)} POIs across {len(dataset.categories)} categories)")
    print(f"  Start Place      : {dataset.poi_names[start_poi]} (ID={start_poi}, Category={dataset.poi_categories[start_poi]})")
    print(f"  End Place        : {dataset.poi_names[end_poi]} (ID={end_poi}, Category={dataset.poi_categories[end_poi]})")
    print(f"  Time Budget      : {budget_hr:.2f} hours (Start Time: {format_clock(start_hour)})")
    print(f"  Number of Stops  : {num_stops} stops")
    print(f"  Must-See Category: {must_see}")

    print("\n[2] CSP CATEGORY-PATTERN GENERATION & PRUNING TRACE")
    csp_modes = [
        ("Backtracking Only (BT)", False, False),
        ("BT + Forward Checking (FC)", True, False),
        ("BT + FC + Arc Consistency (AC-3)", True, True),
    ]
    for label, u_fc, u_ac3 in csp_modes:
        pats, st = planner.csp.solve(
            num_stops=num_stops,
            start_category=dataset.poi_categories[start_poi],
            end_category=dataset.poi_categories[end_poi],
            must_see_category=must_see,
            budget_hr=budget_hr,
            use_fc=u_fc,
            use_ac3=u_ac3,
            max_solutions=10,
        )
        print(
            f"  {label:<34} -> Patterns: {len(pats):2d} | "
            f"Assignments: {st.assignments:4d} | Backtracks: {st.backtracks:4d} | Pruned Values: {st.pruned_values:4d}"
        )

    out_nn = planner.plan_tour(
        start_poi=start_poi,
        end_poi=end_poi,
        time_budget_hr=budget_hr,
        must_see_category=must_see,
        num_stops=num_stops,
        start_hour=start_hour,
        search_algo="astar",
        use_fc=True,
        use_ac3=True,
        cost_mode="nn",
        max_patterns=10,
    )

    print("\n[3] CANDIDATE PATTERNS EVALUATED BY HEURISTIC SEARCH (A*) + NN DURATION CHECK")
    for idx, ev in enumerate(out_nn.pattern_evaluations, start=1):
        pat_str = " -> ".join(ev["pattern"])
        status_tag = "ACCEPTED" if ev["accepted"] else "REJECTED (> Budget)"
        print(
            f"  Pattern {idx:2d}: {pat_str:<54} | "
            f"CSP Exp: {ev['expected_csp_time_hr']:.2f}h | NN Pred: {ev['predicted_nn_time_hr']:.2f}h | "
            f"Cost: {ev['total_cost']:.2f} | Expanded: {ev['nodes_expanded']:2d} | [{status_tag}]"
        )

    best = out_nn.best_itinerary
    assert best is not None
    print("\n[4] BEST RETURNED ITINERARY (NN ON + A* + BT+FC+AC-3)")
    print(f"  Status           : {out_nn.status.upper()} — {out_nn.message}")
    print(f"  Category Pattern : {' -> '.join(best.categories)}")
    print(f"  Total Time       : {best.total_time_hr:.2f} hours ({best.total_time_hr * 60:.1f} min) / {budget_hr:.2f}h budget")
    print(f"  Total Search Cost: {best.total_cost:.4f} (Search Nodes Expanded: {out_nn.total_search_stats.nodes_expanded}, Generated: {out_nn.total_search_stats.nodes_generated})")
    print("-" * 78)
    print(f"  {'Stop':<5} {'Arrival':<8} {'Depart':<8} {'Visit(m)':<9} {'Walk(m)':<8} {'Category':<11} {'Landmark Name'}")
    print("-" * 78)
    for i, pid in enumerate(best.pois):
        arr_h = best.arrival_hours[i]
        vis_h = best.visit_durations_hr[i]
        dep_h = arr_h + vis_h
        walk_m = best.travel_times_hr[i] * 60.0 if i < len(best.travel_times_hr) else 0.0
        print(
            f"  #{i+1:<4} {format_clock(arr_h):<8} {format_clock(dep_h):<8} "
            f"{vis_h*60:6.1f}m   {walk_m:5.1f}m   {best.categories[i]:<11} {dataset.poi_names[pid]}"
        )
    print("-" * 78)

    out_off = planner.plan_tour(
        start_poi=start_poi,
        end_poi=end_poi,
        time_budget_hr=budget_hr,
        must_see_category=must_see,
        num_stops=num_stops,
        start_hour=start_hour,
        search_algo="astar",
        use_fc=True,
        use_ac3=True,
        cost_mode="popularity",
        max_patterns=10,
    )
    best_off = out_off.best_itinerary
    assert best_off is not None
    nn_solver = planner._build_search_solver(cost_mode="nn")
    off_true_cost = sum(
        nn_solver.step_cost(best_off.pois[i], best_off.pois[i + 1], int(best_off.arrival_hours[i] + best_off.visit_durations_hr[i]) % 24)
        for i in range(len(best_off.pois) - 1)
    )
    print("\n[5] ABLATION: SYSTEM WITH NEURAL NETWORK ON vs. OFF")
    print(f"  NN ON  Route : {' -> '.join(dataset.poi_names[p] for p in best.pois)}")
    print(f"                 Plausibility Cost (NLL + Walk): {best.total_cost:.3f} | Predicted Time: {best.total_time_hr:.2f}h")
    print(f"  NN OFF Route : {' -> '.join(dataset.poi_names[p] for p in best_off.pois)}")
    print(f"                 Plausibility Cost (NLL + Walk): {off_true_cost:.3f} | Static Time   : {best_off.total_time_hr:.2f}h")

    tight_budget = 2.4
    out_tight = planner.plan_tour(
        start_poi=start_poi,
        end_poi=end_poi,
        time_budget_hr=tight_budget,
        must_see_category=must_see,
        num_stops=num_stops,
        start_hour=start_hour,
    )
    print("\n[6] IMPOSSIBLE / TIGHT BUDGET FALLBACK DEMO (Budget = 2.40h for 6 stops)")
    print(f"  Status : {out_tight.status.upper()}")
    print(f"  Message: {out_tight.message}")

    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10.0, "axes.titlesize": 11.0, "axes.labelsize": 10.0})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), constrained_layout=True)

    csp_labels = ["BT Only", "BT + FC", "BT+FC+AC-3"]
    bt_vals = []
    asgn_vals = []
    for _, u_fc, u_ac3 in csp_modes:
        _, st = planner.csp.solve(
            num_stops=num_stops,
            start_category=dataset.poi_categories[start_poi],
            end_category=dataset.poi_categories[end_poi],
            must_see_category=must_see,
            budget_hr=budget_hr,
            use_fc=u_fc,
            use_ac3=u_ac3,
            max_solutions=10,
        )
        bt_vals.append(st.backtracks)
        asgn_vals.append(st.assignments)

    x_c = np.arange(len(csp_labels))
    axes[0].bar(x_c - 0.18, asgn_vals, 0.36, label="Assignments", color="#1e3a5f")
    axes[0].bar(x_c + 0.18, bt_vals, 0.36, label="Backtracks", color="#d97706")
    axes[0].set_title("1. CSP Pruning (6-Stop Query)")
    axes[0].set_xticks(x_c)
    axes[0].set_xticklabels(csp_labels)
    axes[0].set_ylabel("Count")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")

    pat_ids = [f"P{i}" for i in range(1, len(out_nn.pattern_evaluations) + 1)]
    csp_times = [ev["expected_csp_time_hr"] for ev in out_nn.pattern_evaluations]
    nn_times = [ev["predicted_nn_time_hr"] for ev in out_nn.pattern_evaluations]
    x_p = np.arange(len(pat_ids))
    axes[1].bar(x_p - 0.18, csp_times, 0.36, label="CSP Expected Time", color="#64748b")
    bar_colors = ["#0f766e" if ev["accepted"] else "#b91c1c" for ev in out_nn.pattern_evaluations]
    axes[1].bar(x_p + 0.18, nn_times, 0.36, label="NN Predicted Time", color=bar_colors)
    axes[1].axhline(budget_hr, color="#b91c1c", linestyle="--", linewidth=1.8, label=f"Budget ({budget_hr:.1f}h)")
    axes[1].set_title("2. CSP Filter vs. NN Duration Check")
    axes[1].set_xticks(x_p)
    axes[1].set_xticklabels(pat_ids)
    axes[1].set_ylabel("Total Tour Time (Hours)")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left", fontsize=8.5)

    lats = [dataset.poi_coords[p][0] for p in best.pois]
    lons = [dataset.poi_coords[p][1] for p in best.pois]
    all_lats = [dataset.poi_coords[p][0] for p in dataset.poi_ids]
    all_lons = [dataset.poi_coords[p][1] for p in dataset.poi_ids]
    axes[2].scatter(all_lons, all_lats, color="#cbd5e1", s=28, label="Other Edinburgh POIs", zorder=2)
    axes[2].plot(lons, lats, color="#0f766e", linewidth=2.2, marker="o", markersize=7, label="Best Plausible Route", zorder=4)
    for idx_stop, (lon_v, lat_v, pid) in enumerate(zip(lons, lats, best.pois), start=1):
        short_name = dataset.poi_names[pid].split()[0]
        axes[2].annotate(
            f"{idx_stop}:{short_name}",
            (lon_v, lat_v),
            textcoords="offset points",
            xytext=(4, 5 if idx_stop % 2 == 1 else -10),
            fontsize=8.0,
            weight="bold",
            color="#0f172a",
        )
    axes[2].set_title("3. Selected Edinburgh Sightseeing Route")
    axes[2].set_xlabel("Longitude (°E)")
    axes[2].set_ylabel("Latitude (°N)")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="lower left", fontsize=8.5)

    out_fig = fig_dir / "demo_cooperation.png"
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[7] SAVED COOPERATION PLOT -> {out_fig}")

    demo_record = {
        "query": {
            "city": "Edinburgh",
            "start_poi": start_poi,
            "start_name": dataset.poi_names[start_poi],
            "end_poi": end_poi,
            "end_name": dataset.poi_names[end_poi],
            "budget_hr": budget_hr,
            "must_see": must_see,
            "num_stops": num_stops,
            "start_hour": start_hour,
        },
        "csp_comparison": {
            "BT": {"assignments": asgn_vals[0], "backtracks": bt_vals[0]},
            "BT+FC": {"assignments": asgn_vals[1], "backtracks": bt_vals[1]},
            "BT+FC+AC3": {"assignments": asgn_vals[2], "backtracks": bt_vals[2]},
        },
        "best_itinerary": {
            "pois": best.pois,
            "poi_names": [dataset.poi_names[p] for p in best.pois],
            "categories": best.categories,
            "arrival_times": [format_clock(h) for h in best.arrival_hours],
            "visit_durations_min": [round(d * 60.0, 1) for d in best.visit_durations_hr],
            "walk_times_min": [round(w * 60.0, 1) for w in best.travel_times_hr],
            "total_time_hr": round(best.total_time_hr, 2),
            "total_cost": round(best.total_cost, 4),
            "nodes_expanded": out_nn.total_search_stats.nodes_expanded,
            "nodes_generated": out_nn.total_search_stats.nodes_generated,
        },
        "nn_on_vs_off": {
            "nn_on_cost": round(best.total_cost, 4),
            "nn_off_cost_under_nn": round(off_true_cost, 4),
            "nn_on_route": [dataset.poi_names[p] for p in best.pois],
            "nn_off_route": [dataset.poi_names[p] for p in best_off.pois],
        },
        "tight_budget_fallback": {
            "requested_budget_hr": tight_budget,
            "status": out_tight.status,
            "minimum_working_budget_hr": out_tight.minimum_working_budget_hr,
            "message": out_tight.message,
        },
    }
    with open("outputs/demo_output.json", "w", encoding="utf-8") as f:
        json.dump(demo_record, f, indent=2)


if __name__ == "__main__":
    main()
