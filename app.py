import html
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from data.loader import TourDataset
from integration.pipeline import IntegratedTourPlanner
from nn.train import train_and_evaluate


st.set_page_config(
    page_title="Edinburgh Tour Itinerary Planner",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
    }
    .badge-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warn {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-err {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    dataset = TourDataset("data")
    nn_model, markov_base, pop_base, nn_eval = train_and_evaluate(
        dataset, seed=42, epochs=55, batch_size=64, lr=0.004, save_dir="outputs/models"
    )
    planner = IntegratedTourPlanner(dataset, nn_model, markov_base, pop_base)
    return dataset, planner, nn_eval


def format_clock(hour_float: float) -> str:
    total_min = int(round(hour_float * 60)) % (24 * 60)
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"


CATEGORY_COLORS = {
    "Park": [34, 197, 94, 200],       # Green
    "Historical": [234, 88, 12, 200],  # Orange
    "Cultural": [168, 85, 247, 200],  # Purple
    "Museum": [59, 130, 246, 200],    # Blue
    "Cafe": [245, 158, 11, 200],      # Amber
    "Castle": [239, 68, 68, 200]      # Red
}


def main():
    st.markdown('<div class="main-header">🏰 Edinburgh Tour Itinerary Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Neuro-Symbolic Tour Planning: CSP (Category Patterns) + Heuristic Search (A*) + Dual-Head Neural Network</div>', unsafe_allow_html=True)

    with st.spinner("Loading Edinburgh dataset and Neural Network model..."):
        dataset, planner, nn_eval = load_system()

    # Sidebar inputs
    st.sidebar.header("🧭 Tour Query Parameters")

    poi_options = {
        f"{dataset.poi_names[pid]} ({dataset.poi_categories[pid]})": pid
        for pid in dataset.poi_ids
    }
    poi_labels = list(poi_options.keys())

    # Default indices: Princes Street Gardens (Park, id=4) and Calton Hill (Park, id=5)
    start_default_idx = next((i for i, k in enumerate(poi_labels) if "Princes Street Gardens" in k), 0)
    end_default_idx = next((i for i, k in enumerate(poi_labels) if "Calton Hill" in k), min(1, len(poi_labels)-1))

    selected_start_label = st.sidebar.selectbox("Start Place (البداية)", poi_labels, index=start_default_idx)
    selected_end_label = st.sidebar.selectbox("End Place (النهاية)", poi_labels, index=end_default_idx)

    start_poi = poi_options[selected_start_label]
    end_poi = poi_options[selected_end_label]

    budget_hr = st.sidebar.slider("Time Budget (hours)", min_value=1.5, max_value=10.0, value=5.0, step=0.25)
    num_stops = st.sidebar.slider("Number of Stops", min_value=2, max_value=8, value=6, step=1)

    must_see_options = ["None"] + sorted(list(dataset.categories))
    must_see_default_idx = must_see_options.index("Museum") if "Museum" in must_see_options else 0
    selected_must_see = st.sidebar.selectbox("Must-See Category (فئة إلزامية)", must_see_options, index=must_see_default_idx)
    must_see = None if selected_must_see == "None" else selected_must_see

    start_hour = st.sidebar.slider("Start Time (وقت الانطلاق)", min_value=8.0, max_value=16.0, value=10.0, step=0.5, format="%0.1f h")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Solver Settings")
    search_algo = st.sidebar.selectbox("Search Algorithm", ["astar", "ucs", "greedy"], format_func=lambda x: {"astar": "A* Search", "ucs": "UCS (Uniform Cost)", "greedy": "Greedy Best-First"}[x])
    csp_mode = st.sidebar.selectbox("CSP Pruning Mode", ["BT+FC+AC-3", "BT+FC", "BT_Only"])
    cost_mode = st.sidebar.selectbox("Cost Model", ["nn", "popularity", "markov"], format_func=lambda x: {"nn": "Dual-Head Neural Network", "popularity": "Popularity Baseline", "markov": "Markov Baseline"}[x])

    use_fc = csp_mode in ["BT+FC+AC-3", "BT+FC"]
    use_ac3 = csp_mode == "BT+FC+AC-3"

    run_btn = st.sidebar.button("🚀 Plan Tour (تخطيط الرحلة)", type="primary", use_container_width=True)

    tab_itinerary, tab_csp, tab_compare, tab_experiments, tab_about = st.tabs([
        "🗺️ Itinerary & Map",
        "🧩 CSP Pruning Analysis",
        "⚖️ NN vs Baseline Ablation",
        "📈 Research Experiments",
        "ℹ️ About & Metrics"
    ])

    # Run planning
    output = planner.plan_tour(
        start_poi=start_poi,
        end_poi=end_poi,
        time_budget_hr=budget_hr,
        must_see_category=must_see,
        num_stops=num_stops,
        start_hour=start_hour,
        search_algo=search_algo,
        use_fc=use_fc,
        use_ac3=use_ac3,
        cost_mode=cost_mode,
        max_patterns=10,
    )

    with tab_itinerary:
        # Status header
        if output.status == "ok":
            st.markdown(f'<span class="badge-ok">✓ Optimal Plausible Tour Found ({output.best_itinerary.total_time_hr:.2f}h / {budget_hr:.2f}h)</span>', unsafe_allow_html=True)
        elif output.status == "budget_warning":
            st.markdown(f'<span class="badge-warn">⚠️ Budget Warning: {output.message}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-err">❌ No Solution: {output.message}</span>', unsafe_allow_html=True)

        st.write("")

        if output.best_itinerary:
            itin = output.best_itinerary

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Duration", f"{itin.total_time_hr:.2f} hrs", f"Budget: {budget_hr:.2f} hrs")
            with col2:
                st.metric("Total Plausibility Cost", f"{itin.total_cost:.3f}")
            with col3:
                st.metric("Stops Count", f"{len(itin.pois)} stops")
            with col4:
                st.metric("Nodes Expanded", f"{output.total_search_stats.nodes_expanded}")

            # Itinerary Table
            table_data = []
            map_points = []
            path_coords = []

            for i, p_id in enumerate(itin.pois):
                name = dataset.poi_names[p_id]
                cat = dataset.poi_categories[p_id]
                arr_hr = itin.arrival_hours[i]
                dur_m = itin.visit_durations_hr[i] * 60.0
                walk_m = itin.travel_times_hr[i] * 60.0 if i < len(itin.travel_times_hr) else 0.0
                dep_hr = arr_hr + itin.visit_durations_hr[i]

                lat, lon = dataset.poi_coords[p_id]
                map_points.append({
                    "name": f"#{i+1}: {name}",
                    "category": cat,
                    "lat": lat,
                    "lon": lon,
                    "color": CATEGORY_COLORS.get(cat, [100, 100, 100, 200]),
                    "stop_num": str(i + 1)
                })
                path_coords.append([lon, lat])

                table_data.append({
                    "Stop #": f"#{i+1}",
                    "Arrival": format_clock(arr_hr),
                    "Departure": format_clock(dep_hr),
                    "Visit Time": f"{dur_m:.1f} min",
                    "Walk Next": f"{walk_m:.1f} min" if walk_m > 0 else "—",
                    "Category": cat,
                    "Landmark Name": name
                })

            st.subheader("📋 Tour Schedule")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            # Map view
            st.subheader("📍 Edinburgh Route Map")
            
            mid_lat = np.mean([p["lat"] for p in map_points])
            mid_lon = np.mean([p["lon"] for p in map_points])

            view_state = pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=13.2,
                pitch=20,
            )

            path_data = [{"path": path_coords, "name": "Tour Path"}]

            line_layer = pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_color=[30, 58, 138, 220],
                width_scale=20,
                width_min_pixels=3,
                pickable=True,
            )

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_points,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=85,
                pickable=True,
            )

            text_layer = pdk.Layer(
                "TextLayer",
                data=map_points,
                get_position=["lon", "lat"],
                get_text="name",
                get_size=13,
                get_color=[0, 0, 0, 255],
                get_angle=0,
                get_text_anchor="'start'",
                get_alignment_baseline="'center'",
                get_pixel_offset=[12, 0]
            )

            deck = pdk.Deck(
                layers=[line_layer, scatter_layer, text_layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>{name}</b><br/>Category: {category}", "style": {"color": "white"}}
            )
            st.pydeck_chart(deck)

            # Download Itinerary Button
            export_payload = {
                "start_place": dataset.poi_names[start_poi],
                "end_place": dataset.poi_names[end_poi],
                "budget_hours": budget_hr,
                "total_time_hours": itin.total_time_hr,
                "total_cost": itin.total_cost,
                "stops": table_data,
            }
            st.download_button(
                label="📥 Download Itinerary (JSON)",
                data=json.dumps(export_payload, indent=2),
                file_name="edinburgh_itinerary.json",
                mime="application/json",
                use_container_width=True
            )

        else:
            st.error(f"Could not generate an itinerary. {output.message}")

    with tab_csp:
        st.subheader("🧩 CSP Category Pattern Generation & Pruning")
        st.write("Comparison of pruning performance under different constraint satisfaction settings for this query:")

        csp_modes = [
            ("Backtracking Only (BT)", False, False),
            ("BT + Forward Checking (FC)", True, False),
            ("BT + FC + Arc Consistency (AC-3)", True, True),
        ]
        csp_data = []
        for label, u_fc, u_ac3 in csp_modes:
            pats, st_res = planner.csp.solve(
                num_stops=num_stops,
                start_category=dataset.poi_categories[start_poi],
                end_category=dataset.poi_categories[end_poi],
                must_see_category=must_see,
                budget_hr=budget_hr,
                use_fc=u_fc,
                use_ac3=u_ac3,
                max_solutions=10,
            )
            csp_data.append({
                "CSP Algorithm": label,
                "Valid Patterns Found": len(pats),
                "Variable Assignments": st_res.assignments,
                "Backtracks": st_res.backtracks,
                "Domain Values Pruned": st_res.pruned_values
            })

        st.dataframe(pd.DataFrame(csp_data), use_container_width=True, hide_index=True)

        st.subheader("Evaluated Candidate Patterns")
        pat_rows = []
        for p_eval in output.pattern_evaluations:
            pat_str = " -> ".join(p_eval.get("pattern", []))
            csp_t = p_eval.get("expected_csp_time_hr", 0.0)
            nn_t = p_eval.get("predicted_nn_time_hr", 0.0)
            cost_val = p_eval.get("total_cost")
            pat_rows.append({
                "Pattern": pat_str,
                "CSP Expected Time": f"{csp_t:.2f}h",
                "NN Predicted Time": f"{nn_t:.2f}h",
                "Total Cost": f"{cost_val:.2f}" if cost_val is not None else "—",
                "Nodes Expanded": p_eval.get("nodes_expanded", "—"),
                "Status": "Accepted" if p_eval.get("accepted") else "Rejected (> Budget)"
            })
        if pat_rows:
            st.dataframe(pd.DataFrame(pat_rows), use_container_width=True, hide_index=True)

    with tab_compare:
        st.subheader("⚖️ Ablation: Neural Network ON vs. OFF (Popularity Baseline)")
        out_off = planner.plan_tour(
            start_poi=start_poi,
            end_poi=end_poi,
            time_budget_hr=budget_hr,
            must_see_category=must_see,
            num_stops=num_stops,
            start_hour=start_hour,
            search_algo=search_algo,
            use_fc=True,
            use_ac3=True,
            cost_mode="popularity",
            max_patterns=10,
        )

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### 🧠 Neural Network Model (NN ON)")
            if output.best_itinerary:
                st.write("**Route:**", " ➔ ".join([dataset.poi_names[p] for p in output.best_itinerary.pois]))
                st.write(f"**Plausibility Cost:** `{output.best_itinerary.total_cost:.3f}`")
                st.write(f"**Predicted Time:** `{output.best_itinerary.total_time_hr:.2f} hours`")
            else:
                st.write("No valid route found.")

        with colB:
            st.markdown("#### 📉 Baseline / Popularity (NN OFF)")
            if out_off.best_itinerary:
                st.write("**Route:**", " ➔ ".join([dataset.poi_names[p] for p in out_off.best_itinerary.pois]))
                st.write(f"**Plausibility Cost:** `{out_off.best_itinerary.total_cost:.3f}`")
                st.write(f"**Predicted Time:** `{out_off.best_itinerary.total_time_hr:.2f} hours`")
            else:
                st.write("No valid route found.")

        st.markdown("---")
        fig_path = Path("outputs/figures/demo_cooperation.png")
        if fig_path.exists():
            st.subheader("🖼️ Three-Module Cooperation Visual Summary")
            st.image(str(fig_path), caption="Generated Cooperation Plot (CSP Pruning + Search Nodes + Plausibility Gap)", use_container_width=True)
    with tab_experiments:
        st.subheader("📈 Scientific Experiments & Evaluation Suite (Seeds 42–46)")
        
        exp_select = st.radio(
            "Select Experiment to View:",
            ["Experiment 1: Search Comparison (UCS vs. A* vs. Greedy)",
             "Experiment 2: CSP Ablation (BT vs. BT+FC vs. BT+FC+AC-3)",
             "Experiment 3: Cost Model & Held-Out Trajectory Accuracy"],
            horizontal=True
        )
        
        if "Experiment 1" in exp_select:
            st.markdown("### Experiment 1: Search Comparison across 4 to 8 stops (5 Random Seeds)")
            st.write("Evaluating Search Nodes Expanded, Generated, Plausibility Cost, and Optimality Gap vs. UCS/Oracle:")
            p1 = Path("outputs/figures/exp1_search_plot.png")
            if p1.exists():
                st.image(str(p1), use_container_width=True)
            csv1 = Path("outputs/exp1_search_comparison.csv")
            if csv1.exists():
                st.dataframe(pd.read_csv(csv1), use_container_width=True)
                
        elif "Experiment 2" in exp_select:
            st.markdown("### Experiment 2: CSP Pruning Ablation across 4 to 8 stops (5 Random Seeds)")
            st.write("Comparing Backtracking (BT) vs. Forward Checking (FC) vs. Arc Consistency (AC-3):")
            p2 = Path("outputs/figures/exp2_csp_plot.png")
            if p2.exists():
                st.image(str(p2), use_container_width=True)
            csv2 = Path("outputs/exp2_csp_ablation.csv")
            if csv2.exists():
                st.dataframe(pd.read_csv(csv2), use_container_width=True)

        elif "Experiment 3" in exp_select:
            st.markdown("### Experiment 3: Cost Model vs. Markov & Popularity on Held-Out Edinburgh Test Set")
            st.write("Comparing Next-Place NLL, Top-1/Top-3 Accuracy, Duration MAE, and Trajectory Point-F1 / Interior-F1 / Pairs-F1:")
            p3 = Path("outputs/figures/exp3_f1_and_nn_plot.png")
            if p3.exists():
                st.image(str(p3), use_container_width=True)
            csv3 = Path("outputs/exp3_trajectory_f1_comparison.csv")
            if csv3.exists():
                st.dataframe(pd.read_csv(csv3), use_container_width=True)

    with tab_about:
        st.subheader("📊 Model & Evaluation Metrics")
        st.write("Pre-computed Dual-Head Neural Network vs. Baselines on Held-Out Edinburgh Test Set:")
        
        nn_te = nn_eval.get("nn_test", {})
        mk_te = nn_eval.get("markov_baseline_test", {})
        pop_te = nn_eval.get("popularity_baseline_test", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("NN Next-Place Top-1 Acc", f"{nn_te.get('top1_acc', 0.0):.1f}%")
            st.metric("NN Next-Place Top-3 Acc", f"{nn_te.get('top3_acc', 0.0):.1f}%")
            st.metric("NN Next-Place NLL", f"{nn_te.get('nll', 0.0):.3f}")
        with c2:
            st.metric("Markov Baseline Top-1 Acc", f"{mk_te.get('top1_acc', 0.0):.1f}%")
            st.metric("Markov Baseline Top-3 Acc", f"{mk_te.get('top3_acc', 0.0):.1f}%")
            st.metric("Markov Baseline NLL", f"{mk_te.get('nll', 0.0):.3f}")
        with c3:
            st.metric("Popularity Baseline Top-1 Acc", f"{pop_te.get('top1_acc', 0.0):.1f}%")
            st.metric("Popularity Baseline Top-3 Acc", f"{pop_te.get('top3_acc', 0.0):.1f}%")
            st.metric("Popularity Baseline NLL", f"{pop_te.get('nll', 0.0):.3f}")


if __name__ == "__main__":
    main()
