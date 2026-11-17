"""Interactive Streamlit interface for the P6 Edinburgh tour planner."""
from __future__ import annotations

import json
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pydeck as pdk
import streamlit as st

from tourplan import csp, data, integration, nn


ROOT = Path(__file__).resolve().parent
AUTH_DB_PATH = Path(os.environ.get("TOUR_PLANNER_AUTH_DB", ROOT / "users.sqlite3")).expanduser()
PASSWORD_ITERATIONS = 600_000
CATEGORY_COLORS = {
    "Park": [34, 197, 94, 210],
    "Historical": [234, 88, 12, 210],
    "Cultural": [168, 85, 247, 210],
    "Museum": [59, 130, 246, 210],
    "Cafe": [245, 158, 11, 210],
    "Castle": [239, 68, 68, 210],
    "Structure": [14, 116, 144, 210],
    "Entertainment": [219, 39, 119, 210],
}


@st.cache_resource(show_spinner="Loading Edinburgh data and training the models…")
def load_system(seed: int):
    city = data.get_city("Edin")
    models = nn.train_all(city, seed, quick=True)
    prior = csp.category_prior(city, models["data"]["tr_t"])
    return city, models, prior


def format_clock(minutes: float) -> str:
    value = int(round(minutes)) % (24 * 60)
    return f"{value // 60:02d}:{value % 60:02d}"


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS, dklen=32
    )


def _open_auth_db():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(AUTH_DB_PATH, timeout=15)


def _initialize_auth_db() -> None:
    with _open_auth_db() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


def _validate_username(username: str) -> tuple[str | None, str | None]:
    username = username.strip()
    if not 3 <= len(username) <= 32:
        return None, "Use a username between 3 and 32 characters."
    if re.fullmatch(r"[A-Za-z0-9_.-]+", username) is None:
        return None, "Usernames may contain letters, numbers, periods, underscores, and hyphens."
    return username, None


def _create_account(username: str, password: str) -> tuple[bool, str]:
    normalized, error = _validate_username(username)
    if error:
        return False, error
    if not 12 <= len(password) <= 128:
        return False, "Use a password between 12 and 128 characters."

    salt = secrets.token_bytes(16)
    verifier = _password_hash(password, salt)
    try:
        with _open_auth_db() as db:
            db.execute(
                "INSERT INTO users (username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (normalized, salt, verifier, datetime.now(timezone.utc).isoformat()),
            )
    except sqlite3.IntegrityError:
        return False, "That username is already registered. Choose another one."
    except sqlite3.Error:
        return False, "Account storage is unavailable. Please try again later."
    return True, normalized


def _verify_account(username: str, password: str) -> str | None:
    normalized, _ = _validate_username(username)
    try:
        with _open_auth_db() as db:
            row = None if normalized is None else db.execute(
                "SELECT username, salt, password_hash FROM users WHERE username = ?",
                (normalized,),
            ).fetchone()
    except sqlite3.Error:
        return None

    # Do a same-cost derivation for unknown usernames to make account discovery harder.
    salt = row[1] if row else bytes(16)
    expected = row[2] if row else bytes(32)
    candidate = _password_hash(password[:128], salt)
    matches = hmac.compare_digest(candidate, expected)
    return row[0] if row and len(password) <= 128 and matches else None


def _render_auth_gate() -> None:
    try:
        _initialize_auth_db()
    except (OSError, sqlite3.Error):
        st.error("Account storage could not be opened. Check the deployment's writable database path.")
        st.stop()

    if st.session_state.get("auth_username"):
        return

    st.markdown('<div class="planner-kicker">P6 · Neuro-symbolic planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="planner-title">Welcome to Edinburgh</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="planner-subtitle">Sign in or create an account to plan a tour.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    sign_in_column, register_column = st.columns(2, gap="large")

    with sign_in_column:
        with st.container(border=True):
            st.subheader("Sign in")
            st.caption("Already have an account? Continue planning your tour.")
            with st.form("sign_in_form"):
                username = st.text_input("Username", autocomplete="username")
                password = st.text_input(
                    "Password", type="password", autocomplete="current-password"
                )
                submitted = st.form_submit_button(
                    "Sign in", type="primary", use_container_width=True
                )
            if submitted:
                account = _verify_account(username, password)
                if account:
                    st.session_state["auth_username"] = account
                    st.rerun()
                st.error("Username or password is incorrect.")

    with register_column:
        with st.container(border=True):
            st.subheader("Create your account")
            st.caption("New here? Register to start building a sightseeing itinerary.")
            with st.form("create_account_form"):
                new_username = st.text_input("Choose a username", autocomplete="username")
                new_password = st.text_input(
                    "Create a password", type="password", autocomplete="new-password"
                )
                confirm_password = st.text_input(
                    "Confirm password", type="password", autocomplete="new-password"
                )
                st.caption(
                    "Username: 3–32 letters, numbers, periods, underscores, or hyphens. "
                    "Password: 12–128 characters."
                )
                submitted = st.form_submit_button(
                    "Create account", type="primary", use_container_width=True
                )
            if submitted:
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    created, result = _create_account(new_username, new_password)
                    if created:
                        st.session_state["auth_username"] = result
                        st.rerun()
                    st.error(result)

    st.caption(
        "Passwords are stored as salted hashes. Accounts use local SQLite storage; "
        "see the README for deployment notes."
    )
    st.stop()


def route_rows(city, itinerary):
    rows, map_points, coordinates = [], [], []
    places = city.pois
    for i, place_idx in enumerate(itinerary.places):
        poi_id = int(places.poi_id.iloc[place_idx])
        category = str(places.category.iloc[place_idx])
        lat, lon = float(places.lat.iloc[place_idx]), float(places.lon.iloc[place_idx])
        stay = int(itinerary.stay[i])
        walk = (float(city.travel_min[place_idx, itinerary.places[i + 1]])
                if i + 1 < len(itinerary.places) else None)
        coordinates.append([lon, lat])
        map_points.append({
            "label": f"Stop {i + 1} · POI {poi_id}",
            "stop": i + 1,
            "poi_id": poi_id,
            "category": category,
            "lat": lat,
            "lon": lon,
            "color": CATEGORY_COLORS.get(category, [71, 85, 105, 210]),
        })
        rows.append({
            "Stop": i + 1,
            "POI": poi_id,
            "Category": category,
            "Arrival": format_clock(itinerary.arrive[i]),
            "Stay (min)": stay,
            "Walk to next (min)": None if walk is None else round(walk, 1),
        })
    return rows, map_points, coordinates


st.set_page_config(
    page_title="Edinburgh Tour Planner",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(100,116,139,.18); }
      .planner-kicker { color: #2563eb; font-size: .76rem; font-weight: 750;
                        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .25rem; }
      .planner-title { color: #0f172a; font-size: 2.15rem; line-height: 1.15;
                       font-weight: 760; margin: 0; }
      .planner-subtitle { color: #475569; font-size: 1rem; margin-top: .55rem; }
      .section-label { color: #64748b; font-size: .78rem; font-weight: 700;
                       letter-spacing: .09em; text-transform: uppercase; }
      .route-note { color: #64748b; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

_render_auth_gate()

st.markdown('<div class="planner-kicker">P6 · Neuro-symbolic planning</div>', unsafe_allow_html=True)
st.markdown('<div class="planner-title">Edinburgh Tour Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="planner-subtitle">Build a time-aware sightseeing itinerary from real Flickr travel patterns, '
    'category constraints, and route search.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state['auth_username']}**")
    if st.button("Sign out", use_container_width=True):
        st.session_state.pop("auth_username", None)
        st.session_state.pop("planner_result", None)
        st.session_state.pop("csp_comparison", None)
        st.session_state.pop("model_comparison", None)
        st.rerun()
    st.divider()
    st.header("🧭 Tour request")
    seed = st.number_input("Model seed", min_value=0, max_value=9999, value=0, step=1)
    city, models, prior = load_system(int(seed))
    places = city.pois.reset_index(drop=True)
    place_ids = [int(x) for x in places.poi_idx]
    label_by_idx = {
        int(row.poi_idx): f"POI {int(row.poi_id)} · {row.category}"
        for row in places.itertuples()
    }
    poi_id_to_idx = {int(row.poi_id): int(row.poi_idx) for row in places.itertuples()}
    start_default = poi_id_to_idx.get(7, place_ids[0])
    end_default = poi_id_to_idx.get(10, place_ids[min(1, len(place_ids) - 1)])
    start = st.selectbox("Start place", place_ids, index=place_ids.index(start_default),
                         format_func=lambda value: label_by_idx[value])
    end = st.selectbox("End place", place_ids, index=place_ids.index(end_default),
                       format_func=lambda value: label_by_idx[value])
    budget_hours = st.slider("Time budget", 1.0, 10.0, 5.0, 0.25, format="%.2f h")
    num_stops = st.slider("Total stops", 3, min(10, len(place_ids)), 6, 1)
    must_options = ["None", *city.categories]
    must_default = must_options.index("Museum") if "Museum" in must_options else 0
    must_label = st.selectbox("Must-see category", must_options, index=must_default)
    must_category = None if must_label == "None" else must_label
    start_hour = st.slider("Start time", 8.0, 16.0, 10.0, 0.5, format="%0.1f h")

    with st.expander("⚙️ Solver settings"):
        search_algo = st.selectbox(
            "Search algorithm",
            ["astar", "ucs", "greedy"],
            format_func={"astar": "A* Search", "ucs": "Uniform Cost Search", "greedy": "Greedy Best-First"}.__getitem__,
        )
        csp_mode = st.selectbox("CSP pruning", ["BT + FC + AC-3", "BT + FC", "Backtracking only"])
        cost_model = st.selectbox(
            "Route cost model",
            ["Neural network (best held-out route F1)",
             "Context-aware counts (higher next-place Top-1)",
             "Markov counts", "Popularity", "No model"],
        )
    solve = st.button("🚀 Plan tour", type="primary", use_container_width=True)

if solve:
    bundle_key = {
        "Neural network (best held-out route F1)": "NN",
        "Context-aware counts (higher next-place Top-1)": "Context Markov",
        "Markov counts": "Markov",
        "Popularity": "Popularity",
        "No model": "Off",
    }[cost_model]
    use_fc = csp_mode != "Backtracking only"
    use_ac3 = csp_mode == "BT + FC + AC-3"
    request = integration.Request(
        start=int(start),
        end=int(end),
        budget_min=float(budget_hours * 60),
        must_cat=must_category,
        n_stops=int(num_stops),
        start_hour=float(start_hour),
    )
    with st.spinner("Searching for a plausible itinerary…"):
        result = integration.plan(
            city,
            models["bundles"][bundle_key],
            request,
            prior,
            algo=search_algo,
            fc=use_fc,
            ac3=use_ac3,
        )
    st.session_state["planner_result"] = {
        "request": request,
        "result": result,
        "cost_model": cost_model,
        "csp_mode": csp_mode,
        "search_algo": search_algo,
    }
    st.session_state.pop("csp_comparison", None)
    st.session_state.pop("model_comparison", None)

saved = st.session_state.get("planner_result")
tabs = st.tabs([
    "🗺️ Itinerary & Map",
    "🧩 CSP Analysis",
    "⚖️ Model Comparison",
    "📈 Experiments",
    "ℹ️ About & Metrics",
])

with tabs[0]:
    if saved is None:
        st.info("Set your request in the sidebar and select **Plan tour** to build an itinerary.")
        st.caption("The default request starts at POI 7, ends at POI 10, and includes a museum stop.")
    else:
        request = saved["request"]
        result = saved["result"]
        if result.status == "ok":
            st.success(result.message)
        elif result.status == "over_budget":
            st.warning(result.message)
        else:
            st.error(result.message)

        itinerary = result.itinerary
        if itinerary is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total duration", f"{itinerary.total_min / 60:.2f} h",
                      f"Budget {request.budget_min / 60:.2f} h")
            c2.metric("Walking time", f"{itinerary.travel_min:.0f} min")
            c3.metric("Stops", f"{len(itinerary.places)}")
            c4.metric("Route cost", f"{itinerary.cost:.2f}")

            rows, map_points, coordinates = route_rows(city, itinerary)
            st.subheader("Tour schedule")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            left, right = st.columns([1.7, 1], gap="large")
            with left:
                st.subheader("Edinburgh route")
                mid_lat = sum(point["lat"] for point in map_points) / len(map_points)
                mid_lon = sum(point["lon"] for point in map_points) / len(map_points)
                path_layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": coordinates, "name": "Planned route"}],
                    get_path="path",
                    get_color=[30, 58, 138, 230],
                    get_width=7,
                    width_scale=20,
                    width_min_pixels=4,
                    pickable=True,
                )
                marker_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_points,
                    get_position=["lon", "lat"],
                    get_color="color",
                    get_radius=75,
                    radius_min_pixels=7,
                    pickable=True,
                )
                text_layer = pdk.Layer(
                    "TextLayer",
                    data=map_points,
                    get_position=["lon", "lat"],
                    get_text="label",
                    get_size=12,
                    get_color=[15, 23, 42, 255],
                    get_text_anchor="'start'",
                    get_alignment_baseline="'center'",
                    get_pixel_offset=[12, 0],
                )
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[path_layer, marker_layer, text_layer],
                        initial_view_state=pdk.ViewState(
                            latitude=mid_lat, longitude=mid_lon, zoom=12.5, pitch=15,
                        ),
                        map_style=None,
                        tooltip={"html": "<b>{label}</b><br/>Category: {category}"},
                    ),
                    use_container_width=True,
                )
                st.caption("The line connects planned stops in order; it is not turn-by-turn road routing.")
            with right:
                st.subheader("Plan details")
                st.markdown(f"**Cost model**  \n{saved['cost_model']}")
                st.markdown(f"**Search**  \n{saved['search_algo'].upper()}")
                st.markdown(f"**CSP pruning**  \n{saved['csp_mode']}")
                st.markdown(f"**Category pattern**  \n{' → '.join(integration.pattern_names(city, result.pattern))}")
                st.markdown(f"**Patterns evaluated**  \n{len(result.patterns_tried)}")
                payload = {
                    "budget_minutes": request.budget_min,
                    "total_minutes": itinerary.total_min,
                    "cost_model": saved["cost_model"],
                    "pattern": integration.pattern_names(city, result.pattern),
                    "stops": rows,
                }
                st.download_button(
                    "Download itinerary JSON",
                    data=json.dumps(payload, indent=2),
                    file_name="edinburgh_itinerary.json",
                    mime="application/json",
                    use_container_width=True,
                )
                with st.expander("Planner diagnostics"):
                    st.write({
                        "Search nodes expanded": result.search_stats.get("expanded", 0),
                        "Search nodes generated": result.search_stats.get("generated", 0),
                        "CSP assignments": result.csp_stats.get("assignments", 0),
                        "CSP backtracks": result.csp_stats.get("backtracks", 0),
                    })

with tabs[1]:
    st.subheader("Category-pattern pruning")
    st.caption("Compare the same request with backtracking, forward checking, and AC-3 enabled separately.")
    if st.button("Compare pruning methods", key="compare_csp"):
        req = saved["request"] if saved else integration.Request(
            int(start), int(end), budget_hours * 60, must_category, int(num_stops), float(start_hour)
        )
        problem = csp.build_csp(city, models["bundles"]["NN"], req.start, req.end,
                                req.n_stops, req.must_cat, req.budget_min, prior=prior)
        variants = [("BT", False, False), ("BT + FC", True, False), ("BT + FC + AC-3", True, True)]
        csp_rows, patterns = [], None
        for label, fc, ac3 in variants:
            solutions, stats = csp.solve(problem, fc=fc, ac3=ac3, max_solutions=10, order_values=True)
            csp_rows.append({
                "Solver": label,
                "Patterns found (up to 10)": len(solutions),
                "Assignments": stats.assignments,
                "Backtracks": stats.backtracks,
                "Prunings": stats.prunings,
                "AC-3 revisions": stats.revisions,
            })
            if label == "BT + FC + AC-3":
                patterns = csp.rank_patterns(solutions, prior, 10)
        st.session_state["csp_comparison"] = (csp_rows, patterns)
    csp_comparison = st.session_state.get("csp_comparison")
    if csp_comparison:
        csp_rows, patterns = csp_comparison
        st.dataframe(pd.DataFrame(csp_rows), use_container_width=True, hide_index=True)
        if patterns:
            st.markdown("#### Highest-ranked valid patterns")
            st.write([" → ".join(integration.pattern_names(city, pattern)) for pattern in patterns])
        else:
            st.info("No valid category pattern exists for this request. Check the start/end categories or stop count.")
    else:
        st.info("Select **Compare pruning methods** to measure assignments, backtracks, and domain pruning for this request.")

with tabs[2]:
    st.subheader("Neural model vs. simpler baselines")
    st.caption("All models use the same request and planner; only the transition-cost bundle changes.")
    if st.button("Compare cost models", key="compare_models"):
        req = saved["request"] if saved else integration.Request(
            int(start), int(end), budget_hours * 60, must_category, int(num_stops), float(start_hour)
        )
        compare_rows = []
        for label, key in [("Context-aware counts", "Context Markov"),
                           ("Neural network", "NN"), ("Markov counts", "Markov"),
                           ("Popularity", "Popularity"), ("No model", "Off")]:
            plan = integration.plan(city, models["bundles"][key], req, prior)
            it = plan.itinerary
            compare_rows.append({
                "Model": label,
                "Status": plan.status.replace("_", " ").title(),
                "Route cost": None if it is None else round(it.cost, 3),
                "Duration (h)": None if it is None else round(it.total_min / 60, 2),
                "Route": " → ".join(str(int(city.pois.poi_id.iloc[x])) for x in it.places) if it else "—",
            })
        st.session_state["model_comparison"] = compare_rows
    model_comparison = st.session_state.get("model_comparison")
    if model_comparison:
        comparison = pd.DataFrame(model_comparison)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        chart_data = comparison.dropna(subset=["Route cost"]).set_index("Model")[["Route cost"]]
        if not chart_data.empty:
            st.bar_chart(chart_data, horizontal=True)
    else:
        st.info("Select **Compare cost models** to run the same request with the neural and baseline models.")

with tabs[3]:
    st.subheader("Research experiments")
    exp_choice = st.radio(
        "Experiment",
        ["Search algorithms", "CSP pruning", "Trajectory F1"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if exp_choice == "Search algorithms":
        summary = pd.read_csv(ROOT / "results" / "exp1_search_summary.csv")
        st.caption("Mean expanded search nodes by itinerary length; error bars are shown in the full table as standard deviation.")
        chart = summary.pivot(index="n", columns="algo", values="expanded_mean")
        st.line_chart(chart)
        st.dataframe(summary, use_container_width=True, hide_index=True)
    elif exp_choice == "CSP pruning":
        summary = pd.read_csv(ROOT / "results" / "exp2_csp_summary.csv")
        chart = summary.pivot(index="n", columns="variant", values="backtracks_mean")
        st.line_chart(chart)
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        summary = pd.read_csv(ROOT / "results" / "exp3_f1_summary.csv")
        chart = summary.set_index("model")[["f1_mid_mean", "f1_all_mean"]]
        st.bar_chart(chart)
        st.dataframe(summary, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Model and dataset metrics")
    test = models["data"]["te_t"]
    metrics = []
    for label, key in [("Neural network", "nn_next"), ("Markov counts", "markov"), ("Popularity", "pop")]:
        score = nn.next_metrics(models[key], test)
        metrics.append({"Model": label, "Test NLL": score["nll"],
                        "Top-1 accuracy": score["top1"], "Top-3 accuracy": score["top3"]})
    context_score = nn.context_metrics(models["context_markov"], models["data"]["te_ctx"])
    metrics.insert(0, {"Model": "Context-aware counts (uses previous stop)",
                       "Test NLL": context_score["nll"],
                       "Top-1 accuracy": context_score["top1"],
                       "Top-3 accuracy": context_score["top3"]})
    st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
    duration = nn.duration_metrics(models["nn_dur"], models["data"]["te_d"])
    st.metric("Duration model MAE", f"{duration['mae_min']:.1f} min")
    st.markdown("#### Dataset")
    info = city.meta
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Places", info.get("n_places", city.n_places))
    m2.metric("Visits", info.get("n_visits", len(city.visits)))
    m3.metric("Trajectories", info.get("n_trajs", int(city.visits.traj.nunique())))
    m4.metric("Transitions", info.get("n_transitions", "—"))
    st.caption("Source: Flickr tour-cikm16 Edinburgh subset. Photo timestamps are noisy duration proxies; walking speed, budgets, stop counts, and must-see categories are generated assumptions.")
