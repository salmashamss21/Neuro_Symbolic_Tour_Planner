"""Optional interactive Edinburgh itinerary interface (Streamlit)."""
import streamlit as st

from tourplan import csp, data, integration, nn


@st.cache_resource(show_spinner="Loading Flickr data and training the neural models…")
def load_planner(seed: int):
    city = data.get_city("Edin")
    models = nn.train_all(city, seed, quick=True)
    prior = csp.category_prior(city, models["data"]["tr_t"])
    return city, models, prior


st.set_page_config(page_title="P6 Tour Planner", page_icon="🗺️", layout="wide")
st.title("P6 · Plausible Tour Itineraries")
st.caption("Edinburgh routes planned with Flickr trajectories, a category CSP, and heuristic search.")

with st.sidebar:
    seed = st.number_input("Training seed", min_value=0, max_value=9999, value=0, step=1)
    city, models, prior = load_planner(int(seed))
    places = city.pois.reset_index(drop=True)
    labels = {int(row.poi_idx): f"{row.poi_id} · {row.category}" for row in places.itertuples()}
    start = st.selectbox("Start place", list(labels), index=0, format_func=lambda x: labels[x])
    end = st.selectbox("End place", list(labels), index=min(1, len(labels) - 1), format_func=lambda x: labels[x])
    budget = st.slider("Time budget (minutes)", 60, 600, 300, step=15)
    stops = st.slider("Total stops, including start and end", 3, min(10, len(labels)), 6)
    must = st.selectbox("Must-see category", ["(none)", *city.categories])
    hour = st.slider("Start hour", 0, 23, 10)
    run = st.button("Plan itinerary", type="primary")

if run:
    req = integration.Request(int(start), int(end), float(budget), None if must == "(none)" else must,
                              int(stops), float(hour))
    result = integration.plan(city, models["bundles"]["NN"], req, prior)
    st.subheader("Plan result")
    st.write(result.message)
    if result.itinerary is not None:
        st.code(integration.format_plan(city, result))
        idx = result.itinerary.places
        route = city.pois.iloc[idx][["lat", "lon"]].rename(columns={"lat": "latitude", "lon": "longitude"})
        st.map(route)
    else:
        st.info("Try a larger time budget, fewer stops, or another must-see category.")
    left, right = st.columns(2)
    left.metric("Search nodes expanded", result.search_stats.get("expanded", 0))
    right.metric("CSP backtracks", result.csp_stats.get("backtracks", 0))
else:
    st.info("Choose the request in the sidebar and select **Plan itinerary**.")

with st.expander("Model performance"):
    test = models["data"]["te_t"]
    nn_score = nn.next_metrics(models["nn_next"], test)
    markov_score = nn.next_metrics(models["markov"], test)
    st.write({"NN test metrics": nn_score, "Markov test metrics": markov_score})
