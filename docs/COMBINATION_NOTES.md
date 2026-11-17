# How the two project versions were combined

The GitHub repository and the `p6_tour_itineraries.zip` archive implement the same P6 pipeline, but they use different data layouts and incompatible APIs. The final project keeps the archive's `tourplan/` implementation as the canonical core because it includes the upstream Flickr raw files, a reproducible multi-city data pipeline, a ten-seed experiment bundle, the report source, and an honest contribution template.

The final project also adopts the GitHub version's clearly named module organization and interactive UI direction:

- Added public package facades under `data/`, `nn/`, `csp/`, `search/`, `integration/`, and `eval/`; the canonical implementations remain in `tourplan/` and the README documents that mapping.
- Added an optional Streamlit interface that calls the archived planner directly, so its inputs, route and test data stay consistent with the report.
- Kept the archive's data dictionary, report, experiment tables, plots, and contribution/log material.
- Included the user's attached PowerPoint deck and report PDF in `docs/`. The attached report PDF matches the archive's PDF byte for byte; the attached presentation is the 12-slide presentation for the same ten-seed result set.

The GitHub repository's original interface imports its own `data.loader` and `integration.pipeline` contracts and its `make_data.py` describes a different data build. Copying those files unchanged would have paired the UI with the wrong data contract. The new `app.py` is therefore a small compatible interface over the canonical archive code.

Before submission, each team member should replace the placeholders in `CONTRIBUTIONS.md` with a truthful account of their own work. The existing source history is retained in `.git` and `GIT_LOG.txt`; no authorship or contribution claims were invented.
