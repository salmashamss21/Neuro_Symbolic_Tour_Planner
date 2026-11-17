# P6: Plausible Tour Itineraries

An end-to-end Edinburgh tour planner that combines three methods:

- **Neural network:** predicts the next place and expected visit duration from Flickr trajectories.
- **Context-aware transition model:** predicts the next place from the previous and current stops, backing off to first-order counts when a pair is rare.
- **CSP:** selects a valid category pattern using backtracking (BT), forward checking (FC), and AC-3.
- **Search:** selects real places using UCS, A*, or Greedy, with counters and a brute-force oracle for small cases.

The integration evaluates up to ten likely patterns, checks predicted visit durations against the time budget, and gives a clear no-solution reason or smallest-budget fallback when needed.

## Setup

Python 3.10 or newer is required. The project uses pinned core dependencies and NumPy-only neural networks; no GPU is needed.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The raw Flickr data is included. To rebuild the Edinburgh processed data with the fixed default seed:

```bash
python make_data.py --city Edin --seed 0
```

The preprocessing script downloads missing source files, verifies/stores checksums, filters POIs, creates derived duration/travel features, and splits by user. If network access is unavailable, the included raw files are used.

## Run the project

```bash
python run_demo.py
python -m pytest -q tests
python run_experiments.py --city Edin --seeds 10
```

The demo prints the request, NN baseline metrics, CSP pruning counters, search trace, integrated itinerary, and NN-on/off comparison. The Streamlit planner opens directly on the planner. It includes the standard NN, goal-aware NN, feature-based NN, context Markov, first-order Markov, popularity, and no-model options. Full model training runs when the app loads; this matches the experiment training setup. It saves cooperation plots in `figures/`. Experiment outputs (raw runs, mean +/- standard deviation tables, and plots) are in `results/` and `figures/`.

Try a different request:

```bash
python run_demo.py --n-stops 7 --budget 240 --must Historical --start-id 7 --end-id 10 --hour 9
```

The optional Streamlit UI adapts the GitHub project's interactive interface to the included multi-module planner. Install it separately and run it with:

```bash
python -m pip install -r requirements-app.txt
streamlit run app.py
```

After selecting **Plan tour**, the map draws a blue line through the stops in order. The line connects POIs directly and does not represent street-level walking directions.

## Modules

The `data/`, `nn/`, `search/`, `csp/`, `integration/`, and `eval/` packages provide clearly named facades. The canonical implementation lives in `tourplan/` to keep the tested modules together.

| Module | Responsibility |
|---|---|
| `data/` and `tourplan/data.py` | Download, preprocessing, checksums, city data, splits, training samples |
| `nn/` and `tourplan/nn.py` | Next-place and duration networks; first/second-order Markov, popularity, and duration baselines |
| `csp/` and `tourplan/csp.py` | Category-pattern constraints; BT, FC, AC-3 and counters |
| `search/` and `tourplan/search.py` | UCS, A*, Greedy, brute-force oracle and search counters |
| `integration/` and `tourplan/integration.py` | CSP -> search -> duration validation and budget fallback |
| `eval/` and `tourplan/evaluation.py` | Trajectory metrics, generated requests and experiments |
| `tests/` | Unit and end-to-end tests, including normal, edge and impossible requests |
| `docs/` | Data dictionary, report, presentation, demo script, contributions template and git log |

## Deliverables

- `docs/DATA_DICTIONARY.md`: columns, data origin, and generated/preprocessed fields.
- `docs/REPORT.md` and `docs/REPORT.pdf`: project report.
- `docs/P6 Plausible Tour Itineraries.pptx`: presentation deck; `docs/PRESENTATION_SCRIPT.md` gives the 10-minute talk and 5-minute live demo sequence.
- `docs/CONTRIBUTIONS.md`: truthful individual-contribution template (complete it with the actual team contributions before submission).
- `docs/GIT_LOG.txt`: source history record.
- `results/` and `figures/`: experiments, summary tables and plots.

## Reproducibility and interpretation

The reported Edinburgh metrics use ten user splits (seeds 0-9). Goal-aware NN reached 47.6% Top-1 and 69.7% Top-3 (test NLL 1.912); feature NN reached 48.0% Top-1 and 64.7% Top-3 (NLL 2.002). Both improve paired next-place NLL and Top-1 versus first-order Markov. The standard NN has the highest mean intermediate-stop F1 (0.525); goal-aware is 0.497 and the feature model is 0.424. Goal-aware hyperparameters are selected by validation NLL on seeds 0-2 only. Because splits differ by seed, mild cross-seed validation/test overlap is possible; within each seed, test users are held out for final reporting. Flickr photo timestamps are noisy duration proxies, while walking speed, clock correction, stop counts, budgets and must-see categories are generated assumptions.
