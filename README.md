# P6: Plausible Tour Itineraries (Search + CSP + Neural Network)

An end-to-end neuro-symbolic tour recommendation and itinerary planning system for **Edinburgh, UK**, built on the Flickr Tour Dataset (`tour-cikm16`, [Lim et al., IJCAI 2015](https://kwanhui.github.io/publications/2015-IJCAI-personalTour.pdf); [Chen et al., CIKM 2016](https://arxiv.org/html/1608.07051v1), repository: `https://github.com/computationalmedia/tour-cikm16`).

Given a tourist's **start place**, **end place**, **time budget**, **number of stops**, and **must-see category** (e.g., `Museum`), the system synthesizes a sightseeing route that real travellers would plausibly follow while strictly respecting category sequencing and time budget constraints.

## 1. Repository Architecture (Clearly Named Modules)

```text
p6_plausible_tour_itineraries/
├── make_data.py                  # Reproducible dataset preprocessing script (fixed seed = 42)
├── run_demo.py                   # Integrated live demo (CSP + Search + Two-Head NN + plots)
├── requirements.txt              # Pinned Python package dependencies
├── README.md                     # Environment setup and exact reproduction commands
├── CONTRIBUTIONS_AND_GITLOG.md   # Individual contribution statements (<=200 words each) + git log
├── data/
│   ├── __init__.py
│   ├── loader.py                 # TourDataset class, Haversine distance & 4.0 km/h travel times
│   ├── DATA_DICTIONARY.md        # Every column documented (real vs. generated & how created)
│   ├── edinburgh_pois.csv        # 28 Edinburgh POIs (>=20 visits) across 6 categories
│   ├── edinburgh_trajectories.csv# 936 trajectories (4,843 visits), durations clipped >= 5 min
│   ├── edinburgh_transitions.csv # Consecutive POI-to-POI transitions with user-level splits
│   └── dataset_summary.json      # Preprocessing summary metadata
├── nn/
│   ├── __init__.py
│   ├── model.py                  # DualHeadTourNN: Head 1 (Next-Place MLP) & Head 2 (Log-Duration MLP)
│   ├── baselines.py              # MarkovCountBaseline & PopularityBaseline models
│   └── train.py                  # Joint Adam training & test evaluation vs. baselines
├── csp/
│   ├── __init__.py
│   └── solver.py                 # CategoryPatternCSP: switchable BT, FC, and AC-3 + counters
├── search/
│   ├── __init__.py
│   └── solver.py                 # TourSearchSolver: UCS, A*, Greedy, and Brute-Force Oracle + counters
├── integration/
│   ├── __init__.py
│   └── pipeline.py               # IntegratedTourPlanner: CSP -> Search -> NN duration check + fallback
├── eval/
│   ├── __init__.py
│   ├── metrics.py                # Point-F1, Interior-F1, and Ordered Pairs-F1 metrics
│   └── run_experiments.py        # Runs all 3 experiments across 5 seeds (42..46) + generates plots
├── tests/
│   ├── __init__.py
│   └── test_all.py               # 12 automated tests (5 normal, 3 edge, 2 impossible + 2 unit tests)
└── outputs/
    ├── demo_output.json          # Detailed trace of run_demo.py
    ├── exp1_search_comparison.csv# Experiment 1 table (UCS vs. A* vs. Greedy, 4-8 stops, 5 seeds)
    ├── exp2_csp_ablation.csv     # Experiment 2 table (BT vs. BT+FC vs. BT+FC+AC-3, 5 seeds)
    ├── exp3_trajectory_f1_comparison.csv # Experiment 3 table (NN vs. Markov vs. Popularity, 5 seeds)
    └── figures/                  # Generated PNG plots for demo and experiments
```

## 2. Environment Setup & Pinned Dependencies

Ensure Python 3.10+ is installed, then install the pinned packages from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pinned `requirements.txt`:
- `numpy==2.1.3`
- `pandas==2.2.3`
- `scipy==1.15.2`
- `matplotlib==3.10.1`
- `seaborn==0.13.2`
- `tabulate==0.9.0`

## 3. Exact Run Commands

Run all commands from the root `p6_plausible_tour_itineraries/` directory with `PYTHONPATH=.`:

### Step 1 — Reproducible Data Preprocessing (`make_data.py`)
```bash
PYTHONPATH=. python3 make_data.py --seed 42 --out-dir data
```
Downloads/builds the Edinburgh Flickr dataset (`28` POIs with $\ge 20$ visits across `6` categories: `Cafe`, `Castle`, `Cultural`, `Historical`, `Museum`, `Park`), clips noisy photo-timestamp durations below `5` minutes (`300s`), computes Haversine walking times at `4.0 km/h`, and partitions users into leak-free Train (`70%`), Validation (`15%`), and Test (`15%`) splits.

### Step 2 — Train & Benchmark the Two-Head Neural Network
```bash
PYTHONPATH=. python3 nn/train.py
```
Trains `DualHeadTourNN` (Head 1: Next-Place Cross-Entropy; Head 2: Log-Duration MSE) and compares test NLL, Top-1/Top-3 accuracy, and duration MAE against the Count-based Markov and Popularity baselines.

### Step 3 — Run the Integrated Demo (`run_demo.py`)
```bash
PYTHONPATH=. python3 run_demo.py
```
Runs the end-to-end Edinburgh 5-hour Museum query (`Princes Street Gardens -> ... -> Calton Hill`), prints the CSP pruning trace (`BT` vs. `BT+FC` vs. `BT+FC+AC-3`), candidate pattern evaluations, the optimal itinerary (`Park -> Historical -> Cultural -> Museum -> Cafe -> Park`, `4.91h`), the `NN ON` vs. `NN OFF` comparison, the tight-budget fallback (`2.40h` budget warning reporting the smallest working budget `4.85h`), and saves `outputs/figures/demo_cooperation.png`.

### Step 4 — Run the Automated Test Suite (5 Normal, 3 Edge, 2 Impossible + Unit Tests)
```bash
PYTHONPATH=. python3 tests/test_all.py
```
Executes all 12 unit and integration tests, verifying that impossible cases return a clear `"No solution"` message and that `UCS` and `A*` match the exhaustive Brute-Force Oracle on a small pool.

### Step 5 — Reproduce All Three Multi-Seed Experiments (Seeds 42–46)
```bash
PYTHONPATH=. python3 eval/run_experiments.py
```
Runs Experiments 1, 2, and 3 across 5 random seeds (`42, 43, 44, 45, 46`), exporting summary CSV tables (`mean ± std`) to `outputs/` and publication plots to `outputs/figures/`.
