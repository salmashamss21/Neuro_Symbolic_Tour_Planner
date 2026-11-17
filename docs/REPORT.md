---
title: "P6 — Plausible Tour Itineraries: neural networks, search and constraint satisfaction on Flickr trajectories"
author: "Team report (Edinburgh, tour-cikm16 data)"
geometry: margin=2cm
fontsize: 10pt
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# 1. Problem

A tourist gives a **start place, an end place, a time budget, a must-see category** and a number of stops.
The system must return an itinerary with arrival times and stay durations that (i) obeys simple rules, (ii) fits the
budget and (iii) is *plausible*, i.e. similar to what real travellers do. We combine three techniques:

* a **neural network** that learns from real trajectories which place people visit next and how long they stay;
* a **CSP** that first chooses the *category pattern* of the tour (e.g. Park → Museum → Structure → Cultural → Structure → Park);
* a **search** (UCS, A*, Greedy) that then picks the real places for a pattern, minimising an NN-based cost.

Rules of a valid pattern: neighbouring stops have different categories; the must-see category appears; no category
more than twice; the (lower bound of the) total time is within the budget. The first and last stop are the given places.

# 2. Data

Flickr tour dataset of Lim et al. (CIKM'16), Edinburgh: 28 places, 7 853 visits by 1 454 users, built from geotagged
photos (full column list and real/generated status in `docs/DATA_DICTIONARY.md`). Following the task we keep places
with ≥ 20 visits (28 → 25 places; Structure 10, Historical 4, Park 4, Cultural 3, Museum 3, Entertainment 1), giving
7 828 visits, 1 448 users, 5 009 trajectories and **2 819 consecutive-place transitions**. Users are split 70/15/15 % into
train/validation/test (no user appears in two sets); the seed changes the split.

Three properties of the data matter for everything below. (1) **Durations are noisy:** they are last-photo minus
first-photo time, so 68 % of visits are below 5 minutes (a single photo). We clip to [5 min, 4 h]. (2) **Timestamps are not
local time:** photo activity peaks at stored hour 03–04 in all five cities, so we add 11 h (an assumption; documented).
(3) Walking times (4.5 km/h, ×1.3 detour), budgets, number of stops, start hours and must-see categories in the
experiments are **generated**, not observed.

# 3. Design

## 3.1 Neural networks and transition models (`tourplan/nn.py`)

Both neural heads are small ReLU MLPs written in NumPy with manual back-propagation and Adam. Input to the next-place NN is a one-hot current place plus sine/cosine hour features (24 h and 12 h harmonics); its softmax masks the current POI. The duration head predicts log visit duration from place and arrival hour. Weight decay 0.01 and architecture were selected on validation data; training uses early stopping.

We compare the NN with popularity and two count models. First-order Markov counts use the current POI. Context Markov uses the previous and current POIs, backs off to first-order counts for unseen pairs, and selects its backoff strength on validation Top-1 accuracy. It is a discrete count model, not a second neural head. This uses a little more route context than the project minimum while keeping the NN's requested input unchanged. The duration baselines are per-place and global means.

## 3.2 CSP (`tourplan/csp.py`)

Variables x0 through x(n-1) are the categories of the stops; x0 and x(n-1) are fixed by the start/end place.
Constraints: neighbours differ (binary), must-see present, per-category cap (≤ 2 and ≤ number of places in the category),
and total time ≥ lower bound ≤ budget, where the bound adds, per stop, the shortest predicted stay in the category and,
per link, the shortest walk between categories (so no valid pattern is ever lost; the exact check is made by the search).
Three switchable solvers share the code: **BT** (constraints checked when their variables are assigned),
**BT+FC** (forward checking also for the global rules: saturated categories removed, must-see forced into its only
remaining slot, budget bound) and **BT+FC+AC-3** (AC-3 on the neighbour constraints at the root and re-established after
every assignment). Counters: assignments, backtracks, checks, revisions, prunings. All three return identical solution sets
(asserted in the experiment and in the tests). Value ordering uses the category-transition frequencies of the training data;
the 10 patterns passed on are the 10 most likely under that category chain.

## 3.3 Search (`tourplan/search.py`)

State = (position in the plan, previous place, current place, visited set, elapsed minutes); action = go to an unvisited place whose category is the one the pattern requires next (the last position is forced to be the end place). Step cost is -log P(next | previous, current) + lambda times walking minutes, with lambda = 0.05 nats/min. The Streamlit planner defaults to the neural model because it has the best held-out itinerary F1; context Markov is selectable for stronger exact next-place Top-1. The first transition has no previous POI and backs off to first-order counts.

Stays are whole minutes, max(5, rounded predicted minutes); states exceeding the budget are pruned using an admissible bound on remaining minutes. **UCS** orders by g, **A*** by g+h, and **Greedy** by h. The heuristic is a backward dynamic programme that relaxes the visited set and minimizes each transition over available context/hour values, so it remains admissible. Counters include nodes expanded/generated, pruned, and maximum open list.

## 3.4 Integration (`tourplan/integration.py`)

CSP (without budget) → if empty: **"No solution"** with the reason (unknown must-see, too many stops for the 2-per-category
rule, two stops of the same category…). Otherwise CSP (with budget) → ≤ 10 most likely patterns → search each → patterns
whose predicted stays exceed the budget are rejected → the lowest-cost itinerary is returned. If all 10 are rejected the
planner first computes the **exact smallest feasible budget** (branch and bound over all valid patterns with a time-cost A*);
if that budget is within the request it widens the pattern list, otherwise it returns the **shortest itinerary with a warning**
that states the smallest budget that works. (This widening was added after a test showed that rejecting the top-10 patterns
could wrongly report "over budget".)

![Demo (`python run_demo.py`): itinerary, predicted-stay timeline, CSP effort, search effort and the cost of the 10 patterns.](../figures/demo.png){width=100%}

# 4. Results

All numbers are mean ± standard deviation over **10 seeds** (each seed = different user split, NN initialisation and generated requests).

## 4.1 Next-place and duration metrics (held-out users)

All next-place metrics are mean ± standard deviation across 10 user splits (seeds 0-9). Top-1 means the exact next POI is ranked first; Top-3 means it appears among the first three. The full Edinburgh data has 2,819 transitions before splitting; each held-out test fold contains only a few hundred examples.

| next-place model | Test NLL (lower) | Top-1 accuracy | Top-3 accuracy |
|---|---:|---:|---:|
| Context Markov (previous + current POI) | **2.538 ± 0.049** | **0.270 ± 0.025** | **0.511 ± 0.033** |
| NN (current POI + hour) | 2.566 ± 0.052 | 0.235 ± 0.022 | 0.481 ± 0.030 |
| Markov counts (current POI) | 2.551 ± 0.054 | 0.244 ± 0.024 | 0.505 ± 0.029 |
| Popularity | 2.836 ± 0.037 | 0.139 ± 0.011 | 0.329 ± 0.029 |

The context model raises held-out Top-1 by 3.4 percentage points over the NN and 2.5 points over first-order Markov counts. It reaches 51.1% Top-3, but 27.0% Top-1 is still modest; the model should not be described as highly accurate. The NN remains the learned component required by the project, but it does not outperform counts on next-place prediction.

| duration model | RMSE (log seconds) | MAE (log seconds) | MAE (minutes) |
|---|---:|---:|---:|
| NN | **1.214 ± 0.053** | **0.949 ± 0.039** | 27.2 ± 2.9 |
| per-place mean | 1.226 ± 0.055 | 0.959 ± 0.039 | 27.3 ± 3.0 |
| global mean | 1.241 ± 0.059 | 0.983 ± 0.039 | 27.4 ± 3.0 |

Duration predictions are slightly better than the per-place mean, but still noisy because Flickr photo gaps are an imperfect proxy for time spent at a location. See `figures/nn_training.png` for training curves.

## 4.2 Experiment 1 - UCS vs A* vs Greedy (4-8 stops; 150 requests per size)

| stops | UCS expanded | A* expanded | Greedy expanded | A* time (ms) | UCS time (ms) | Greedy cost gap | Greedy optimal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 21.3 ± 3.0 | 3.7 ± 0.4 | 3.0 | 0.06 | 0.18 | 0.99 ± 0.33 | 40% |
| 5 | 71.5 ± 15.3 | 6.3 ± 0.7 | 4.0 | 0.10 | 0.71 | 1.82 ± 0.33 | 16% |
| 6 | 179.0 ± 26.4 | 10.3 ± 1.4 | 5.0 | 0.16 | 1.97 | 2.26 ± 0.42 | 9% |
| 7 | 346.6 ± 98.2 | 14.6 ± 2.3 | 6.0 | 0.21 | 3.81 | 2.91 ± 0.56 | 5% |
| 8 | 801.8 ± 200.7 | 29.9 ± 7.5 | 7.0 | 0.40 | 9.46 | 2.64 ± 0.50 | 9% |

UCS and A* agree on the optimum for all 750 requests. At 8 stops A* expands about 27 times fewer nodes. Greedy is fastest, but its path is costlier and optimal in only 5-40% of cases. Timing is environment-dependent; node counts and cost gaps are the more stable comparisons. See `figures/exp1_search.png`.

## 4.3 Experiment 2 - BT vs BT+FC vs BT+FC+AC-3 (full enumeration; 250 requests per size)

| stops | variant | assignments | backtracks | time (ms) |
|---|---|---:|---:|---:|
| 6 | BT | 1,423 ± 14 | 739 ± 28 | 1.0 |
| 6 | BT+FC | 759 ± 37 | 74 ± 5 | 4.6 |
| 6 | BT+FC+AC-3 | 699 ± 33 | 14 ± 5 | 5.1 |
| 8 | BT | 22,807 ± 1,018 | 14,787 ± 594 | 21 |
| 8 | BT+FC | 9,421 ± 861 | 1,402 ± 87 | 65 |
| 8 | BT+FC+AC-3 | 8,435 ± 785 | 416 ± 71 | 71 |
| 10 | BT | 230,736 ± 19,174 | 189,001 ± 14,803 | 176 |
| 10 | BT+FC | 54,196 ± 6,066 | 12,460 ± 1,010 | 409 |
| 10 | BT+FC+AC-3 | 46,322 ± 5,159 | 4,587 ± 500 | 429 |

At 8 stops forward checking reduces backtracks by about 10 times and AC-3 by about 36 times, while all variants return identical solutions. On this small six-category problem, propagation takes longer in wall-clock time: plain BT is about 3 times faster at 8 stops. See `figures/exp2_csp.png`.

## 4.4 Experiment 3 - route costs vs held-out real trajectories

Each test trajectory with 3-8 distinct places and different start/end POIs becomes a request (925 requests over 10 seeds, 92.5 per seed): start/end are the first/last POIs, the must-see is a random intermediate category, the budget is at least the real span, and the start hour is real. F1 is measured on intermediate stops by set overlap. All models use the same CSP, category prior, search and duration model; only transition costs differ. "No model" uses uniform transitions and therefore favors short walks.

| cost model | F1 (intermediate stops) | F1 (all stops) |
|---|---:|---:|
| NN costs | **0.525 ± 0.035** | **0.776 ± 0.015** |
| Markov counts | 0.512 ± 0.038 | 0.769 ± 0.017 |
| Context Markov | 0.509 ± 0.045 | 0.768 ± 0.019 |
| Popularity | 0.453 ± 0.044 | 0.746 ± 0.020 |
| No model (travel only) | 0.439 ± 0.040 | 0.737 ± 0.019 |

Context Markov has the highest next-step Top-1 accuracy, but it does not produce the highest whole-itinerary F1. The NN gives the best average trajectory F1, while first-order and context counts are close. This shows why next-place accuracy and end-to-end route similarity must be reported separately. Every tested plan met its generated budget. See `figures/exp3_f1.png`.

## 4.5 Tests

31 pytest cases (`python -m pytest -q tests`): CSP agreement with brute force and between solvers, counters, AC-3 on a hand-made chain; search with a hand-computed answer,
agreement of UCS/A* with brute force on real-model instances in two cost modes, admissible heuristic, budget pruning; planner: **5 normal**
(4–8 stops), **3 edge** (3 stops, start = end, budget exactly at / 1 minute below the reported minimum), **2 impossible** (unknown category, 14 stops) plus a third
(2 stops of the same category), each returning a clear "No solution: …" message.

# 5. Limits

* **Next-place accuracy remains modest.** Context Markov reaches 27.0% Top-1 on held-out users, not a high absolute score. The data contains only 25 Edinburgh POIs and sparse transitions; it omits user preferences, trip purpose, weather, opening hours and route context beyond two stops.
* **Noisy duration targets.** Visit duration is inferred from photo timestamps; many visits have only one photo. The +11 h clock correction is an assumption.
* **Generated inputs.** Walking speed, budgets, stop counts and must-see categories are generated. Experiment 3 uses a filtered subset of trajectories, and its budgets rarely bind. A random-itinerary F1 baseline was not run.
* **Shared category prior.** All route variants rank category patterns with the same training-count prior.
* **Search settings.** lambda = 0.05, the ten-pattern cap and CSP lower bound were not tuned. FC and AC-3 reduce node counts but cost time for tiny domains.
* **Scope.** The reported evaluation uses Edinburgh only. The data pipeline supports additional cities, but cross-city accuracy is not established.

# 6. Conclusions

The planner combines neural predictions, category constraints and search to produce valid, time-aware tours. Context Markov raises exact next-place Top-1 from 23.5% for the NN to 27.0% by using the previous and current POIs; its Top-3 is 51.1%. That is the strongest one-step accuracy measured here, but it is not high in absolute terms. For complete itineraries, the NN has the strongest held-out F1 (0.525), slightly ahead of first-order counts (0.512) and context counts (0.509). The app exposes both models and labels the different objectives. More data and user/trip context are needed for substantially higher accuracy. Flickr durations remain noisy, while the CSP and A* search enforce route rules and budget constraints.

*Reproduction:* `python make_data.py --city Edin --seed 0` then `python run_experiments.py --city Edin --seeds 10` (README).
