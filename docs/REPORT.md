---
title: "P6: Plausible Tour Itineraries: neural networks, search and constraint satisfaction on Flickr trajectories"
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
* a **CSP** that first chooses the *category pattern* of the tour (e.g. Park -> Museum -> Structure -> Cultural -> Structure -> Park);
* a **search** (UCS, A*, Greedy) that then picks the real places for a pattern, minimising an NN-based cost.

Rules of a valid pattern: neighbouring stops have different categories; the must-see category appears; no category
more than twice; the (lower bound of the) total time is within the budget. The first and last stop are the given places.

# 2. Data

Flickr tour dataset of Lim et al. (CIKM'16), Edinburgh: 28 places, 7 853 visits by 1 454 users, built from geotagged
photos (full column list and real/generated status in `docs/DATA_DICTIONARY.md`). Following the task we keep places
with >= 20 visits (28 -> 25 places; Structure 10, Historical 4, Park 4, Cultural 3, Museum 3, Entertainment 1), giving
7 828 visits, 1 448 users, 5 009 trajectories and **2 819 consecutive-place transitions**. Users are split 70/15/15 % into
train/validation/test (no user appears in two sets); the seed changes the split.

Three properties of the data matter for everything below. (1) **Durations are noisy:** they are last-photo minus
first-photo time, so 68 % of visits are below 5 minutes (a single photo). We clip to [5 min, 4 h]. (2) **Timestamps are not
local time:** photo activity peaks at stored hour 03-04 in all five cities, so we add 11 h (an assumption; documented).
(3) Walking times (4.5 km/h, x1.3 detour), budgets, number of stops, start hours and must-see categories in the
experiments are **generated**, not observed.

# 3. Design

## 3.1 Neural networks and transition models (`tourplan/nn.py`)

All neural heads are NumPy ReLU MLPs with manual back-propagation and Adam. The unchanged reference next-place NN takes current POI and cyclic hour features; the duration head predicts log visit duration from POI and arrival hour. The optional goal-aware NN adds request end POI and remaining stops. A shared candidate scorer uses category one-hot, normalized walking-time distances from current and end POIs, and normalized position in the trajectory. Goal-model architecture and learning rate are selected by validation NLL averaged across seeds 0-2 only; early stopping also uses validation data. Test users are used only for final metric reporting.

Comparisons include popularity, first-order Markov counts, and a second-order context Markov model using previous and current POIs with first-order backoff for unseen pairs. The context model tunes backoff on validation Top-1. Duration baselines are per-place and global means.

## 3.2 CSP (`tourplan/csp.py`)

Variables x0 through x(n-1) are the categories of the stops; x0 and x(n-1) are fixed by the start/end place.
Constraints: neighbours differ (binary), must-see present, per-category cap (<= 2 and <= number of places in the category),
and total time >= lower bound <= budget, where the bound adds, per stop, the shortest predicted stay in the category and,
per link, the shortest walk between categories (so no valid pattern is ever lost; the exact check is made by the search).
Three switchable solvers share the code: **BT** (constraints checked when their variables are assigned),
**BT+FC** (forward checking also for the global rules: saturated categories removed, must-see forced into its only
remaining slot, budget bound) and **BT+FC+AC-3** (AC-3 on the neighbour constraints at the root and re-established after
every assignment). Counters: assignments, backtracks, checks, revisions, prunings. All three return identical solution sets
(asserted in the experiment and in the tests). Value ordering uses the category-transition frequencies of the training data;
the 10 patterns passed on are the 10 most likely under that category chain.

## 3.3 Search (`tourplan/search.py`)

State = (position in the plan, previous place, current place, visited set, elapsed minutes); action = go to an unvisited place whose category is the one the pattern requires next (the last position is forced to be the end place). Step cost is -log P(next | context) + lambda times walking minutes, with lambda = 0.05 nats/min. Goal-aware costs use the requested end POI and stops remaining at each position; feature costs use normalized route position. The standard neural model remains the UI reference option. Model choice is not tuned on test users. The first transition in the context-count model has no previous POI and backs off to first-order counts.

Stays are whole minutes, max(5, rounded predicted minutes); states exceeding the budget are pruned using an admissible bound on remaining minutes. **UCS** orders by g, **A*** by g+h, and **Greedy** by h. The heuristic is a backward dynamic programme that relaxes the visited set and minimizes each transition over available context/hour values, so it remains admissible. Counters include nodes expanded/generated, pruned, and maximum open list.

## 3.4 Integration (`tourplan/integration.py`)

CSP (without budget) -> if empty: **"No solution"** with the reason (unknown must-see, too many stops for the 2-per-category
rule, two stops of the same category...). Otherwise CSP (with budget) -> <= 10 most likely patterns -> search each -> patterns
whose predicted stays exceed the budget are rejected -> the lowest-cost itinerary is returned. If all 10 are rejected the
planner first computes the **exact smallest feasible budget** (branch and bound over all valid patterns with a time-cost A*);
if that budget is within the request it widens the pattern list, otherwise it returns the **shortest itinerary with a warning**
that states the smallest budget that works. (This widening was added after a test showed that rejecting the top-10 patterns
could wrongly report "over budget".)

![Demo (`python run_demo.py`): itinerary, predicted-stay timeline, CSP effort, search effort and the cost of the 10 patterns.](../figures/demo.png){width=100%}

# 4. Results

All numbers are mean +/- standard deviation over **10 seeds** (each seed = different user split, NN initialisation and generated requests).

## 4.1 Next-place and duration metrics (held-out users)

All values are means ± standard deviations across 10 user splits (seeds 0-9). Top-1 is the exact next POI ranked first; Top-3 means it is among the first three. The full Edinburgh data has 2,819 transitions before splitting; each test fold contains a few hundred examples.

| next-place model | Test NLL (lower) | Top-1 accuracy | Top-3 accuracy |
|---|---:|---:|---:|
| Goal-aware NN | 1.912 ± 0.099 | 0.476 ± 0.034 | 0.697 ± 0.031 |
| Feature NN | 2.002 ± 0.092 | 0.480 ± 0.041 | 0.647 ± 0.034 |
| NN (current POI + hour) | 2.566 ± 0.052 | 0.235 ± 0.022 | 0.481 ± 0.030 |
| Markov counts (current POI) | 2.551 ± 0.054 | 0.244 ± 0.024 | 0.505 ± 0.029 |
| Context Markov (previous + current POI) | 2.538 ± 0.049 | 0.270 ± 0.025 | 0.511 ± 0.033 |
| Popularity | 2.836 ± 0.037 | 0.139 ± 0.011 | 0.329 ± 0.029 |

Paired differences are model minus first-order Markov, resampled by seed with a deterministic percentile bootstrap (20,000 resamples). Intervals excluding zero indicate a paired difference under this procedure. Negative NLL difference is better; positive Top-1 and Top-3 differences are better. Both new models improve NLL and Top-1 over Markov with intervals excluding zero. The reference NN's Top-1 and NLL intervals include zero; its Top-3 is lower.

| model vs first-order Markov | NLL difference (95% paired bootstrap CI) | Top-1 difference (95% CI) | Top-3 difference (95% CI) |
|---|---:|---:|---:|
| NN | +0.015 [-0.002, +0.029] | -0.009 [-0.020, +0.002] | -0.024 [-0.037, -0.010] |
| Context Markov | -0.013 [-0.026, +0.004] | +0.025 [+0.017, +0.033] | +0.005 [-0.000, +0.010] |
| Goal-aware NN | -0.639 [-0.710, -0.574] | +0.231 [+0.204, +0.262] | +0.191 [+0.165, +0.218] |
| Feature NN | -0.549 [-0.616, -0.481] | +0.236 [+0.203, +0.270] | +0.141 [+0.111, +0.169] |
| Popularity | +0.285 [+0.264, +0.306] | -0.105 [-0.119, -0.089] | -0.176 [-0.195, -0.158] |

| duration model | RMSE (log seconds) | MAE (log seconds) | MAE (minutes) |
|---|---:|---:|---:|
| NN | 1.214 ± 0.053 | 0.949 ± 0.039 | 27.2 ± 2.9 |
| per-place mean | 1.226 ± 0.055 | 0.959 ± 0.039 | 27.3 ± 3.0 |
| global mean | 1.241 ± 0.059 | 0.983 ± 0.039 | 27.4 ± 3.0 |

Duration estimates remain noisy because Flickr photo gaps are an imperfect proxy for time spent at a location. See `figures/nn_training.png` for training curves.

## 4.2 Experiment 1 - UCS vs A* vs Greedy (4-8 stops; 150 requests per size)

| stops | UCS expanded | A* expanded | Greedy expanded | A* time (ms) | UCS time (ms) | Greedy cost gap | Greedy optimal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 21.3 +/- 3.0 | 3.7 +/- 0.4 | 3.0 | 0.06 | 0.18 | 0.99 +/- 0.33 | 40% |
| 5 | 71.5 +/- 15.3 | 6.3 +/- 0.7 | 4.0 | 0.10 | 0.71 | 1.82 +/- 0.33 | 16% |
| 6 | 179.0 +/- 26.4 | 10.3 +/- 1.4 | 5.0 | 0.16 | 1.97 | 2.26 +/- 0.42 | 9% |
| 7 | 346.6 +/- 98.2 | 14.6 +/- 2.3 | 6.0 | 0.21 | 3.81 | 2.91 +/- 0.56 | 5% |
| 8 | 801.8 +/- 200.7 | 29.9 +/- 7.5 | 7.0 | 0.40 | 9.46 | 2.64 +/- 0.50 | 9% |

UCS and A* agree on the optimum for all 750 requests. At 8 stops A* expands about 27 times fewer nodes. Greedy is fastest, but its path is costlier and optimal in only 5-40% of cases. Timing is environment-dependent; node counts and cost gaps are the more stable comparisons. See `figures/exp1_search.png`.

## 4.3 Experiment 2 - BT vs BT+FC vs BT+FC+AC-3 (full enumeration; 250 requests per size)

| stops | variant | assignments | backtracks | time (ms) |
|---|---|---:|---:|---:|
| 6 | BT | 1,423 +/- 14 | 739 +/- 28 | 1.0 |
| 6 | BT+FC | 759 +/- 37 | 74 +/- 5 | 4.6 |
| 6 | BT+FC+AC-3 | 699 +/- 33 | 14 +/- 5 | 5.1 |
| 8 | BT | 22,807 +/- 1,018 | 14,787 +/- 594 | 21 |
| 8 | BT+FC | 9,421 +/- 861 | 1,402 +/- 87 | 65 |
| 8 | BT+FC+AC-3 | 8,435 +/- 785 | 416 +/- 71 | 71 |
| 10 | BT | 230,736 +/- 19,174 | 189,001 +/- 14,803 | 176 |
| 10 | BT+FC | 54,196 +/- 6,066 | 12,460 +/- 1,010 | 409 |
| 10 | BT+FC+AC-3 | 46,322 +/- 5,159 | 4,587 +/- 500 | 429 |

At 8 stops forward checking reduces backtracks by about 10 times and AC-3 by about 36 times, while all variants return identical solutions. On this small six-category problem, propagation takes longer in wall-clock time: plain BT is about 3 times faster at 8 stops. See `figures/exp2_csp.png`.

## 4.4 Experiment 3 - route costs vs held-out real trajectories

Each test trajectory with 3-8 distinct places and different start/end POIs becomes a request (925 requests over 10 seeds; 92.5 per seed). The must-see is a random intermediate category, the budget is at least the observed span, and the start hour is real. F1 is measured on intermediate stops by set overlap. Every model uses the same CSP, category prior, search and duration model; only transition costs differ.

| cost model | F1 (intermediate stops) | F1 (all stops) |
|---|---:|---:|
| NN costs | 0.525 ± 0.035 | 0.776 ± 0.015 |
| Goal-aware NN costs | 0.497 ± 0.045 | 0.763 ± 0.021 |
| Feature NN costs | 0.424 ± 0.049 | 0.727 ± 0.021 |
| Markov counts | 0.512 ± 0.038 | 0.769 ± 0.017 |
| Context Markov | 0.509 ± 0.045 | 0.768 ± 0.019 |
| Popularity | 0.453 ± 0.044 | 0.746 ± 0.020 |
| No model (travel only) | 0.439 ± 0.040 | 0.737 ± 0.019 |

Paired bootstrap differences versus first-order Markov are across seeds; a confidence interval containing zero does not support a difference. The standard NN has the highest mean F1, but its paired interval versus Markov includes zero. Goal-aware route F1 is also not significantly different from Markov by this interval; the feature model is significantly lower despite stronger one-step accuracy. This is why next-place metrics and end-to-end route similarity must be reported separately. Every evaluated plan met its generated budget. See `figures/exp3_f1.png`.

| model vs first-order Markov | paired F1 difference (intermediate stops), 95% bootstrap CI |
|---|---:|
| NN | +0.012 [-0.001, +0.029] |
| Goal-aware NN | -0.015 [-0.031, +0.003] |
| Feature NN | -0.088 [-0.110, -0.067] |
| Context Markov | -0.003 [-0.011, +0.006] |
| Popularity | -0.059 [-0.085, -0.034] |
| No model | -0.073 [-0.094, -0.052] |

## 4.5 Tests

32 pytest cases (`python -m pytest -q tests`): CSP agreement with brute force and between solvers, counters, AC-3 on a hand-made chain; search with a hand-computed answer,
agreement of UCS/A* with brute force on real-model instances in two cost modes, admissible heuristic, budget pruning; planner: **5 normal**
(4-8 stops), **3 edge** (3 stops, start = end, budget exactly at / 1 minute below the reported minimum), **2 impossible** (unknown category, 14 stops) plus a third
(2 stops of the same category), each returning a clear "No solution: ..." message.

# 5. Limits

* **One-step accuracy and itinerary quality differ.** Goal-aware Top-1 is 47.6% and feature-model Top-1 is 48.0%; both are significantly above first-order Markov by paired bootstrap intervals. These are useful gains but not perfect predictions. Feature-model route F1 is lower than Markov; the goal model's paired route-F1 interval includes zero.
* **Cross-seed overlap.** Hyperparameters are selected using validation NLL on seeds 0-2 only. Since each seed uses a different user split, a user held out for final reporting under one seed may appear in validation under another; this creates mild cross-seed tuning/reporting overlap. Within each seed, test users remain held out.
* **Noisy duration targets.** Visit duration is inferred from photo timestamps; many visits have only one photo. The +11 h clock correction is an assumption.
* **Generated inputs.** Walking speed, budgets, stop counts and must-see categories are generated. Experiment 3 uses a filtered subset of trajectories, and its budgets rarely bind. A random-itinerary F1 baseline was not run.
* **Shared category prior and search settings.** All route variants share the same training-count category prior. Lambda = 0.05, the ten-pattern cap and CSP lower bound were not tuned. FC and AC-3 reduce node counts but cost time for tiny domains.
* **Scope.** The reported evaluation uses Edinburgh only. Cross-city accuracy is not established.

# 6. Conclusions

The added goal-aware and feature-based models improve next-place prediction when request end and route position are available: feature NN reaches 48.0% Top-1 and goal-aware NN reaches 69.7% Top-3. Paired bootstrap intervals versus first-order Markov show significant NLL and Top-1 improvements. These one-step gains do not improve whole-route matching: standard NN has the highest mean intermediate-stop F1 (0.525), but its paired interval versus Markov includes zero; goal-aware scores 0.497 and feature-based scores 0.424. The feature model's route F1 is significantly below Markov, and goal-aware is not significantly different by the paired interval. We keep the standard NN as the reference route-cost choice, report the trade-off, and retain goal/feature options for request-conditioned next-place prediction. More route and user context is needed for stronger full-itinerary results.

*Reproduction:* `python make_data.py --city Edin --seed 0` then `python run_experiments.py --city Edin --seeds 10` (README).


**AI-assistance disclosure:** AI assistance was used to implement and revise code, tests, experiments, the report, and presentation. The team should describe its own review and contributions in `docs/CONTRIBUTIONS.md`.
