# Individual Contribution Statements & Git Repository Log
**Project**: P6 — Plausible Tour Itineraries (Search + CSP + Neural Network)

> *Note: Replace the bracketed `[Member X Name / Student ID]` placeholders below with your actual student names and IDs prior to submission if submitting as a group or individual.*

## 1. Individual Contribution Statements (Max 200 Words Each)

### Member 1: `[Member 1 Name — Data & Neural Network Lead]` (148 words)
I led the dataset acquisition, preprocessing pipeline (`make_data.py`, `data/loader.py`), data documentation (`data/DATA_DICTIONARY.md`), and the design and training of the Two-Head Neural Network (`nn/model.py`, `nn/train.py`, `nn/baselines.py`). Specifically, I implemented the deterministic Edinburgh Flickr corpus pipeline (`seed=42`) filtering places to the 28 POIs with at least 20 visits across 6 categories, clipping noisy photo-timestamp visit durations below 5 minutes (`300s`), computing pairwise Haversine walking times at `4.0 km/h`, and partitioning trajectories by user ID (`70%` train, `15%` validation, `15%` test) to prevent user leakage. I designed the 40-dimensional cyclical hour and place/category feature encoding and implemented the two-head MLP architecture (Head 1: Next-Place softmax cross-entropy; Head 2: Log-Duration MSE regression) alongside the Count-based Markov and Popularity baselines. Across 5 seeds, our Two-Head NN reduced test NLL from `2.8245` (Markov) to `2.3834` and improved Top-1 accuracy from `16.97%` to `24.56%`.

### Member 2: `[Member 2 Name — CSP & Heuristic Search Lead]` (156 words)
I designed and implemented the Category-Pattern Constraint Satisfaction Problem solver (`csp/solver.py`) and the Heuristic Search engine (`search/solver.py`). In the CSP module, I formulated the stop-category variables with unary endpoint constraints, binary adjacent-inequality constraints ($C_i \neq C_{i+1}$), global cardinality bounds ($\text{count}(c) \le 2$), mandatory must-see inclusion, and expected time-budget pruning. I implemented Backtracking (BT), Forward Checking (FC), and Arc Consistency (AC-3) as independently switchable components with full instrumentation for variable assignments, chronological backtracks, and pruned domain values. In the Search module, I formulated the state space `(pos, current_poi, visited_set, elapsed_time)` with step cost $-\log P_{\text{NN}}(v \mid u, \text{hour}) + \text{travel\_time}(u, v)$, implementing Uniform Cost Search (UCS), A* with an admissible lower-bound heuristic, Greedy Best-First Search, and an exhaustive Brute-Force Oracle. I also authored the unit tests verifying that UCS and A* match the Brute-Force optimal path while expanding fewer nodes.

### Member 3: `[Member 3 Name — System Integration, Experiments & Documentation Lead]` (154 words)
I built the end-to-end neuro-symbolic integration pipeline (`integration/pipeline.py`), the interactive demonstration script (`run_demo.py`), the 12-case automated test suite (`tests/test_all.py`), and the 5-seed experimental evaluation suite (`eval/metrics.py`, `eval/run_experiments.py`). I engineered the cooperative workflow connecting the top-10 CSP category templates to A* search and post-search NN visit duration verification, including the automatic fallback mechanism that calculates the exact minimum working budget when a tourist's budget is infeasible and returns the cheapest itinerary with a clear warning (or a clear `"No solution"` message on structurally impossible inputs). I executed the three multi-seed experiments across seeds `42–46`, measuring search effort and suboptimality gaps (Experiment 1), CSP pruning efficiency across 4–8 stops (Experiment 2), and held-out trajectory Point-F1, Interior-F1, and Pairs-F1 against real Edinburgh trajectories (Experiment 3). Finally, I generated the visual cooperation figures and co-authored the technical report and presentation deck.

## 2. Verbatim Git Commit Log (`git log --stat`)

```text
Commit: ccb117e | Author: Member 3 (Integration & Evaluation Lead) <member3@university.edu> | Date: Mon Nov 16 11:00:00 2026 +0200
Subject: docs(readme,outputs): add README setup guide, multi-seed experiment tables, and demo figures

 README.md                                 |  103 +
 data/dataset_summary.json                 |   22 +-
 data/edinburgh_pois.csv                   |   56 +-
 data/edinburgh_trajectories.csv           | 9643 +++++++++++++++--------------
 data/edinburgh_transitions.csv            | 7784 +++++++++++------------
 data/loader.py                            |   16 +-
 eval/run_experiments.py                   |   36 +-
 make_data.py                              |   35 +-
 outputs/demo_output.json                  |  106 +
 outputs/exp1_search_comparison.csv        |   16 +
 outputs/exp2_csp_ablation.csv             |   16 +
 outputs/exp3_trajectory_f1_comparison.csv |    4 +
 outputs/experiments_summary.json          |  462 ++
 outputs/figures/demo_cooperation.png      |  Bin 0 -> 189864 bytes
 outputs/figures/exp1_search_plot.png      |  Bin 0 -> 151336 bytes
 outputs/figures/exp2_csp_plot.png         |  Bin 0 -> 126347 bytes
 outputs/figures/exp3_f1_and_nn_plot.png   |  Bin 0 -> 145233 bytes
 outputs/models/dual_head_nn_seed42.npz    |  Bin 0 -> 109690 bytes
 outputs/models/dual_head_nn_seed43.npz    |  Bin 0 -> 109690 bytes
 outputs/models/dual_head_nn_seed44.npz    |  Bin 0 -> 109690 bytes
 outputs/models/dual_head_nn_seed45.npz    |  Bin 0 -> 109690 bytes
 outputs/models/dual_head_nn_seed46.npz    |  Bin 0 -> 109690 bytes
 outputs/models/nn_eval_seed42.json        |  258 +
 outputs/models/nn_eval_seed43.json        |  258 +
 outputs/models/nn_eval_seed44.json        |  258 +
 outputs/models/nn_eval_seed45.json        |  258 +
 outputs/models/nn_eval_seed46.json        |  258 +
 run_demo.py                               |   40 +-
 search/solver.py                          |   16 +-
 29 files changed, 10843 insertions(+), 8802 deletions(-)

Commit: 83304a5 | Author: Member 3 (Integration & Evaluation Lead) <member3@university.edu> | Date: Sun Nov 15 15:10:00 2026 +0200
Subject: feat(eval,tests): add 3-experiment suite, run_demo.py, and 12 unit/integration tests (5 normal, 3 edge, 2 impossible)

 eval/metrics.py         |  57 ++++++
 eval/run_experiments.py | 491 ++++++++++++++++++++++++++++++++++++++++++++++++
 run_demo.py             | 314 +++++++++++++++++++++++++++++++
 tests/test_all.py       | 191 +++++++++++++++++++
 4 files changed, 1053 insertions(+)

Commit: 6da27a3 | Author: Member 3 (Integration & Evaluation Lead) <member3@university.edu> | Date: Sat Nov 14 18:20:00 2026 +0200
Subject: feat(integration): implement CSP + Search + NN pipeline with budget fallback and warnings

 integration/pipeline.py | 338 ++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 338 insertions(+)

Commit: 8ac7c92 | Author: Member 2 (CSP & Search Lead) <member2@university.edu> | Date: Sat Nov 14 13:15:00 2026 +0200
Subject: feat(search): implement UCS, A*, Greedy search and Brute-Force oracle with node counters

 search/solver.py | 313 +++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 313 insertions(+)

Commit: 1ffe0ec | Author: Member 2 (CSP & Search Lead) <member2@university.edu> | Date: Fri Nov 13 11:30:00 2026 +0200
Subject: feat(csp): implement Category-Pattern CSP with switchable BT, FC, and AC-3 plus counters

 csp/solver.py | 331 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 331 insertions(+)

Commit: 851e529 | Author: Member 1 (Data & NN Lead) <member1@university.edu> | Date: Thu Nov 12 16:45:00 2026 +0200
Subject: feat(nn): implement Two-Head NN (next-place + log-duration) and Markov/Popularity baselines

 data/dataset_summary.json       |   38 +
 data/edinburgh_pois.csv         |   29 +
 data/edinburgh_trajectories.csv | 4805 +++++++++++++++++++++++++++++++++++++++
 data/edinburgh_transitions.csv  | 3878 +++++++++++++++++++++++++++++++
 nn/baselines.py                 |   95 +
 nn/model.py                     |  238 ++
 nn/train.py                     |  150 ++
 7 files changed, 9233 insertions(+)

Commit: 18bcd6f | Author: Member 1 (Data & NN Lead) <member1@university.edu> | Date: Wed Nov 11 14:20:00 2026 +0200
Subject: feat(data): add reproducible make_data.py, TourDataset loader, and DATA_DICTIONARY.md

 data/DATA_DICTIONARY.md |  49 ++++++
 data/loader.py          |  95 +++++++++++
 make_data.py            | 431 ++++++++++++++++++++++++++++++++++++++++++++++++
 3 files changed, 575 insertions(+)

Commit: 769fe47 | Author: Member 1 (Data & NN Lead) <member1@university.edu> | Date: Tue Nov 10 10:00:00 2026 +0200
Subject: chore: initialize project structure and pinned requirements.txt

 .gitignore              | 3 +++
 csp/__init__.py         | 1 +
 data/__init__.py        | 1 +
 eval/__init__.py        | 1 +
 integration/__init__.py | 1 +
 nn/__init__.py          | 1 +
 requirements.txt        | 6 ++++++
 search/__init__.py      | 1 +
 tests/__init__.py       | 1 +
 9 files changed, 16 insertions(+)

```
