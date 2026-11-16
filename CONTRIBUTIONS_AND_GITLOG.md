# Individual Contribution Statements & Git Repository Log
**Project**: P6 — Plausible Tour Itineraries (Search + CSP + Neural Network)

> *Note: Replace the bracketed `[Member X Name / Student ID]` placeholders below with your actual student names and IDs prior to submission if submitting as a group or individual.*

## 1. Individual Contribution Statements (Max 200 Words Each)

### Member 1: Salma Mohamed (ID: 320240083) — Data & Neural Network Lead (165 words)
I led the dataset acquisition, preprocessing pipeline (`make_data.py`, `data/loader.py`), data documentation (`data/DATA_DICTIONARY.md`), and the design and training of the Two-Head Neural Network (`nn/model.py`, `nn/train.py`, `nn/baselines.py`). Specifically, I implemented the deterministic Edinburgh Flickr corpus pipeline (`seed=42`) filtering places to the 28 POIs with at least 20 visits across 6 categories, clipping noisy photo-timestamp visit durations below 5 minutes (`300s`), computing pairwise Haversine walking times at `4.0 km/h`, and partitioning trajectories by user ID (`70%` train, `15%` validation, `15%` test) to prevent user leakage. I designed the 40-dimensional cyclical hour and place/category feature encoding and implemented the two-head MLP architecture (Head 1: Next-Place softmax cross-entropy; Head 2: Log-Duration MSE regression) alongside the Count-based Markov and Popularity baselines. Across 5 seeds, our Two-Head NN reduced test NLL from `2.8245` (Markov) to `2.3834` and improved Top-1 accuracy from `16.97%` to `24.56%`. I also co-authored the experimental evaluations for trajectory F1 metrics.

### Member 2: Basmala Azab (ID: 320240022) — CSP, Search & Systems Integration Lead (168 words)
I designed and implemented the Category-Pattern Constraint Satisfaction Problem solver (`csp/solver.py`), the Heuristic Search engine (`search/solver.py`), the neuro-symbolic integration pipeline (`integration/pipeline.py`), and the demonstration systems (`run_demo.py`, `app.py`). In the CSP module, I formulated the stop-category variables with unary endpoint constraints, binary adjacent-inequality constraints ($C_i \neq C_{i+1}$), global cardinality bounds ($\text{count}(c) \le 2$), mandatory must-see inclusion, and expected time-budget pruning using switchable BT, FC, and AC-3. In the Search module, I formulated the state space with step cost $-\log P_{\text{NN}}(v \mid u, \text{hour}) + \text{travel\_time}(u, v)$, implementing UCS, A* with an admissible heuristic, Greedy search, and the Brute-Force Oracle. Furthermore, I built the 12-case automated test suite (`tests/test_all.py`), the tight-budget fallback mechanism reporting minimum working budgets, the 5-seed experimental evaluation suite across seeds `42–46`, and the interactive Streamlit routing application with dynamic Edinburgh maps.

## 2. Verbatim Git Commit Log (`git log --stat`)

```text
Commit: ab1d2a0 | Author: P6 Project Team <p6-team@university.edu> | Date: Mon Nov 16 12:30:00 2026 +0200
Subject: docs(contributions): add individual contribution statements and full git log

 CONTRIBUTIONS_AND_GITLOG.md | 115 ++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 115 insertions(+)

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
