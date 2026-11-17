# 10-minute talk + 5-minute live demo

Slides: the "P6 Plausible Tour Itineraries" deck (Slides artifact; also exportable to PDF/PPTX). Timings below.

## Talk (10 min)
| min | slide | say |
|---|---|---|
| 0:00 | 1 Title / problem | tourist gives start, end, budget, must-see → plausible route that fits time |
| 1:00 | 2 Three parts | NN = what is plausible; CSP = which category pattern; Search = which real places |
| 2:00 | 3 Data | tour-cikm16, Edinburgh, 25 places, 2 819 transitions; durations noisy (68 % < 5 min); generated parts |
| 3:00 | 4 NN and transition accuracy | context counts reach 27.0% Top-1 vs 23.5% for the NN; Top-3 reaches 51.1% |
| 4:00 | 5 CSP | variables/constraints; BT vs FC vs AC-3 |
| 5:00 | 6 Search | state includes previous and current stop; DP heuristic; A* optimal |
| 6:00 | 7 Integration | 10 patterns → search → budget check → exact smallest budget / warning |
| 7:00 | 8 Exp 1 | A* 27× fewer nodes than UCS; Greedy cheap but 5–40 % optimal |
| 8:00 | 9 Exp 2 | FC/AC-3: 10×/36× fewer backtracks but 3× slower in Python |
| 8:45 | 10 Exp 3 | F1: NN .525, Markov .512, context counts .509, Popularity .453, no model .439 |
| 9:30 | 11 Limits & conclusion | context helps, but exact next-place prediction remains difficult on sparse Flickr data |

## Live demo (5 min) — all commands run from the project folder
1. **Whole system, with search trace and CSP pruning** (≈ 2 min): `python run_demo.py`
   Point at: section 3 (BT vs FC vs AC-3 counters and domains), section 4 (UCS/A*/Greedy counters + A* trace f = g + h),
   section 5 (itinerary), `figures/demo.png`.
2. **Model comparison** (about 1 min): choose "Neural network (best held-out route F1)" for the route demo, then compare with "Context-aware counts (higher next-place Top-1)". The CLI demo shows NN-on versus NN-off in section 6.
3. **Different request** (≈ 1 min): `python run_demo.py --n-stops 7 --budget 240 --must Historical --hour 9 --no-plots`
4. **Failure messages** (≈ 1 min), in a Python shell:
   ```python
   from tourplan import data as D, nn, csp as C, integration as I
   cd = D.get_city("Edin"); m = nn.train_all(cd, 0); prior = C.category_prior(cd, m["data"]["tr_t"]); b = m["bundles"]["NN"]
   print(I.format_plan(cd, I.plan(cd, b, I.Request(6, 9, 60,  "Museum", 6, 10), prior)))   # budget too small -> smallest budget + warning
   print(I.format_plan(cd, I.plan(cd, b, I.Request(6, 9, 300, "Castle", 6, 10), prior)))   # no such category -> No solution
   ```
Backup if the live run fails: `results/demo_output.txt` and `figures/demo.png` contain a recorded run.
