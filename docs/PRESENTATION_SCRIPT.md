# 10-minute talk + 5-minute live demo

Slides: the "P6 Plausible Tour Itineraries" 12-slide deck.

## Talk (10 minutes)

| Time | Slide | Talking points |
|---|---:|---|
| 0:00 | 1 Title | P6: plausible tour itineraries |
| 0:30 | 2 Problem | Tourist inputs, constraints, and plausible routes |
| 1:15 | 3 Idea | NN predicts places, CSP selects categories, search selects POIs |
| 2:00 | 4 Data | Edinburgh Flickr data and generated assumptions |
| 2:45 | 5 NN | Goal-aware: Top-1 0.476 / Top-3 0.697; feature model: 0.480 / 0.647. Both beat Markov on paired NLL and Top-1. |
| 3:45 | 6 CSP | Variables, constraints, BT, FC, and AC-3 |
| 4:45 | 7 Search | State, costs, UCS, A*, and Greedy |
| 5:45 | 8 Integration | CSP patterns, search, duration, and budget checks |
| 6:30 | 9 Experiment 1 | Search algorithm comparison |
| 7:30 | 10 Experiment 2 | CSP pruning comparison |
| 8:30 | 11 Experiment 3 | Mean route F1: NN 0.525, Markov 0.512, context 0.509, goal-aware 0.497, feature 0.424 |
| 9:30 | 12 Limits | One-step gains do not guarantee better route F1; goal-aware F1 is not significantly different from Markov, feature F1 is lower |

## Live demo (5 minutes)

Run these commands from the repository root.

1. **System trace and CSP pruning (about 2 min):** `python run_demo.py`. Show BT/FC/AC-3 counters, the UCS/A*/Greedy trace, the itinerary, and `figures/demo.png`.
2. **Model comparison (about 1 min):** In the planner compare Goal-aware neural network, Feature-based neural network, and Context-aware counts. Explain that higher next-place accuracy does not guarantee route F1.
3. **Different request (about 1 min):** `python run_demo.py --n-stops 7 --budget 240 --must Historical --start-id 7 --end-id 10 --hour 9 --no-plots`.
4. **Failure messages (about 1 min):** In a Python shell, set up the system and show a too-small budget and an impossible category request:

   ```python
   from tourplan import data as D, nn, csp as C, integration as I
   cd = D.get_city("Edin")
   models = nn.train_all(cd, 0)
   prior = C.category_prior(cd, models["data"]["tr_t"])
   bundle = models["bundles"]["NN"]
   print(I.format_plan(cd, I.plan(cd, bundle, I.Request(6, 9, 60, "Museum", 6, 10), prior)))
   print(I.format_plan(cd, I.plan(cd, bundle, I.Request(6, 9, 300, "Castle", 6, 10), prior)))
   ```

Backup outputs, if a live command fails: `results/experiments_log.txt` and `figures/demo.png`.
