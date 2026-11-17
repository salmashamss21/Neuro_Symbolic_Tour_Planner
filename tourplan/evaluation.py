"""Evaluation helpers: F1 against held-out real trajectories, request generators."""
from __future__ import annotations

import numpy as np

from . import data as D
from . import integration as I


def f1_sets(pred, truth) -> tuple:
    pred, truth = set(pred), set(truth)
    if not pred or not truth:
        return 0.0, 0.0, 0.0
    hit = len(pred & truth)
    p, r = hit / len(pred), hit / len(truth)
    f = 0.0 if hit == 0 else 2 * p * r / (p + r)
    return p, r, f


def requests_from_trajectories(cd, trajs, rng, min_len=3, max_len=8, min_budget_per_stop=30.0):
    """Turn real test trajectories into tourist requests (generated: budget, must-see)."""
    out = []
    for t in trajs:
        pl = t["places"]
        n = len(pl)
        if not (min_len <= n <= max_len) or pl[0] == pl[-1] or len(set(pl)) != n:
            continue
        mid = pl[1:-1]
        must = cd.categories[cd.cat_of[mid[rng.integers(len(mid))]]]
        v = cd.visits[cd.visits["traj"] == t["traj"]]
        span = (v["end_ts"].max() - v["start_ts"].min()) / 60.0
        budget = max(span, min_budget_per_stop * n)           # generated budget (minutes)
        out.append((t, I.Request(pl[0], pl[-1], float(budget), must, n, float(t["hours_start"][0]))))
    return out


def random_requests(cd, rng, n_stops, count, budget_range=(120, 480)):
    """Synthetic requests for the search / CSP experiments (all generated)."""
    P = cd.n_places
    out = []
    while len(out) < count:
        s, e = (int(x) for x in rng.choice(P, 2, replace=False))
        must = cd.categories[int(rng.integers(len(cd.categories)))]
        out.append(I.Request(s, e, float(rng.uniform(*budget_range)), must, n_stops,
                             float(rng.uniform(8, 14))))
    return out
