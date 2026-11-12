"""Baseline models for comparison against the Two-Head Neural Network.

1. Count-based Markov Baseline:
   - Transition probabilities P(p_{t+1} | p_t) estimated via Laplace-smoothed
     empirical transition counts (ignoring hour-of-day context).
   - Visit duration predicted as the empirical mean log-duration per POI.
2. Popularity / Global Mean Baseline:
   - Transition probabilities proportional to global POI visit popularity P(p_{t+1}).
   - Visit duration predicted as the category-level mean log-duration.
"""

from __future__ import annotations

import numpy as np


class MarkovCountBaseline:
    """First-order count-based Markov chain for next-place + POI mean for duration."""

    def __init__(self, n_pois: int = 28, smoothing: float = 1.0):
        self.n_pois = n_pois
        self.smoothing = smoothing
        self.trans_probs = np.full((n_pois, n_pois), 1.0 / max(1, n_pois - 1), dtype=np.float64)
        np.fill_diagonal(self.trans_probs, 0.0)
        self.poi_mean_log_dur = np.full(n_pois, 8.0, dtype=np.float64)

    def fit(self, curr_idx: np.ndarray, next_idx: np.ndarray, log_durs: np.ndarray) -> None:
        counts = np.full((self.n_pois, self.n_pois), self.smoothing, dtype=np.float64)
        np.fill_diagonal(counts, 0.0)
        for u, v in zip(curr_idx, next_idx):
            if u != v:
                counts[u, v] += 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
        self.trans_probs = counts / np.maximum(row_sums, 1e-12)

        for p in range(self.n_pois):
            mask = curr_idx == p
            if np.any(mask):
                self.poi_mean_log_dur[p] = float(np.mean(log_durs[mask]))
            else:
                self.poi_mean_log_dur[p] = float(np.mean(log_durs))

    def predict_next_probs_batch(self, curr_idx: np.ndarray) -> np.ndarray:
        return self.trans_probs[curr_idx]

    def predict_next_probs(self, poi_idx: int, hour: int = 12) -> np.ndarray:
        del hour
        return self.trans_probs[poi_idx]

    def predict_log_duration_batch(self, curr_idx: np.ndarray) -> np.ndarray:
        return self.poi_mean_log_dur[curr_idx]

    def predict_duration_hours(self, poi_idx: int, hour: int = 12) -> float:
        del hour
        return max(300.0, float(np.exp(self.poi_mean_log_dur[poi_idx]))) / 3600.0


class PopularityBaseline:
    """Popularity-based next-place predictor + Category mean duration predictor."""

    def __init__(self, n_pois: int = 28, n_cats: int = 6, smoothing: float = 1.0):
        self.n_pois = n_pois
        self.n_cats = n_cats
        self.smoothing = smoothing
        self.pop_probs = np.full(n_pois, 1.0 / n_pois, dtype=np.float64)
        self.cat_mean_log_dur = np.full(n_cats, 8.0, dtype=np.float64)

    def fit(self, curr_idx: np.ndarray, next_idx: np.ndarray, cat_idx: np.ndarray, log_durs: np.ndarray) -> None:
        del curr_idx
        counts = np.full(self.n_pois, self.smoothing, dtype=np.float64)
        for v in next_idx:
            counts[v] += 1.0
        self.pop_probs = counts / counts.sum()

        for c in range(self.n_cats):
            mask = cat_idx == c
            if np.any(mask):
                self.cat_mean_log_dur[c] = float(np.mean(log_durs[mask]))
            else:
                self.cat_mean_log_dur[c] = float(np.mean(log_durs))

    def predict_next_probs_batch(self, curr_idx: np.ndarray) -> np.ndarray:
        out = np.tile(self.pop_probs, (len(curr_idx), 1))
        out[np.arange(len(curr_idx)), curr_idx] = 0.0
        out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
        return out

    def predict_next_probs(self, poi_idx: int, hour: int = 12) -> np.ndarray:
        del hour
        p = self.pop_probs.copy()
        p[poi_idx] = 0.0
        return p / p.sum()

    def predict_log_duration_batch(self, cat_idx: np.ndarray) -> np.ndarray:
        return self.cat_mean_log_dur[cat_idx]
