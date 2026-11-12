"""Two-Head Neural Network for Plausible Tour Itineraries.

Architecture:
- Shared Input Representation: One-hot POI ID (28 dims) + One-hot Category (6 dims)
  + Cyclical Hour Encoding (sin(2*pi*h/24), cos(2*pi*h/24)) + 4-Bucket Time-of-Day
  encoding -> 40 input features.
- Head 1 (Next-Place Prediction): 2-Layer MLP (40 -> 64 -> ReLU -> 64 -> ReLU -> 28)
  with Softmax output trained via Cross-Entropy (Negative Log-Likelihood) + Adam.
- Head 2 (Visit Duration Prediction): 2-Layer MLP (40 -> 64 -> ReLU -> 32 -> ReLU -> 1)
  predicting log(visit_duration_sec) trained via Mean Squared Error (MSE) + Adam.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np


def encode_features(
    poi_indices: np.ndarray,
    cat_indices: np.ndarray,
    hours: np.ndarray,
    n_pois: int = 28,
    n_cats: int = 6,
) -> np.ndarray:
    """Encodes (place, category, hour) into a 40-dimensional feature vector per sample."""
    batch_size = len(poi_indices)
    x_poi = np.zeros((batch_size, n_pois), dtype=np.float64)
    x_poi[np.arange(batch_size), poi_indices] = 1.0

    x_cat = np.zeros((batch_size, n_cats), dtype=np.float64)
    x_cat[np.arange(batch_size), cat_indices] = 1.0

    # Cyclical hour encoding + 4 time-of-day regime indicators
    rad = 2.0 * math.pi * (hours.astype(np.float64) / 24.0)
    sin_h = np.sin(rad)[:, None]
    cos_h = np.cos(rad)[:, None]

    buckets = np.zeros((batch_size, 4), dtype=np.float64)
    for idx, h in enumerate(hours):
        if 8 <= h < 11:
            buckets[idx, 0] = 1.0
        elif 11 <= h < 14:
            buckets[idx, 1] = 1.0
        elif 14 <= h < 17:
            buckets[idx, 2] = 1.0
        else:
            buckets[idx, 3] = 1.0

    return np.hstack([x_poi, x_cat, sin_h, cos_h, buckets])


class DualHeadTourNN:
    """Two-head neural network predicting next-place probabilities and log visit duration."""

    def __init__(
        self,
        n_pois: int = 28,
        n_cats: int = 6,
        hidden_dim: int = 64,
        seed: int = 42,
    ):
        self.n_pois = n_pois
        self.n_cats = n_cats
        self.in_dim = n_pois + n_cats + 6
        self.hidden_dim = hidden_dim
        self.seed = seed
        rng = np.random.default_rng(seed)

        # Head 1: Next-Place MLP (in_dim -> hidden_dim -> hidden_dim -> n_pois)
        self.w1_next = rng.normal(0.0, math.sqrt(2.0 / self.in_dim), (self.in_dim, hidden_dim))
        self.b1_next = np.zeros(hidden_dim, dtype=np.float64)
        self.w2_next = rng.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim))
        self.b2_next = np.zeros(hidden_dim, dtype=np.float64)
        self.w3_next = rng.normal(0.0, math.sqrt(1.0 / hidden_dim), (hidden_dim, n_pois))
        self.b3_next = np.zeros(n_pois, dtype=np.float64)

        # Head 2: Log-Duration MLP (in_dim -> hidden_dim -> 32 -> 1)
        self.w1_dur = rng.normal(0.0, math.sqrt(2.0 / self.in_dim), (self.in_dim, hidden_dim))
        self.b1_dur = np.zeros(hidden_dim, dtype=np.float64)
        self.w2_dur = rng.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim, 32))
        self.b2_dur = np.zeros(32, dtype=np.float64)
        self.w3_dur = rng.normal(0.0, math.sqrt(1.0 / 32), (32, 1))
        self.b3_dur = np.array([8.0], dtype=np.float64)  # Initialize near ln(3000s) ~ 8.0

        # Adam optimizer state
        self._adam_t = 0
        self._m = {}
        self._v = {}

    def _adam_step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float, weight_decay: float = 1e-4) -> None:
        self._adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, p in params.items():
            g = grads[k] + (weight_decay * p if k.startswith("w") else 0.0)
            if k not in self._m:
                self._m[k] = np.zeros_like(p)
                self._v[k] = np.zeros_like(p)
            self._m[k] = b1 * self._m[k] + (1.0 - b1) * g
            self._v[k] = b2 * self._v[k] + (1.0 - b2) * (g * g)
            m_hat = self._m[k] / (1.0 - b1 ** self._adam_t)
            v_hat = self._v[k] / (1.0 - b2 ** self._adam_t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def forward_next_place(self, x: np.ndarray, current_poi_indices: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Forward pass for Head 1 returning (probabilities, cache)."""
        z1 = x @ self.w1_next + self.b1_next
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.w2_next + self.b2_next
        h2 = np.maximum(0.0, z2)
        logits = h2 @ self.w3_next + self.b3_next
        if current_poi_indices is not None:
            # Disallow self-transitions (pi -> pi)
            logits = logits.copy()
            logits[np.arange(len(x)), current_poi_indices] = -1e9
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits_shifted)
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "probs": probs}
        return probs, cache

    def forward_duration(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward pass for Head 2 returning (predicted_log_duration, cache)."""
        z1 = x @ self.w1_dur + self.b1_dur
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.w2_dur + self.b2_dur
        h2 = np.maximum(0.0, z2)
        pred = (h2 @ self.w3_dur + self.b3_dur).squeeze(-1)
        cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "pred": pred}
        return pred, cache

    def train_step(
        self,
        x: np.ndarray,
        curr_idx: np.ndarray,
        target_next_idx: np.ndarray,
        target_log_dur: np.ndarray,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
    ) -> tuple[float, float]:
        """Performs one joint minibatch update on both heads; returns (nll_loss, mse_loss)."""
        bs = len(x)
        # Head 1 forward & backward
        probs, c1 = self.forward_next_place(x, curr_idx)
        nll = -float(np.mean(np.log(np.clip(probs[np.arange(bs), target_next_idx], 1e-12, 1.0))))
        dlogits = probs.copy()
        dlogits[np.arange(bs), target_next_idx] -= 1.0
        dlogits /= bs

        dw3_next = c1["h2"].T @ dlogits
        db3_next = np.sum(dlogits, axis=0)
        dh2 = dlogits @ self.w3_next.T
        dz2 = dh2 * (c1["z2"] > 0.0)
        dw2_next = c1["h1"].T @ dz2
        db2_next = np.sum(dz2, axis=0)
        dh1 = dz2 @ self.w2_next.T
        dz1 = dh1 * (c1["z1"] > 0.0)
        dw1_next = x.T @ dz1
        db1_next = np.sum(dz1, axis=0)

        # Head 2 forward & backward
        pred_dur, c2 = self.forward_duration(x)
        diff = pred_dur - target_log_dur
        mse = float(np.mean(diff ** 2))
        dpred = (2.0 * diff / bs)[:, None]

        dw3_dur = c2["h2"].T @ dpred
        db3_dur = np.sum(dpred, axis=0)
        dh2_d = dpred @ self.w3_dur.T
        dz2_d = dh2_d * (c2["z2"] > 0.0)
        dw2_dur = c2["h1"].T @ dz2_d
        db2_dur = np.sum(dz2_d, axis=0)
        dh1_d = dz2_d @ self.w2_dur.T
        dz1_d = dh1_d * (c2["z1"] > 0.0)
        dw1_dur = x.T @ dz1_d
        db1_dur = np.sum(dz1_d, axis=0)

        params = {
            "w1_next": self.w1_next, "b1_next": self.b1_next,
            "w2_next": self.w2_next, "b2_next": self.b2_next,
            "w3_next": self.w3_next, "b3_next": self.b3_next,
            "w1_dur": self.w1_dur, "b1_dur": self.b1_dur,
            "w2_dur": self.w2_dur, "b2_dur": self.b2_dur,
            "w3_dur": self.w3_dur, "b3_dur": self.b3_dur,
        }
        grads = {
            "w1_next": dw1_next, "b1_next": db1_next,
            "w2_next": dw2_next, "b2_next": db2_next,
            "w3_next": dw3_next, "b3_next": db3_next,
            "w1_dur": dw1_dur, "b1_dur": db1_dur,
            "w2_dur": dw2_dur, "b2_dur": db2_dur,
            "w3_dur": dw3_dur, "b3_dur": db3_dur,
        }
        self._adam_step(params, grads, lr=lr, weight_decay=weight_decay)
        return nll, mse

    def predict_next_probs(self, poi_idx: int, cat_idx: int, hour: int) -> np.ndarray:
        """Returns probability distribution over all POIs for a single (place, hour) state."""
        x = encode_features(
            np.array([poi_idx]),
            np.array([cat_idx]),
            np.array([hour]),
            n_pois=self.n_pois,
            n_cats=self.n_cats,
        )
        probs, _ = self.forward_next_place(x, np.array([poi_idx]))
        return probs[0]

    def predict_duration_hours(self, poi_idx: int, cat_idx: int, hour: int) -> float:
        """Returns predicted visit duration in hours for a single (place, hour) state."""
        x = encode_features(
            np.array([poi_idx]),
            np.array([cat_idx]),
            np.array([hour]),
            n_pois=self.n_pois,
            n_cats=self.n_cats,
        )
        log_dur, _ = self.forward_duration(x)
        sec = max(300.0, float(np.exp(log_dur[0])))
        return sec / 3600.0

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            w1_next=self.w1_next, b1_next=self.b1_next,
            w2_next=self.w2_next, b2_next=self.b2_next,
            w3_next=self.w3_next, b3_next=self.b3_next,
            w1_dur=self.w1_dur, b1_dur=self.b1_dur,
            w2_dur=self.w2_dur, b2_dur=self.b2_dur,
            w3_dur=self.w3_dur, b3_dur=self.b3_dur,
        )

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        for k in ["w1_next", "b1_next", "w2_next", "b2_next", "w3_next", "b3_next",
                  "w1_dur", "b1_dur", "w2_dur", "b2_dur", "w3_dur", "b3_dur"]:
            setattr(self, k, data[k])
