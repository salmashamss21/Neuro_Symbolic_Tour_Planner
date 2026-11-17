"""Two small neural heads (pure NumPy) and the count-based baselines.

Head 1 (NextPlaceNet): P(next place | current place, hour)  -> softmax over places.
Head 2 (DurationNet) : E[log visit duration | place, hour]  -> regression on log seconds.

A "model bundle" packages a transition model and a duration model and exposes the two
look-up tables the search needs:
    logp_table(): (B, P, P)  log P(next=j | cur=i, hour bucket b); -inf on the diagonal
    dur_table():  (B, P)     expected stay in minutes at place i when arriving in bucket b
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_BUCKETS = 48            # half-hour buckets for the look-up tables


def hour_bucket(minutes_of_day: float, n_buckets: int = N_BUCKETS) -> int:
    return int(((minutes_of_day % 1440.0) / 1440.0) * n_buckets) % n_buckets


def bucket_centres(n_buckets: int = N_BUCKETS) -> np.ndarray:
    return (np.arange(n_buckets) + 0.5) * 24.0 / n_buckets


def featurize(place: np.ndarray, hour: np.ndarray, n_places: int) -> np.ndarray:
    place = np.asarray(place, dtype=int)
    hour = np.asarray(hour, dtype=float)
    x = np.zeros((len(place), n_places + 4))
    x[np.arange(len(place)), place] = 1.0
    ang = 2 * np.pi * hour / 24.0
    x[:, n_places] = np.sin(ang)
    x[:, n_places + 1] = np.cos(ang)
    x[:, n_places + 2] = np.sin(2 * ang)
    x[:, n_places + 3] = np.cos(2 * ang)
    return x


# ------------------------------------------------------------------ tiny MLP -------
class MLP:
    """Fully-connected ReLU network with manual back-prop and Adam."""

    def __init__(self, sizes, rng: np.random.Generator):
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            acts.append(z if i == len(self.W) - 1 else np.maximum(z, 0.0))
        return acts[-1], acts

    def backward(self, acts, d_out):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        d = d_out
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    def step(self, gW, gb, lr, wd=0.0, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self.W + self.b
        grads = [g + wd * p if k < len(self.W) else g
                 for k, (g, p) in enumerate(zip(gW + gb, params))]
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def get(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set(self, state):
        self.W, self.b = [w.copy() for w in state[0]], [b.copy() for b in state[1]]


def _masked_log_softmax(logits: np.ndarray, cur: np.ndarray) -> np.ndarray:
    z = logits.copy()
    z[np.arange(len(cur)), cur] = -np.inf           # a tourist never "moves" to the same place
    z -= z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


# ------------------------------------------------------------- head 1: next place --
class NextPlaceNet:
    name = "NN"

    def __init__(self, n_places: int, hidden=(64, 64), seed: int = 0, lr=3e-3, wd=1e-2,
                 batch=64, max_epochs=200, patience=20):
        self.P, self.hidden, self.seed = n_places, tuple(hidden), seed
        self.lr, self.wd, self.batch, self.max_epochs, self.patience = lr, wd, batch, max_epochs, patience
        self.net = MLP([n_places + 4, *self.hidden, n_places], np.random.default_rng(seed))
        self.history = {"train_nll": [], "val_nll": []}

    def logp(self, cur, hour) -> np.ndarray:
        cur = np.asarray(cur, dtype=int)
        out, _ = self.net.forward(featurize(cur, hour, self.P))
        return _masked_log_softmax(out, cur)

    def nll(self, cur, hour, nxt) -> float:
        lp = self.logp(cur, hour)
        return float(-lp[np.arange(len(nxt)), nxt].mean())

    def fit(self, train, val):
        cur, hour, nxt = train
        rng = np.random.default_rng(self.seed + 1)
        best, best_state, bad = np.inf, self.net.get(), 0
        n = len(cur)
        for epoch in range(self.max_epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                x = featurize(cur[idx], hour[idx], self.P)
                out, acts = self.net.forward(x)
                lp = _masked_log_softmax(out, cur[idx])
                grad = np.exp(lp)
                grad[np.arange(len(idx)), nxt[idx]] -= 1.0
                grad[np.arange(len(idx)), cur[idx]] = 0.0
                gW, gb = self.net.backward(acts, grad / len(idx))
                self.net.step(gW, gb, self.lr, self.wd)
            tr, va = self.nll(*train), self.nll(*val)
            self.history["train_nll"].append(tr)
            self.history["val_nll"].append(va)
            if va < best - 1e-4:
                best, best_state, bad = va, self.net.get(), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.set(best_state)
        return self

    def table(self, n_buckets=N_BUCKETS) -> np.ndarray:
        P = self.P
        out = np.empty((n_buckets, P, P))
        for b, h in enumerate(bucket_centres(n_buckets)):
            out[b] = self.logp(np.arange(P), np.full(P, h))
        return out


class GoalAwareNextPlaceNet(NextPlaceNet):
    """Optional goal-conditioned MLP: P(next | current, hour, end, stops left)."""
    name = "Goal-aware NN"

    def __init__(self, n_places: int, seed=0, hidden=(64, 32), lr=1e-3, wd=1e-2,
                 batch=64, max_epochs=120, patience=15):
        self.P, self.hidden, self.seed = n_places, tuple(hidden), seed
        self.lr, self.wd, self.batch, self.max_epochs, self.patience = lr, wd, batch, max_epochs, patience
        self.net = MLP([2 * n_places + 5, *self.hidden, n_places], np.random.default_rng(seed + 500))
        self.history = {"train_nll": [], "val_nll": []}
        self._goal_table_cache = {}

    def _features(self, cur, hour, end, remaining):
        cur, end = np.asarray(cur, int), np.asarray(end, int)
        x = np.zeros((len(cur), 2 * self.P + 5))
        x[np.arange(len(cur)), cur] = 1
        x[np.arange(len(cur)), self.P + end] = 1
        ang = 2 * np.pi * np.asarray(hour, float) / 24
        x[:, 2*self.P:2*self.P+4] = np.column_stack((np.sin(ang), np.cos(ang), np.sin(2*ang), np.cos(2*ang)))
        x[:, -1] = np.asarray(remaining, float) / 10.0
        return x

    def logp_goal(self, cur, hour, end, remaining, position=None):
        out, _ = self.net.forward(self._features(cur, hour, end, remaining))
        return _masked_log_softmax(out, np.asarray(cur, int))

    def fit_goal(self, train, val):
        cur, hour, end, remaining, _position, nxt = train
        vc, vh, ve, vr, _vposition, vn = val
        rng = np.random.default_rng(self.seed + 501)
        best, state, bad = np.inf, self.net.get(), 0
        for _ in range(self.max_epochs):
            order = rng.permutation(len(cur))
            for s in range(0, len(cur), self.batch):
                ix = order[s:s+self.batch]
                out, acts = self.net.forward(self._features(cur[ix], hour[ix], end[ix], remaining[ix]))
                lp = _masked_log_softmax(out, cur[ix]); grad = np.exp(lp)
                grad[np.arange(len(ix)), nxt[ix]] -= 1; grad[np.arange(len(ix)), cur[ix]] = 0
                gw, gb = self.net.backward(acts, grad / len(ix)); self.net.step(gw, gb, self.lr, self.wd)
            tr = self._nll_goal(cur, hour, end, remaining, nxt)
            va = self._nll_goal(vc, vh, ve, vr, vn)
            self.history['train_nll'].append(tr); self.history['val_nll'].append(va)
            if va < best - 1e-4: best, state, bad = va, self.net.get(), 0
            else:
                bad += 1
                if bad >= self.patience: break
        self.net.set(state); self._goal_table_cache.clear(); return self

    def _nll_goal(self, cur, hour, end, remaining, nxt, position=None):
        lp = self.logp_goal(cur, hour, end, remaining, position)
        return float(-lp[np.arange(len(nxt)), nxt].mean())

    def table_goal(self, end, remaining, n_buckets=N_BUCKETS, position=None):
        position = 0.0 if position is None else float(position)
        key = (int(end), int(remaining), int(n_buckets), round(position, 8))
        if key in self._goal_table_cache:
            return self._goal_table_cache[key]
        out = np.empty((n_buckets, self.P, self.P))
        for b, h in enumerate(bucket_centres(n_buckets)):
            out[b] = self.logp_goal(np.arange(self.P), np.full(self.P, h), np.full(self.P, end),
                                    np.full(self.P, remaining), np.full(self.P, position))
        self._goal_table_cache[key] = out
        return out


class FeatureNextPlaceNet(GoalAwareNextPlaceNet):
    """Shared candidate scorer using categories, both distances and route position."""
    name = "Feature NN"

    def __init__(self, n_places, categories, travel_min, seed=0, hidden=(32,), lr=1e-3,
                 wd=1e-2, batch=64, max_epochs=120, patience=15):
        self.P, self.categories = n_places, np.asarray(categories, int)
        self.n_categories = int(self.categories.max()) + 1
        self.travel_min = np.asarray(travel_min, float)
        self.seed, self.hidden, self.lr, self.wd = seed, tuple(hidden), lr, wd
        self.batch, self.max_epochs, self.patience = batch, max_epochs, patience
        self.net = MLP([self.n_categories + 4, *self.hidden, 1], np.random.default_rng(seed + 900))
        self.history = {"train_nll": [], "val_nll": []}
        self._goal_table_cache = {}

    def _candidate_features(self, cur, end, position):
        cur, end = np.asarray(cur, int), np.asarray(end, int)
        n = len(cur); x = np.zeros((n, self.P, self.n_categories + 4))
        x[np.arange(n)[:, None], np.arange(self.P)[None, :], self.categories[None, :]] = 1
        scale = max(float(self.travel_min.max()), 1.0)
        x[:, :, self.n_categories] = self.travel_min[cur] / scale
        x[:, :, self.n_categories+1] = self.travel_min[end] / scale
        x[:, :, self.n_categories+2] = np.asarray(position, float)[:, None]
        x[:, :, self.n_categories+3] = self.categories[cur, None] / max(1, self.n_categories-1)
        return x

    def _logp_candidate(self, cur, hour, end, remaining, position=None):
        if position is None:
            position = 1.0 - np.asarray(remaining, float) / (np.asarray(remaining, float) + 1)
        feat = self._candidate_features(cur, end, position)
        raw, _ = self.net.forward(feat.reshape(-1, feat.shape[-1]))
        return _masked_log_softmax(raw.reshape(len(cur), self.P), np.asarray(cur, int))

    def logp_goal(self, cur, hour, end, remaining, position=None):
        return self._logp_candidate(cur, hour, end, remaining, position)

    def fit_goal(self, train, val):
        cur, hour, end, remaining, position, nxt = train
        vc, vh, ve, vr, vposition, vn = val
        rng = np.random.default_rng(self.seed + 901); best, state, bad = np.inf, self.net.get(), 0
        for _ in range(self.max_epochs):
            order = rng.permutation(len(cur))
            for s in range(0, len(cur), self.batch):
                ix = order[s:s+self.batch]; feat = self._candidate_features(cur[ix], end[ix], position[ix])
                raw, acts = self.net.forward(feat.reshape(-1, feat.shape[-1])); logits = raw.reshape(len(ix), self.P)
                lp = _masked_log_softmax(logits, cur[ix]); grad = np.exp(lp)
                grad[np.arange(len(ix)), nxt[ix]] -= 1; grad[np.arange(len(ix)), cur[ix]] = 0
                grad = grad.reshape(-1, 1) / len(ix); gw, gb = self.net.backward(acts, grad)
                self.net.step(gw, gb, self.lr, self.wd)
            tr = self._nll_goal(cur, hour, end, remaining, nxt, position)
            va = self._nll_goal(vc, vh, ve, vr, vn, vposition)
            self.history['train_nll'].append(tr); self.history['val_nll'].append(va)
            if va < best - 1e-4: best, state, bad = va, self.net.get(), 0
            else:
                bad += 1
                if bad >= self.patience: break
        self.net.set(state); return self


# --------------------------------------------------------------- head 2: duration ---
class DurationNet:
    name = "NN"

    def __init__(self, n_places: int, hidden=(16,), seed: int = 0, lr=3e-3, wd=1e-2,
                 batch=64, max_epochs=200, patience=20):
        self.P, self.hidden, self.seed = n_places, tuple(hidden), seed
        self.lr, self.wd, self.batch, self.max_epochs, self.patience = lr, wd, batch, max_epochs, patience
        self.net = MLP([n_places + 4, *self.hidden, 1], np.random.default_rng(seed + 100))
        self.mean_ = 0.0
        self.history = {"train_mse": [], "val_mse": []}

    def predict_log(self, place, hour) -> np.ndarray:
        out, _ = self.net.forward(featurize(place, hour, self.P))
        return out[:, 0] + self.mean_

    def mse(self, place, hour, y) -> float:
        return float(((self.predict_log(place, hour) - y) ** 2).mean())

    def fit(self, train, val):
        place, hour, y = train
        self.mean_ = float(y.mean())                       # centre targets for easier training
        rng = np.random.default_rng(self.seed + 2)
        best, best_state, bad = np.inf, self.net.get(), 0
        n = len(place)
        for epoch in range(self.max_epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                out, acts = self.net.forward(featurize(place[idx], hour[idx], self.P))
                grad = 2.0 * (out[:, 0] + self.mean_ - y[idx])[:, None] / len(idx)
                gW, gb = self.net.backward(acts, grad)
                self.net.step(gW, gb, self.lr, self.wd)
            tr, va = self.mse(*train), self.mse(*val)
            self.history["train_mse"].append(tr)
            self.history["val_mse"].append(va)
            if va < best - 1e-5:
                best, best_state, bad = va, self.net.get(), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.set(best_state)
        return self

    def table_minutes(self, n_buckets=N_BUCKETS) -> np.ndarray:
        P = self.P
        out = np.empty((n_buckets, P))
        for b, h in enumerate(bucket_centres(n_buckets)):
            out[b] = np.exp(self.predict_log(np.arange(P), np.full(P, h))) / 60.0
        return out


# ------------------------------------------------------------------- baselines -----
class MarkovCounts:
    """Count-based first-order Markov chain, additive smoothing, hour-independent."""
    name = "Markov"

    def __init__(self, n_places: int, alpha: float = 0.5):
        self.P, self.alpha = n_places, alpha
        self.counts = np.zeros((n_places, n_places))

    def fit(self, train, val=None, alphas=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)):
        cur, _, nxt = train
        np.add.at(self.counts, (cur, nxt), 1.0)
        if val is not None:                                # tune smoothing on validation NLL
            best = min(alphas, key=lambda a: self._nll(val, a))
            self.alpha = best
        return self

    def _logp_mat(self, alpha):
        c = self.counts + alpha
        np.fill_diagonal(c, 0.0)
        return np.log(c / c.sum(1, keepdims=True) + 1e-300)

    def _nll(self, data, alpha):
        cur, _, nxt = data
        return float(-self._logp_mat(alpha)[cur, nxt].mean())

    def logp(self, cur, hour=None):
        return self._logp_mat(self.alpha)[np.asarray(cur, dtype=int)]

    def nll(self, cur, hour, nxt):
        return self._nll((cur, hour, nxt), self.alpha)

    def table(self, n_buckets=N_BUCKETS):
        m = self._logp_mat(self.alpha)
        return np.repeat(m[None], n_buckets, axis=0)


class ContextMarkovCounts:
    """Smoothed second-order transitions P(next | previous, current).

    Sparse previous/current pairs back off to a validation-smoothed first-order
    Markov model. The backoff strength is selected on validation Top-1 accuracy.
    """
    name = "Context Markov"
    ALPHAS = (0.5, 1, 2, 5, 10, 20, 50, 100)

    def __init__(self, n_places: int):
        self.P = n_places
        self.base = MarkovCounts(n_places)
        self.trigram = np.zeros((n_places, n_places, n_places), dtype=float)
        self.alpha = 10.0

    def fit(self, train, context_train, val, context_val):
        self.base.fit(train, val)
        prev, cur, _, nxt = context_train
        valid = prev >= 0
        np.add.at(self.trigram, (prev[valid], cur[valid], nxt[valid]), 1.0)
        for i in range(self.P):
            self.trigram[:, i, i] = 0.0

        vprev, vcur, vhour, vnxt = context_val
        self.alpha = max(
            self.ALPHAS,
            key=lambda alpha: self._top1(vprev, vcur, vhour, vnxt, alpha),
        )
        return self

    def _probabilities(self, alpha):
        bigram = np.exp(self.base._logp_mat(self.base.alpha))
        out = np.empty((self.P + 1, self.P, self.P), dtype=float)
        out[self.P] = bigram  # sentinel previous place: first-order backoff
        totals = self.trigram.sum(axis=2, keepdims=True)
        out[:self.P] = (self.trigram + alpha * bigram[None, :, :]) / (totals + alpha)
        for previous in range(self.P):
            for current in range(self.P):
                if totals[previous, current, 0] == 0:
                    out[previous, current] = bigram[current]
        return out

    def _top1(self, prev, cur, hour, nxt, alpha):
        probabilities = self._probabilities(alpha)
        prev_idx = np.where(prev < 0, self.P, prev)
        scores = probabilities[prev_idx, cur].copy()
        scores[np.arange(len(cur)), cur] = -1.0
        return float((scores.argmax(axis=1) == nxt).mean())

    def logp_context(self, prev, cur, hour=None):
        probabilities = self._probabilities(self.alpha)
        prev = np.asarray(prev, dtype=int)
        cur = np.asarray(cur, dtype=int)
        prev_idx = np.where(prev < 0, self.P, prev)
        return np.log(np.maximum(probabilities[prev_idx, cur], 1e-300))

    def logp(self, cur, hour=None):
        return self.logp_context(np.full(len(cur), -1), cur, hour)

    def nll(self, cur, hour, nxt):
        lp = self.logp(cur, hour)
        return float(-lp[np.arange(len(nxt)), nxt].mean())

    def table(self, n_buckets=N_BUCKETS):
        probs = self._probabilities(self.alpha)
        lp = np.log(np.maximum(probs, 1e-300))
        return np.repeat(lp[None, :, :, :], n_buckets, axis=0)


class Popularity:
    """P(next=j) proportional to how often j was visited as a next place (ignores current)."""
    name = "Popularity"

    def __init__(self, n_places: int, alpha: float = 1.0):
        self.P, self.alpha = n_places, alpha
        self.counts = np.zeros(n_places)

    def fit(self, train, val=None):
        np.add.at(self.counts, train[2], 1.0)
        return self

    def _mat(self):
        c = np.tile(self.counts + self.alpha, (self.P, 1))
        np.fill_diagonal(c, 0.0)
        return np.log(c / c.sum(1, keepdims=True) + 1e-300)

    def logp(self, cur, hour=None):
        return self._mat()[np.asarray(cur, dtype=int)]

    def nll(self, cur, hour, nxt):
        return float(-self._mat()[cur, nxt].mean())

    def table(self, n_buckets=N_BUCKETS):
        return np.repeat(self._mat()[None], n_buckets, axis=0)


class MeanDuration:
    """Per-place mean of log duration (shrunk toward the global mean); hour-independent."""
    name = "MeanDur"

    def __init__(self, n_places: int, shrink: float = 5.0):
        self.P, self.shrink = n_places, shrink
        self.glob = 0.0
        self.place_mean = np.zeros(n_places)

    def fit(self, train, val=None):
        place, _, y = train
        self.glob = float(y.mean())
        s = np.bincount(place, weights=y, minlength=self.P)
        n = np.bincount(place, minlength=self.P)
        self.place_mean = (s + self.shrink * self.glob) / (n + self.shrink)
        return self

    def predict_log(self, place, hour=None):
        return self.place_mean[np.asarray(place, dtype=int)]

    def mse(self, place, hour, y):
        return float(((self.predict_log(place) - y) ** 2).mean())

    def table_minutes(self, n_buckets=N_BUCKETS):
        return np.repeat((np.exp(self.place_mean) / 60.0)[None], n_buckets, axis=0)


class GlobalMeanDuration(MeanDuration):
    name = "GlobalMean"

    def fit(self, train, val=None):
        super().fit(train, val)
        self.place_mean = np.full(self.P, self.glob)
        return self


# ---------------------------------------------------------------------- metrics -----
def next_metrics(model, data) -> dict:
    cur, hour, nxt = data
    lp = model.logp(cur, hour)
    top = np.argsort(-lp, axis=1)
    return dict(nll=float(-lp[np.arange(len(nxt)), nxt].mean()),
                top1=float((top[:, 0] == nxt).mean()),
                top3=float((top[:, :3] == nxt[:, None]).any(1).mean()))


def context_metrics(model, data) -> dict:
    prev, cur, hour, nxt = data
    lp = model.logp_context(prev, cur, hour)
    top = np.argsort(-lp, axis=1)
    return dict(nll=float(-lp[np.arange(len(nxt)), nxt].mean()),
                top1=float((top[:, 0] == nxt).mean()),
                top3=float((top[:, :3] == nxt[:, None]).any(1).mean()))


def goal_metrics(model, data) -> dict:
    cur, hour, end, remaining, position, nxt = data
    lp = model.logp_goal(cur, hour, end, remaining, position)
    top = np.argsort(-lp, axis=1)
    return dict(nll=float(-lp[np.arange(len(nxt)), nxt].mean()),
                top1=float((top[:, 0] == nxt).mean()),
                top3=float((top[:, :3] == nxt[:, None]).any(1).mean()))


def duration_metrics(model, data) -> dict:
    place, hour, y = data
    pred = model.predict_log(place, hour)
    return dict(mae_log=float(np.abs(pred - y).mean()), rmse_log=float(np.sqrt(((pred - y) ** 2).mean())),
                mae_min=float(np.abs(np.exp(pred) - np.exp(y)).mean() / 60.0))


# ---------------------------------------------------------------------- bundles -----
@dataclass
class ModelBundle:
    """Transition model + duration model, as the search sees them."""
    name: str
    trans: object
    dur: object
    n_buckets: int = N_BUCKETS
    _logp: np.ndarray = field(default=None, repr=False)
    _dur: np.ndarray = field(default=None, repr=False)

    def logp_table(self):
        if self._logp is None:
            self._logp = self.trans.table(self.n_buckets)
        return self._logp

    def dur_table(self):
        if self._dur is None:
            self._dur = self.dur.table_minutes(self.n_buckets)
        return self._dur


@dataclass
class GoalAwareBundle:
    """Transition bundle whose probabilities depend on the request end and route position."""
    name: str
    trans: object
    dur: object
    n_buckets: int = N_BUCKETS
    _dur: np.ndarray = field(default=None, repr=False)

    def dur_table(self):
        if self._dur is None:
            self._dur = self.dur.table_minutes(self.n_buckets)
        return self._dur


def train_all(cd, seed: int, quick: bool = False, goal_config: dict | None = None):
    """Train every model for one seed (user split + initialisation).  Returns dict."""
    from . import data as D
    sp = D.split_users(cd.visits, seed)
    tr_t, va_t, te_t = (D.transitions(cd.visits, sp[k]) for k in ("train", "val", "test"))
    tr_ctx, va_ctx, te_ctx = (
        D.context_transitions(cd.visits, sp[k]) for k in ("train", "val", "test")
    )
    tr_goal, va_goal, te_goal = (D.goal_transitions(cd.visits, sp[k]) for k in ("train", "val", "test"))
    tr_d, va_d, te_d = (D.duration_samples(cd.visits, sp[k]) for k in ("train", "val", "test"))
    P = cd.n_places
    kw = dict(max_epochs=40) if quick else {}
    goal_config = goal_config or {}
    nn_next = NextPlaceNet(P, seed=seed, **kw).fit(tr_t, va_t)
    nn_dur = DurationNet(P, seed=seed, **kw).fit(tr_d, va_d)
    markov = MarkovCounts(P).fit(tr_t, va_t)
    context_markov = ContextMarkovCounts(P).fit(tr_t, tr_ctx, va_t, va_ctx)
    epochs = 40 if quick else 120
    goal_next = GoalAwareNextPlaceNet(P, seed=seed, max_epochs=epochs,
                                      hidden=goal_config.get("goal_hidden", (64, 32)),
                                      lr=goal_config.get("goal_lr", 1e-3)).fit_goal(tr_goal, va_goal)
    feature_next = FeatureNextPlaceNet(P, cd.cat_of, cd.travel_min, seed=seed, max_epochs=epochs,
                                       hidden=goal_config.get("feature_hidden", (32,)),
                                       lr=goal_config.get("feature_lr", 1e-3)).fit_goal(tr_goal, va_goal)
    pop = Popularity(P).fit(tr_t)
    mean_dur = MeanDuration(P).fit(tr_d)
    glob_dur = GlobalMeanDuration(P).fit(tr_d)
    return dict(split=sp, data=dict(tr_t=tr_t, va_t=va_t, te_t=te_t,
                                    tr_ctx=tr_ctx, va_ctx=va_ctx, te_ctx=te_ctx,
                                    tr_goal=tr_goal, va_goal=va_goal, te_goal=te_goal,
                                    tr_d=tr_d, va_d=va_d, te_d=te_d),
                nn_next=nn_next, nn_dur=nn_dur, markov=markov,
                context_markov=context_markov, pop=pop, mean_dur=mean_dur,
                goal_next=goal_next, feature_next=feature_next,
                glob_dur=glob_dur,
                bundles={"NN": ModelBundle("NN", nn_next, nn_dur),
                         "Markov": ModelBundle("Markov", markov, mean_dur),
                         "Context Markov": ModelBundle("Context Markov", context_markov, mean_dur),
                         "Goal-aware NN": GoalAwareBundle("Goal-aware NN", goal_next, mean_dur),
                         "Feature NN": GoalAwareBundle("Feature NN", feature_next, mean_dur),
                         "Popularity": ModelBundle("Popularity", pop, mean_dur),
                         "NN-trans+MeanDur": ModelBundle("NN-trans+MeanDur", nn_next, mean_dur),
                         "Off": ModelBundle("Off", UniformTrans(P), glob_dur)})


class UniformTrans:
    """'NN off' transition model: every other place is equally likely (search then only sees travel)."""
    name = "Uniform"

    def __init__(self, n_places: int):
        self.P = n_places

    def _mat(self):
        m = np.full((self.P, self.P), -np.log(self.P - 1))
        np.fill_diagonal(m, -np.inf)
        return m

    def logp(self, cur, hour=None):
        return self._mat()[np.asarray(cur, dtype=int)]

    def table(self, n_buckets=N_BUCKETS):
        return np.repeat(self._mat()[None], n_buckets, axis=0)
