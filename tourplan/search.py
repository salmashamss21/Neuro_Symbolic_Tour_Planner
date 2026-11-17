"""Itinerary search: UCS, A* and Greedy best-first over (position, place, visited, elapsed).

Given a category pattern (from the CSP) the search picks the actual places.
  state  = (pos, current place, visited places, elapsed minutes)
  action = move to an unvisited place whose category equals pattern[pos+1]
           (the last position is forced to be the end place)
  cost   = -log P_NN(next | current, clock hour on leaving) + lam * travel_minutes   (mode "nll")
           or minutes (travel + stay)                                                    (mode "time")
Times are whole minutes: travel = round(walk minutes), stay = max(5, round(predicted minutes)).
A path whose total time exceeds the budget is pruned (the pattern is then "rejected").

Heuristic for A* / Greedy: exact cost-to-go in the relaxed problem that ignores the visited set and
takes the cheapest clock hour for every step (backward DP over positions).  It is admissible and
consistent, so A* is optimal and expands no more nodes than UCS.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

from .nn import N_BUCKETS, hour_bucket


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0
    pruned_budget: int = 0
    max_open: int = 0

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class Itinerary:
    places: list                     # poi_idx per stop
    arrive: list                     # arrival clock minutes (since midnight)
    stay: list                       # stay minutes per stop
    total_min: float                 # total minutes from first arrival to last departure
    cost: float                      # search cost (nll + lam*travel, or minutes in "time" mode)
    nll: float = 0.0                 # sum of -log p along the path
    travel_min: float = 0.0


@dataclass
class SearchProblem:
    cd: object
    bundle: object
    start: int
    end: int
    pattern: tuple                   # category index per position
    start_hour: float = 9.0          # clock hour when the tour starts
    budget: float | None = None      # minutes
    lam: float = 0.05                # nats per walking minute
    mode: str = "nll"                # "nll" | "time"

    def __post_init__(self):
        P = self.cd.n_places
        self.n = len(self.pattern)
        self.cats = self.cd.cat_of
        self.logp = self.bundle.logp_table()               # (B, P, P)
        self.nb = self.logp.shape[0]
        self.stay = np.maximum(5, np.round(self.bundle.dur_table())).astype(int)   # (B, P)
        self.trav = np.round(self.cd.travel_min).astype(int)
        self.t0 = self.start_hour * 60.0
        self.cands = [np.flatnonzero(self.cats == c) for c in self.pattern]
        self.cands[0] = np.array([self.start])
        self.cands[-1] = np.array([self.end])
        for k in range(1, self.n - 1):    # start / end places are not re-used as intermediate stops
            self.cands[k] = np.array([j for j in self.cands[k] if j not in (self.start, self.end)], dtype=int)
        self._build_heuristic()

    # ------------------------------------------------------------- model pieces ----
    def bucket(self, minutes_since_start: float) -> int:
        return hour_bucket(self.t0 + minutes_since_start, self.nb)

    def stay_at(self, place: int, elapsed: int) -> int:
        return int(self.stay[self.bucket(elapsed), place])

    def step_cost(self, i: int, j: int, depart_elapsed: int):
        """cost of moving i->j, leaving i at elapsed minutes. Returns (cost, nll, travel)."""
        tr = int(self.trav[i, j])
        nll = float(-self.logp[self.bucket(depart_elapsed), i, j])
        if self.mode == "time":
            return float(tr), nll, tr        # stay is added separately in 'time' mode
        return nll + self.lam * tr, nll, tr

    # ------------------------------------------------------------- heuristic -------
    def _build_heuristic(self):
        """h[k][i] = lower bound of cost to finish from place i at position k  (k = n-1 -> 0)."""
        P = self.cd.n_places
        n = self.n
        nll_min = (-self.logp).min(0)                      # min over clock hours, (P, P)
        if self.mode == "time":
            step = self.trav.astype(float)
            stay_min = self.stay.min(0).astype(float)      # (P,)
        else:
            step = nll_min + self.lam * self.trav
            stay_min = np.zeros(P)
        self.step_min = step
        h = [np.full(P, np.inf) for _ in range(n)]
        h[n - 1][self.end] = 0.0 if self.mode != "time" else 0.0
        tl = [np.full(P, np.inf) for _ in range(n)]       # lower bound of remaining *minutes*
        tl[n - 1][self.end] = 0.0
        tmin = self.trav.astype(float) + self.stay.min(0)[None, :]
        for k in range(n - 2, -1, -1):
            nxt = self.cands[k + 1]
            for i in self.cands[k]:
                best, bt = np.inf, np.inf
                for j in nxt:
                    if j == i:
                        continue
                    c = step[i, j] + stay_min[j] + h[k + 1][j]
                    best = min(best, c)
                    bt = min(bt, tmin[i, j] + tl[k + 1][j])
                h[k][i] = best
                tl[k][i] = bt
        self.h, self.tl = h, tl

    def h_of(self, pos: int, place: int) -> float:
        return float(self.h[pos][place])

    def feasible_lb(self) -> bool:
        return np.isfinite(self.h[0][self.start])

    def min_total_minutes_lb(self) -> float:
        return float(self.stay[:, self.start].min() + self.tl[0][self.start])


def _expand(pb: SearchProblem, pos, cur, visited, elapsed, g, nll_acc, trav_acc):
    """Yield successor tuples (place, elapsed', g', nll', trav')."""
    nxt = pb.cands[pos + 1]
    last = (pos + 1 == pb.n - 1)
    for j in nxt:
        j = int(j)
        if j == cur:
            continue                      # never "move" to the same place
        if not last and (j in visited or j == pb.end):
            continue                      # intermediate stops: unvisited and not the end place
        c, nll, tr = pb.step_cost(cur, j, elapsed)
        arrive = elapsed + tr
        el2 = arrive + pb.stay_at(j, arrive)
        g2 = g + c + (pb.stay_at(j, arrive) if pb.mode == "time" else 0.0)
        yield j, el2, g2, nll_acc + nll, trav_acc + tr, arrive


def search(pb: SearchProblem, algo: str = "astar", max_expansions: int = 2_000_000, trace: list | None = None):
    """Run 'ucs' | 'astar' | 'greedy'. Returns (Itinerary | None, SearchStats)."""
    stats = SearchStats()
    if not pb.feasible_lb():
        return None, stats
    counter = itertools.count()
    s0 = pb.stay_at(pb.start, 0)
    g0 = float(s0) if pb.mode == "time" else 0.0

    def prio(g, pos, place):
        h = pb.h_of(pos, place)
        if algo == "ucs":
            return g
        if algo == "astar":
            return g + h
        if algo == "greedy":
            return h
        raise ValueError(algo)

    start_state = (0, pb.start, frozenset([pb.start]), s0)
    # per state: best g seen (exact: state includes elapsed minutes)
    best_g = {start_state: g0}
    parent = {}
    heap = [(prio(g0, 0, pb.start), 0.0, next(counter), start_state, g0, 0.0, 0)]
    stats.generated = 1
    # bookkeeping to rebuild the path
    info = {start_state: (None, 0.0, 0)}                  # parent state, nll_acc, trav_acc
    while heap:
        stats.max_open = max(stats.max_open, len(heap))
        f, _, _, st, g, _, _ = heapq.heappop(heap)
        if g > best_g.get(st, np.inf) + 1e-12:
            continue                                      # stale entry
        pos, cur, visited, elapsed = st
        if pos == pb.n - 1:
            return _rebuild(pb, st, g, info), stats
        stats.expanded += 1
        if trace is not None:
            trace.append(dict(n=stats.expanded, pos=pos, place=cur, elapsed=elapsed, g=g,
                              h=pb.h_of(pos, cur), open=len(heap)))
        if stats.expanded > max_expansions:
            return None, stats
        _, nll_acc, trav_acc = info[st]
        for j, el2, g2, nll2, trav2, arrive in _expand(pb, pos, cur, visited, elapsed, g, nll_acc, trav_acc):
            # budget pruning with an admissible bound on the remaining minutes
            if pb.budget is not None and el2 + pb.tl[pos + 1][j] > pb.budget + 1e-9:
                stats.pruned_budget += 1
                continue
            v2 = visited | {j} if pos + 1 < pb.n - 1 else visited
            st2 = (pos + 1, j, v2, el2)
            if g2 < best_g.get(st2, np.inf) - 1e-12:
                best_g[st2] = g2
                info[st2] = (st, nll2, trav2)
                stats.generated += 1
                heapq.heappush(heap, (prio(g2, pos + 1, j), g2, next(counter), st2, g2, 0.0, 0))
    return None, stats


def _rebuild(pb: SearchProblem, goal, g, info) -> Itinerary:
    chain = []
    st = goal
    while st is not None:
        chain.append(st)
        st = info[st][0]
    chain.reverse()
    places = [s[1] for s in chain]
    arrive, stay = [], []
    el = 0
    for k, s in enumerate(chain):
        if k == 0:
            arr = 0
        else:
            arr = chain[k - 1][3] + int(pb.trav[places[k - 1], places[k]])
        arrive.append(pb.t0 + arr)
        stay.append(s[3] - arr)
    _, nll_acc, trav_acc = info[goal]
    return Itinerary(places=places, arrive=arrive, stay=stay, total_min=float(chain[-1][3]),
                     cost=float(g), nll=float(nll_acc), travel_min=float(trav_acc))


# -------------------------------------------------------------------- brute force ---
def path_cost(pb: SearchProblem, places) -> tuple:
    """Exact cost / total minutes of a concrete path under the same model (used for checking)."""
    el = pb.stay_at(places[0], 0)
    g = float(el) if pb.mode == "time" else 0.0
    for a, b in zip(places[:-1], places[1:]):
        c, _, tr = pb.step_cost(a, b, el)
        arr = el + tr
        s = pb.stay_at(b, arr)
        el = arr + s
        g += c + (s if pb.mode == "time" else 0.0)
    return g, el


def brute_force(pb: SearchProblem):
    """Enumerate every admissible path for the pattern; return the best (cost, places) or None."""
    n = pb.n
    best = None
    mids = [list(map(int, pb.cands[k])) for k in range(1, n - 1)]
    for combo in itertools.product(*mids):
        path = [pb.start, *combo, pb.end]
        mid = list(combo)
        if len(set(mid)) != len(mid) or pb.start in mid or pb.end in mid:
            continue
        if any(a == b for a, b in zip(path[:-1], path[1:])):
            continue
        g, el = path_cost(pb, path)
        if pb.budget is not None and el > pb.budget + 1e-9:
            continue
        if best is None or g < best[0] - 1e-12:
            best = (g, path)
    return best
