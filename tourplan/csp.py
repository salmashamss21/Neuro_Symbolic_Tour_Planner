"""Category-pattern CSP.

Variables   x_0 .. x_{n-1}: the category of each stop (x_0 / x_{n-1} are fixed by the start / end place).
Constraints (1) neighbouring stops differ                      (binary, used by AC-3)
            (2) the must-see category appears at least once    (global)
            (3) no category more than `max_per_cat` times      (global; also limited by #places)
            (4) lower bound of the total time <= budget        (global)

Three switchable solvers (all return the same set of solutions; they differ in work done):
    BT          plain chronological backtracking, constraints checked when their variables are assigned
    BT+FC       + forward checking (prunes future domains after each assignment, incl. global rules)
    BT+FC+AC3   + AC-3 arc consistency at the root and re-established after each assignment (MAC)

Counters in `CSPStats`: assignments, backtracks, checks, revisions, prunings, solutions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CSPStats:
    assignments: int = 0      # tentative variable assignments tried
    backtracks: int = 0       # assignments whose subtree produced no solution
    checks: int = 0           # constraint checks performed
    revisions: int = 0        # AC-3 arc revisions
    prunings: int = 0         # domain values removed (FC + AC-3)
    solutions: int = 0

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class CSPProblem:
    n: int
    domains: list                    # list of frozensets of category indices
    cap: list                        # max number of stops per category
    must: int | None = None
    budget: float | None = None      # minutes; None = unconstrained
    stay_lb: np.ndarray | None = None    # per-category lower bound on stay (min)
    trav_lb: np.ndarray | None = None    # category x category lower bound on travel (min)
    trav_any: float = 0.0
    prior: np.ndarray | None = None      # category x category preference for value ordering
    n_cats: int = 0


def build_csp(cd, bundle, start: int, end: int, n: int, must_cat: str | None,
              budget: float | None, max_per_cat: int = 2, prior=None) -> CSPProblem:
    """Create the CSP for one request from city data and the duration model."""
    cats = cd.cat_of
    n_cats = len(cd.categories)
    n_in_cat = np.bincount(cats, minlength=n_cats)
    cap = [int(min(max_per_cat, c)) for c in n_in_cat]
    if start == end:                                    # same place may open and close the tour
        cap[cats[start]] = min(max_per_cat, int(n_in_cat[cats[start]]) + 1)
    full = frozenset(c for c in range(n_cats) if cap[c] > 0)
    domains = [full] * n
    domains[0] = frozenset([int(cats[start])])
    domains[-1] = frozenset([int(cats[end])])
    stay = np.maximum(5.0, np.round(bundle.dur_table())).min(0)   # (P,) min over time of day
    stay_lb = np.array([stay[cats == c].min() if (cats == c).any() else np.inf for c in range(n_cats)])
    t = cd.travel_min
    trav_lb = np.full((n_cats, n_cats), np.inf)
    P = cd.n_places
    for i in range(P):
        for j in range(P):
            if i != j:
                trav_lb[cats[i], cats[j]] = min(trav_lb[cats[i], cats[j]], np.round(t[i, j]))
    trav_any = float(np.nanmin(np.where(np.eye(P, dtype=bool), np.inf, np.round(t))))
    must = None if must_cat is None else cd.categories.index(must_cat)
    return CSPProblem(n, domains, cap, must, budget, stay_lb, trav_lb, trav_any, prior, n_cats)


# ---------------------------------------------------------------------------------
def _lb_total(p: CSPProblem, doms, assigned: int) -> float:
    """Lower bound on total minutes given the (partially assigned) pattern."""
    tot = 0.0
    for i in range(p.n):
        tot += min(p.stay_lb[c] for c in doms[i]) if i >= assigned else p.stay_lb[next(iter(doms[i]))]
    for i in range(p.n - 1):
        if i + 1 < assigned:
            tot += p.trav_lb[next(iter(doms[i])), next(iter(doms[i + 1]))]
        else:
            tot += p.trav_any
    return tot


def _ac3(p: CSPProblem, doms: list, stats: CSPStats, queue=None) -> bool:
    """AC-3 on the 'neighbours differ' constraints. Returns False on domain wipe-out."""
    if queue is None:
        queue = deque((i, j) for i in range(p.n) for j in (i - 1, i + 1) if 0 <= j < p.n)
    else:
        queue = deque(queue)
    inq = set(queue)
    while queue:
        i, j = queue.popleft()
        inq.discard((i, j))
        stats.revisions += 1
        # arc (i, j): remove x from D_i if D_j == {x}   (no y != x left in D_j)
        if len(doms[j]) == 1:
            x = next(iter(doms[j]))
            stats.checks += 1
            if x in doms[i]:
                doms[i] = doms[i] - {x}
                stats.prunings += 1
                if not doms[i]:
                    return False
                for k in (i - 1, i + 1):
                    if 0 <= k < p.n and k != j and (k, i) not in inq:
                        queue.append((k, i))
                        inq.add((k, i))
    return True


def _fc_global(p: CSPProblem, doms: list, counts: list, assigned: int, stats: CSPStats) -> bool:
    """Forward-checking for the global constraints. Mutates doms (list of frozensets)."""
    # (3) saturated categories vanish from future domains
    for c in range(p.n_cats):
        stats.checks += 1
        if counts[c] >= p.cap[c]:
            for i in range(assigned, p.n):
                if c in doms[i]:
                    doms[i] = doms[i] - {c}
                    stats.prunings += 1
                    if not doms[i]:
                        return False
    # (2) must-see still reachable / forced
    if p.must is not None and counts[p.must] == 0:
        stats.checks += 1
        holders = [i for i in range(assigned, p.n) if p.must in doms[i]]
        if not holders:
            return False
        if len(holders) == 1 and len(doms[holders[0]]) > 1:
            stats.prunings += len(doms[holders[0]]) - 1
            doms[holders[0]] = frozenset([p.must])
    # (4) budget lower bound
    if p.budget is not None:
        stats.checks += 1
        if _lb_total(p, doms, assigned) > p.budget + 1e-9:
            return False
    return True


def solve(p: CSPProblem, fc: bool = False, ac3: bool = False, max_solutions: int | None = None,
          order_values: bool = False):
    """Enumerate patterns (tuples of category indices). Returns (solutions, CSPStats)."""
    stats = CSPStats()
    sols: list = []
    n = p.n
    doms = [frozenset(d) for d in p.domains]
    counts0 = [0] * p.n_cats

    def ordered(i, prev, dom):
        if order_values and p.prior is not None and prev is not None:
            return sorted(dom, key=lambda c: (-p.prior[prev, c], c))
        return sorted(dom)

    if ac3:
        if not _ac3(p, doms, stats):
            return sols, stats
    if fc:
        if not _fc_global(p, doms, counts0, 0, stats):
            return sols, stats
        if ac3 and not _ac3(p, doms, stats):
            return sols, stats

    def rec(i, assign, doms, counts, part_cost) -> bool:
        """Returns True if the subtree yielded at least one solution."""
        found = False
        prev = assign[-1] if assign else None
        for v in ordered(i, prev, doms[i]):
            if max_solutions is not None and stats.solutions >= max_solutions:
                return found
            stats.assignments += 1
            # ---- constraint checks against already-assigned variables ----
            stats.checks += 1
            if prev is not None and prev == v:
                stats.backtracks += 1
                continue
            stats.checks += 1
            if counts[v] + 1 > p.cap[v]:
                stats.backtracks += 1
                continue
            new_cost = part_cost + p.stay_lb[v] + (p.trav_lb[prev, v] if prev is not None else 0.0)
            if p.budget is not None:
                stats.checks += 1
                if new_cost > p.budget + 1e-9:
                    stats.backtracks += 1
                    continue
            new_counts = counts.copy()
            new_counts[v] += 1
            new_assign = assign + [v]
            nd = doms
            if fc:
                nd = list(doms)
                nd[i] = frozenset([v])
                ok = True
                if i + 1 < n and v in nd[i + 1]:
                    nd[i + 1] = nd[i + 1] - {v}
                    stats.prunings += 1
                    ok = bool(nd[i + 1])
                if ok:
                    ok = _fc_global(p, nd, new_counts, i + 1, stats)
                if ok and ac3:
                    ok = _ac3(p, nd, stats, queue=[(i + 1, i)] if i + 1 < n else [])
                if not ok:
                    stats.backtracks += 1
                    continue
            if i == n - 1:
                stats.checks += 1
                if p.must is None or new_counts[p.must] > 0:
                    sols.append(tuple(new_assign))
                    stats.solutions += 1
                    found = True
                else:
                    stats.backtracks += 1
                continue
            if rec(i + 1, new_assign, nd, new_counts, new_cost):
                found = True
            else:
                stats.backtracks += 1
        return found

    rec(0, [], doms, counts0, 0.0)
    return sols, stats


def category_prior(cd, trans_data, alpha: float = 0.5) -> np.ndarray:
    """Smoothed category->category transition frequencies from training transitions."""
    cur, _, nxt = trans_data
    k = len(cd.categories)
    m = np.full((k, k), alpha)
    np.add.at(m, (cd.cat_of[cur], cd.cat_of[nxt]), 1.0)
    return m / m.sum(1, keepdims=True)


def rank_patterns(patterns, prior: np.ndarray | None, top: int = 10):
    """Keep the `top` patterns with the highest category-level log-likelihood (ties: lexicographic)."""
    if prior is None:
        return sorted(patterns)[:top]

    def score(pt):
        return sum(np.log(prior[a, b]) for a, b in zip(pt[:-1], pt[1:]))
    return sorted(patterns, key=lambda pt: (-score(pt), pt))[:top]


def root_domains(p: CSPProblem, fc: bool = False, ac3: bool = False):
    """Domains after root-level propagation only (for demos): returns (domains, CSPStats) or (None, stats)."""
    stats = CSPStats()
    doms = [frozenset(d) for d in p.domains]
    if ac3 and not _ac3(p, doms, stats):
        return None, stats
    if fc:
        if not _fc_global(p, doms, [0] * p.n_cats, 0, stats):
            return None, stats
        if ac3 and not _ac3(p, doms, stats):
            return None, stats
    return doms, stats
