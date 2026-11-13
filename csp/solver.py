"""Category-Pattern Constraint Satisfaction Problem (CSP) Solver.

Variables:
  C_0, C_1, ..., C_{L-1} representing the category of each stop in an L-stop tour.
Domains:
  - D(C_0) = { Category(start_place) }
  - D(C_{L-1}) = { Category(end_place) }
  - D(C_i) = All available POI categories in the city for 0 < i < L-1.
Constraints:
  1. Unary endpoint constraints: C_0 = Cat(start), C_{L-1} = Cat(end).
  2. Binary adjacent-difference constraint: C_i != C_{i+1} for all 0 <= i < L-1
     (neighbouring stops must belong to different categories).
  3. Global cardinality constraint: Count(c in pattern) <= 2 for every category c
     (no category appears more than twice in the itinerary).
  4. Global mandatory constraint: must_see_category appears at least once in
     (C_0, ..., C_{L-1}).
  5. Global expected time-budget constraint:
     sum_{i=0}^{L-1} E[visit_duration(C_i)] + (L - 1) * E[walk_time] <= time_budget_hours.

Switchable Algorithmic Components:
  - Backtracking (BT): Standard depth-first backtracking checking constraints on
    partial/complete assignments.
  - Forward Checking (FC): Prunes future unassigned variables' domains immediately
    upon each assignment (pruning adjacent category match, categories that hit
    count==2, mandatory category reachability, and minimum remaining expected time).
  - Arc Consistency (AC-3): Enforces binary arc consistency across all adjacent
    variable pairs (X_i, X_j) and propagates domain reductions before and during
    search.

Counters tracked:
  - assignments: Number of variable assignments attempted.
  - backtracks: Number of dead-ends / chronological backtracks triggered.
  - pruned_values: Number of domain values pruned by FC / AC-3.
"""

from __future__ import annotations

from collections import deque
import dataclasses
from typing import Sequence


@dataclasses.dataclass
class CSPStats:
    """Tracks CSP solver effort and pruning statistics."""
    assignments: int = 0
    backtracks: int = 0
    pruned_values: int = 0
    patterns_found: int = 0


class CategoryPatternCSP:
    """CSP solver for valid sightseeing category sequences with switchable BT, FC, and AC-3."""

    def __init__(
        self,
        categories: Sequence[str],
        category_mean_durations_hr: dict[str, float],
        expected_walk_hr: float = 0.22,
    ):
        self.categories = sorted(list(categories))
        self.cat_durations = dict(category_mean_durations_hr)
        self.expected_walk_hr = float(expected_walk_hr)
        self.min_cat_duration = min(self.cat_durations.values()) if self.cat_durations else 0.25

    def expected_pattern_time(self, pattern: Sequence[str]) -> float:
        """Computes expected total time (hours) for a category pattern."""
        if not pattern:
            return 0.0
        visit_hr = sum(self.cat_durations[c] for c in pattern)
        walk_hr = max(0, len(pattern) - 1) * self.expected_walk_hr
        return visit_hr + walk_hr

    def _ac3(
        self,
        domains: dict[int, list[str]],
        num_stops: int,
        stats: CSPStats,
        assigned: dict[int, str],
    ) -> bool:
        """Enforces AC-3 arc consistency over adjacent inequality constraints and cardinality bounds."""
        queue = deque()
        for i in range(num_stops - 1):
            queue.append((i, i + 1))
            queue.append((i + 1, i))

        # Also add cardinality-coupling arcs between all pairs when a domain becomes singleton
        for i in range(num_stops):
            for j in range(num_stops):
                if i != j and abs(i - j) > 1:
                    queue.append((i, j))

        while queue:
            xi, xj = queue.popleft()
            if self._revise(domains, xi, xj, stats, assigned):
                if len(domains[xi]) == 0:
                    return False
                for xk in range(num_stops):
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))
        return True

    def _revise(
        self,
        domains: dict[int, list[str]],
        xi: int,
        xj: int,
        stats: CSPStats,
        assigned: dict[int, str],
    ) -> bool:
        """Revises domain of xi with respect to xj."""
        revised = False
        to_remove = []
        is_adjacent = abs(xi - xj) == 1

        # Count how many times each category is already fixed in other singleton/assigned positions
        singleton_counts: dict[str, int] = {}
        for k, dom in domains.items():
            if k != xi and len(dom) == 1:
                c = dom[0]
                singleton_counts[c] = singleton_counts.get(c, 0) + 1

        for val_i in domains[xi]:
            if is_adjacent:
                # Must exist some val_j in D(xj) such that val_j != val_i
                if not any(val_j != val_i for val_j in domains[xj]):
                    to_remove.append(val_i)
                    continue
            # Cardinality consistency: if two other variables already have singleton {val_i}, xi cannot be val_i
            if singleton_counts.get(val_i, 0) >= 2:
                to_remove.append(val_i)

        if to_remove:
            domains[xi] = [v for v in domains[xi] if v not in to_remove]
            stats.pruned_values += len(to_remove)
            revised = True
        return revised

    def _is_partial_valid(
        self,
        assignment: dict[int, str],
        num_stops: int,
        must_see: str | None,
        budget_hr: float,
    ) -> bool:
        """Checks constraints on the current partial (or complete) assignment."""
        # 1. Adjacent stops differ
        for i in range(num_stops - 1):
            if i in assignment and (i + 1) in assignment:
                if assignment[i] == assignment[i + 1]:
                    return False

        # 2. No category more than twice
        counts: dict[str, int] = {}
        for c in assignment.values():
            counts[c] = counts.get(c, 0) + 1
            if counts[c] > 2:
                return False

        # 3. Must-see category check
        unassigned_slots = num_stops - len(assignment)
        if must_see is not None:
            has_must_see = any(c == must_see for c in assignment.values())
            if not has_must_see and unassigned_slots == 0:
                return False

        # 4. Budget check (exact on complete assignment; optimistic bound only when complete in pure BT)
        if unassigned_slots == 0:
            pattern = [assignment[i] for i in range(num_stops)]
            if self.expected_pattern_time(pattern) > budget_hr + 1e-9:
                return False

        return True

    def _forward_check(
        self,
        var_idx: int,
        val: str,
        domains: dict[int, list[str]],
        assignment: dict[int, str],
        num_stops: int,
        must_see: str | None,
        budget_hr: float,
        stats: CSPStats,
    ) -> bool:
        """Prunes future domains using Forward Checking and lookahead bounds."""
        # Count current category occurrences
        counts: dict[str, int] = {}
        elapsed_visit_hr = 0.0
        for c in assignment.values():
            counts[c] = counts.get(c, 0) + 1
            elapsed_visit_hr += self.cat_durations[c]

        # 1. Prune adjacent unassigned neighbours
        for nbr in (var_idx - 1, var_idx + 1):
            if 0 <= nbr < num_stops and nbr not in assignment:
                if val in domains[nbr]:
                    domains[nbr] = [c for c in domains[nbr] if c != val]
                    stats.pruned_values += 1
                    if not domains[nbr]:
                        return False

        # 2. If val reached count == 2, prune val from all remaining unassigned variables
        if counts.get(val, 0) >= 2:
            for u in range(num_stops):
                if u not in assignment and val in domains[u]:
                    domains[u] = [c for c in domains[u] if c != val]
                    stats.pruned_values += 1
                    if not domains[u]:
                        return False

        # 3. Must-see reachability check
        unassigned = [u for u in range(num_stops) if u not in assignment]
        if must_see is not None and counts.get(must_see, 0) == 0:
            can_place_must_see = any(must_see in domains[u] for u in unassigned)
            if not can_place_must_see:
                return False
            # If only 1 unassigned variable can take must_see, force its domain to [must_see]
            holders = [u for u in unassigned if must_see in domains[u]]
            if len(holders) == 1:
                u_only = holders[0]
                pruned = len(domains[u_only]) - 1
                if pruned > 0:
                    domains[u_only] = [must_see]
                    stats.pruned_values += pruned

        # 4. Budget lookahead pruning on remaining unassigned domains
        walk_total_hr = max(0, num_stops - 1) * self.expected_walk_hr
        min_other_visits = sum(
            min(self.cat_durations[c] for c in domains[u]) for u in unassigned
        ) if unassigned else 0.0

        if elapsed_visit_hr + walk_total_hr + min_other_visits > budget_hr + 1e-9:
            return False

        # Prune individual expensive categories from unassigned domains if selecting them exceeds budget
        for u in unassigned:
            min_u = min(self.cat_durations[c] for c in domains[u])
            base_without_u = elapsed_visit_hr + walk_total_hr + (min_other_visits - min_u)
            allowed = [
                c for c in domains[u]
                if base_without_u + self.cat_durations[c] <= budget_hr + 1e-9
            ]
            pruned = len(domains[u]) - len(allowed)
            if pruned > 0:
                domains[u] = allowed
                stats.pruned_values += pruned
            if not domains[u]:
                return False

        return True

    def solve(
        self,
        num_stops: int,
        start_category: str,
        end_category: str,
        must_see_category: str | None,
        budget_hr: float,
        use_fc: bool = True,
        use_ac3: bool = True,
        max_solutions: int = 10,
    ) -> tuple[list[tuple[str, ...]], CSPStats]:
        """Solves the Category-Pattern CSP and returns (up to max_solutions patterns, CSPStats)."""
        stats = CSPStats()
        if num_stops < 2:
            return [], stats
        if must_see_category is not None and must_see_category not in self.categories:
            return [], stats
        if start_category not in self.categories or end_category not in self.categories:
            return [], stats

        # Order categories by ascending duration / balanced variety
        ordered_cats = sorted(self.categories, key=lambda c: (self.cat_durations[c], c))

        domains: dict[int, list[str]] = {
            i: list(ordered_cats) for i in range(num_stops)
        }
        domains[0] = [start_category]
        domains[num_stops - 1] = [end_category]

        assignment: dict[int, str] = {}
        solutions: list[tuple[str, ...]] = []

        # Initial AC-3 preprocessing if enabled
        if use_ac3:
            if not self._ac3(domains, num_stops, stats, assignment):
                return [], stats

        def backtrack(idx: int, cur_domains: dict[int, list[str]]) -> None:
            if len(solutions) >= max_solutions:
                return
            if idx == num_stops:
                if self._is_partial_valid(assignment, num_stops, must_see_category, budget_hr):
                    solutions.append(tuple(assignment[i] for i in range(num_stops)))
                    stats.patterns_found = len(solutions)
                else:
                    stats.backtracks += 1
                return

            for val in list(cur_domains[idx]):
                if len(solutions) >= max_solutions:
                    return
                stats.assignments += 1
                assignment[idx] = val

                if not self._is_partial_valid(assignment, num_stops, must_see_category, budget_hr):
                    del assignment[idx]
                    stats.backtracks += 1
                    continue

                next_domains = {k: list(v) for k, v in cur_domains.items()}
                next_domains[idx] = [val]

                consistent = True
                if use_fc:
                    consistent = self._forward_check(
                        idx, val, next_domains, assignment, num_stops, must_see_category, budget_hr, stats
                    )
                if consistent and use_ac3:
                    consistent = self._ac3(next_domains, num_stops, stats, assignment)

                if consistent:
                    backtrack(idx + 1, next_domains)
                else:
                    stats.backtracks += 1

                del assignment[idx]

        backtrack(0, domains)
        return solutions, stats
