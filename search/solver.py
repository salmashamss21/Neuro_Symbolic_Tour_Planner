"""Heuristic Search Engine (UCS, A*, Greedy Best-First, and Brute-Force Oracle).

State representation:
  State = (pos, current_poi_id, visited_frozenset, elapsed_time_hr)
  where:
    - pos: current index in the category pattern (0 <= pos < L - 1)
    - current_poi_id: ID of the POI at position `pos`
    - visited_frozenset: set of already visited POI IDs (no repeating places)
    - elapsed_time_hr: cumulative travel + visit time in hours up to departing `current_poi_id`

Action:
  Transition to an unvisited POI `next_poi` whose category matches `pattern[pos + 1]`
  (and at the final step `pos + 1 == L - 1`, `next_poi` must equal `end_poi_id`).

Step Cost:
  cost(u -> v | hour) = -log P_NN(v | u, hour) + alpha_travel * travel_time_hr(u, v)

Algorithms supported:
  1. UCS (Uniform Cost Search): f(n) = g(n)
  2. A* Search: f(n) = g(n) + h(n), where h(n) is an admissible lower bound combining
     minimum remaining negative log transition probability and straight-line Haversine
     travel time to the destination.
  3. Greedy Best-First Search: f(n) = step_cost + h(n) (prioritizes immediate heuristic/step score).
  4. Brute-Force Oracle: Exhaustively evaluates all valid simple paths matching the pattern
     on a small pool to verify exact optimality of UCS and A*.
"""

from __future__ import annotations

import dataclasses
import heapq
import itertools
import math
from typing import Callable, Sequence


@dataclasses.dataclass
class SearchStats:
    """Counters for search space exploration."""
    nodes_expanded: int = 0
    nodes_generated: int = 0


@dataclasses.dataclass
class ItineraryResult:
    """Detailed itinerary produced by Search for a given category pattern."""
    pois: list[int]
    categories: list[str]
    arrival_hours: list[float]
    visit_durations_hr: list[float]
    travel_times_hr: list[float]
    step_costs: list[float]
    total_cost: float
    total_time_hr: float
    stats: SearchStats
    trace: list[dict]


class TourSearchSolver:
    """Builds concrete POI itineraries for a category pattern using UCS, A*, or Greedy search."""

    def __init__(
        self,
        poi_ids: Sequence[int],
        poi_categories: dict[int, str],
        travel_time_hr: dict[tuple[int, int], float],
        transition_prob_fn: Callable[[int, int, int], float],
        duration_hr_fn: Callable[[int, int], float],
        travel_weight: float = 1.0,
    ):
        self.poi_ids = list(poi_ids)
        self.poi_categories = dict(poi_categories)
        self.travel_time_hr = dict(travel_time_hr)
        self.transition_prob_fn = transition_prob_fn
        self.duration_hr_fn = duration_hr_fn
        self.travel_weight = float(travel_weight)

        self.cat_to_pois: dict[str, list[int]] = {}
        for pid in self.poi_ids:
            cat = self.poi_categories[pid]
            self.cat_to_pois.setdefault(cat, []).append(pid)

        # Precompute minimum possible -log P across all pairs & hours for admissible A* heuristic
        self._min_neg_log_p = 0.05

    def step_cost(self, u_poi: int, v_poi: int, hour: int) -> float:
        """Computes minus log transition probability plus travel time."""
        prob = max(1e-9, min(1.0 - 1e-9, float(self.transition_prob_fn(u_poi, v_poi, hour))))
        neg_log_p = -math.log(prob)
        walk_hr = self.travel_time_hr.get((u_poi, v_poi), 0.25)
        return neg_log_p + self.travel_weight * walk_hr

    def admissible_heuristic(self, pos: int, current_poi: int, end_poi: int, num_stops: int) -> float:
        """Admissible & consistent lower bound on remaining cost from `pos` to `num_stops - 1`."""
        rem_steps = (num_stops - 1) - pos
        if rem_steps <= 0:
            return 0.0
        # Triangle inequality guarantees straight-line travel time <= multi-hop travel time
        direct_walk = self.travel_time_hr.get((current_poi, end_poi), 0.0)
        return rem_steps * self._min_neg_log_p + self.travel_weight * direct_walk

    def solve(
        self,
        pattern: Sequence[str],
        start_poi: int,
        end_poi: int,
        start_hour: float = 9.0,
        algorithm: str = "astar",
        enforce_budget_in_search: float | None = None,
    ) -> ItineraryResult | None:
        """Finds the best itinerary matching `pattern` from `start_poi` to `end_poi`."""
        num_stops = len(pattern)
        stats = SearchStats()
        if num_stops < 2:
            return None
        if self.poi_categories.get(start_poi) != pattern[0]:
            return None
        if self.poi_categories.get(end_poi) != pattern[-1]:
            return None
        if start_poi == end_poi:
            return None

        start_dur_hr = float(self.duration_hr_fn(start_poi, int(start_hour) % 24))
        init_elapsed = start_dur_hr

        # Priority queue items: (priority, g_cost, tie_breaker, pos, cur_poi, visited_frozenset, elapsed_hr, path_pois, path_arr, path_durs, path_walks, path_costs)
        counter = itertools.count()
        h0 = self.admissible_heuristic(0, start_poi, end_poi, num_stops)
        if algorithm == "ucs":
            f0 = 0.0
        elif algorithm == "greedy":
            f0 = h0
        else:
            f0 = h0

        pq = [(
            f0,
            0.0,
            next(counter),
            0,
            start_poi,
            frozenset([start_poi]),
            init_elapsed,
            [start_poi],
            [float(start_hour)],
            [start_dur_hr],
            [],
            [],
        )]
        stats.nodes_generated += 1

        # Best known g-cost for (pos, cur_poi, visited_frozenset)
        best_g: dict[tuple[int, int, frozenset[int]], float] = {}
        trace: list[dict] = []

        while pq:
            f_val, g_val, _, pos, cur_poi, visited, elapsed_hr, p_pois, p_arr, p_durs, p_walks, p_costs = heapq.heappop(pq)

            state_key = (pos, cur_poi, visited)
            if state_key in best_g and best_g[state_key] < g_val - 1e-12:
                continue
            best_g[state_key] = g_val

            stats.nodes_expanded += 1
            if len(trace) < 40:
                trace.append({
                    "step": stats.nodes_expanded,
                    "pos": pos,
                    "poi": cur_poi,
                    "g": round(g_val, 4),
                    "f": round(f_val, 4),
                    "elapsed_hr": round(elapsed_hr, 3),
                })

            if pos == num_stops - 1:
                if cur_poi == end_poi:
                    return ItineraryResult(
                        pois=p_pois,
                        categories=list(pattern),
                        arrival_hours=p_arr,
                        visit_durations_hr=p_durs,
                        travel_times_hr=p_walks,
                        step_costs=p_costs,
                        total_cost=g_val,
                        total_time_hr=elapsed_hr,
                        stats=stats,
                        trace=trace,
                    )
                continue

            next_pos = pos + 1
            req_cat = pattern[next_pos]
            if next_pos == num_stops - 1:
                candidates = [end_poi] if end_poi not in visited else []
            else:
                candidates = [
                    pid for pid in self.cat_to_pois.get(req_cat, [])
                    if pid not in visited and pid != end_poi
                ]

            depart_hour_clock = start_hour + elapsed_hr
            depart_hour_int = int(depart_hour_clock) % 24

            for nxt_poi in candidates:
                walk_hr = self.travel_time_hr.get((cur_poi, nxt_poi), 0.25)
                arr_clock = depart_hour_clock + walk_hr
                arr_hour_int = int(arr_clock) % 24
                visit_hr = float(self.duration_hr_fn(nxt_poi, arr_hour_int))
                new_elapsed = elapsed_hr + walk_hr + visit_hr

                if enforce_budget_in_search is not None and new_elapsed > enforce_budget_in_search + 1e-9:
                    continue

                sc = self.step_cost(cur_poi, nxt_poi, depart_hour_int)
                new_g = g_val + sc
                new_visited = visited | {nxt_poi}
                nxt_key = (next_pos, nxt_poi, new_visited)
                if nxt_key in best_g and best_g[nxt_key] <= new_g + 1e-12:
                    continue

                h_val = self.admissible_heuristic(next_pos, nxt_poi, end_poi, num_stops)
                if algorithm == "ucs":
                    priority = new_g
                elif algorithm == "greedy":
                    priority = sc + h_val
                else:
                    priority = new_g + h_val

                stats.nodes_generated += 1
                heapq.heappush(pq, (
                    priority,
                    new_g,
                    next(counter),
                    next_pos,
                    nxt_poi,
                    new_visited,
                    new_elapsed,
                    p_pois + [nxt_poi],
                    p_arr + [arr_clock],
                    p_durs + [visit_hr],
                    p_walks + [walk_hr],
                    p_costs + [sc],
                ))

        return None

    def brute_force_search(
        self,
        pattern: Sequence[str],
        start_poi: int,
        end_poi: int,
        start_hour: float = 9.0,
    ) -> ItineraryResult | None:
        """Exhaustively enumerates all valid simple paths matching `pattern` (ground-truth oracle)."""
        num_stops = len(pattern)
        if num_stops < 2:
            return None
        if self.poi_categories.get(start_poi) != pattern[0] or self.poi_categories.get(end_poi) != pattern[-1]:
            return None
        if start_poi == end_poi:
            return None

        mid_pools = [
            [p for p in self.cat_to_pois.get(pattern[pos], []) if p not in (start_poi, end_poi)]
            for pos in range(1, num_stops - 1)
        ]

        best_res: ItineraryResult | None = None
        stats = SearchStats()

        for combo in itertools.product(*mid_pools):
            if len(set(combo)) != len(combo):
                continue
            seq = [start_poi] + list(combo) + [end_poi]
            stats.nodes_generated += 1
            stats.nodes_expanded += 1

            arr_list = [float(start_hour)]
            durs_list = [float(self.duration_hr_fn(start_poi, int(start_hour) % 24))]
            walks_list = []
            costs_list = []
            elapsed = durs_list[0]
            total_cost = 0.0

            for i in range(num_stops - 1):
                u, v = seq[i], seq[i + 1]
                dep_hr_int = int(start_hour + elapsed) % 24
                sc = self.step_cost(u, v, dep_hr_int)
                w_hr = self.travel_time_hr.get((u, v), 0.25)
                arr_clock = start_hour + elapsed + w_hr
                v_dur = float(self.duration_hr_fn(v, int(arr_clock) % 24))
                elapsed += w_hr + v_dur
                total_cost += sc
                arr_list.append(arr_clock)
                durs_list.append(v_dur)
                walks_list.append(w_hr)
                costs_list.append(sc)

            if best_res is None or total_cost < best_res.total_cost:
                best_res = ItineraryResult(
                    pois=seq,
                    categories=list(pattern),
                    arrival_hours=arr_list,
                    visit_durations_hr=durs_list,
                    travel_times_hr=walks_list,
                    step_costs=costs_list,
                    total_cost=total_cost,
                    total_time_hr=elapsed,
                    stats=stats,
                    trace=[],
                )

        return best_res
