"""End-to-End Integrated Pipeline: CSP -> Heuristic Search -> NN Duration Verification.

Workflow:
1. CSP produces up to 10 valid category patterns satisfying:
   - Start and end categories match `start_poi` and `end_poi`
   - Neighbouring stops have different categories
   - `must_see_category` appears at least once
   - No category appears more than twice
   - Expected pattern duration is within `time_budget_hours`
2. For each candidate category pattern, Search (UCS / A* / Greedy) finds the most
   plausible sequence of concrete POIs minimizing sum(-log P_NN + travel_time).
3. Predicted NN visit durations and exact walking times are summed:
   - If total predicted time <= `time_budget_hours`, the candidate is accepted.
   - If total predicted time > `time_budget_hours`, the candidate is rejected.
4. The accepted itinerary with the lowest total cost (highest plausibility) is returned.
5. Fallback & Minimum Working Budget Handling:
   - If no solution fits `time_budget_hours` (either CSP returns 0 patterns under
     budget or all Search candidates exceed budget after adding NN durations),
     the system relaxes the CSP time budget to find the cheapest feasible itinerary,
     reports the exact `minimum_working_budget_hours` that works, and returns the
     cheapest itinerary with a clear warning message (or returns `status="no_solution"`
     when structural constraints like impossible must-see or invalid endpoints cannot
     be satisfied under any finite time budget).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from csp.solver import CategoryPatternCSP, CSPStats
from data.loader import TourDataset
from nn.baselines import MarkovCountBaseline, PopularityBaseline
from nn.model import DualHeadTourNN
from search.solver import ItineraryResult, SearchStats, TourSearchSolver


@dataclasses.dataclass
class PipelineOutput:
    """Complete result of the integrated CSP + Search + NN tour planner."""
    status: str  # "ok", "budget_warning", or "no_solution"
    message: str
    best_itinerary: ItineraryResult | None
    minimum_working_budget_hr: float | None
    csp_patterns_generated: list[tuple[str, ...]]
    pattern_evaluations: list[dict]
    csp_stats: CSPStats
    total_search_stats: SearchStats


class IntegratedTourPlanner:
    """Co-ordinates CSP category planning, NN transition/duration heads, and heuristic search."""

    def __init__(
        self,
        dataset: TourDataset,
        nn_model: DualHeadTourNN,
        markov_baseline: MarkovCountBaseline | None = None,
        popularity_baseline: PopularityBaseline | None = None,
    ):
        self.dataset = dataset
        self.nn_model = nn_model
        self.markov_baseline = markov_baseline
        self.popularity_baseline = popularity_baseline
        self.cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}

        self.csp = CategoryPatternCSP(
            categories=dataset.categories,
            category_mean_durations_hr=dataset.category_mean_duration_hr,
            expected_walk_hr=dataset.mean_walk_time_hr,
        )

    def _build_search_solver(self, cost_mode: str = "nn", travel_weight: float = 1.0) -> TourSearchSolver:
        """Creates a TourSearchSolver wired to NN, Markov-count, or Popularity costs."""
        # Cache probability rows per (poi_id, hour) for fast search expansion
        prob_cache: dict[tuple[int, int], object] = {}
        dur_cache: dict[tuple[int, int], float] = {}

        def prob_fn(u_poi: int, v_poi: int, hour: int) -> float:
            u_idx = self.dataset.poi_to_idx[u_poi]
            v_idx = self.dataset.poi_to_idx[v_poi]
            h_mod = int(hour) % 24
            key = (u_idx, h_mod)
            if key not in prob_cache:
                if cost_mode == "nn":
                    cat_idx = self.cat_to_idx[self.dataset.poi_categories[u_poi]]
                    prob_cache[key] = self.nn_model.predict_next_probs(u_idx, cat_idx, h_mod)
                elif cost_mode == "markov" and self.markov_baseline is not None:
                    prob_cache[key] = self.markov_baseline.predict_next_probs(u_idx, h_mod)
                elif cost_mode == "popularity" and self.popularity_baseline is not None:
                    prob_cache[key] = self.popularity_baseline.predict_next_probs(u_idx, h_mod)
                else:
                    # Uniform / distance fallback when NN is switched off
                    n = len(self.dataset.poi_ids)
                    prob_cache[key] = [1.0 / max(1, n - 1)] * n
            return float(prob_cache[key][v_idx])

        def dur_fn(poi_id: int, hour: int) -> float:
            p_idx = self.dataset.poi_to_idx[poi_id]
            h_mod = int(hour) % 24
            key = (p_idx, h_mod)
            if key not in dur_cache:
                if cost_mode == "nn":
                    cat_idx = self.cat_to_idx[self.dataset.poi_categories[poi_id]]
                    dur_cache[key] = self.nn_model.predict_duration_hours(p_idx, cat_idx, h_mod)
                elif cost_mode == "markov" and self.markov_baseline is not None:
                    dur_cache[key] = self.markov_baseline.predict_duration_hours(p_idx, h_mod)
                else:
                    dur_cache[key] = float(self.dataset.poi_mean_duration_hr.get(poi_id, 0.6))
            return dur_cache[key]

        return TourSearchSolver(
            poi_ids=self.dataset.poi_ids,
            poi_categories=self.dataset.poi_categories,
            travel_time_hr=self.dataset.travel_time_hr,
            transition_prob_fn=prob_fn,
            duration_hr_fn=dur_fn,
            travel_weight=travel_weight,
        )

    def plan_tour(
        self,
        start_poi: int,
        end_poi: int,
        time_budget_hr: float,
        must_see_category: str | None = "Museum",
        num_stops: int = 5,
        start_hour: float = 10.0,
        search_algo: str = "astar",
        use_fc: bool = True,
        use_ac3: bool = True,
        cost_mode: str = "nn",
        max_patterns: int = 10,
    ) -> PipelineOutput:
        """Executes the full CSP -> Search -> NN Duration Verification pipeline."""
        total_search_stats = SearchStats()

        # Validate basic structural inputs
        if start_poi not in self.dataset.poi_to_idx or end_poi not in self.dataset.poi_to_idx:
            return PipelineOutput(
                status="no_solution",
                message=f"No solution: Unknown start ({start_poi}) or end ({end_poi}) POI ID.",
                best_itinerary=None,
                minimum_working_budget_hr=None,
                csp_patterns_generated=[],
                pattern_evaluations=[],
                csp_stats=CSPStats(),
                total_search_stats=total_search_stats,
            )

        if start_poi == end_poi:
            return PipelineOutput(
                status="no_solution",
                message="No solution: Start POI and End POI must be distinct places.",
                best_itinerary=None,
                minimum_working_budget_hr=None,
                csp_patterns_generated=[],
                pattern_evaluations=[],
                csp_stats=CSPStats(),
                total_search_stats=total_search_stats,
            )

        if num_stops < 2 or num_stops > 2 * len(self.dataset.categories):
            return PipelineOutput(
                status="no_solution",
                message=(
                    f"No solution: Requested {num_stops} stops violates category cardinality "
                    f"rules (at most 2 stops per category across {len(self.dataset.categories)} categories)."
                ),
                best_itinerary=None,
                minimum_working_budget_hr=None,
                csp_patterns_generated=[],
                pattern_evaluations=[],
                csp_stats=CSPStats(),
                total_search_stats=total_search_stats,
            )

        if must_see_category is not None and must_see_category not in self.dataset.categories:
            return PipelineOutput(
                status="no_solution",
                message=f"No solution: Must-see category '{must_see_category}' does not exist in dataset.",
                best_itinerary=None,
                minimum_working_budget_hr=None,
                csp_patterns_generated=[],
                pattern_evaluations=[],
                csp_stats=CSPStats(),
                total_search_stats=total_search_stats,
            )

        if time_budget_hr <= 0.0:
            return PipelineOutput(
                status="no_solution",
                message=f"No solution: Time budget ({time_budget_hr:.2f}h) must be strictly positive.",
                best_itinerary=None,
                minimum_working_budget_hr=None,
                csp_patterns_generated=[],
                pattern_evaluations=[],
                csp_stats=CSPStats(),
                total_search_stats=total_search_stats,
            )

        start_cat = self.dataset.poi_categories[start_poi]
        end_cat = self.dataset.poi_categories[end_poi]

        # Step 1: Run CSP with user's budget
        patterns, csp_stats = self.csp.solve(
            num_stops=num_stops,
            start_category=start_cat,
            end_category=end_cat,
            must_see_category=must_see_category,
            budget_hr=time_budget_hr,
            use_fc=use_fc,
            use_ac3=use_ac3,
            max_solutions=max_patterns,
        )

        search_solver = self._build_search_solver(cost_mode=cost_mode)
        pattern_evals: list[dict] = []
        accepted: list[ItineraryResult] = []
        rejected_over_budget: list[ItineraryResult] = []

        # Step 2 & 3: Search for each pattern and check predicted duration against budget
        for pat in patterns:
            res = search_solver.solve(
                pattern=pat,
                start_poi=start_poi,
                end_poi=end_poi,
                start_hour=start_hour,
                algorithm=search_algo,
            )
            if res is None:
                continue
            total_search_stats.nodes_expanded += res.stats.nodes_expanded
            total_search_stats.nodes_generated += res.stats.nodes_generated

            fits = res.total_time_hr <= time_budget_hr + 1e-9
            pattern_evals.append({
                "pattern": list(pat),
                "expected_csp_time_hr": round(self.csp.expected_pattern_time(pat), 3),
                "predicted_nn_time_hr": round(res.total_time_hr, 3),
                "total_cost": round(res.total_cost, 4),
                "nodes_expanded": res.stats.nodes_expanded,
                "accepted": fits,
                "pois": [self.dataset.poi_names[p] for p in res.pois],
            })
            if fits:
                accepted.append(res)
            else:
                rejected_over_budget.append(res)

        if accepted:
            best = min(accepted, key=lambda r: r.total_cost)
            min_budget = min(r.total_time_hr for r in accepted)
            return PipelineOutput(
                status="ok",
                message=(
                    f"Found optimal plausible itinerary within {time_budget_hr:.2f}h budget "
                    f"(predicted duration: {best.total_time_hr:.2f}h, cost: {best.total_cost:.3f})."
                ),
                best_itinerary=best,
                minimum_working_budget_hr=round(min_budget, 2),
                csp_patterns_generated=patterns,
                pattern_evaluations=pattern_evals,
                csp_stats=csp_stats,
                total_search_stats=total_search_stats,
            )

        # Step 4: Fallback if no pattern fits budget — check if unbudgeted patterns exist
        relaxed_patterns, relaxed_csp_stats = self.csp.solve(
            num_stops=num_stops,
            start_category=start_cat,
            end_category=end_cat,
            must_see_category=must_see_category,
            budget_hr=100.0,  # Unconstrained time budget to find minimum working budget
            use_fc=use_fc,
            use_ac3=use_ac3,
            max_solutions=max_patterns,
        )
        if not patterns:
            csp_stats = relaxed_csp_stats

        for pat in relaxed_patterns:
            if pat in patterns:
                continue
            res = search_solver.solve(
                pattern=pat,
                start_poi=start_poi,
                end_poi=end_poi,
                start_hour=start_hour,
                algorithm=search_algo,
            )
            if res is None:
                continue
            total_search_stats.nodes_expanded += res.stats.nodes_expanded
            total_search_stats.nodes_generated += res.stats.nodes_generated
            rejected_over_budget.append(res)
            pattern_evals.append({
                "pattern": list(pat),
                "expected_csp_time_hr": round(self.csp.expected_pattern_time(pat), 3),
                "predicted_nn_time_hr": round(res.total_time_hr, 3),
                "total_cost": round(res.total_cost, 4),
                "nodes_expanded": res.stats.nodes_expanded,
                "accepted": False,
                "pois": [self.dataset.poi_names[p] for p in res.pois],
            })

        if rejected_over_budget:
            cheapest_time_itin = min(rejected_over_budget, key=lambda r: (r.total_time_hr, r.total_cost))
            min_working_budget = round(cheapest_time_itin.total_time_hr, 2)
            return PipelineOutput(
                status="budget_warning",
                message=(
                    f"No solution fits within {time_budget_hr:.2f}h budget. "
                    f"Smallest budget that works is {min_working_budget:.2f}h; "
                    f"returning cheapest itinerary ({cheapest_time_itin.total_time_hr:.2f}h) with warning."
                ),
                best_itinerary=cheapest_time_itin,
                minimum_working_budget_hr=min_working_budget,
                csp_patterns_generated=relaxed_patterns,
                pattern_evaluations=pattern_evals,
                csp_stats=csp_stats,
                total_search_stats=total_search_stats,
            )

        return PipelineOutput(
            status="no_solution",
            message=(
                f"No solution exists for {num_stops} stops from {self.dataset.poi_names[start_poi]} "
                f"({start_cat}) to {self.dataset.poi_names[end_poi]} ({end_cat}) "
                f"with must-see '{must_see_category}'."
            ),
            best_itinerary=None,
            minimum_working_budget_hr=None,
            csp_patterns_generated=[],
            pattern_evaluations=[],
            csp_stats=csp_stats,
            total_search_stats=total_search_stats,
        )
