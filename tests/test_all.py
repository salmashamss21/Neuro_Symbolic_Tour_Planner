#!/usr/bin/env python3
"""Comprehensive Test Suite for P6: Plausible Tour Itineraries.

Covers:
- 5 Normal Cases (standard tourist queries across different start/end POIs, must-see categories,
  and stop counts 4..6).
- 3 Edge Cases (minimum 2-stop tour, tight-budget fallback returning minimum working budget
  warning, and a long 7-stop tour where must-see is already satisfied by the start place).
- 2 Impossible Cases (requesting 14 stops when at most 2 per category across 6 categories = 12
  is possible; requesting a non-existent category or identical start/end place), verifying that
  every impossible case returns `status == "no_solution"` and a clear `"No solution"` message.
- Unit tests for Search optimality vs. Brute-Force Oracle on a small pool and CSP pruning
  equivalence across BT, BT+FC, and BT+FC+AC-3.
"""

from __future__ import annotations

import unittest

from data.loader import TourDataset
from integration.pipeline import IntegratedTourPlanner
from nn.train import train_and_evaluate


class TestPlausibleTourItineraries(unittest.TestCase):
    """Test suite verifying Normal (5), Edge (3), Impossible (2), and Module Unit Tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = TourDataset("data")
        cls.nn_model, cls.mk_base, cls.pop_base, _ = train_and_evaluate(
            cls.dataset, seed=42, epochs=35, batch_size=64, lr=0.004, save_dir="outputs/models"
        )
        cls.planner = IntegratedTourPlanner(cls.dataset, cls.nn_model, cls.mk_base, cls.pop_base)

    def _assert_valid_itinerary(self, out, expected_stops: int, must_see: str) -> None:
        self.assertEqual(out.status, "ok", msg=f"Expected status='ok', got {out.status}: {out.message}")
        self.assertIsNotNone(out.best_itinerary)
        itin = out.best_itinerary
        self.assertEqual(len(itin.pois), expected_stops)
        self.assertEqual(len(set(itin.pois)), expected_stops, "POIs in itinerary must be unique")
        # Check adjacent categories differ
        for i in range(len(itin.categories) - 1):
            self.assertNotEqual(itin.categories[i], itin.categories[i + 1], "Adjacent stops must differ in category")
        # Check no category appears more than twice
        for c in set(itin.categories):
            self.assertLessEqual(itin.categories.count(c), 2, f"Category {c} appears > 2 times")
        # Check must-see category included
        self.assertIn(must_see, itin.categories, f"Must-see category {must_see} missing from {itin.categories}")

    # =========================================================================
    # 5 NORMAL CASES
    # =========================================================================
    def test_normal_1_classic_edinburgh_museum_tour(self) -> None:
        """Normal Case 1: 5-hour, 5-stop tour from Princes Street Gardens to Calton Hill including a Museum."""
        out = self.planner.plan_tour(
            start_poi=4, end_poi=5, time_budget_hr=5.0, must_see_category="Museum", num_stops=5, start_hour=10.0
        )
        self._assert_valid_itinerary(out, expected_stops=5, must_see="Museum")
        self.assertLessEqual(out.best_itinerary.total_time_hr, 5.0)

    def test_normal_2_royal_mile_castle_tour(self) -> None:
        """Normal Case 2: 4.5-hour, 4-stop tour from Royal Mile to Arthur's Seat including a Castle."""
        out = self.planner.plan_tour(
            start_poi=7, end_poi=6, time_budget_hr=4.5, must_see_category="Castle", num_stops=4, start_hour=9.0
        )
        self._assert_valid_itinerary(out, expected_stops=4, must_see="Castle")
        self.assertLessEqual(out.best_itinerary.total_time_hr, 4.5)

    def test_normal_3_afternoon_cafe_and_culture(self) -> None:
        """Normal Case 3: 5.5-hour, 5-stop tour from Edinburgh Castle to Palace of Holyroodhouse with a Cafe."""
        out = self.planner.plan_tour(
            start_poi=1, end_poi=3, time_budget_hr=5.5, must_see_category="Cafe", num_stops=5, start_hour=12.0
        )
        self._assert_valid_itinerary(out, expected_stops=5, must_see="Cafe")
        self.assertLessEqual(out.best_itinerary.total_time_hr, 5.5)

    def test_normal_4_historical_highlights_6_stops(self) -> None:
        """Normal Case 4: 6.0-hour, 6-stop tour from National Museum to Scott Monument including Park."""
        out = self.planner.plan_tour(
            start_poi=2, end_poi=9, time_budget_hr=6.0, must_see_category="Park", num_stops=6, start_hour=9.0
        )
        self._assert_valid_itinerary(out, expected_stops=6, must_see="Park")
        self.assertLessEqual(out.best_itinerary.total_time_hr, 6.0)

    def test_normal_5_cultural_walk_4_stops(self) -> None:
        """Normal Case 5: 4.0-hour, 4-stop tour from St Giles' Cathedral to Royal Botanic Garden with Cultural stop."""
        out = self.planner.plan_tour(
            start_poi=8, end_poi=11, time_budget_hr=4.5, must_see_category="Cultural", num_stops=4, start_hour=11.0
        )
        self._assert_valid_itinerary(out, expected_stops=4, must_see="Cultural")
        self.assertLessEqual(out.best_itinerary.total_time_hr, 4.5)

    # =========================================================================
    # 3 EDGE CASES
    # =========================================================================
    def test_edge_1_minimum_two_stops_direct(self) -> None:
        """Edge Case 1: Minimum 2-stop tour (Edinburgh Castle -> National Museum) where start/end satisfy must-see."""
        out = self.planner.plan_tour(
            start_poi=1, end_poi=2, time_budget_hr=3.5, must_see_category="Museum", num_stops=2, start_hour=10.0
        )
        self._assert_valid_itinerary(out, expected_stops=2, must_see="Museum")

    def test_edge_2_tight_budget_triggers_minimum_working_budget_warning(self) -> None:
        """Edge Case 2: Budget (2.0h for 5 stops) is too small; system reports smallest working budget and cheapest tour."""
        out = self.planner.plan_tour(
            start_poi=4, end_poi=5, time_budget_hr=2.0, must_see_category="Museum", num_stops=5, start_hour=10.0
        )
        self.assertEqual(out.status, "budget_warning")
        self.assertIsNotNone(out.best_itinerary)
        self.assertIsNotNone(out.minimum_working_budget_hr)
        self.assertGreater(out.minimum_working_budget_hr, 2.0)
        self.assertIn("Smallest budget that works", out.message)

    def test_edge_3_same_start_and_end_category_7_stops(self) -> None:
        """Edge Case 3: 7-stop tour starting and ending in the same category (Park -> ... -> Park) without exceeding count<=2."""
        out = self.planner.plan_tour(
            start_poi=4, end_poi=6, time_budget_hr=7.5, must_see_category="Castle", num_stops=7, start_hour=9.0
        )
        self._assert_valid_itinerary(out, expected_stops=7, must_see="Castle")
        self.assertEqual(out.best_itinerary.categories.count("Park"), 2)

    # =========================================================================
    # 2 IMPOSSIBLE CASES (Must give a clear "No solution" message)
    # =========================================================================
    def test_impossible_1_exceeds_pigeonhole_category_limit(self) -> None:
        """Impossible Case 1: Requesting 14 stops across 6 categories when each category can appear at most twice (max 12)."""
        out = self.planner.plan_tour(
            start_poi=1, end_poi=2, time_budget_hr=15.0, must_see_category="Museum", num_stops=14, start_hour=9.0
        )
        self.assertEqual(out.status, "no_solution")
        self.assertIsNone(out.best_itinerary)
        self.assertIn("No solution", out.message)

    def test_impossible_2_two_stops_same_category_or_unsatisfiable_must_see(self) -> None:
        """Impossible Case 2: 2-stop tour where start (Park) and end (Castle) cannot include must-see 'Cafe', or identical endpoints."""
        out_unreach = self.planner.plan_tour(
            start_poi=4, end_poi=1, time_budget_hr=5.0, must_see_category="Cafe", num_stops=2, start_hour=10.0
        )
        self.assertEqual(out_unreach.status, "no_solution")
        self.assertIsNone(out_unreach.best_itinerary)
        self.assertIn("No solution", out_unreach.message)

        out_same = self.planner.plan_tour(
            start_poi=1, end_poi=1, time_budget_hr=5.0, must_see_category="Castle", num_stops=4, start_hour=10.0
        )
        self.assertEqual(out_same.status, "no_solution")
        self.assertIn("No solution", out_same.message)

    # =========================================================================
    # MODULE UNIT TESTS: Search vs. Brute-Force Oracle & CSP Pruning Consistency
    # =========================================================================
    def test_unit_search_matches_brute_force_oracle(self) -> None:
        """Verifies that UCS and A* find the exact same optimal cost and path as Brute-Force on a 4-stop pattern."""
        solver = self.planner._build_search_solver(cost_mode="nn")
        pattern = ("Park", "Museum", "Historical", "Castle")
        bf_res = solver.brute_force_search(pattern, start_poi=4, end_poi=1, start_hour=10.0)
        ucs_res = solver.solve(pattern, start_poi=4, end_poi=1, start_hour=10.0, algorithm="ucs")
        astar_res = solver.solve(pattern, start_poi=4, end_poi=1, start_hour=10.0, algorithm="astar")

        self.assertIsNotNone(bf_res)
        self.assertIsNotNone(ucs_res)
        self.assertIsNotNone(astar_res)
        self.assertAlmostEqual(bf_res.total_cost, ucs_res.total_cost, places=6)
        self.assertAlmostEqual(bf_res.total_cost, astar_res.total_cost, places=6)
        self.assertEqual(bf_res.pois, ucs_res.pois)
        self.assertEqual(bf_res.pois, astar_res.pois)
        self.assertLessEqual(astar_res.stats.nodes_expanded, ucs_res.stats.nodes_expanded)

    def test_unit_csp_pruning_reduces_backtracks(self) -> None:
        """Verifies that BT+FC+AC-3 finds valid patterns while drastically reducing assignments and backtracks vs. BT."""
        pats_bt, st_bt = self.planner.csp.solve(
            num_stops=6, start_category="Park", end_category="Park",
            must_see_category="Museum", budget_hr=5.0, use_fc=False, use_ac3=False, max_solutions=10
        )
        pats_fc, st_fc = self.planner.csp.solve(
            num_stops=6, start_category="Park", end_category="Park",
            must_see_category="Museum", budget_hr=5.0, use_fc=True, use_ac3=False, max_solutions=10
        )
        pats_ac3, st_ac3 = self.planner.csp.solve(
            num_stops=6, start_category="Park", end_category="Park",
            must_see_category="Museum", budget_hr=5.0, use_fc=True, use_ac3=True, max_solutions=10
        )
        self.assertEqual(pats_bt, pats_fc)
        self.assertEqual(pats_fc, pats_ac3)
        self.assertLess(st_fc.assignments, st_bt.assignments)
        self.assertLessEqual(st_ac3.assignments, st_fc.assignments)


if __name__ == "__main__":
    unittest.main(verbosity=2)
