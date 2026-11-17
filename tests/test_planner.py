"""End-to-end planner tests: 5 normal, 3 edge, 2 impossible cases."""
import numpy as np
import pytest

from tourplan import integration as I


def req(parks, **kw):
    d = dict(start=parks[0], end=parks[1], budget_min=300, must_cat="Museum", n_stops=6, start_hour=10.0)
    d.update(kw)
    return I.Request(**d)


def check_valid(cd, r, pl):
    it = pl.itinerary
    assert it is not None and len(it.places) == r.n_stops
    assert it.places[0] == r.start and it.places[-1] == r.end
    cats = [cd.categories[cd.cat_of[p]] for p in it.places]
    assert r.must_cat in cats
    assert all(a != b for a, b in zip(cats, cats[1:]))
    assert max(cats.count(c) for c in set(cats)) <= 2
    mid = it.places[1:-1]
    assert len(set(mid)) == len(mid)
    assert all(a < b for a, b in zip(it.arrive, it.arrive[1:]))


# ------------------------------------------------------------------- 5 normal -----
@pytest.mark.parametrize("n,must,budget", [(4, "Museum", 240), (5, "Historical", 300), (6, "Cultural", 300),
                                           (7, "Museum", 420), (8, "Structure", 480)])
def test_normal(city, bundle, prior, parks, n, must, budget):
    r = req(parks, n_stops=n, must_cat=must, budget_min=budget)
    pl = I.plan(city, bundle, r, prior)
    assert pl.status == "ok", pl.message
    check_valid(city, r, pl)
    assert pl.itinerary.total_min <= budget + 1e-9


# --------------------------------------------------------------------- 3 edge ------
def test_edge_minimum_three_stops(city, bundle, prior, parks):
    r = req(parks, n_stops=3, must_cat="Museum")
    pl = I.plan(city, bundle, r, prior)
    assert pl.status == "ok"
    check_valid(city, r, pl)
    assert city.cat_of[pl.itinerary.places[1]] == city.categories.index("Museum")   # only free slot


def test_edge_start_equals_end(city, bundle, prior, parks):
    r = req(parks, end=parks[0], n_stops=5)
    pl = I.plan(city, bundle, r, prior)
    assert pl.status == "ok" and pl.itinerary.places[0] == pl.itinerary.places[-1]


def test_edge_budget_just_below_minimum_warns_and_returns_cheapest(city, bundle, prior, parks):
    r0 = req(parks, budget_min=1000)
    pl0 = I.plan(city, bundle, r0, prior)
    lo = I.plan(city, bundle, req(parks, budget_min=10), prior)                    # certainly too small
    assert lo.status == "over_budget" and lo.min_budget is not None and "smallest budget" in lo.message
    assert lo.itinerary.total_min == pytest.approx(lo.min_budget)
    # the reported minimum budget really is the threshold
    ok = I.plan(city, bundle, req(parks, budget_min=lo.min_budget), prior)
    assert ok.status == "ok" and ok.itinerary.total_min <= lo.min_budget + 1e-9
    below = I.plan(city, bundle, req(parks, budget_min=lo.min_budget - 1), prior)
    assert below.status == "over_budget"
    assert pl0.status == "ok"


# ----------------------------------------------------------------- 2 impossible ----
def test_impossible_unknown_must_see(city, bundle, prior, parks):
    pl = I.plan(city, bundle, req(parks, must_cat="Castle"), prior)
    assert pl.status == "no_solution" and pl.itinerary is None
    assert pl.message.startswith("No solution") and "Castle" in pl.message


def test_impossible_too_many_stops(city, bundle, prior, parks):
    pl = I.plan(city, bundle, req(parks, n_stops=14), prior)
    assert pl.status == "no_solution" and pl.message.startswith("No solution") and "impossible" in pl.message


def test_two_stops_same_category_impossible(city, bundle, prior, parks):
    pl = I.plan(city, bundle, req(parks, n_stops=2, must_cat="Park"), prior)
    assert pl.status == "no_solution" and "neighbouring stops must differ" in pl.message
