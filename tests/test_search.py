"""Search tests: known answer on a hand-made world, brute force on the real model, counters."""
import numpy as np
import pandas as pd
import pytest

from tourplan import data as D
from tourplan import nn
from tourplan import search as S


class FixedBundle:
    """Hand-made model: explicit -log p matrix, constant 10-minute stays."""
    def __init__(self, logp, P):
        self._lp = np.repeat(logp[None], 4, axis=0)
        self._st = np.full((4, P), 10.0)

    def logp_table(self):
        return self._lp

    def dur_table(self):
        return self._st


def tiny_world():
    """4 places on a line: S(0)=cat0, A(1),B(2)=cat1, E(3)=cat2.  Known optimum S->A->E."""
    pois = pd.DataFrame(dict(poi_idx=range(4), category=["x", "y", "y", "z"], cat_idx=[0, 1, 1, 2]))
    cd = D.CityData("toy", pois, pd.DataFrame(), ["x", "y", "z"], np.zeros((4, 4)))
    cd.travel_min = np.array([[0, 5, 5, 20], [5, 0, 5, 5], [5, 5, 0, 5], [20, 5, 5, 0]], float)
    lp = np.log(np.array([[0.0001, 0.7, 0.2, 0.1], [0.1, 0.0001, 0.2, 0.7],
                          [0.1, 0.2, 0.0001, 0.7], [0.3, 0.3, 0.3, 0.1]]))
    np.fill_diagonal(lp, -np.inf)
    return cd, FixedBundle(lp, 4)


@pytest.mark.parametrize("algo", ["ucs", "astar", "greedy"])
def test_known_answer_tiny(algo):
    cd, b = tiny_world()
    pb = S.SearchProblem(cd, b, 0, 3, (0, 1, 2), start_hour=9, budget=None, lam=0.0)
    it, st = S.search(pb, algo)
    assert it.places == [0, 1, 3]                      # picks A (p=0.7) not B (p=0.2)
    assert it.cost == pytest.approx(-np.log(0.7) - np.log(0.7))
    assert st.expanded >= 2 and st.generated >= st.expanded


def test_budget_prunes_to_none():
    cd, b = tiny_world()
    pb = S.SearchProblem(cd, b, 0, 3, (0, 1, 2), 9, budget=40.0, lam=0.0)     # needs 30 stay + 10 travel = 40
    assert S.search(pb, "astar")[0] is not None
    pb = S.SearchProblem(cd, b, 0, 3, (0, 1, 2), 9, budget=39.0, lam=0.0)
    it, st = S.search(pb, "astar")
    assert it is None and st.pruned_budget > 0


def test_astar_ucs_match_brute_force_real_model(city, bundle, parks):
    cats = city.categories
    ci = cats.index
    patterns = [(ci("Park"), ci("Museum"), ci("Historical"), ci("Park")),
                (ci("Park"), ci("Structure"), ci("Museum"), ci("Cultural"), ci("Park")),
                (ci("Park"), ci("Museum"), ci("Structure"), ci("Historical"), ci("Museum"), ci("Park"))]
    for mode in ("nll", "time"):
        for pat in patterns:
            pb = S.SearchProblem(city, bundle, parks[0], parks[1], pat, 10.0, 600.0, mode=mode)
            bf = S.brute_force(pb)
            for algo in ("ucs", "astar"):
                it, _ = S.search(pb, algo)
                assert it.cost == pytest.approx(bf[0], abs=1e-9), (mode, algo)
            g, _ = S.search(pb, "greedy")
            assert g.cost >= bf[0] - 1e-9


def test_astar_expands_no_more_than_ucs_and_path_is_valid(city, bundle, parks):
    ci = city.categories.index
    pat = (ci("Park"), ci("Structure"), ci("Museum"), ci("Structure"), ci("Historical"), ci("Park"))
    pb = S.SearchProblem(city, bundle, parks[0], parks[1], pat, 10.0, 600.0)
    ia, sa = S.search(pb, "astar")
    iu, su = S.search(pb, "ucs")
    assert sa.expanded <= su.expanded and ia.cost == pytest.approx(iu.cost)
    mid = ia.places[1:-1]
    assert len(set(mid)) == len(mid) and parks[0] not in mid and parks[1] not in mid
    assert [int(city.cat_of[p]) for p in ia.places] == list(pat)
    assert ia.total_min == pytest.approx(sum(ia.stay) + ia.travel_min)


def test_heuristic_admissible(city, bundle, parks):
    ci = city.categories.index
    pat = (ci("Park"), ci("Museum"), ci("Structure"), ci("Park"))
    pb = S.SearchProblem(city, bundle, parks[0], parks[1], pat, 10.0, 600.0)
    opt, _ = S.search(pb, "ucs")
    assert pb.h_of(0, parks[0]) <= opt.cost + 1e-9
