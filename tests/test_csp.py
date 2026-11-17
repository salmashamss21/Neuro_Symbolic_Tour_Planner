"""CSP unit tests: constraints, solver agreement, counters, known tiny answers."""
import itertools

import numpy as np
import pytest

from tourplan import csp as C


def toy(n, cap, must=None, dom=None, budget=None):
    k = len(cap)
    full = frozenset(range(k))
    domains = [frozenset(dom[i]) if dom and dom.get(i) else full for i in range(n)]
    return C.CSPProblem(n, domains, list(cap), must, budget, np.full(k, 10.0), np.full((k, k), 5.0), 5.0, None, k)


def brute(p):
    out = set()
    for a in itertools.product(*[sorted(d) for d in p.domains]):
        if any(x == y for x, y in zip(a, a[1:])):
            continue
        if any(a.count(c) > p.cap[c] for c in range(p.n_cats)):
            continue
        if p.must is not None and p.must not in a:
            continue
        if p.budget is not None:
            tot = sum(p.stay_lb[c] for c in a) + 5.0 * (p.n - 1)
            if tot > p.budget + 1e-9:
                continue
        out.add(a)
    return out


VARIANTS = [dict(), dict(fc=True), dict(fc=True, ac3=True)]


@pytest.mark.parametrize("kw", VARIANTS)
def test_matches_brute_force_small(kw):
    p = toy(5, [2, 2, 2], must=1, dom={0: [0], 4: [2]})
    s, st = C.solve(p, **kw)
    assert set(s) == brute(p) and st.solutions == len(s)


@pytest.mark.parametrize("kw", VARIANTS)
def test_budget_rule(kw):
    p = toy(4, [2, 2, 2], budget=62.0)           # 4*10 + 3*5 = 55 fits, so patterns exist
    assert set(C.solve(p, **kw)[0]) == brute(p) != set()
    p2 = toy(4, [2, 2, 2], budget=50.0)          # lower bound 55 > 50 -> impossible
    assert C.solve(p2, **kw)[0] == []


@pytest.mark.parametrize("kw", VARIANTS)
def test_all_rules_hold(kw, city, bundle, parks):
    p = C.build_csp(city, bundle, parks[0], parks[1], 6, "Museum", 400)
    sols, _ = C.solve(p, **kw)
    assert sols
    must = city.categories.index("Museum")
    for s in sols:
        assert all(a != b for a, b in zip(s, s[1:]))             # neighbours differ
        assert must in s                                         # must-see present
        assert max(np.bincount(s)) <= 2                          # <= 2 per category
        assert s[0] == city.cat_of[parks[0]] and s[-1] == city.cat_of[parks[1]]


def test_three_solvers_agree_and_counters_improve(city, bundle, parks):
    p = C.build_csp(city, bundle, parks[0], parks[1], 8, "Museum", 400)
    r = {n: C.solve(p, **kw) for n, kw in zip(["bt", "fc", "ac3"], VARIANTS)}
    assert set(r["bt"][0]) == set(r["fc"][0]) == set(r["ac3"][0])
    assert r["fc"][1].assignments < r["bt"][1].assignments
    assert r["ac3"][1].backtracks <= r["fc"][1].backtracks < r["bt"][1].backtracks
    assert r["bt"][1].revisions == 0 and r["ac3"][1].revisions > 0


def test_ac3_prunes_singleton_chain():
    # x0={0}, x1={0,1}, x2={1}: AC-3 must reduce x1 to {1}? no: x1 != x0 -> {1}, and x2 != x1 -> conflict
    p = toy(3, [3, 3], dom={0: [0], 1: [0, 1], 2: [1]})
    assert C.solve(p, fc=True, ac3=True)[0] == []
    doms = [frozenset(d) for d in p.domains]
    st = C.CSPStats()
    assert C._ac3(p, doms, st) is False


def test_max_solutions_and_ordering(city, bundle, parks, prior):
    p = C.build_csp(city, bundle, parks[0], parks[1], 6, "Museum", 400, prior=prior)
    s, _ = C.solve(p, fc=True, ac3=True, max_solutions=10, order_values=True)
    assert len(s) == 10
    top = C.rank_patterns(C.solve(p, fc=True)[0], prior, 10)
    assert len(top) == 10 and len(set(top)) == 10


def test_start_equals_end_cap(city, bundle):
    ent = int(city.pois.index[city.pois.category == "Entertainment"][0])   # only one place in category
    p = C.build_csp(city, bundle, ent, ent, 5, "Museum", None)
    s, _ = C.solve(p, fc=True, ac3=True)
    assert s and all(x[0] == x[-1] == city.cat_of[ent] and x.count(x[0]) == 2 for x in s)
