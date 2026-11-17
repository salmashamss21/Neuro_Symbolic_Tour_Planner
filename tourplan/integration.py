"""CSP -> Search -> duration check pipeline (the full planner)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import csp as C
from . import search as S


@dataclass
class Request:
    start: int                       # poi_idx of the start place
    end: int                         # poi_idx of the end place
    budget_min: float                # time budget in minutes
    must_cat: str | None             # must-see category name
    n_stops: int                     # total stops incl. start and end
    start_hour: float = 9.0          # clock hour at which the tour starts


@dataclass
class Plan:
    status: str                      # "ok" | "over_budget" | "no_solution"
    message: str
    itinerary: S.Itinerary | None = None
    pattern: tuple | None = None
    min_budget: float | None = None  # smallest budget (min) for which a plan exists
    patterns_tried: list = field(default_factory=list)   # [(pattern, cost or None, reason)]
    csp_stats: dict = field(default_factory=dict)
    search_stats: dict = field(default_factory=dict)
    seconds: float = 0.0


def validate(cd, req: Request) -> str | None:
    P = cd.n_places
    if not (0 <= req.start < P and 0 <= req.end < P):
        return "start/end place is not in the place set"
    if req.n_stops < 2:
        return "at least 2 stops (start and end) are needed"
    if req.must_cat is not None and req.must_cat not in cd.categories:
        return f"must-see category '{req.must_cat}' does not exist (categories: {', '.join(cd.categories)})"
    if req.budget_min <= 0:
        return "budget must be positive"
    return None


def pattern_names(cd, pattern):
    return [cd.categories[c] for c in pattern]


def plan(cd, bundle, req: Request, prior=None, algo: str = "astar", fc: bool = True, ac3: bool = True,
         max_patterns: int = 10, lam: float = 0.05, enum_cap: int = 50_000) -> Plan:
    t0 = time.time()
    err = validate(cd, req)
    if err:
        return Plan("no_solution", f"No solution: {err}.", seconds=time.time() - t0)

    # 1) CSP without the budget rule: does ANY valid pattern exist at all?
    p_free = C.build_csp(cd, bundle, req.start, req.end, req.n_stops, req.must_cat, None, prior=prior)
    all_pat, st_free = C.solve(p_free, fc=fc, ac3=ac3, max_solutions=enum_cap, order_values=True)
    if not all_pat:
        why = _explain_empty(cd, req)
        return Plan("no_solution", f"No solution: {why}.", csp_stats=st_free.as_dict(),
                    seconds=time.time() - t0)

    # 2) CSP with the budget rule: up to `max_patterns` best patterns
    p = C.build_csp(cd, bundle, req.start, req.end, req.n_stops, req.must_cat, req.budget_min, prior=prior)
    pats, st = C.solve(p, fc=fc, ac3=ac3, max_solutions=enum_cap, order_values=True)
    pats = C.rank_patterns(pats, prior, max_patterns)

    tried, best, best_pat = [], None, None
    sstats = S.SearchStats()
    for pat in pats:
        pb = S.SearchProblem(cd, bundle, req.start, req.end, pat, req.start_hour, req.budget_min, lam)
        it, ss = S.search(pb, algo)
        for k in ("expanded", "generated", "pruned_budget"):
            setattr(sstats, k, getattr(sstats, k) + getattr(ss, k))
        sstats.max_open = max(sstats.max_open, ss.max_open)
        if it is None:
            tried.append((pat, None, "rejected: predicted durations exceed the budget"))
            continue
        tried.append((pat, it.cost, "ok"))
        if best is None or it.cost < best.cost:
            best, best_pat = it, pat
    if best is not None:
        return Plan("ok", f"Itinerary found ({len(tried)} pattern(s) tried).", best, best_pat,
                    None, tried, st.as_dict(), sstats.as_dict(), time.time() - t0)

    # 3) none of the top patterns fits.  Exact smallest budget over ALL valid patterns first.
    min_budget, cheapest, cheapest_pat = _min_time_plan(cd, bundle, req, all_pat, p_free, lam)
    if min_budget is not None and min_budget <= req.budget_min + 1e-9:
        # a feasible plan exists, just not among the top patterns: widen the pattern list
        rest_all, _ = C.solve(p, fc=fc, ac3=ac3, max_solutions=enum_cap, order_values=True)
        rest = [q for q in C.rank_patterns(rest_all, prior, len(rest_all)) if q not in set(pats)][:300]
        for pat in rest:
            pb = S.SearchProblem(cd, bundle, req.start, req.end, pat, req.start_hour, req.budget_min, lam)
            it, ss = S.search(pb, algo)
            sstats.expanded += ss.expanded
            sstats.generated += ss.generated
            if it is None:
                tried.append((pat, None, "rejected: predicted durations exceed the budget"))
                continue
            tried.append((pat, it.cost, "ok"))
            if best is None or it.cost < best.cost:
                best, best_pat = it, pat
            break
        if best is None:                                  # fall back to the shortest pattern
            pb = S.SearchProblem(cd, bundle, req.start, req.end, cheapest_pat, req.start_hour,
                                 req.budget_min, lam)
            best, _ = S.search(pb, algo)
            best_pat = cheapest_pat
        return Plan("ok", f"Itinerary found after widening to {len(tried)} pattern(s) "
                          f"(the 10 most typical patterns did not fit the budget).", best, best_pat,
                    None, tried, st.as_dict(), sstats.as_dict(), time.time() - t0)

    msg = (f"Warning: no itinerary fits {req.budget_min:.0f} min. The smallest budget that works is "
           f"{min_budget:.0f} min; the cheapest (shortest) itinerary is returned and exceeds the budget "
           f"by {min_budget - req.budget_min:.0f} min.")
    return Plan("over_budget", msg, cheapest, cheapest_pat, min_budget, tried,
                st.as_dict(), sstats.as_dict(), time.time() - t0)


def _min_time_plan(cd, bundle, req, patterns, p_free, lam):
    """Exact shortest total time over all valid patterns (branch and bound on a pattern lower bound)."""
    def lb(pat):
        return (sum(p_free.stay_lb[c] for c in pat)
                + sum(p_free.trav_lb[a, b] for a, b in zip(pat[:-1], pat[1:])))
    best, best_pat = None, None
    for pat in sorted(patterns, key=lambda q: (lb(q), q)):
        if best is not None and lb(pat) >= best.total_min - 1e-9:
            break
        pb = S.SearchProblem(cd, bundle, req.start, req.end, pat, req.start_hour, None, lam, mode="time")
        it, _ = S.search(pb, "astar")
        if it is not None and (best is None or it.total_min < best.total_min):
            best, best_pat = it, pat
    return (float(best.total_min) if best else None), best, best_pat


def _explain_empty(cd, req: Request) -> str:
    cats = cd.cat_of
    must = None if req.must_cat is None else cd.categories.index(req.must_cat)
    if req.n_stops == 2 and cats[req.start] == cats[req.end]:
        return "start and end are in the same category but neighbouring stops must differ"
    if must is not None and req.n_stops == 2 and must not in (cats[req.start], cats[req.end]):
        return "with only two stops there is no room for the must-see category"
    max_stops = sum(min(2, int((cats == c).sum())) for c in range(len(cd.categories)))
    if req.n_stops > max_stops:
        return (f"{req.n_stops} stops are impossible: with at most 2 stops per category this city "
                f"allows at most {max_stops} distinct stops")
    return "no category pattern satisfies all rules (neighbours differ, must-see present, max 2 per category)"


def format_plan(cd, plan_: Plan) -> str:
    lines = [plan_.message]
    if plan_.itinerary is None:
        return "\n".join(lines)
    it = plan_.itinerary
    lines.append(f"pattern: {' -> '.join(pattern_names(cd, plan_.pattern))}")
    for k, (pl, ar, sy) in enumerate(zip(it.places, it.arrive, it.stay)):
        h, m = divmod(int(round(ar)), 60)
        lines.append(f"  {k + 1}. {h % 24:02d}:{m:02d}  place {int(cd.pois.poi_id[pl]):>2} "
                     f"[{cd.pois.category[pl]}]  stay {sy:>3} min")
    lines.append(f"total {it.total_min:.0f} min (walking {it.travel_min:.0f} min)")
    return "\n".join(lines)
