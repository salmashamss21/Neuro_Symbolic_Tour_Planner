"""Trajectory evaluation metrics: Point-F1, Pairs-F1, and Category-F1.

Implements both:
- Standard Tour Point-F1 (Lim et al., IJCAI 2015; Chen et al., CIKM 2016)
- Interior-Only Point-F1 (excluding known start and end query POIs)
- Ordered Pairs-F1 (Chen et al., CIKM 2016) capturing sequential visit order.
"""

from __future__ import annotations

from typing import Sequence


def compute_point_f1(pred_pois: Sequence[int], true_pois: Sequence[int], exclude_endpoints: bool = False) -> dict[str, float]:
    """Computes Precision, Recall, and F1 on visited POIs."""
    if exclude_endpoints:
        p_set = set(pred_pois[1:-1])
        t_set = set(true_pois[1:-1])
    else:
        p_set = set(pred_pois)
        t_set = set(true_pois)

    if not p_set and not t_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not p_set or not t_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    inter = len(p_set & t_set)
    prec = inter / len(p_set)
    rec = inter / len(t_set)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def compute_pairs_f1(pred_pois: Sequence[int], true_pois: Sequence[int]) -> float:
    """Computes Pairs-F1 (Chen et al., CIKM 2016) over all ordered POI pairs."""
    n_p = len(pred_pois)
    n_t = len(true_pois)
    if n_p < 2 or n_t < 2:
        return 0.0

    pred_pairs = set()
    for i in range(n_p):
        for j in range(i + 1, n_p):
            pred_pairs.add((pred_pois[i], pred_pois[j]))

    true_pairs = set()
    for i in range(n_t):
        for j in range(i + 1, n_t):
            true_pairs.add((true_pois[i], true_pois[j]))

    nc = len(pred_pairs & true_pairs)
    if nc == 0:
        return 0.0
    p_pair = nc / len(pred_pairs)
    r_pair = nc / len(true_pairs)
    return 2.0 * p_pair * r_pair / (p_pair + r_pair)
