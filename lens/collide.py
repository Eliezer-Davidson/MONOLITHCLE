# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Exact intersection test for two convex hulls.

HISTORY. This replaces an earlier GJK-style routine (`lens/gjk.py`, never
committed) that answered by walking a simplex toward the origin with an
iteration cap. What follows uses a different method entirely: a linear
feasibility problem solved with scipy's `linprog` (numpy and scipy are both
BSD-3-Clause).

METHOD. Two convex hulls intersect exactly when some point is a convex
combination of BOTH vertex sets, i.e. when

    A^T u == B^T v,   sum(u) == sum(v) == 1,   u >= 0, v >= 0

has a solution. That is a linear feasibility problem, which `linprog` decides
exactly (up to LP tolerance) with no iteration cap and no possibility of
cycling. Touching hulls count as intersecting, matching the closed-set
convention MuJoCo and the CEM's penetration count both assume.

WHY NOT GJK. The code this replaces answered by walking a simplex toward the
origin and bailing out after 64 iterations, returning "no overlap" when it ran
out. On real scene hulls it cycled -- not converged slowly, cycled -- on 12% of
pairs, and so reported no overlap for 31 of 111 genuine overlaps. Raising the
cap to 1024 changed nothing except making it 9x slower. Since the CEM calls
this to price objects buried inside one another, a missed overlap is a free
pass for precisely the failure being optimized away, so a slower exact test is
worth more here than a fast one that is wrong in that direction.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def _aabb_disjoint(a: np.ndarray, b: np.ndarray) -> bool:
    """True when the axis-aligned bounding boxes do not overlap.

    Exact in one direction only, which is the direction used: separated boxes
    imply separated hulls, so this can rule an intersection OUT but never in.
    """
    return bool(np.any(a.max(axis=0) < b.min(axis=0)) or
                np.any(b.max(axis=0) < a.min(axis=0)))


def _axis_separates(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> bool:
    """True when the hulls' projections onto `axis` do not overlap.

    A single separating direction is a certificate of disjointness (separating
    hyperplane theorem), so a hit here is conclusive. A miss proves nothing --
    the hulls may still be disjoint along some direction not tested.
    """
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return False
    axis = axis / n
    pa, pb = a @ axis, b @ axis
    return bool(pa.max() < pb.min() or pb.max() < pa.min())


def _intersect_lp(a: np.ndarray, b: np.ndarray) -> bool:
    """The exact test, with no prefilter. Kept separate so tests can check that
    the prefilter never changes an answer."""
    na, nb = len(a), len(b)
    a_eq = np.zeros((4, na + nb))
    a_eq[:3, :na] = a.T
    a_eq[:3, na:] = -b.T
    a_eq[3, :na] = 1.0                                        # sum(u) == 1
    b_eq = np.array([0.0, 0.0, 0.0, 1.0])
    # sum(v) == 1 as its own row. Both normalisations are required: without
    # them u = v = 0 satisfies A^T u == B^T v and every pair reads as a hit.
    a_eq = np.vstack([a_eq, np.concatenate([np.zeros(na), np.ones(nb)])])
    b_eq = np.append(b_eq, 1.0)
    res = linprog(np.zeros(na + nb), A_eq=a_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    return bool(res.success)


def hulls_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    """Do the convex hulls of point sets `a` and `b` share any point?

    Touching counts as intersecting. Cheap exact rejections are tried first --
    bounding boxes, then the coordinate axes and the centroid-difference
    direction -- because most pairs the CEM scores are well separated and the
    LP is ~30 ms on the ~10^3-vertex hulls of a room scan.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return False

    if _aabb_disjoint(a, b):
        return False
    for axis in (*np.eye(3), b.mean(axis=0) - a.mean(axis=0)):
        if _axis_separates(a, b, np.asarray(axis, dtype=np.float64)):
            return False

    return _intersect_lp(a, b)
