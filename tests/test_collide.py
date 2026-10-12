"""Exact convex-hull intersection test.

Replaces an earlier GJK-style routine (`lens/gjk.py`, never committed);
`lens/collide.py` decides hull intersection by linear-programming feasibility on
numpy + scipy (BSD-3-Clause).

Correctness matters in a specific direction here. The replaced code answered
by iterating a simplex and giving up after 64 steps, and "gave up" was returned
as "no overlap" -- on real scene hulls it cycled on 12% of cases and so MISSED
28% of genuine overlaps (31 of 111). Since the CEM's whole purpose in calling
this is to penalise objects buried inside each other, a missed overlap is a
free pass for exactly the failure being optimized away.
"""
import numpy as np
import pytest

from lens.collide import hulls_intersect


def cube(center, size=1.0):
    h = size / 2.0
    return np.array([[x, y, z] for x in (-h, h) for y in (-h, h)
                     for z in (-h, h)], float) + np.asarray(center, float)


@pytest.mark.parametrize("name,A,B,expected", [
    ("identical",             cube([0, 0, 0]),      cube([0, 0, 0]),           True),
    ("overlapping by half",   cube([0, 0, 0]),      cube([0.5, 0, 0]),         True),
    ("touching face to face", cube([0, 0, 0]),      cube([1.0, 0, 0]),         True),
    ("1 cm apart",            cube([0, 0, 0]),      cube([1.01, 0, 0]),        False),
    ("far apart",             cube([0, 0, 0]),      cube([5, 0, 0]),           False),
    ("touching at a corner",  cube([0, 0, 0]),      cube([1, 1, 1]),           True),
    ("nested",                cube([0, 0, 0], 4.0), cube([0, 0, 0], 0.5),      True),
    ("diagonal near miss",    cube([0, 0, 0]),      cube([1.01, 1.01, 1.01]),  False),
    ("apart in y only",       cube([0, 0, 0]),      cube([0, 1.5, 0]),         False),
    ("apart in z only",       cube([0, 0, 0]),      cube([0, 0, 1.5]),         False),
])
def test_hand_checked_cases(name, A, B, expected):
    assert hulls_intersect(A, B) is expected, name


def test_a_tetrahedron_inside_a_cube():
    """Non-box geometry, and a case where no vertex of either shape lies inside
    the other's vertex set -- vertex-containment shortcuts get this wrong."""
    tet = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]], float)
    assert hulls_intersect(cube([0, 0, 0], 2.0), tet) is True


def test_crossing_slabs_intersect_with_no_vertex_inside_either():
    """Two long thin boxes crossing like an X. They clearly intersect, yet no
    vertex of either lies within the other -- the classic case a naive
    point-in-hull test misses."""
    a = np.array([[x, y, z] for x in (-2.0, 2.0) for y in (-0.1, 0.1)
                  for z in (-0.1, 0.1)])
    b = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-2.0, 2.0)
                  for z in (-0.1, 0.1)])
    assert hulls_intersect(a, b) is True


def test_result_is_symmetric():
    a, b = cube([0, 0, 0]), cube([0.7, 0.2, 0])
    assert hulls_intersect(a, b) == hulls_intersect(b, a)


def test_returns_a_plain_bool_not_a_numpy_scalar():
    """The CEM sums these into a penetration count; a numpy bool_ or a 0-d
    array silently changes the arithmetic downstream."""
    r = hulls_intersect(cube([0, 0, 0]), cube([5, 0, 0]))
    assert type(r) is bool


def test_the_cheap_prefilter_never_changes_the_answer():
    """`hulls_intersect` rejects obviously-separated pairs before paying for the
    LP. That shortcut must be exact: it may only answer False when the hulls are
    genuinely disjoint, never when they touch."""
    from lens import collide
    rng = np.random.default_rng(0)
    for _ in range(300):
        A = rng.normal(size=(10, 3)) * 0.4
        B = rng.normal(size=(10, 3)) * 0.4 + rng.uniform(-1.6, 1.6, 3)
        fast = collide.hulls_intersect(A, B)
        slow = collide._intersect_lp(A, B)          # LP with no prefilter
        assert fast == slow, f"prefilter disagreed: fast={fast} slow={slow}"


def test_terminates_on_the_inputs_that_made_the_old_gjk_cycle():
    """The replaced implementation looped forever on ~12% of real scene-hull
    pairs and returned 'no overlap' by default. Hulls of this shape and scale
    must now be answered, and answered correctly."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(400, 3)) * np.array([0.33, 0.27, 0.37])
    b = rng.normal(size=(400, 3)) * np.array([1.29, 0.90, 0.66])
    assert hulls_intersect(a, b) is True             # concentric: must overlap
    assert hulls_intersect(a, b + np.array([50.0, 0, 0])) is False
