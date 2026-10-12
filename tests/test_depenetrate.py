# MONOLITHCLE (package `lens`). De-penetration pass tests (overnight 2026-08-14).
"""Tests for lens.depenetrate — minimal-displacement resolution of
root-vs-root interpenetration (option B in the held-out campaign findings §4).
Target is
`dist >= -epsilon`, NOT "no contact" and NOT "free space": legitimate
tucking (chair under desk) must survive. Fixtures are synthetic MJCF
strings, same style as tests/test_verify_scene.py.
"""
import mujoco
import numpy as np

from lens.depenetrate import resolve_penetrations


# Two free boxes overlapping 50 mm in x. B is lighter (density 100 vs 1000),
# so B is the one that moves; the minimal separation is ~50 mm along +x.
OVERLAP_FIXTURE = """
<mujoco model="overlap_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <body name="heavy_a" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" density="1000"/>
    </body>
    <body name="light_b" pos="0.15 0 0.1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" density="100"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_overlapping_boxes_separate_by_about_the_overlap_depth(tmp_path):
    m = mujoco.MjModel.from_xml_string(OVERLAP_FIXTURE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    report = resolve_penetrations(m, d)

    assert report["resolved"] is True
    # only the lighter body moved
    assert set(report["displacements_m"]) == {"light_b"}
    disp = np.linalg.norm(report["displacements_m"]["light_b"])
    # ≈ overlap depth (50 mm) plus at most a couple of resolution steps,
    # and emphatically NOT "moved to free space"
    assert 0.045 <= disp <= 0.075
    # residual penetration at the new pose is within epsilon
    mujoco.mj_forward(m, d)
    worst = min((d.contact[i].dist for i in range(d.ncon)), default=0.0)
    assert worst >= -0.0021


# A chair seat legitimately TUCKED under a thin tabletop, but overlapping the
# table LEG by ~40 mm in x (MTV axis). The pass must clear the leg overlap by
# a small -x translation while the seat STAYS under the tabletop — target is
# dist >= -epsilon, not "no contact" and not "free space".
TUCKED_FIXTURE = """
<mujoco model="tucked_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <body name="table" pos="0 0 0">
      <freejoint/>
      <geom name="tabletop" type="box" size="0.3 0.3 0.01" pos="0 0 0.5" density="1000"/>
      <geom name="leg" type="box" size="0.02 0.02 0.25" pos="0.28 0 0.25" density="1000"/>
    </body>
    <body name="chair" pos="0 0 0">
      <freejoint/>
      <geom name="seat" type="box" size="0.12 0.1 0.02" pos="0.2 0 0.3" density="100"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_tucked_chair_clears_leg_but_stays_under_tabletop(tmp_path):
    m = mujoco.MjModel.from_xml_string(TUCKED_FIXTURE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    report = resolve_penetrations(m, d)

    assert report["resolved"] is True
    assert set(report["displacements_m"]) == {"chair"}
    disp = report["displacements_m"]["chair"]
    # cleared along -x by ≈ the 60 mm x-overlap, nothing gratuitous
    assert -0.10 <= disp[0] <= -0.055
    assert abs(disp[1]) < 0.02 and abs(disp[2]) < 0.02
    # the tuck survives: seat (half-size 0.12, geom offset x=0.2) still inside
    # the tabletop's x-footprint (|x| <= 0.3) and still below the top's z
    chair_adr = m.jnt_qposadr[m.body_jntadr[m.body("chair").id]]
    cx = d.qpos[chair_adr]
    assert cx + 0.2 + 0.12 <= 0.30 + 1e-9        # seat max-x under the top
    assert abs(d.qpos[chair_adr + 2]) < 0.02      # no vertical ejection


# Two boxes 100 mm apart: nothing to do, poses must be BIT-identical.
CLEAN_FIXTURE = """
<mujoco model="clean_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <body name="a" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" density="500"/>
    </body>
    <body name="b" pos="0.4 0 0.1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_clean_scene_is_untouched_bit_identical(tmp_path):
    m = mujoco.MjModel.from_xml_string(CLEAN_FIXTURE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qpos_before = d.qpos.copy()

    report = resolve_penetrations(m, d)

    assert report["resolved"] is True
    assert report["displacements_m"] == {}
    assert np.array_equal(d.qpos, qpos_before)


# The 526 failure mode in miniature: a plate nested between the two jaws of a
# heavy C-clamp, overlapping BOTH jaws vertically. The deepest contact's
# normal alternates ±z (clearing one jaw deepens the other), so pure
# normal-stepping oscillates forever; the only resolution is LATERAL escape
# through the clamp's open side (+x). Diagnosed live on lens_526's chair-desk
# pair (normals (−0.02,0.06,−1.0)/(−0.03,−0.07,1.0) flip-flopping ~2000×).
CLAMP_FIXTURE = """
<mujoco model="clamp_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <body name="clamp" pos="0 0 0">
      <freejoint/>
      <geom name="jaw_top" type="box" size="0.2 0.2 0.02" pos="0 0 0.30" density="2000"/>
      <geom name="jaw_bottom" type="box" size="0.2 0.2 0.02" pos="0 0 0.10" density="2000"/>
    </body>
    <body name="plate" pos="0 0 0">
      <freejoint/>
      <geom name="plate_geom" type="box" size="0.05 0.05 0.11" pos="0.05 0 0.20" density="100"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_vertically_trapped_pair_escapes_laterally(tmp_path):
    m = mujoco.MjModel.from_xml_string(CLAMP_FIXTURE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    report = resolve_penetrations(m, d)

    assert report["resolved"] is True
    assert set(report["displacements_m"]) == {"plate"}
    disp = report["displacements_m"]["plate"]
    # escape is lateral (+x through the open side), not vertical thrash
    assert disp[0] > 0.10
    assert abs(disp[2]) < 0.06


def test_cap_on_one_body_hands_remainder_to_the_other(tmp_path):
    """526's chair-desk fusion: the lighter body's displacement cap can be
    smaller than the overlap. The pass must then move the OTHER body for the
    remainder (minimal per-body movement) rather than giving up."""
    m = mujoco.MjModel.from_xml_string(OVERLAP_FIXTURE)   # 50 mm x-overlap
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    report = resolve_penetrations(m, d, max_disp_per_body=0.03)

    assert report["resolved"] is True
    assert set(report["displacements_m"]) == {"light_b", "heavy_a"}
    db = np.linalg.norm(report["displacements_m"]["light_b"])
    da = np.linalg.norm(report["displacements_m"]["heavy_a"])
    assert db <= 0.0301                      # light body stopped at its cap
    assert 0.015 <= da <= 0.0301             # heavy body covered the rest


# A LIGHT body under a HEAVY one, both above an infinite floor.  The pass moves
# the lighter body along the contact normal, away from the heavier — and here
# "away" is straight DOWN.  World contacts are excluded from the pair scan
# (depenetrate.py: "floor/world is not a root pair"), so nothing stops it: the
# object-object overlap is traded for an object-FLOOR overlap.
# Measured on real scenes at spawn (t=0, no simulation), 2026-08-17 held-out
# sweep: lens_527 webcam box -4.5 cm => world penetration 4.8 -> 45.2 mm;
# lens_530 mouse -2.0 cm => 9.4 -> 29.3 mm; lens_526 desk -3.5 cm => 0.1 ->
# 30.8 mm.  All past the ship-gate's own 20 mm FAIL threshold, on a pass that
# is ON by default.
FLOOR_FIXTURE = """
<mujoco model="floor_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="light_lower" pos="0 0 0.10">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" density="100"/>
    </body>
    <body name="heavy_upper" pos="0 0 0.28">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""


def _worst_world_penetration_mm(m, d):
    """Deepest body-vs-world penetration at the current pose, in mm."""
    mujoco.mj_forward(m, d)
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if 0 not in (b1, b2):
            continue
        worst = max(worst, -min(0.0, float(c.dist)))
    return worst * 1000.0


def test_depenetration_never_pushes_a_body_into_the_floor():
    """Resolving a root-root overlap must not create a world-contact overlap.

    A scene that ships an object sunk into the floor is not de-penetrated; its
    penetration has been moved somewhere the pass does not look.
    """
    m = mujoco.MjModel.from_xml_string(FLOOR_FIXTURE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    before = _worst_world_penetration_mm(m, d)
    assert before < 1.0, "fixture must start clear of the floor"

    report = resolve_penetrations(m, d)

    after = _worst_world_penetration_mm(m, d)
    assert report["resolved"] is True
    assert after <= 2.1, (
        f"pass drove a body into the floor: world penetration "
        f"{before:.1f} -> {after:.1f} mm")
