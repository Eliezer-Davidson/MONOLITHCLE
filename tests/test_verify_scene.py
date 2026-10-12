# MONOLITHCLE (package `lens`). Scene ship-gate tests (overnight 2026-08-14).
"""Tests for lens.verify_scene — the ship-gate that decides whether an exported
MJCF is fit to ship. All fixtures are tiny synthetic MJCF strings: primitives
only, no mesh assets, deterministic outcomes.
"""
from pathlib import Path

from lens.verify_scene import verify_scene


# A free box wedged between two static walls with a 40 mm interference fit per
# side. The walls are static, so physics can never resolve the overlap: the two
# opposing horizontal contact normals cancel and the box just sits there,
# permanently penetrated ~40 mm >> the 20 mm ship threshold. This is the
# deterministic form of the plan's "two-box deep-overlap" fixture — a free box
# spawned overlapping a free box simply gets pushed out during settle, which
# tests nothing.
PEN_FIXTURE = """
<mujoco model="pen_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="wall_left" type="box" pos="-0.13 0 0.2" size="0.02 0.3 0.2"/>
    <geom name="wall_right" type="box" pos="0.13 0 0.2" size="0.02 0.3 0.2"/>
    <body name="squeezed_box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.15 0.1 0.1" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_deep_overlap_fails_on_penetration(tmp_path):
    p = tmp_path / "pen.xml"
    p.write_text(PEN_FIXTURE)
    v = verify_scene(p, seconds=3.0)
    assert v.verdict == "FAIL"
    assert v.worst_penetration_mm > 20.0
    joined = " ".join(v.reasons)
    assert "penetration" in joined
    assert "squeezed_box" in joined


# A free sphere with nothing under it: it free-falls forever, leaving its
# spawn point far behind. (A plane geom is INFINITE for collision whatever its
# `size`, so "spawned past the floor edge" is not expressible with a plane —
# the floor is simply absent here.) The plan's "sphere launched at 100 m/s"
# needs a keyframe qvel, and honoring keyframes would change verify semantics
# for shipped scenes (which have none) — free fall exercises the same escape
# detector without that side effect.
ESCAPE_FIXTURE = """
<mujoco model="escape_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <body name="runaway_ball" pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_runaway_body_fails_on_escape(tmp_path):
    p = tmp_path / "escape.xml"
    p.write_text(ESCAPE_FIXTURE)
    v = verify_scene(p, seconds=10.0)
    assert v.verdict == "FAIL"
    joined = " ".join(v.reasons)
    assert "escape" in joined
    assert "runaway_ball" in joined
    assert v.escapes["runaway_ball"] > 5.0


# A tall thin box spawned tilted 15° about y — past its ~5.7° tipping angle
# (atan(half_width/half_height) = atan(0.02/0.2)) — so it deterministically
# falls flat. Final tilt vs spawn ≈ 75°, far over the 30° topple threshold,
# with no divergence, escape, or deep penetration: verdict must be WARN.
TOPPLE_FIXTURE = """
<mujoco model="topple_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="tippy_box" pos="0 0 0.22" quat="0.991445 0 0.130526 0">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.2" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_toppled_body_warns(tmp_path):
    p = tmp_path / "topple.xml"
    p.write_text(TOPPLE_FIXTURE)
    v = verify_scene(p, seconds=5.0)
    assert v.verdict == "WARN"
    joined = " ".join(v.reasons)
    assert "topple" in joined
    assert "tippy_box" in joined
    assert v.topples["tippy_box"] > 30.0


# A small box resting centered on a big box resting on the floor: nothing
# penetrates, escapes, topples, or jitters. The one shape that must sail
# through the gate — guards against overzealous FAIL/WARN logic.
STACK_FIXTURE = """
<mujoco model="stack_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base_box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.1" density="500"/>
    </body>
    <body name="top_box" pos="0 0 0.25">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_stable_stack_passes_and_reports_low_jitter(tmp_path):
    p = tmp_path / "stack.xml"
    p.write_text(STACK_FIXTURE)
    v = verify_scene(p, seconds=5.0)
    assert v.verdict == "PASS"
    assert v.reasons == []
    # late jitter (mean sum|qvel| over the final second) must be measured and tiny
    assert v.late_jitter is not None
    assert v.late_jitter < 1.0


def test_cli_exit_code_zero_only_on_pass(tmp_path):
    """tools/verify_scene.py exits 0 on PASS and nonzero otherwise, and writes
    the verdict JSON when asked — the property pipeline/CI hooks rely on."""
    import json
    import subprocess
    import sys
    cli = str(Path(__file__).resolve().parent.parent / "tools" / "verify_scene.py")

    good = tmp_path / "stack.xml"
    good.write_text(STACK_FIXTURE)
    out_json = tmp_path / "good.json"
    r = subprocess.run([sys.executable, cli, str(good), "--seconds", "3",
                        "--json", str(out_json)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(out_json.read_text())["verdict"] == "PASS"

    bad = tmp_path / "pen.xml"
    bad.write_text(PEN_FIXTURE)
    r = subprocess.run([sys.executable, cli, str(bad), "--seconds", "3"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "FAIL" in r.stdout


def test_cli_usage_error_does_not_collide_with_the_warn_exit_code(tmp_path):
    """A malformed invocation must NOT be reportable as a verdict.

    argparse exits 2 on a usage error, and 2 is the WARN verdict code — so a
    caller that only reads the exit status records a mistyped command as a
    real WARN.  This bit a live gate sweep on 2026-08-17:
    nine gate runs were argparse failures and every one read as WARN.
    Usage errors get EX_USAGE (64), disjoint from every verdict code.
    """
    import subprocess
    import sys
    cli = str(Path(__file__).resolve().parent.parent / "tools" / "verify_scene.py")

    good = tmp_path / "stack.xml"
    good.write_text(STACK_FIXTURE)
    # --json declared but its argument omitted => argparse usage error
    r = subprocess.run([sys.executable, cli, str(good), "--seconds", "3", "--json"],
                       capture_output=True, text=True)
    assert r.returncode == 64, (
        f"usage error exited {r.returncode}; 2 would be indistinguishable "
        f"from WARN. stderr={r.stderr}")
    assert r.returncode not in (0, 1, 2)
