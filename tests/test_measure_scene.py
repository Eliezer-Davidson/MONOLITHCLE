# MONOLITHCLE (package `lens`). Measure-harness tests (overnight 2026-08-14).
"""Tests for lens.measure — the maintained settle-measurement harness that
replaces the frozen `_evidence/.../settle_delivered.py` (kept untouched for
comparability). Fixtures are tiny synthetic MJCF strings, same style as
tests/test_verify_scene.py.
"""
from lens.measure import measure_scene


# A tall thin box tilted past its ~5.7° tipping angle: it falls flat. Its
# CENTER barely translates (the 534 lamp measured "2.2 cm = stable" while
# lying on the floor — held-out campaign findings §3); only tilt_deg exposes the topple.
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


def test_measure_reports_tilt_not_just_drift(tmp_path):
    p = tmp_path / "topple.xml"
    p.write_text(TOPPLE_FIXTURE)
    res = measure_scene(p, windows=(0.5, 3.0))
    assert res["diverged"] is None
    final = res["measurements"]["3.0"]
    assert final["valid"] is True
    body = final["bodies"]["tippy_box"]
    # the topple: rotation is the signal, translation stays modest
    assert body["tilt_deg"] > 30.0
    assert body["drift_cm"] < 40.0


# A stable brick plus a hinge with an explicit spring too stiff for the
# timestep (omega·dt > 2): semi-implicit Euler amplifies the oscillation and
# BADQACC fires at t≈0.22 s — after the first window, before the second
# (stiffness empirically tuned; k=1200 never diverges, k=1600 dies at 0.04 s).
DIVERGE_FIXTURE = """
<mujoco model="diverge_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="brick" pos="1 0 0.1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" density="500"/>
    </body>
    <body name="flail" pos="0 0 0.5">
      <joint name="h" type="hinge" axis="1 0 0" stiffness="1300" springref="0.02" damping="0"/>
      <geom type="capsule" fromto="0 0 0 0 0.3 0" size="0.02" density="100"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_divergence_stops_the_clock_and_invalidates_later_windows(tmp_path):
    """Numbers measured after a divergence are garbage. The
    harness must record the divergence (named body), keep windows measured
    BEFORE it, and mark every later window invalid with no body values."""
    p = tmp_path / "diverge.xml"
    p.write_text(DIVERGE_FIXTURE)
    res = measure_scene(p, windows=(0.1, 1.0, 3.0))
    div = res["diverged"]
    assert div is not None
    assert 0.1 < div["time_s"] < 1.0
    assert div["body"] == "flail"
    assert res["measurements"]["0.1"]["valid"] is True
    assert "brick" in res["measurements"]["0.1"]["bodies"]
    assert res["measurements"]["1.0"] == {"valid": False, "bodies": None}
    assert res["measurements"]["3.0"] == {"valid": False, "bodies": None}


# A free sphere with nothing under it free-falls forever (a plane geom is
# infinite, so there is no "floor edge" to fall off — the floor is absent).
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


def test_escaped_bodies_are_flagged(tmp_path):
    """A body far from spawn at the last valid window gets an escape flag —
    the harness-level form of the gate's escape FAIL (held-out 530's ballistic
    cup measured '6337 cm drift' with no flag saying the number means GONE)."""
    p = tmp_path / "escape.xml"
    p.write_text(ESCAPE_FIXTURE)
    res = measure_scene(p, windows=(0.5, 2.0))
    assert res["diverged"] is None
    assert "runaway_ball" in res["escaped"]
    assert res["escaped"]["runaway_ball"] > 5.0


def test_cli_writes_json_and_prints_table(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    cli = str(Path(__file__).resolve().parent.parent / "tools" / "measure_scene.py")
    p = tmp_path / "topple.xml"
    p.write_text(TOPPLE_FIXTURE)
    out = tmp_path / "m.json"
    r = subprocess.run([sys.executable, cli, str(p), "--label", "fixture",
                        "--out", str(out), "--windows", "0.5,3.0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    data = json.loads(out.read_text())
    assert data["label"] == "fixture"
    assert data["measurements"]["3.0"]["bodies"]["tippy_box"]["tilt_deg"] > 30.0
    assert "tippy_box" in r.stdout and "tilt" in r.stdout
