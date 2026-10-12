# MONOLITHCLE (package `lens`). Settle-measurement harness (overnight 2026-08-14).
"""Delivered-physics settle measurement of a shipped MJCF.

Successor to the 5 s `settle_delivered.py` harness used for the 08-12/08-13
numbers (not shipped). What the old harness missed, per the held-out campaign
(docs/evidence/heldout-campaign-2026-08-13/findings.md §3):

- 5 s window: lens_526 diverges at 9.45 s, the 534 lamps slide off after 5 s
  -> default windows now reach 30 s.
- translation-only drift: a toppling lamp moves its COM barely centimeters
  -> per-body tilt_deg at every window.
- stepping blind through NaN: measurements taken after a divergence are
  garbage -> the guarded loop stops the clock at the first
  MuJoCo warning and every later window is marked invalid.

Loads the MJCF exactly as shipped -- no configure_physics, no mass/damping
rewrite. Depends on mujoco + numpy only. Read-only: writes nothing.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

WINDOWS = (1.5, 3.0, 5.0, 10.0, 30.0)
ESCAPE_M = 5.0


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def measure_scene(xml: str | Path, windows: tuple[float, ...] = WINDOWS,
                  escape_m: float = ESCAPE_M) -> dict:
    """Measure per-body drift/tilt at each window of a guarded settle.

    Returns a JSON-ready dict. `measurements` maps str(window) ->
    {"valid": bool, "bodies": {name: {"drift_cm", "tilt_deg"}}}; windows at or
    after a divergence are invalid and carry no body values.
    """
    m = mujoco.MjModel.from_xml_path(str(Path(xml).resolve()))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bodyname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
    names = {b: bodyname(b) for b in range(1, m.nbody)}

    is_free = {}
    for b in range(1, m.nbody):
        jnt = m.body_jntadr[b]
        is_free[b] = jnt != -1 and m.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE

    xipos0 = d.xipos.copy()
    xquat0 = d.xquat.copy()
    windows = tuple(sorted(windows))

    def _snapshot() -> dict:
        return {names[b]: {
            "drift_cm": round(float(np.linalg.norm(d.xipos[b] - xipos0[b])) * 100.0, 2),
            "tilt_deg": round(_quat_angle_deg(d.xquat[b], xquat0[b]), 1),
        } for b in range(1, m.nbody)}

    measurements: dict[str, dict] = {}
    diverged = None
    for w in windows:
        while d.time < w and diverged is None:
            mujoco.mj_step(m, d)
            if d.warning.number.sum() > 0:
                wi = int(np.argmax(d.warning.number))
                dof = int(d.warning[wi].lastinfo)
                body = bodyname(int(m.dof_bodyid[dof])) if 0 <= dof < m.nv else "?"
                diverged = {"time_s": round(float(d.time), 3),
                            "warning": mujoco.mjtWarning(wi).name,
                            "dof": dof, "body": body}
        if diverged is None:
            measurements[str(w)] = {"valid": True, "bodies": _snapshot()}
        else:
            measurements[str(w)] = {"valid": False, "bodies": None}

    # Escape census over the LAST VALID window: a body this far from spawn is
    # gone, not "drifting" — the flag says the drift number means GONE.
    escaped: dict[str, float] = {}
    for w in reversed(windows):
        entry = measurements[str(w)]
        if entry["valid"]:
            for name, vals in entry["bodies"].items():
                if vals["drift_cm"] / 100.0 > escape_m:
                    escaped[name] = round(vals["drift_cm"] / 100.0, 2)
            break

    return {
        "xml": str(Path(xml).resolve()),
        "timestep": float(m.opt.timestep),
        "nbody": int(m.nbody),
        "free_bodies": sorted(names[b] for b in is_free if is_free[b]),
        "static_bodies": sorted(names[b] for b in is_free if not is_free[b]),
        "masses_kg": {names[b]: round(float(m.body_mass[b]), 4) for b in range(1, m.nbody)},
        "damping_nonzero": bool(np.any(m.dof_damping != 0)),
        "gravity": [float(g) for g in m.opt.gravity],
        "windows_s": list(windows),
        "diverged": diverged,
        "escaped": escaped,
        "measurements": measurements,
    }
