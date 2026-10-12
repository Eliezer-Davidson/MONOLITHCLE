# MONOLITHCLE (package `lens`). Scene ship-gate (overnight 2026-08-14).
"""Ship-gate for exported MJCF scenes.

`verify_scene(xml)` runs a guarded settle (the forensic_probe loop: autoreset
disabled, stop at the FIRST MuJoCo warning — stepping through NaN can hang for
minutes) and returns a verdict on whether the scene is fit to ship:

- FAIL: divergence (any mjWARN), NaN, any body far from spawn (escape), or
  residual interpenetration above threshold at settle. Overlap must never ship
  silently again (held-out campaign findings §4, option A).
- WARN: topple (tilt above threshold) or late jitter above threshold.
- PASS: none of the above.

Deliberately depends on mujoco + numpy only, so it runs in any env and never
drags the reconstruction stack into a verification step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

PEN_THRESHOLD_MM = 20.0
ESCAPE_M = 5.0
TOPPLE_DEG = 30.0
JITTER_THRESHOLD = 1.0            # mean sum|qvel| over the final settle second


@dataclass
class Verdict:
    verdict: str                      # "PASS" | "WARN" | "FAIL"
    reasons: list[str] = field(default_factory=list)
    worst_penetration_mm: float = 0.0
    residual_penetrations_mm: dict[str, float] = field(default_factory=dict)
    escapes: dict[str, float] = field(default_factory=dict)
    topples: dict[str, float] = field(default_factory=dict)
    late_jitter: float | None = None  # None if the settle ended early
    settled_s: float = 0.0
    diverged: dict | None = None


def _body_namer(m: mujoco.MjModel):
    return lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def verify_scene(xml: str | Path, seconds: float = 30.0,
                 pen_threshold_mm: float = PEN_THRESHOLD_MM,
                 escape_m: float = ESCAPE_M,
                 topple_deg: float = TOPPLE_DEG,
                 jitter_threshold: float = JITTER_THRESHOLD) -> Verdict:
    """Settle the scene at `xml` for up to `seconds` and return a ship verdict."""
    m = mujoco.MjModel.from_xml_path(str(Path(xml).resolve()))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    d = mujoco.MjData(m)
    bodyname = _body_namer(m)
    mujoco.mj_forward(m, d)

    spawn_pos = d.xpos[1:].copy()     # all non-world bodies
    spawn_quat = d.xquat[1:].copy()

    def _escapes() -> dict[str, float]:
        dist = np.linalg.norm(d.xpos[1:] - spawn_pos, axis=1)
        return {bodyname(1 + i): round(float(dist[i]), 2)
                for i in np.flatnonzero(dist > escape_m)}

    # Guarded settle (forensic_probe pattern): keep a periodic finite snapshot,
    # stop at the first warning, name the diverging body via m.dof_bodyid.
    # An escaped body also ends the settle early — it can only get further away,
    # and the verdict is already decided.
    snap = (d.qpos.copy(), d.qvel.copy(), d.time)
    diverged = None
    escapes: dict[str, float] = {}
    qvel_tail: list[float] = []
    while d.time < seconds:
        step_no = int(d.time / m.opt.timestep)
        if step_no % 250 == 0 and not np.isnan(d.qpos).any():
            snap = (d.qpos.copy(), d.qvel.copy(), d.time)
        mujoco.mj_step(m, d)
        if d.time > seconds - 1.0:
            qvel_tail.append(float(np.abs(d.qvel).sum()))
        if d.warning.number.sum() > 0:
            w = int(np.argmax(d.warning.number))
            dof = int(d.warning[w].lastinfo)
            body = bodyname(int(m.dof_bodyid[dof])) if 0 <= dof < m.nv else "?"
            diverged = {"time_s": round(float(d.time), 3),
                        "warning": mujoco.mjtWarning(w).name,
                        "dof": dof, "body": body}
            break
        if step_no % 500 == 0:
            escapes = _escapes()
            if escapes:
                break

    # Measure the settled state — or, on divergence, the last finite snapshot.
    if diverged:
        d.qpos[:], d.qvel[:], _ = snap
        settled_s = float(snap[2])
    else:
        settled_s = float(d.time)
    mujoco.mj_forward(m, d)

    # Residual interpenetration census at settle (contacts with dist < 0).
    pens: dict[tuple, float] = {}
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0:
            continue
        pair = tuple(sorted((bodyname(int(m.geom_bodyid[c.geom1])),
                             bodyname(int(m.geom_bodyid[c.geom2])))))
        pens[pair] = min(pens.get(pair, 0.0), float(c.dist))
    residual = {" | ".join(k): round(-v * 1000, 1)
                for k, v in sorted(pens.items(), key=lambda kv: kv[1])}
    worst_mm = max(residual.values(), default=0.0)

    if not diverged:
        escapes = _escapes()

    # Topple census: rotation at settle vs spawn, per body.
    topples = {}
    for i in range(1, m.nbody):
        tilt = _quat_angle_deg(d.xquat[i], spawn_quat[i - 1])
        if tilt > topple_deg:
            topples[bodyname(i)] = round(tilt, 1)

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    if diverged:
        fail_reasons.append(f"divergence: {diverged['warning']} at t={diverged['time_s']}s "
                            f"on body '{diverged['body']}' (dof {diverged['dof']})")
    for body, dist in escapes.items():
        fail_reasons.append(f"escape: body '{body}' is {dist:.2f} m from spawn (> {escape_m:.0f} m)")
    for pair, mm in residual.items():
        if mm > pen_threshold_mm:
            fail_reasons.append(f"residual penetration {mm:.1f} mm > {pen_threshold_mm:.0f} mm: {pair}")
    for body, tilt in topples.items():
        warn_reasons.append(f"topple: body '{body}' tilted {tilt:.1f}° (> {topple_deg:.0f}°) vs spawn")
    late_jitter = round(float(np.mean(qvel_tail)), 4) if qvel_tail and not diverged else None
    if late_jitter is not None and late_jitter > jitter_threshold:
        warn_reasons.append(f"late jitter {late_jitter:.3f} sum|qvel| > {jitter_threshold} "
                            f"over the final second")

    verdict = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    return Verdict(verdict=verdict, reasons=fail_reasons + warn_reasons,
                   worst_penetration_mm=worst_mm,
                   residual_penetrations_mm=residual, escapes=escapes,
                   topples=topples, late_jitter=late_jitter,
                   settled_s=round(settled_s, 2), diverged=diverged)
