"""Scene forensics for the held-out campaign analysis (2026-08-14).

Usage: forensic_probe.py <scene.xml> <out.json>

Measures, per shipped MJCF scene:
- spawn penetration census (contacts with dist<0 at t=0)
- geometry: per-body min collision half-thickness + max extent (thin/small classes)
- 10 s settle with a divergence guard: STOPS at the first MuJoCo warning
  (stepping through NaN can hang for minutes), names the diverging body via
  m.dof_bodyid, and reports the last finite snapshot
- topple census: per-body rotation vs spawn (catches fallen-over objects the
  translation-only drift metric misses)
- late-time jitter: mean sum|qvel| over the final second (only if it survived)
Autoreset is disabled at runtime so divergence is observable.
"""
import json
import os
import sys

import mujoco
import numpy as np


def quat_angle_deg(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def main():
    xml, out_path = sys.argv[1], sys.argv[2]
    os.chdir(os.path.dirname(os.path.abspath(xml)))
    m = mujoco.MjModel.from_xml_path(os.path.abspath(xml))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    d = mujoco.MjData(m)
    bodyname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"

    res = {"xml": os.path.abspath(xml), "nbody": int(m.nbody), "ngeom": int(m.ngeom)}

    # geometry census over collision geoms (contype or conaffinity nonzero)
    geo = {}
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        b = bodyname(int(m.geom_bodyid[g]))
        if b == "world":
            continue
        aabb = m.geom_aabb[g][3:6]  # local half-sizes
        e = geo.setdefault(b, {"min_half_thickness_cm": 1e9, "max_extent_cm": 0.0, "n_hulls": 0})
        e["min_half_thickness_cm"] = min(e["min_half_thickness_cm"], float(min(aabb)) * 100)
        e["max_extent_cm"] = max(e["max_extent_cm"], float(max(aabb)) * 2 * 100)
        e["n_hulls"] += 1
    for b, e in geo.items():
        e["mass_kg"] = round(float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)]), 3)
        e["min_half_thickness_cm"] = round(e["min_half_thickness_cm"], 2)
        e["max_extent_cm"] = round(e["max_extent_cm"], 1)
    res["bodies"] = geo
    res["n_thin_bodies"] = sum(1 for e in geo.values() if e["min_half_thickness_cm"] < 1.5)
    res["n_small_bodies"] = sum(1 for e in geo.values() if e["max_extent_cm"] < 12.0)

    # spawn penetration census
    mujoco.mj_forward(m, d)
    pens = {}
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0:
            continue
        pair = tuple(sorted((bodyname(int(m.geom_bodyid[c.geom1])), bodyname(int(m.geom_bodyid[c.geom2])))))
        pens[pair] = min(pens.get(pair, 0.0), float(c.dist))
    res["spawn_penetrations_mm"] = {" | ".join(k): round(-v * 1000, 1) for k, v in
                                    sorted(pens.items(), key=lambda kv: kv[1])}
    res["spawn_worst_pen_mm"] = round(-min(pens.values()) * 1000, 1) if pens else 0.0

    spawn_quat = {b: d.xquat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy() for b in geo}
    spawn_pos = {b: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy() for b in geo}

    # settle with divergence guard
    snap = (d.qpos.copy(), d.qvel.copy(), d.time)
    diverged = None
    qvel_tail = []
    while d.time < 10.0:
        if int(d.time / m.opt.timestep) % 250 == 0 and not np.isnan(d.qpos).any():
            snap = (d.qpos.copy(), d.qvel.copy(), d.time)
        mujoco.mj_step(m, d)
        if d.warning.number.sum() > 0:
            w = int(np.argmax(d.warning.number))
            dof = int(d.warning[w].lastinfo)
            body = bodyname(int(m.dof_bodyid[dof])) if 0 <= dof < m.nv else "?"
            diverged = {"time_s": round(float(d.time), 3),
                        "warning": mujoco.mjtWarning(w).name,
                        "dof": dof, "body": body}
            break
        if d.time > 9.0:
            qvel_tail.append(float(np.abs(d.qvel).sum()))
    res["diverged"] = diverged
    res["late_jitter_sum_qvel"] = round(float(np.mean(qvel_tail)), 4) if qvel_tail else None

    # final state (last finite snapshot if diverged)
    if diverged:
        d.qpos[:], d.qvel[:], _ = snap
        mujoco.mj_forward(m, d)
        res["state_at_s"] = round(float(snap[2]), 2)
    else:
        res["state_at_s"] = 10.0
    final = {}
    for b in geo:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        final[b] = {
            "drift_cm": round(float(np.linalg.norm(d.xpos[i] - spawn_pos[b])) * 100, 1),
            "tilt_deg": round(quat_angle_deg(d.xquat[i], spawn_quat[b]), 1),
        }
    res["final"] = final
    res["n_toppled_gt30deg"] = sum(1 for v in final.values() if v["tilt_deg"] > 30)

    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("nbody", "n_thin_bodies", "n_small_bodies",
                                          "spawn_worst_pen_mm", "diverged", "n_toppled_gt30deg")}))


if __name__ == "__main__":
    main()
