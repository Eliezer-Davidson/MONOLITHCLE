# MONOLITHCLE (package `lens`). Root-vs-root de-penetration pass (overnight 2026-08-14).
"""Minimal-displacement resolution of root-vs-root interpenetration.

The CEM plausibility gate freezes all roots and only optimizes children on
surfaces, so furniture-vs-furniture overlap ships unexamined (held-out
campaign findings §4: chair-desk 114 mm in scene 526). This pass runs
after layout, before the ship-gate: iteratively translate the LIGHTER body of
the deepest interpenetrating root pair along that contact's normal in small
steps, re-checking contacts each step, until every root-vs-root contact
satisfies `dist >= -epsilon`.

Deliberately NOT "move to free space": the photo legitimately has the chair
TUCKED under the desk. Only interpenetration is removed — contacts at or
above -epsilon are left exactly where the layout put them.

Purely kinematic (writes qpos, never steps physics), deterministic, and
mujoco + numpy only.
"""
from __future__ import annotations

import mujoco
import numpy as np

EPSILON_M = 0.002        # residual penetration target (2 mm)
STEP_M = 0.005           # translation per iteration (5 mm)
MAX_DISP_PER_BODY_M = 0.30
MAX_ITERS = 2000
PATIENCE = 6             # non-improving selections of one pair before the
                         # lateral fallback kicks in (see below)


def _is_free_root(m: mujoco.MjModel, b: int) -> bool:
    jnt = m.body_jntadr[b]
    return jnt != -1 and m.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE


def _body_center(m: mujoco.MjModel, d: mujoco.MjData, b: int) -> np.ndarray:
    """Volume-weighted world center of a body's collision-geom AABBs.

    NOT d.xpos: the pipeline bakes world position into mesh vertices and
    leaves every body frame at the origin, so xpos says nothing about where
    the geometry is. geom_aabb (local center + half-sizes) mapped through the
    geom frame does."""
    num = np.zeros(3)
    den = 0.0
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != b:
            continue
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        aabb = m.geom_aabb[g]
        center_w = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ aabb[:3]
        vol = float(np.prod(np.maximum(aabb[3:6], 1e-9)))
        num += vol * center_w
        den += vol
    return num / den if den > 0 else d.xpos[b].copy()


def _world_penetration(m: mujoco.MjModel, d: mujoco.MjData, b: int) -> float:
    """Deepest penetration of body `b` into the world/floor, in metres.

    The pair scan skips world contacts (the floor is not a root), so without
    this the pass will happily resolve an object-object overlap by driving the
    lighter body straight down into the ground — measured on real scenes at
    spawn, up to 45 mm, past the ship-gate's own 20 mm FAIL threshold.
    """
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if 0 not in (b1, b2) or b not in (b1, b2):
            continue
        worst = max(worst, -min(0.0, float(c.dist)))
    return worst


def resolve_penetrations(m: mujoco.MjModel, d: mujoco.MjData,
                         epsilon: float = EPSILON_M, step: float = STEP_M,
                         max_disp_per_body: float = MAX_DISP_PER_BODY_M,
                         max_iters: int = MAX_ITERS) -> dict:
    """Resolve root-vs-root penetrations in place (mutates d.qpos).

    Returns {"resolved", "displacements_m": {body: xyz np.ndarray},
    "iterations", "worst_residual_mm", and "reason" when unresolved}.
    World/floor contacts and non-free bodies are never touched.
    """
    bodyname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
    disp: dict[str, np.ndarray] = {}
    # Per-pair progress bookkeeping. A pair whose deepest contact stops
    # improving under normal-stepping (the 526 chair-desk mode: near-vertical
    # normals flip sign every step, clearing one contact deepens another)
    # switches permanently to LATERAL escape: step the mover away from the
    # other body's geometry center in the xy-plane. Lateral keeps the object
    # grounded and matches what furniture separation physically means; a pair
    # that cannot make progress either way is marked stuck and skipped.
    best_depth: dict[tuple, float] = {}
    stall: dict[tuple, int] = {}
    lateral: set[tuple] = set()
    stuck: set[tuple] = set()

    def _report(resolved: bool, iters: int, reason: str | None = None) -> dict:
        mujoco.mj_forward(m, d)
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 == 0 or b2 == 0:
                continue
            if _is_free_root(m, b1) and _is_free_root(m, b2):
                worst = max(worst, -min(0.0, float(c.dist)))
        out = {"resolved": resolved, "iterations": iters,
               "displacements_m": disp,
               "worst_residual_mm": round(worst * 1000, 1)}
        if stuck:
            out["stuck_pairs"] = sorted(
                f"{bodyname(a)} | {bodyname(b)}" for a, b in stuck)
        if reason:
            out["reason"] = reason
        return out

    for it in range(max_iters):
        mujoco.mj_forward(m, d)
        worst = None
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= -epsilon:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 == 0 or b2 == 0:
                continue                      # floor/world is not a root pair
            if not (_is_free_root(m, b1) and _is_free_root(m, b2)):
                continue
            if tuple(sorted((b1, b2))) in stuck:
                continue
            if worst is None or c.dist < worst[0].dist:
                worst = (c, b1, b2)
        if worst is None:
            return _report(not stuck, it,
                           "unresolvable pairs remain" if stuck else None)

        c, b1, b2 = worst
        pair = tuple(sorted((b1, b2)))
        depth = -float(c.dist)
        # Patience applies to NORMAL mode only. Lateral stepping strictly
        # grows the pair's center distance, so it terminates on its own
        # (resolution or the displacement cap) — but its contact depth stays
        # CONSTANT until the exit is near (box MTV stays on the trapped axis),
        # so depth-stall is meaningless there and would kill valid escapes.
        if pair not in lateral:
            if depth < best_depth.get(pair, np.inf) - 1e-4:
                best_depth[pair] = depth
                stall[pair] = 0
            else:
                stall[pair] = stall.get(pair, 0) + 1
                if stall[pair] >= PATIENCE:
                    lateral.add(pair)

        # Lighter body moves first; when its cap is exhausted the OTHER body
        # takes over for the remainder (526's chair-desk fusion overflows a
        # single body's budget). Stuck only when both are out of budget.
        n = np.array(c.frame[0:3], dtype=float)               # geom1 -> geom2
        if m.body_mass[b2] <= m.body_mass[b1]:
            candidates = [(b2, b1, n), (b1, b2, -n)]
        else:
            candidates = [(b1, b2, -n), (b2, b1, n)]

        stepped = False
        for mover, other, direction in candidates:
            if pair in lateral:
                away = _body_center(m, d, mover) - _body_center(m, d, other)
                away[2] = 0.0
                norm = float(np.linalg.norm(away))
                if norm < 1e-8:
                    continue                  # coincident centers: no lateral escape
                direction = away / norm
            name = bodyname(mover)
            cur = disp.setdefault(name, np.zeros(3))
            if float(np.linalg.norm(cur + direction * step)) > max_disp_per_body:
                continue                      # this body is out of budget
            adr = m.jnt_qposadr[m.body_jntadr[mover]]
            before_world = _world_penetration(m, d, mover)
            d.qpos[adr:adr + 3] += direction * step
            mujoco.mj_forward(m, d)
            if _world_penetration(m, d, mover) > max(before_world, epsilon):
                d.qpos[adr:adr + 3] -= direction * step   # would sink it
                continue
            cur += direction * step
            stepped = True
            break
        if not stepped:
            stuck.add(pair)

    return _report(False, max_iters, "max iterations reached")
