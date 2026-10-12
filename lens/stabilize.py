# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Settle-in-MuJoCo + CEM pose refinement.

Design: single-image reconstruction gets object poses roughly right, but
objects end up slightly floating or interpenetrating. A cross-entropy search
over small 6-DoF pose deltas -- per support group, then globally -- finds the
placement that rests with near-zero penetration and minimal drift when
simulated, and the settled poses are baked back into a new SceneSpec.

Settling happens in MuJoCo specifically because MuJoCo is the engine the
exported scene is meant to be stepped in: a scene that rests stably in one
physics engine can fall over in another, so "stable" has to mean "stable in
the engine the scene ships for."

The cross-entropy method itself is standard stochastic optimization
(textbook). `settle()` is the independent check on the result:
it re-loads the exported MJCF and measures drift and penetration, so a
stabilized scene is verified rather than asserted.
"""
from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import mujoco
import numpy as np
import trimesh

from lens.scenespec import SceneSpec

# ---------------------------------------------------------------------------
# MATH / GEOMETRY UTILS
# ---------------------------------------------------------------------------

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])

def _axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    half = angle / 2.0
    s = math.sin(half)
    return np.array([math.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])

def _quat_angle_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2.0 * math.acos(abs(dot))

# ---------------------------------------------------------------------------
# STABILIZE / CEM
# ---------------------------------------------------------------------------

# Each CEM rollout settles for 1.0 s of simulated time and reads velocity at
# 0.25 s. What has to be held fixed is
# the SIMULATED DURATION, not the step count -- this implementation's MJCF runs
# dt=0.002, where a literal 60 steps is 0.12 s and free fall covers only 7 cm.
COARSE_SETTLE_SECONDS = 0.25
SETTLE_SECONDS = 1.0
FINE_SETTLE_SECONDS = SETTLE_SECONDS
VEL_READ_SECONDS = 0.25

# The whole scene is lifted by `-scene_min + 0.002` before settling, so
# nothing starts interpenetrating the ground plane.
BASE_CLEARANCE = 0.002

# CEM search knobs. Module-level so a test can shrink them: at the shipped
# values one `stabilize_scene` call is ~7680 rollouts x 500 mj_steps per pass.
POP_SIZE = 512
ELITE_FRAC = 0.025
N_ITERS = 15
# Per-DoF sampling sigma, [dx, dy, dz, ax, ay, az] (m, m, m, rad, rad, rad).
# Wide in the ground plane and in yaw, near-rigid vertically and in
# roll/pitch -- vertical placement is physics' job, not the optimizer's.
INIT_STD = np.array([0.05, 0.05, 0.005, 0.005, 0.005, 0.05])


def _budget() -> Tuple[int, int]:
    """CEM population/iteration budget, env-overridable per run
    (LENS_CEM_POP / LENS_CEM_ITERS). The campaign runs used pop 2048; the
    shipped default here is 512 -- the override exists so a campaign can set
    its budget without editing code."""
    import os
    return (int(os.environ.get("LENS_CEM_POP", POP_SIZE)),
            int(os.environ.get("LENS_CEM_ITERS", N_ITERS)))


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 4 << 30


# Interpreter + mujoco + a collision-only scene model; deliberately generous.
_WORKER_RSS_ESTIMATE = 400 << 20


def _worker_cap(cpus: int | None = None, avail_bytes: int | None = None,
                per_worker_bytes: int = _WORKER_RSS_ESTIMATE) -> int:
    """How many CEM worker processes to spawn. NEVER bare cpu_count(): 24
    workers each loading the full visual scene (~3.3 GB RSS) is what OOM'd
    the 30 GB box on 08-05. Spend at most half of currently-available RAM,
    keep 4 cores for the system, floor at 1. LENS_CEM_WORKERS overrides."""
    import multiprocessing
    import os
    env = os.environ.get("LENS_CEM_WORKERS")
    if env:
        return max(1, int(env))
    if cpus is None:
        cpus = multiprocessing.cpu_count()
    if avail_bytes is None:
        avail_bytes = _mem_available_bytes()
    ram_cap = int((avail_bytes // 2) // per_worker_bytes)
    return max(1, min(cpus - 4, ram_cap))


def _scoring_xml(xml_path) -> Path:
    """Collision-only copy of the scene MJCF for CEM worker processes.

    Visual geoms are contype/conaffinity 0 and mass 0 (`mjcf.py`'s `visual`
    default class), so they contribute nothing to dynamics -- but their
    full-res mesh assets are ~255 MB per scene, and every spawned worker paid
    that to load a model it only steps physics on. Drops every
    `class="visual"` geom, then every mesh asset no remaining geom references.
    A derived file, rewritten unconditionally (stale-asset lesson 2026-07-31).
    """
    import xml.etree.ElementTree as ET
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _prune(el):
        for g in list(el.findall("geom")):
            if g.get("class") == "visual":
                el.remove(g)
        for b in el.findall("body"):
            _prune(b)

    wb = root.find("worldbody")
    if wb is not None:
        _prune(wb)

    referenced = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for asset in root.findall("asset"):
        for m in list(asset.findall("mesh")):
            if m.get("name") not in referenced:
                asset.remove(m)

    out = xml_path.with_name(xml_path.stem + "_scoring.xml")
    tree.write(out)
    return out


# The scoring physics gives every body a dimensionless velocity-proportional
# damping coefficient of 0.3: it integrates dv/dt = -0.3 v, a 3.33 s decay for
# every body whatever its mass.
DAMPING = 0.3

# The scoring physics gives every body a uniform mass of 1 kg, one kilogram
# for a sofa and a tumbler alike. It is a stabilization-time
# choice, not a physical claim, and it applies only while the CEM runs;
# `ObjectSpec.mass` keeps its density-derived value for everything downstream.
STABILIZE_MASS = 1.0


def configure_physics(model: mujoco.MjModel) -> None:
    """Put a compiled model on the scoring-physics footing, in place.

    Damping is the part that does not transfer as a number. MuJoCo's
    `dof_damping` is a force (torque) per unit velocity, so the dimensionless
    0.3 becomes `0.3 * mass` on the translational DoFs and
    `0.3 * principal moment` on the rotational ones -- only then does each DoF
    decay with the 1/0.3 s constant. Writing one scalar onto the freejoint
    instead is right for a 1 kg body's translation and ~20x over-damped for its
    rotation, so this has to be written per DoF rather than in the MJCF.
    """
    model.opt.gravity[:] = [0.0, 0.0, -9.8]
    model.opt.impratio = 1.0

    for b in range(model.nbody):
        jnt = model.body_jntadr[b]
        if jnt == -1 or model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        # Uniform mass first, with inertia rescaled by the same factor so the
        # body keeps its shape-correct distribution (a uniform-mass override
        # that leaves the inertia shape alone). Damping is derived from the FINAL
        # mass, so the order matters.
        scale = STABILIZE_MASS / float(model.body_mass[b])
        model.body_mass[b] = STABILIZE_MASS
        model.body_inertia[b] *= scale

        adr = model.jnt_dofadr[jnt]
        model.dof_damping[adr:adr + 3] = DAMPING * model.body_mass[b]
        model.dof_damping[adr + 3:adr + 6] = DAMPING * model.body_inertia[b]

    # mjModel caches quantities derived from the mass matrix -- body_invweight0,
    # dof_invweight0 -- and the constraint solver scales its impedance by them.
    # Rewriting mass without recomputing these leaves the solver tuned for the
    # masses the model was COMPILED with, which here differ by up to 27x.
    mujoco.mj_setConst(model, mujoco.MjData(model))


def _steps_for(model: mujoco.MjModel, seconds: float) -> int:
    """Number of mj_step calls covering `seconds` of simulated time at this
    model's timestep."""
    return max(1, int(round(seconds / model.opt.timestep)))


def _run(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    for _ in range(_steps_for(model, seconds)):
        mujoco.mj_step(model, data)


def _max_penetration(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Max contact penetration depth (m) at the CURRENT qpos. Callers must
    have already called mj_forward (or mj_step) at this qpos so data.contact
    reflects it -- this only reads mujoco's own narrow-phase distances, it
    does not recompute kinematics."""
    pen = 0.0
    for i in range(data.ncon):
        pen = max(pen, -min(0.0, data.contact[i].dist))
    return pen


@dataclass
class SettleReport:
    drift: Dict[str, float]
    max_penetration: float
    qpos: Dict[str, Tuple[np.ndarray, np.ndarray]]


def settle(xml: str | Path, seconds: float = 2.0) -> SettleReport:
    """Load `xml`, run physics for `seconds`, and report what happened to
    every body: how far it drifted from its starting pose, how much
    penetration remains, and its final freejoint (pos, quat).

    This is the objective check on a stabilized scene -- a scene that is
    actually at rest drifts ~0 when re-simulated. It is deliberately
    independent of the CEM: it re-loads the exported MJCF rather than
    trusting any in-memory state or provenance flag."""
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    p0 = data.xipos[1:].copy()

    _run(model, data, seconds)
    # mj_step's contact buffer reflects the qpos it started that final step
    # from (one step stale). Re-forward at the settled qpos -- no time
    # integration, just refreshed contacts/kinematics -- so max_penetration
    # describes the actual settled state, matching drift/qpos below.
    mujoco.mj_forward(model, data)
    pen = _max_penetration(model, data)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(1, model.nbody)]
    drift = {n: float(np.linalg.norm(d)) for n, d in zip(names, data.xipos[1:] - p0)}
    qpos = {}
    for i, n in enumerate(names):
        adr = model.jnt_qposadr[model.body_jntadr[1 + i]]
        qpos[n] = (data.qpos[adr:adr + 3].copy(), data.qpos[adr + 3:adr + 7].copy())
    return SettleReport(drift=drift, max_penetration=pen, qpos=qpos)


def settle_isaac(usd: str | Path, seconds: float = 1.5) -> SettleReport:
    """Settle the exported USD in Isaac Sim instead of MuJoCo. NOT IMPLEMENTED.

    The scene exporter writes USD (`lens.export.usd`) and that path is tested,
    but the Isaac-side settle runner was never written, so there is no second
    engine to cross-check against today. Kept as a named seam because the
    cross-engine gate below is the only honest way to certify a layout: a
    layout that is stable in one engine is not thereby stable in another --
    measured, a scene that passed a stability gate in one simulator went NaN
    inside half a second of MuJoCo.
    """
    raise NotImplementedError(
        "Isaac Sim settle backend is not implemented in this repository; "
        "only the MuJoCo backend (`settle`) exists."
    )


def cross_engine_settle_gate(xml_path: str | Path, usd_path: str | Path,
                             seconds: float = 1.5, threshold: float = 0.05) -> bool:
    """Re-settle in both MuJoCo and Isaac Sim; True only if stable in both.

    Blocked on `settle_isaac`, which is not implemented -- calling this raises.
    """
    mj_report = settle(xml_path, seconds)
    isaac_report = settle_isaac(usd_path, seconds)

    stable = True
    for name, drift in mj_report.drift.items():
        if drift > threshold:
            print(f"MuJoCo unstable: {name} drifted {drift:.3f}m")
            stable = False

    for name, drift in isaac_report.drift.items():
        if drift > threshold:
            print(f"Isaac Sim unstable: {name} drifted {drift:.3f}m")
            stable = False

    return stable


class FaithfulCEM:
    def __init__(self, spec: SceneSpec, model: mujoco.MjModel, scene_dir: Path):
        self.spec = spec
        self.model = model
        self.scene_dir = scene_dir
        
        # Determine fixed vs movable
        self.roots = [o.id for o in spec.objects if not o.provenance.get("parent")]
        self.fixed_set = set()
        self.movable_set = set()
        
        for o in spec.objects:
            p = o.provenance.get("parent")
            r = o.provenance.get("relation")
            if p:
                parent_cat = next((po.category for po in spec.objects if po.id == p), "")
                if r == "hang" or parent_cat in {"wall", "ceiling"}:
                    self.fixed_set.add(o.id)
                else:
                    self.movable_set.add(o.id)
            else:
                if o.category in {"wall", "ceiling"} or o.mass == 0:
                    self.fixed_set.add(o.id)
                else:
                    self.movable_set.add(o.id)

        # Build shallow->deep groups
        children_map = {}
        for o in spec.objects:
            p = o.provenance.get("parent")
            if p:
                children_map.setdefault(p, []).append(o.id)
        
        self.groups = []
        for node_id in [o.id for o in spec.objects]:
            movable_kids = [c for c in children_map.get(node_id, []) if c in self.movable_set]
            if movable_kids:
                self.groups.append([node_id] + movable_kids)
                
        def get_depth(nid):
            d = 0
            curr = nid
            while True:
                parent = next((o.provenance.get("parent") for o in spec.objects if o.id == curr), None)
                if not parent: break
                curr = parent
                d += 1
            return d
        
        self.groups.sort(key=lambda g: get_depth(g[0]))

        # Meshes first: base_dz is derived from their true vertex minima, and
        # each object's AABB centre is its `centroid` -- see orig_center.
        self.hull_verts = {}
        mesh_min_z = {}
        mesh_center = {}
        for o in spec.objects:
            mesh = trimesh.load(scene_dir / "meshes" / f"{o.id}.obj", force="mesh", process=False)
            self.hull_verts[o.id] = mesh.convex_hull.vertices
            amin, amax = mesh.bounds
            mesh_min_z[o.id] = float(amin[2])
            mesh_center[o.id] = (np.asarray(amin) + np.asarray(amax)) * 0.5

        # base_dz = -scene_min_z + 0.002, taken from mesh AABB minima. It must
        # NOT come from `pos[2] - dims[2]/2`:
        # lens.pipeline.assemble_scene always writes pos=(0,0,0) and bakes
        # world position into the mesh vertices, so that formula degenerates
        # to half the tallest object -- +0.43 m on office_lounge, which
        # teleports the entire scene into the air before every rollout.
        floor_z = 0.0
        scene_min_z = min(mesh_min_z.values()) if mesh_min_z else floor_z
        self.base_dz = floor_z - scene_min_z + BASE_CLEARANCE

        # The optimiser carries TWO distinct per-object reference points, and they are
        # NOT interchangeable:
        #
        #   ref_root_pos[name] = [0, base_dy, 0]                -> orig_center
        #   ref_pos[name]      = centroid + [0, base_dy, 0]     -> self.ref_pos
        #        with centroid = (aabb_min + aabb_max) * 0.5
        #
        # `orig_center` is the BODY-FRAME anchor. assemble_scene always writes
        # pos=(0,0,0) and bakes world position into the mesh vertices, so a
        # freejoint's qpos starts at (0,0,base_dz) -- and the layout term of the
        # cost, `norm(settled_qpos[:3] - orig_center)`, measures how far that
        # body frame has moved. Both sides must be body-frame quantities.
        #
        # `ref_pos` is the object's WORLD centre, used only as the pivot a
        # rotation turns about. Using orig_center there made every object pivot
        # about (0, 0, base_dz) -- the world origin -- so with the body frame at
        # the origin and the geometry metres away, a sampled rotation swung
        # objects bodily across the scene: on office_lounge the backpack came
        # out yawed 44 deg with a 1.18 m compensating translation, and the CEM
        # made placement WORSE than not running it (median 3.1 cm -> 13.8 cm).
        #
        # Conflating them the other way is just as bad: pointing orig_center at
        # the centroid makes the cost compare a body-frame position against a
        # world centroid, a ~2 m constant whose gradient drags whole groups
        # sideways (measured: median 38.1 cm, groups translated ~0.4 m in y).
        #
        # A third quantity, `mesh_center`, is `ref_pos` without the base drop:
        # `centroid_ofs`, the centroid in the BODY frame. It is what
        # the reward rotates to find where the geometry actually is
        # (`_world_centroid`).
        self.mesh_center = mesh_center
        self.orig_center = {}
        self.ref_pos = {}
        drop = np.array([0.0, 0.0, self.base_dz])
        for o in spec.objects:
            p = np.array(o.pos, dtype=np.float64)
            if o.id in self.movable_set or o.id in self.roots:
                p[2] += self.base_dz
            self.orig_center[o.id] = p
            self.ref_pos[o.id] = mesh_center[o.id] + drop

def _qpos_adr(model, name: str) -> int:
    try:
        body_id = model.body(name).id
        jnt_id = model.body_jntadr[body_id]
        if jnt_id != -1:
            return model.jnt_qposadr[jnt_id]
        return -1
    except KeyError:
        return -1

def _axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(aa)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    s = np.sin(angle / 2.0)
    c = np.cos(angle / 2.0)
    return np.array([c, axis[0]*s, axis[1]*s, axis[2]*s])

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_angle_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, 0.0, 1.0)
    return 2.0 * np.arccos(dot)

def _world_centroid(pos: np.ndarray, quat: np.ndarray,
                    body_centroid: np.ndarray) -> np.ndarray:
    """`placed_root + R @ centroid_ofs` -- where the
    object's geometry actually sits, as opposed to where its body frame sits.

    This is the quantity both position terms of the reward are measured on, and
    it is what makes a rotation cost anything: `assemble_scene` leaves every
    body frame at the origin with the geometry metres away, so `qpos[:3]` is
    invariant under a pure rotation while the centroid sweeps an arc.
    """
    return pos + _quat_to_rotmat(quat) @ body_centroid


def _place_children(model: mujoco.MjModel, data: mujoco.MjData,
                    children: List[str], action: np.ndarray,
                    orig_center: Dict[str, np.ndarray],
                    orig_quat: Dict[str, np.ndarray]) -> None:
    """Write one CEM sample onto a group's children. The composition:

        placed_pos  = child_ref + delta_pos          (NO pivot)
        placed_quat = ref_quat * delta_quat          (delta in the body frame)

    `action` is a flat `n_children * 6`; child `j` reads slice `6j:6j+6` and
    nothing else, so the optimizer can move one child without touching its
    siblings. Sharing a single 6-vector across the group -- what this replaces
    -- means no sample can ever separate a backpack from a laptop.

    Position is never rotated about a pivot. Rotating the body-frame position
    about the object's centroid swings a body whose frame is at the origin and
    whose geometry is metres away right across the scene.
    """
    for j, c in enumerate(children):
        adr = _qpos_adr(model, c)
        if adr == -1:
            continue
        d = action[j * 6:(j + 1) * 6]
        data.qpos[adr:adr + 3] = orig_center[c] + d[:3]
        data.qpos[adr + 3:adr + 7] = _quat_mul(
            orig_quat[c], _axis_angle_to_quat(np.asarray(d[3:6])))


@contextlib.contextmanager
def _isolate(model: mujoco.MjModel, active: Set[str]):
    """Restrict collision to `active` for the duration of the block.

    Each pass is optimised in a FRESH env holding only that pass's actors:
    `set(group) | set(support_movables)` for a local group,
    `set(roots) | set(singletons)` for the global pass, and the filter gates
    both the fixed and the movable actor lists. On office_lounge
    that is 3 bodies locally and 2 globally, out of 6.

    MuJoCo cannot drop bodies from a compiled model, so absence is approximated
    by clearing contype/conaffinity. Geoms on the worldbody -- the floor -- are
    always kept: everything rests on it in every pass.

    Pair this with `_inactive_pins`. Disabling contact alone would leave the
    excluded bodies falling through the floor under gravity.
    """
    contype = model.geom_contype.copy()
    conaff = model.geom_conaffinity.copy()
    try:
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            if b == 0:                                  # worldbody / floor
                continue
            if model.body(b).name not in active:
                model.geom_contype[g] = 0
                model.geom_conaffinity[g] = 0
        yield
    finally:
        model.geom_contype[:] = contype
        model.geom_conaffinity[:] = conaff


def _inactive_pins(model: mujoco.MjModel, active: Set[str],
                   data: mujoco.MjData) -> list:
    """Pin rows for every free body this pass does not simulate, holding it at
    the pose it currently has. See `_isolate` for why they also stop colliding.

    Derived from the model rather than the SceneSpec so it covers exactly the
    bodies that can move -- a body without a freejoint needs no pin.
    """
    pins = []
    for b in range(model.nbody):
        jnt = model.body_jntadr[b]
        if jnt == -1 or model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        if model.body(b).name in active:
            continue
        adr = model.jnt_qposadr[jnt]
        pins.append((adr, model.jnt_dofadr[jnt],
                     np.array(data.qpos[adr:adr + 7])))
    return pins


def _rider_offsets(model: mujoco.MjModel, data: mujoco.MjData,
                   groups: List[List[str]]) -> Dict[str, list]:
    """Freeze each group's post-local-pass shape into parent-frame offsets.

    Parent-frame child offsets: `ofs = R_root^T @
    (p_child - p_root)` and `q_rel = q_root^-1 * q_child`, captured once the
    local pass has settled the group. The global pass then moves only the root
    and re-derives the children from these, so a group cannot deform.
    """
    out: Dict[str, list] = {}
    for g in groups:
        root = g[0]
        r_adr = _qpos_adr(model, root)
        if r_adr == -1:
            continue
        p_r = np.array(data.qpos[r_adr:r_adr + 3])
        q_r = np.array(data.qpos[r_adr + 3:r_adr + 7])
        R_r = _quat_to_rotmat(q_r)
        rows = []
        for c in g[1:]:
            c_adr = _qpos_adr(model, c)
            if c_adr == -1:
                continue
            p_c = np.array(data.qpos[c_adr:c_adr + 3])
            q_c = np.array(data.qpos[c_adr + 3:c_adr + 7])
            q_inv = np.array([q_r[0], -q_r[1], -q_r[2], -q_r[3]])
            rows.append((c, R_r.T @ (p_c - p_r), _quat_mul(q_inv, q_c)))
        out[root] = rows
    return out


def _place_entities(model: mujoco.MjModel, data: mujoco.MjData,
                    entities: List[str], action: np.ndarray,
                    orig_center: Dict[str, np.ndarray],
                    riders: Dict[str, list]) -> None:
    """Write one global-pass sample: every entity gets its own 6 numbers, and
    each group's children ride along rigidly.

        placed_pos  = ref_root_pos + delta_pos       (NO pivot)
        placed_quat = delta_quat                     (ABSOLUTE, not composed)

    The absolute quaternion is `placed_quat_all =
    axis_angle_to_quat(delta_aa)` with no reference factor, which is correct
    because every group root sat pinned at identity through the local pass.

    Riders are re-derived from the root, never transformed in place
    (`place_all_descendants`): `p_root + R_root @ ofs`, `q_root * q_rel`.
    """
    for e_idx, e in enumerate(entities):
        adr = _qpos_adr(model, e)
        if adr == -1:
            continue
        d = action[e_idx * 6:(e_idx + 1) * 6]
        q_e = _axis_angle_to_quat(np.asarray(d[3:6]))
        p_e = orig_center[e] + d[:3]
        data.qpos[adr:adr + 3] = p_e
        data.qpos[adr + 3:adr + 7] = q_e

        R_e = _quat_to_rotmat(q_e)
        for c, ofs, q_rel in riders.get(e, []):
            c_adr = _qpos_adr(model, c)
            if c_adr == -1:
                continue
            data.qpos[c_adr:c_adr + 3] = p_e + R_e @ ofs
            data.qpos[c_adr + 3:c_adr + 7] = _quat_mul(q_e, q_rel)


def _build_transform(p, q):
    T = np.eye(4)
    T[:3, :3] = _quat_to_rotmat(q)
    T[:3, 3] = p
    return T

LAMBDA_POSE_STAB = 1.0
LAMBDA_ROT_STAB = 1.0
LAMBDA_POSE_LAYOUT = 6.0
LAMBDA_ROT_LAYOUT = 1.0
LAMBDA_VEL = 1.0
LAMBDA_SETTLED_GEO_PEN = 0.5
LAMBDA_PLACE_GEO_PEN = 0.5


def _step_pinned(model: mujoco.MjModel, data: mujoco.MjData, n_steps: int,
                 pins: List[tuple]) -> None:
    """Advance `n_steps`, holding every body in `pins` exactly where it is.

    The group root is frozen for the whole local pass: the
    group root is a static collider that children rest on and push against, and
    it never moves or is scored. MuJoCo has no post-compile freeze, so the pin
    is enforced by rewriting the root's qpos and zeroing its qvel after each
    step, which is what "kinematic" means here.

    Leaving the root free instead lets contact push it around and drags the
    whole group with it -- the office_lounge bench rose 43.8 cm over one 1 s
    settle.
    """
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        for adr, dof, q in pins:
            data.qpos[adr:adr + 7] = q
            data.qvel[dof:dof + 6] = 0.0


def _compute_cost(
    model: mujoco.MjModel,
    group: List[str],
    samples: np.ndarray,
    orig_center: Dict[str, np.ndarray],
    mesh_center: Dict[str, np.ndarray],
    ref_pos: Dict[str, np.ndarray],
    orig_quat: Dict[str, np.ndarray],
    hull_verts: Dict[str, np.ndarray],
    movable_set: Set[str],
    qpos0: np.ndarray = None,
    settle_steps: int | None = None,
    vel_read_step: int | None = None
) -> np.ndarray:
    """Score a population of local-pass actions (the local-pass objective).

    `samples` is `(pop, n_children * 6)` -- child `j` owns columns `6j:6j+6`.
    `group` is `[root, *children]`; the root is pinned throughout and never
    contributes to the cost: the reward loop runs over the children and the
    root appears only in the penetration term.

    Both position terms are measured on WORLD CENTROIDS
    (`root_pos + R @ centroid_ofs`), not on `qpos[:3]`. That is what prices a
    rotation: `assemble_scene` leaves each body frame at the origin with the
    geometry metres away, so a pure yaw leaves `qpos[:3]` untouched while
    sweeping the object through an arc.

    Cost is the reward negated -- this minimizes -- with
    weights: 1 / 1 / 6 / 1 for pose-stab, rot-stab,
    pose-layout, rot-layout, 1 for velocity and 0.5 for each penetration count.
    """
    # Derive from simulated DURATION, not a literal step count: a 1.0 s
    # settle at dt=1/60 is 60 steps, while this model runs dt=0.002, where 60
    # steps is 0.12 s -- too short for anything to reach its support.
    if settle_steps is None:
        settle_steps = _steps_for(model, SETTLE_SECONDS)
    if vel_read_step is None:
        vel_read_step = _steps_for(model, VEL_READ_SECONDS)

    pop_size = samples.shape[0]
    costs = np.zeros(pop_size, dtype=np.float32)

    data = mujoco.MjData(model)

    # Use custom PyTorch GJK for intersection tests
    from lens.collide import hulls_intersect

    root_id = group[0]
    children = group[1:]
    # The penetration term covers the root plus this group's children --
    # not every movable in the scene, which would score objects no sample in
    # this pass can move.
    pen_objs = [o for o in group if o in hull_verts]

    root_adr = _qpos_adr(model, root_id)
    root_dof = model.jnt_dofadr[model.body_jntadr[model.body(root_id).id]] \
        if root_adr != -1 else -1

    # This pass exists only for this group. Everything else stops colliding
    # (`_isolate`) and is held still (`_inactive_pins`), reproducing the fresh,
    # filtered env built per group. Both are derived from
    # qpos0, which every rollout resets to, so they are computed once.
    active = set(group)
    data.qpos[:] = qpos0
    other_pins = _inactive_pins(model, active, data)

    with _isolate(model, active):
        for i in range(pop_size):
            # Full reset, not just qpos: MjData is reused across the population, so
            # qvel / qacc_warmstart / time would otherwise carry each rollout's
            # terminal state into the next and make the objective history-dependent.
            # Velocities are zeroed on every pose write.
            mujoco.mj_resetData(model, data)
            data.qpos[:] = qpos0

            _place_children(model, data, children, samples[i], orig_center, orig_quat)

            mujoco.mj_forward(model, data)
            placed_xpos = data.xpos.copy()
            placed_xquat = data.xquat.copy()

            pins = other_pins + (
                [(root_adr, root_dof, data.qpos[root_adr:root_adr + 7].copy())]
                if root_adr != -1 else [])

            _step_pinned(model, data, vel_read_step, pins)

            linvel = 0.0
            for c in children:
                v_adr = model.jnt_dofadr[model.body_jntadr[model.body(c).id]]
                linvel += np.linalg.norm(data.qvel[v_adr:v_adr + 3])

            _step_pinned(model, data, settle_steps - vel_read_step, pins)

            mujoco.mj_forward(model, data)
            settled_xpos = data.xpos.copy()
            settled_xquat = data.xquat.copy()

            def transform_hull(h, p, q):
                return (h @ _quat_to_rotmat(q).T) + p

            pen_placed = 0
            pen_settled = 0
            hulls_placed = {}
            hulls_settled = {}
            for obj_id in pen_objs:
                b_id = model.body(obj_id).id
                hulls_placed[obj_id] = transform_hull(
                    hull_verts[obj_id], placed_xpos[b_id], placed_xquat[b_id])
                hulls_settled[obj_id] = transform_hull(
                    hull_verts[obj_id], settled_xpos[b_id], settled_xquat[b_id])

            for a in range(len(pen_objs)):
                for b in range(a + 1, len(pen_objs)):
                    o1, o2 = pen_objs[a], pen_objs[b]
                    if hulls_intersect(hulls_placed[o1], hulls_placed[o2]):
                        pen_placed += 1
                    if hulls_intersect(hulls_settled[o1], hulls_settled[o2]):
                        pen_settled += 1

            cost = 0.0
            for c in children:
                b_id = model.body(c).id
                ofs = mesh_center[c]
                placed_w = _world_centroid(placed_xpos[b_id], placed_xquat[b_id], ofs)
                settled_w = _world_centroid(settled_xpos[b_id], settled_xquat[b_id], ofs)
                placed_q = placed_xquat[b_id]
                settled_q = settled_xquat[b_id]

                cost += LAMBDA_POSE_STAB * np.linalg.norm(settled_w - placed_w)
                cost += LAMBDA_ROT_STAB * _quat_angle_distance(settled_q, placed_q)
                cost += LAMBDA_POSE_LAYOUT * np.linalg.norm(settled_w - ref_pos[c])
                cost += LAMBDA_ROT_LAYOUT * _quat_angle_distance(settled_q, orig_quat[c])

            cost += LAMBDA_VEL * linvel
            cost += LAMBDA_SETTLED_GEO_PEN * pen_settled
            cost += LAMBDA_PLACE_GEO_PEN * pen_placed
            costs[i] = cost

    return costs


def _eval_local_chunk(args):
    xml_path, group, samples_chunk, orig_center, mesh_center, ref_pos, orig_quat, hull_verts, movable_set, qpos0, settle_steps, vel_read_step = args
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    configure_physics(model)
    return _compute_cost(
        model, group, samples_chunk, orig_center, mesh_center, ref_pos, orig_quat,
        hull_verts, movable_set, qpos0, settle_steps, vel_read_step
    )


def _compute_cost_global_func(
    model: mujoco.MjModel,
    samples_global: np.ndarray,
    global_entities: List[str],
    orig_center: Dict[str, np.ndarray],
    mesh_center: Dict[str, np.ndarray],
    ref_pos: Dict[str, np.ndarray],
    orig_quat: Dict[str, np.ndarray],
    hull_verts: Dict[str, np.ndarray],
    riders: Dict[str, list],
    qpos0: np.ndarray,
    global_pins: List[tuple],
    pen_objs: List[str],
    settle_steps: int,
    vel_read_step: int
) -> np.ndarray:
    data = mujoco.MjData(model)
    costs = np.zeros(samples_global.shape[0], dtype=np.float32)
    from lens.collide import hulls_intersect
    for i in range(samples_global.shape[0]):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = qpos0

        _place_entities(model, data, global_entities, samples_global[i],
                        orig_center, riders)

        mujoco.mj_forward(model, data)
        placed_xpos = data.xpos.copy()
        placed_xquat = data.xquat.copy()

        _step_pinned(model, data, vel_read_step, global_pins)

        linvel = 0.0
        for e in global_entities:
            v_adr = model.jnt_dofadr[model.body_jntadr[model.body(e).id]]
            linvel += np.linalg.norm(data.qvel[v_adr:v_adr + 3])

        _step_pinned(model, data, settle_steps - vel_read_step, global_pins)

        mujoco.mj_forward(model, data)
        settled_xpos = data.xpos.copy()
        settled_xquat = data.xquat.copy()

        def transform_hull(h, pp, qq):
            return (h @ _quat_to_rotmat(qq).T) + pp

        hulls_placed = {}
        hulls_settled = {}
        for obj_id in pen_objs:
            b_id = model.body(obj_id).id
            hulls_placed[obj_id] = transform_hull(
                hull_verts[obj_id], placed_xpos[b_id], placed_xquat[b_id])
            hulls_settled[obj_id] = transform_hull(
                hull_verts[obj_id], settled_xpos[b_id], settled_xquat[b_id])

        pen_placed = 0
        pen_settled = 0
        for a in range(len(pen_objs)):
            for b in range(a + 1, len(pen_objs)):
                o1, o2 = pen_objs[a], pen_objs[b]
                if hulls_intersect(hulls_placed[o1], hulls_placed[o2]):
                    pen_placed += 1
                if hulls_intersect(hulls_settled[o1], hulls_settled[o2]):
                    pen_settled += 1

        cost = 0.0
        for e in global_entities:
            b_id = model.body(e).id
            ofs = mesh_center[e]
            placed_w = _world_centroid(placed_xpos[b_id], placed_xquat[b_id], ofs)
            settled_w = _world_centroid(settled_xpos[b_id], settled_xquat[b_id], ofs)
            placed_q = placed_xquat[b_id]
            settled_q = settled_xquat[b_id]

            cost += LAMBDA_POSE_STAB * np.linalg.norm(settled_w - placed_w)
            cost += LAMBDA_ROT_STAB * _quat_angle_distance(settled_q, placed_q)
            cost += LAMBDA_POSE_LAYOUT * np.linalg.norm(settled_w - ref_pos[e])
            cost += LAMBDA_ROT_LAYOUT * _quat_angle_distance(settled_q, orig_quat[e])

        cost += LAMBDA_VEL * linvel
        cost += LAMBDA_SETTLED_GEO_PEN * pen_settled
        cost += LAMBDA_PLACE_GEO_PEN * pen_placed
        costs[i] = cost
    return costs


def _eval_global_chunk(args):
    xml_path, samples_chunk, global_entities, orig_center, mesh_center, ref_pos, orig_quat, hull_verts, riders, qpos0, global_pins, pen_objs, settle_steps, vel_read_step = args
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    configure_physics(model)
    active_global = set(global_entities)
    with _isolate(model, active_global):
        return _compute_cost_global_func(
            model, samples_chunk, global_entities, orig_center, mesh_center, ref_pos,
            orig_quat, hull_verts, riders, qpos0, global_pins, pen_objs, settle_steps, vel_read_step
        )

def stabilize_scene(spec: SceneSpec, scene_dir: str | Path, **kwargs) -> SceneSpec:
    from lens.export import mjcf
    xml = mjcf.write_scene(spec, Path(scene_dir))
    model = mujoco.MjModel.from_xml_path(str(xml))
    
    configure_physics(model)

    # Wait until each iteration to calculate settle steps for coarse-to-fine
    _vel_read_steps = _steps_for(model, VEL_READ_SECONDS)
    _final_settle_steps = _steps_for(model, FINE_SETTLE_SECONDS)

    cem = FaithfulCEM(spec, model, Path(scene_dir))
    
    # Setup base position
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # set initial poses
    orig_quat = {}
    for o in spec.objects:
        adr = _qpos_adr(model, o.id)
        if adr != -1:
            # Apply base_dz
            data.qpos[adr+2] += cem.base_dz
            orig_quat[o.id] = data.qpos[adr+3:adr+7].copy()
            
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    
    # Local Pass
    pop, iters = _budget()
    n_elite = max(1, int(pop * ELITE_FRAC))

    # Workers score on the collision-only model: identical dynamics (visual
    # geoms are massless and contactless), ~none of the memory.
    scoring = _scoring_xml(xml)

    rng = np.random.default_rng(42)

    # ------------------ LOCAL PASS ------------------
    # One group at a time, shallow to deep. The root is pinned for the whole
    # pass and only its children are posed and scored; the group's product is
    # the SETTLED pose, which the next group starts from.
    for group in cem.groups:
        root_id = group[0]
        children = group[1:]
        act_dim = 6 * len(children)
        if act_dim == 0:
            continue

        mean = np.zeros(act_dim)
        std = np.tile(INIT_STD, len(children))
        best_action = np.zeros(act_dim)
        best_cost = float('inf')

        for _iter in range(iters):
            # Coarse-to-fine window
            progress = max(0.0, (_iter - iters/2.0) / (iters/2.0))
            current_settle = COARSE_SETTLE_SECONDS + progress * (FINE_SETTLE_SECONDS - COARSE_SETTLE_SECONDS)
            settle_steps = _steps_for(model, current_settle)

            samples = mean + rng.normal(size=(pop, act_dim)) * std
            
            import multiprocessing
            import concurrent.futures
            n_workers = _worker_cap()
            chunk_size = math.ceil(pop / n_workers)
            chunks = [samples[i:i+chunk_size] for i in range(0, pop, chunk_size)]

            args_list = [
                (str(scoring), group, chunk, cem.orig_center, cem.mesh_center, cem.ref_pos, orig_quat,
                 cem.hull_verts, cem.movable_set, initial_qpos, settle_steps, _vel_read_steps)
                for chunk in chunks
            ]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                results = list(executor.map(_eval_local_chunk, args_list))
                
            costs = np.concatenate(results)

            order = np.argsort(costs)
            elites = samples[order[:n_elite]].copy()

            if costs[order[0]] < best_cost:
                best_cost = costs[order[0]]
                best_action = samples[order[0]].copy()

            # `keep_best`: the running best displaces the WORST elite
            # before the fit, so a good action found early cannot be lost to a
            # later iteration's sampling.
            elites[-1] = best_action

            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-5

            if best_cost < 0.01:
                break

        # Carry the SETTLED pose forward, not the pose the action placed.
        # The settled pose is read AFTER the settle
        # (`settled_pos_registry[c] = fin_pos`), so a child that drops onto its
        # support during scoring keeps that drop. Writing the pre-settle pose
        # instead discards the physics the sample was scored on -- and its log
        # shows exactly such a drop, black_laptop_on_sofa_000 settling to
        # root_y = -0.2108.
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        _place_children(model, data, children, best_action,
                        cem.orig_center, orig_quat)
        mujoco.mj_forward(model, data)
        r_adr = _qpos_adr(model, root_id)
        r_dof = model.jnt_dofadr[model.body_jntadr[model.body(root_id).id]]
        active = set(group)
        pins = _inactive_pins(model, active, data) + (
            [(r_adr, r_dof, data.qpos[r_adr:r_adr + 7].copy())]
            if r_adr != -1 else [])
        with _isolate(model, active):
            _step_pinned(model, data, _final_settle_steps, pins)
        mujoco.mj_forward(model, data)

        initial_qpos = data.qpos.copy()
        
    # ------------------ GLOBAL PASS ------------------
    # Entities are the group roots plus the singleton movables, jointly
    # searched over 6 DoF each. A group's children get no action of their own:
    # they ride on their root through the offsets the local pass settled them
    # into, so the group translates and yaws as one rigid body.
    global_entities = cem.roots + [o for o in cem.movable_set
                                   if not any(o in g for g in cem.groups)]
    n_entities = len(global_entities)
    mean_global = np.zeros(6 * n_entities)
    std_global = np.tile(INIT_STD, n_entities)
    best_action_global = np.zeros(6 * n_entities)
    best_cost_global = float('inf')

    from lens.collide import hulls_intersect

    riders = _rider_offsets(model, data, cem.groups)
    # The global pass scores the ENTITIES only, and its penetration term uses
    # `entity_hull_dev` -- the same list. Children are absent from both because
    # no sample can move them independently; including them prices collisions
    # the optimizer has no way to resolve, and double-counts a group's members
    # against each other.
    pen_objs = [e for e in global_entities if e in cem.hull_verts]

    # Only the entities are in this pass's simulation -- the global
    # filter is `roots | singletons`, so a group's children are not
    # physics bodies here at all. They are carried kinematically by
    # `_place_entities` and re-derived from the settled root afterwards.
    active_global = set(global_entities)
    data.qpos[:] = initial_qpos
    global_pins = _inactive_pins(model, active_global, data)

    import multiprocessing
    import concurrent.futures
    n_workers = _worker_cap()

    with _isolate(model, active_global):
        for _iter in range(iters):
            # Coarse-to-fine window
            progress = max(0.0, (_iter - iters/2.0) / (iters/2.0))
            current_settle = COARSE_SETTLE_SECONDS + progress * (FINE_SETTLE_SECONDS - COARSE_SETTLE_SECONDS)
            settle_steps = _steps_for(model, current_settle)

            samples = mean_global + rng.normal(size=(pop, 6 * n_entities)) * std_global
            
            chunk_size = math.ceil(pop / n_workers)
            chunks = [samples[i:i+chunk_size] for i in range(0, pop, chunk_size)]
            
            args_list = [
                (str(scoring), chunk, global_entities, cem.orig_center, cem.mesh_center, cem.ref_pos,
                 orig_quat, cem.hull_verts, riders, initial_qpos, global_pins, pen_objs, settle_steps, _vel_read_steps)
                for chunk in chunks
            ]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                results = list(executor.map(_eval_global_chunk, args_list))
                
            costs = np.concatenate(results)

            order = np.argsort(costs)
            elites = samples[order[:n_elite]].copy()

            if costs[order[0]] < best_cost_global:
                best_cost_global = costs[order[0]]
                best_action_global = samples[order[0]].copy()

            elites[-1] = best_action_global          # keep_best

            mean_global = elites.mean(axis=0)
            std_global = elites.std(axis=0) + 1e-5

            if best_cost_global < 0.01:
                break

    # Apply final best action
    mujoco.mj_resetData(model, data)
    data.qpos[:] = initial_qpos
    _place_entities(model, data, global_entities, best_action_global,
                    cem.orig_center, riders)
    mujoco.mj_forward(model, data)

    # Settle one last time. Must use the SAME window every rollout was scored
    # with: a literal 60 steps here is 0.12 s at dt=0.002, so the winning action
    # -- chosen because it settled well over 1.0 s -- got written out mid-fall,
    # frozen 0.5*9.8*0.12**2 = 7.1 cm below wherever it started.
    with _isolate(model, active_global):
        _step_pinned(model, data, _final_settle_steps, global_pins)
    mujoco.mj_forward(model, data)

    # Children were never in this pass's simulation, so their final pose is
    # re-derived from where their root actually settled:
    # `pos[i] + root_R @ c_ofs` for every
    # child of every entity. Reading the children's own qpos instead would
    # report the pose they were parked at before the root moved under them.
    for e in global_entities:
        adr = _qpos_adr(model, e)
        if adr == -1:
            continue
        q_e = np.array(data.qpos[adr + 3:adr + 7])
        R_e = _quat_to_rotmat(q_e)
        p_e = np.array(data.qpos[adr:adr + 3])
        for c, ofs, q_rel in riders.get(e, []):
            c_adr = _qpos_adr(model, c)
            if c_adr == -1:
                continue
            data.qpos[c_adr:c_adr + 3] = p_e + R_e @ ofs
            data.qpos[c_adr + 3:c_adr + 7] = _quat_mul(q_e, q_rel)
    mujoco.mj_forward(model, data)
    
    import copy
    out = copy.deepcopy(spec)
    for o in out.objects:
        b_id = model.body(o.id).id
        o.pos = tuple(float(v) for v in data.xpos[b_id])
        q = data.xquat[b_id]
        o.quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        o.provenance["stabilize"] = "cem+settle(mujoco)"
        
    return out
