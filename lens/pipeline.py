# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""End-to-end orchestration: photo -> physics-valid SceneSpec + sim exports.

Pipeline: capture -> VLM object list (`lens.prompts.LIST_OBJECTS`)
-> scene tree -> masks -> reconstruction -> per-object visual mesh + CoACD
collision hulls (written to `<out>/meshes/`) -> ObjectSpecs (PCA-in-xy dims,
density-based mass, parent provenance) -> stabilize (CEM + MuJoCo settle,
unless disabled) -> MJCF/USD export -> `<out>/scene.json`.

An object the segmenter found no mask for is simply absent from `masks`
(handled in `lens.scenegraph.masks.segment_objects`) and therefore absent
from `recon`'s output too -- the loop below only builds a `ReconObject` for
ids that made it all the way through, so a dropped object never produces a
dangling/duplicate id downstream.

USD export is best-effort: its `pxr` dependency may be missing on some
machines, and a broken export must not take down an otherwise-good run --
the MJCF (written first) plus `scene.json` are the primary artifacts.

The module is split at the reconstruction boundary. `assemble_scene`
is the model-free downstream half (CoACD, ObjectSpec build, stabilize,
export) shared by both call paths:
  - `run_pipeline` -- everything in ONE process, driven by pluggable
    `vlm`/`segmenter`/`recon` backends. Used by tests with Fake* backends;
    a real run here would hold segmentation + reconstruction model memory
    at once, which is exactly what OOM-killed the machine at the stabilize
    stage (see `run_pipeline_isolated` and `lens.recon_worker`).
  - `run_pipeline_isolated` -- the real orchestrator. Spawns `lens.recon_worker`
    as a SUBPROCESS to do the model-heavy half; that process exits before
    `assemble_scene` (and therefore stabilize) ever runs, so the OS actually
    reclaims the model RAM instead of relying on in-process `release()`,
    which frees Python objects but not the process's heap. This module
    NEVER imports torch/sam3/sam3d_objects -- it only shells out to the
    worker module by name.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from lens.capture import load_capture
from lens.export import collision, mjcf
from lens.export import usd as usd_export
from lens.prompts import LIST_OBJECTS
from lens.scenegraph.masks import segment_objects
from lens.scenegraph.tree import build_tree, parse_object_lines
from lens.scenespec import ObjectSpec, SceneSpec
from lens.stabilize import stabilize_scene

log = logging.getLogger(__name__)

DENSITY_DEFAULT = 200.0   # kg/m^3 -- ObjectSpec.mass_source = "default"
FRICTION_DEFAULT = 0.5    # no VLM/measured source for this yet; flat default
MASS_FLOOR = 1e-3         # kg -- M2: a near-flat object (rug/mat/sheet) can
                          # decompose to ~0 CoACD volume, and SceneSpec.validate()
                          # / MuJoCo both reject a non-positive freejoint mass.
CHILD_GAP = 0.005         # m -- a re-seated "on" child sits this far above its
                          # parent's top surface (5 mm).


@dataclass
class ReconObject:
    """One already-reconstructed object: the model-free handoff record
    between the reconstruction half of the pipeline (in-process `recon.
    reconstruct`, or the isolated worker's `recon_state.json`) and
    `assemble_scene`. `mesh` is a posed `trimesh.Trimesh` -- assemble_scene
    never touches a recon/segmenter object, only this."""
    id: str
    category: str
    tags: list[str]
    parent: str | None
    mesh: trimesh.Trimesh
    confidence: float
    axes: np.ndarray | None = None   # (3,3) clean model box axes (columns) from
    #                                  recon, the orientation reference
    #                                  `_gravity_align`'s per-object snap needs.


def _pca_dims(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    """Tape-measure-like (length, depth, height) for `mesh`: extents along
    the mesh's own principal axes in the xy-plane for L/D, plain z-extent
    for H. An axis-aligned bounding box on a rotated object over-reports
    L/D (it measures the rotated diagonal); PCA finds the object's true
    in-plane axes first, so a 45-degree-rotated box still reports its real
    side lengths.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    xy = V[:, :2]
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    proj = centered @ eigvecs[:, order]
    ext0 = float(proj[:, 0].max() - proj[:, 0].min())
    ext1 = float(proj[:, 1].max() - proj[:, 1].min())
    length, depth = max(ext0, ext1), min(ext0, ext1)
    height = float(V[:, 2].max() - V[:, 2].min())
    return (length, depth, height)


def _trimesh_obb_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    """Fallback (3,3) box axes (columns) from trimesh's geometric OBB, used only
    when a record carries no clean model axes -- e.g. Fake* backends or a legacy
    recon_state.json. Noisier than the model axes; see `_obb_axes_from_pose`."""
    T = np.asarray(mesh.bounding_box_oriented.primitive.transform)
    A = T[:3, :3]
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def _obb_up_axis_from_axes(axes: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """The box axis closest to `ref`, sign-aligned to `ref`: the selection
    rule is argmax |axesᵀ·ref|, then flip so it points along +ref."""
    up = axes[:, int(np.argmax(np.abs(axes.T @ ref)))]
    return -up if np.dot(up, ref) < 0 else up


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Proper rotation (det=+1) mapping unit vector `a` onto unit vector `b`
    (Rodrigues); handles the parallel and antiparallel degenerate cases."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)          # 180 deg about `axis`
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _record_axes(r: ReconObject) -> np.ndarray:
    """The record's clean model box axes if present, else a trimesh-OBB fallback."""
    if r.axes is not None:
        return np.asarray(r.axes, dtype=np.float64)
    return _trimesh_obb_axes(r.mesh)


def _gravity_align(records: list[ReconObject]) -> np.ndarray:
    """Gravity-canonicalize the scene BEFORE grounding, targeting +Z (this
    pipeline is z-up; the reconstruction backend delivers y-up).

    Single-image SAM-3D reconstructions land in the CAMERA frame, which is not
    gravity-aligned (~40 deg off vertical on office_lounge). Two passes fix it,
    and BOTH use the model's CLEAN box axes -- not a geometric OBB:

      1. coarse: estimate one scene up-axis (size-weighted, each object's box
         axis nearest +Z, sign-aligned) from the floor/carpet references, or from
         every object when none was reconstructed, and rotate the whole scene so
         it is vertical.  (Can only ever be a small rotation -- every contributing
         axis is in the +Z hemisphere -- so it never flips anything.)
      2. fine: per object, snap its OWN box up-axis exactly onto +Z.

    The fine snap is the load-bearing step -- it is what actually lands each
    object upright, and it was the one we had disabled.  Validated to land every
    object upright on office_lounge (table vertical mass ratio 0.898, sofa
    0.478, ...), with or without a floor reference.  Returns
    the coarse global rotation (identity if no records).
    """
    Z = np.array([0.0, 0.0, 1.0])
    if not records:
        return np.eye(3)

    axes_map = {r.id: _record_axes(r) for r in records}

    refs = [r for r in records
            if "floor" in r.category.lower() or "carpet" in r.category.lower()
            or any("floor" in t.lower() or "carpet" in t.lower() for t in r.tags)]
    basis = refs if refs else records

    # --- coarse: one global un-tilt from the (weighted) reference up-axes ---
    ups, weights = [], []
    for r in basis:
        ups.append(_obb_up_axis_from_axes(axes_map[r.id], Z))
        weights.append(max(float(np.prod(r.mesh.bounding_box.extents)), 1e-12))
    scene_up = (np.asarray(ups) * np.asarray(weights)[:, None]).sum(axis=0)
    scene_up = scene_up / (np.linalg.norm(scene_up) + 1e-12)

    R = _rotation_between(scene_up, Z)
    centroid = np.vstack([np.asarray(r.mesh.vertices) for r in records]).mean(axis=0)
    for r in records:
        r.mesh.vertices = (np.asarray(r.mesh.vertices) - centroid) @ R.T + centroid
        axes_map[r.id] = R @ axes_map[r.id]

    # --- fine: snap each object's own box up-axis onto +Z (skip soft objects) ---
    for r in records:
        if "pillow" in r.category.lower() or "pillow" in r.id.lower():
            continue                                     # pillows are not snapped
        up = _obb_up_axis_from_axes(axes_map[r.id], Z)
        Rf = _rotation_between(up, Z)
        c = np.asarray(r.mesh.vertices).mean(axis=0)
        r.mesh.vertices = (np.asarray(r.mesh.vertices) - c) @ Rf.T + c
        axes_map[r.id] = Rf @ axes_map[r.id]
    return R


def _top_footprint(mesh: trimesh.Trimesh, top_z: float, eps: float = 0.02) -> tuple[float, float, float, float]:
    """Returns (min_x, min_y, max_x, max_y) of the vertices within eps of top_z."""
    v = np.asarray(mesh.vertices)
    top_v = v[v[:, 2] >= top_z - eps]
    if len(top_v) == 0:
        return float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 0].max()), float(v[:, 1].max())
    return float(top_v[:, 0].min()), float(top_v[:, 1].min()), float(top_v[:, 0].max()), float(top_v[:, 1].max())


def _seat_child_xy(child: trimesh.Trimesh, footprint: tuple[float, float, float, float]) -> tuple[float, float]:
    """Computes XY translation to clamp child centroid inside parent's top footprint."""
    cv = np.asarray(child.vertices)
    cx, cy = float(cv[:, 0].mean()), float(cv[:, 1].mean())
    min_x, min_y, max_x, max_y = footprint
    
    target_x = max(min_x, min(cx, max_x))
    target_y = max(min_y, min(cy, max_y))
    
    return target_x - cx, target_y - cy


def _support_top(parent: trimesh.Trimesh, child: trimesh.Trimesh) -> float:
    """Height of the parent's surface directly beneath the child's footprint.
    Off by default (`LENS_SUPPORT_TOP=1` enables it). `_ground_records` uses the
    global-top rule instead: `target_min = parent_global_max + 0.005`, i.e. a
    child's lowest vertex is seated on its already-grounded parent's highest
    vertex plus 5 mm. Enable this together with
    `test_child_is_seated_on_the_support_surface_under_its_own_footprint`.

    The parent's GLOBAL max-z is the wrong seat for anything but a box: a sofa's
    global top is the backrest, so a laptop lying on the SEAT gets lifted the
    full height of the backrest (measured +31.7 cm on office_lounge -- the
    only object there worse than 2.6 cm). Restricting the
    query to parent vertices inside the child's XY bounding box picks the seat
    for the laptop and the backrest for the backpack, from one rule.

    The query is over FACES, not vertices. Testing "is a parent vertex inside
    the footprint" looks equivalent and is, on the ~10^5-vertex scan meshes
    this runs on (both agree to the 4th decimal there), but it degrades
    silently on coarse geometry: a box has 8 vertices and none of them need
    lie under the child, so a small object on a big slab finds no support at
    all. A face is included when its XY bounding box overlaps the footprint,
    which is exact for flat supports and elsewhere overshoots by at most one
    face width (millimetres here).

    Falls back to the global max only when NO face overlaps -- a child
    overhanging its parent entirely, where there is no support surface to
    find and the old behaviour is as good a guess as any.
    """
    pv = np.asarray(parent.vertices)
    faces = np.asarray(parent.faces)
    cv = np.asarray(child.vertices)
    lo, hi = cv[:, :2].min(axis=0), cv[:, :2].max(axis=0)
    fx, fy = pv[:, 0][faces], pv[:, 1][faces]           # (F, 3) per-face corners
    under = ((fx.max(axis=1) >= lo[0]) & (fx.min(axis=1) <= hi[0]) &
             (fy.max(axis=1) >= lo[1]) & (fy.min(axis=1) <= hi[1]))
    if not under.any():
        return float(pv[:, 2].max())
    return float(pv[:, 2][faces[under]].max())


def _ground_records(records: list[ReconObject]) -> dict[str, float]:
    """Floor each object to z=0 PER FLOOR-ROOTED
    OBJECT. Single-image reconstruction places every object in its OWN metric
    frame, so the relative z between objects is arbitrary; that is discarded
    by dropping each floor-rooted object to the floor independently.

    An earlier revision applied ONE global offset (the min over floor-rooted
    records) to every mesh -- that only put the LOWEST floor object on z=0 and
    left taller ones floating by their (arbitrary) reconstructed relative-z,
    which is exactly the ~0.6 m "sofa floats above the table" bug.

    Each floor-rooted record (`parent is None`) is dropped by its OWN `-min_z`.
    A child is NOT trusted at its reconstructed relative-z -- single-image recon
    routinely places a child below its parent's base or below the floor (measured
    up to 143 mm under the floor on office_lounge), which detonates the physics
    settle as a contact explosion. Instead each child is RE-SEATED so its bottom
    sits `CHILD_GAP` above the parent's HIGHEST VERTEX: a child's lowest
    vertex lands on its already-grounded parent's highest vertex plus 5 mm,
    `target_min = parent_global_max + 0.005`.

    That rule is crude for non-box furniture: a sofa's global top is its
    backrest, so a laptop on the SEAT is lifted the whole height of the
    backrest. `_support_top` fixes exactly that and is OFF by default
    (`LENS_SUPPORT_TOP=1`), so that the campaigns measure one configuration;
    the cubby analysis (phase 6, 2026-08-15) showed the footprint query is not
    enough for shelves either -- a cubby needs nearest-surface-below.

    Falls back to one global offset only when nothing is floor-rooted. Returns
    the per-id applied z offset for provenance.
    """
    from collections import deque
    offsets: dict[str, float] = {}
    if not records:
        return offsets

    by_id = {r.id: r for r in records}
    
    # Orphan re-parenting
    surfaces = [r for r in records if "surface" in r.tags]
    for r in records:
        if r.parent is not None and r.parent not in by_id:
            best_surface = None
            best_dist = float('inf')
            cx, cy = float(r.mesh.vertices[:, 0].mean()), float(r.mesh.vertices[:, 1].mean())
            for surf in surfaces:
                if surf.id == r.id:
                    continue
                sz = float(surf.mesh.vertices[:, 2].max())
                smin_x, smin_y, smax_x, smax_y = _top_footprint(surf.mesh, sz)
                dx = max(0.0, smin_x - cx, cx - smax_x)
                dy = max(0.0, smin_y - cy, cy - smax_y)
                dist = (dx*dx + dy*dy)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_surface = surf
            if best_surface is not None:
                r.parent = best_surface.id
            else:
                r.parent = None
                
    children: dict[str, list[ReconObject]] = {}
    for r in records:
        if r.parent is not None and r.parent in by_id:
            children.setdefault(r.parent, []).append(r)

    floor_rooted = [r for r in records if r.parent is None]
    if not floor_rooted:
        # No floor anchor at all -- keep the old single global-offset behaviour.
        floor_z = min(float(r.mesh.vertices[:, 2].min()) for r in records)
        for r in records:
            r.mesh.apply_translation((0.0, 0.0, -floor_z))
            offsets[r.id] = -floor_z
        return offsets

    # Breadth-first from the floor roots so a parent is grounded before its
    # children: floor-rooted objects drop to z=0; each child is RE-SEATED onto
    # its (already-grounded) parent's support surface, discarding the
    # reconstructed relative-z. A z-only translation preserves x/y and the
    # object's own shape.
    queue: deque[ReconObject] = deque()
    for root in floor_rooted:
        dz = -float(root.mesh.vertices[:, 2].min())
        root.mesh.apply_translation((0.0, 0.0, dz))
        offsets[root.id] = dz
        queue.append(root)

    while queue:
        parent = queue.popleft()
        for child in children.get(parent.id, []):
            if child.id in offsets:            # guard against a cyclic parent chain
                continue
            p_min_x, p_min_y = float(parent.mesh.vertices[:, 0].min()), float(parent.mesh.vertices[:, 1].min())
            p_max_x, p_max_y = float(parent.mesh.vertices[:, 0].max()), float(parent.mesh.vertices[:, 1].max())
            cv = np.asarray(child.mesh.vertices)
            cx, cy = float(cv[:, 0].mean()), float(cv[:, 1].mean())
            
            if cx < p_min_x or cx > p_max_x or cy < p_min_y or cy > p_max_y:
                parent_top_z = float(parent.mesh.vertices[:, 2].max())
                footprint = _top_footprint(parent.mesh, parent_top_z)
                dx, dy = _seat_child_xy(child.mesh, footprint)
                if abs(dx) > 1e-5 or abs(dy) > 1e-5:
                    child.mesh.apply_translation((dx, dy, 0.0))
            
            # DEFAULT (since 2026-08-12): seat the child at the parent's
            # GLOBAL highest vertex, so every campaign measures one
            # configuration. LENS_SUPPORT_TOP=1 enables the parked alternative
            # (_support_top: the parent surface under the child's footprint),
            # which is better for sofas; it is not the default because the
            # campaigns were run without it.
            import os as _os
            if _os.environ.get("LENS_SUPPORT_TOP") == "1":
                support = _support_top(parent.mesh, child.mesh)
            else:
                support = float(parent.mesh.vertices[:, 2].max())
            dz = support + CHILD_GAP - float(child.mesh.vertices[:, 2].min())
            child.mesh.apply_translation((0.0, 0.0, dz))
            offsets[child.id] = dz
            queue.append(child)

    # An orphan whose parent id isn't among the records (or a cyclic group with
    # no floor-rooted member) is never reached above -- ground it on its own.
    for r in records:
        if r.id not in offsets:
            dz = -float(r.mesh.vertices[:, 2].min())
            r.mesh.apply_translation((0.0, 0.0, dz))
            offsets[r.id] = dz

    return offsets


def assemble_scene(records: list[ReconObject], out_dir, scene_name: str,
                    capture_meta: dict, stabilize: bool = True) -> SceneSpec:
    """The model-free downstream half of the pipeline: already-reconstructed
    `records` -> ObjectSpecs -> (optional) stabilize -> MJCF/USD export ->
    `scene.json`. Must never reference a `recon`/`segmenter` object or call
    `.release()` -- by the time this runs, no model should be resident.
    """
    out_dir = Path(out_dir)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Gravity-canonicalize FIRST: the raw SAM-3D recon
    # frame is camera-tilted (~40 deg on office_lounge), and grounding along a
    # tilted axis drags objects sideways and scrambles the horizontal layout.
    # Un-tilt the whole scene to true +Z, THEN ground (vertical-only).
    _gravity_align(records)

    # Stage-1 Fix: Laptop parenting swap.
    # The VLM is instance-blind and often assigns laptops to the wrong surface.
    # Re-parent laptops to their nearest physical support before grounding.
    surfaces = [r for r in records if "surface" in r.tags]
    laptops = [r for r in records if "laptop" in r.category.lower()]
    if len(laptops) == 2:
        for l in laptops:
            best_dist = float('inf')
            best_surf = l.parent
            cx, cy = float(l.mesh.vertices[:, 0].mean()), float(l.mesh.vertices[:, 1].mean())
            for surf in surfaces:
                sz = float(surf.mesh.vertices[:, 2].max())
                smin_x, smin_y, smax_x, smax_y = _top_footprint(surf.mesh, sz)
                dx = max(0.0, smin_x - cx, cx - smax_x)
                dy = max(0.0, smin_y - cy, cy - smax_y)
                dist = (dx*dx + dy*dy)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_surf = surf.id
            l.parent = best_surf

    # Ground the scene BEFORE _pca_dims/export/ObjectSpec/stabilize, so every
    # downstream artifact (exported meshes, scene.json, MJCF, USD) sees
    # consistently floor-referenced geometry.
    ground_offsets = _ground_records(records)

    object_specs: list[ObjectSpec] = []
    for r in records:
        mesh = r.mesh

        vis_rel = f"meshes/{r.id}.obj"
        mesh.export(out_dir / vis_rel)

        # Per-object budget. A flat 32 gives a sofa four times the concavity
        # it needs, and small objects then settle into the creases of the scan
        # instead of resting on the surface.
        hulls = collision.decompose(
            mesh, max_hulls=collision.hull_budget(r.id, parent=r.parent),
            threshold=0.05)
        col_rel = []
        for i, h in enumerate(hulls):
            rel = f"meshes/{r.id}_col{i:02d}.obj"
            h.export(out_dir / rel)
            col_rel.append(rel)

        volume = sum(abs(h.volume) for h in hulls)
        # Floor the mass: a near-flat object's CoACD hulls can have ~0 total
        # volume, and mass<=0 fails SceneSpec.validate() and crashes MuJoCo
        # (a freejoint body needs positive mass) at stabilize/export.
        mass = max(DENSITY_DEFAULT * volume, MASS_FLOOR)

        object_specs.append(ObjectSpec(
            id=r.id,
            category=r.category,
            visual_mesh=vis_rel,
            collision_meshes=col_rel,
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            dims=_pca_dims(mesh),
            mass=mass,
            friction=FRICTION_DEFAULT,
            mass_source="default",
            scale_source="pointmap",
            confidence=float(r.confidence),
            tags=list(r.tags),
            provenance={"parent": r.parent, "ground_offset_z": ground_offsets.get(r.id, 0.0)},
        ))
    log.info("meshes: wrote %d objects to %s", len(object_specs), meshes_dir)

    spec = SceneSpec(
        name=scene_name,
        objects=object_specs,
        capture=dict(capture_meta),
        floor_z=0.0,
    )

    if stabilize:
        log.info("stabilize: settling scene (CEM + MuJoCo)")
        spec = stabilize_scene(spec, out_dir)

    xml_path = mjcf.write_scene(spec, out_dir)
    log.info("export: wrote MJCF %s", xml_path)

    # De-penetration pass (held-out campaign findings, docs/evidence/
    # heldout-campaign-2026-08-13/findings.md §4). ON by default since 2026-08-17,
    # after a sliding A/B: small nudges strictly improve the settle; the one
    # big move traded penetration-locked fake stability for a real slide.
    # LENS_DEPENETRATE=0 opts out. Resolves root-vs-root
    # interpenetration by minimal translation of the lighter body, updates
    # the spec poses, and re-exports so XML, USD and scene.json all carry
    # the corrected layout.
    import os as _os_dep
    if _os_dep.environ.get("LENS_DEPENETRATE", "1") != "0":
        import mujoco as _mujoco

        from lens.depenetrate import resolve_penetrations
        _m = _mujoco.MjModel.from_xml_path(str(xml_path))
        _d = _mujoco.MjData(_m)
        _mujoco.mj_forward(_m, _d)
        rep = resolve_penetrations(_m, _d)
        if rep["displacements_m"]:
            for o in spec.objects:
                dvec = rep["displacements_m"].get(o.id)
                if dvec is not None:
                    o.pos = tuple(float(p) + float(v) for p, v in zip(o.pos, dvec))
                    o.provenance["depenetrate"] = [round(float(v), 4) for v in dvec]
            xml_path = mjcf.write_scene(spec, out_dir)
        log.info("depenetrate: resolved=%s moved=%d worst_residual=%.1fmm",
                 rep["resolved"], len(rep["displacements_m"]),
                 rep["worst_residual_mm"])

    try:
        usd_path = usd_export.write_scene(spec, out_dir)
        log.info("export: wrote USD %s", usd_path)
    except Exception as e:
        log.warning("export: USD export failed (%s) -- continuing with MJCF only", e)

    # Ship-gate (held-out campaign findings §3-§4): settle the exported MJCF under a
    # divergence guard and record the verdict. The gate REPORTS -- it never
    # alters or deletes the scene; the CLI turns a FAIL into a nonzero exit.
    import os as _os

    from lens.verify_scene import verify_scene as _verify
    verify_s = float(_os.environ.get("LENS_VERIFY_SECONDS", "30"))
    verdict = _verify(xml_path, seconds=verify_s)
    spec.verify = dataclasses.asdict(verdict)
    if verdict.verdict == "FAIL":
        log.error("SHIP-GATE FAIL for %s:", xml_path)
        for r in verdict.reasons:
            log.error("  - %s", r)
    else:
        log.info("ship-gate %s (%s)", verdict.verdict,
                 "; ".join(verdict.reasons) or "clean")

    scene_json = out_dir / "scene.json"
    spec.to_json(scene_json)
    log.info("scene.json written to %s", scene_json)

    return spec


def run_pipeline(photo, out_dir, vlm, segmenter, recon,
                  stabilize: bool = True) -> SceneSpec:
    """Run the full photo -> physics-valid scene pipeline in ONE process.

    `vlm`, `segmenter`, `recon` are the pluggable stage backends (real or
    Fake*, see the module docstrings of `lens.vlm`, `lens.scenegraph.masks`,
    `lens.recon`). Returns the final `SceneSpec`; also writes
    `<out_dir>/meshes/`, `<out_dir>/mujoco/*.xml`, `<out_dir>/usd/*.usdc`
    (best-effort) and `<out_dir>/scene.json` as a side effect.

    Kept for the Fake*-backed test suite and as the direct-call path when a
    caller already has live backend objects. The real `lens run` CLI uses
    `run_pipeline_isolated` instead, which spawns the model-heavy half as a
    subprocess so its memory is actually returned to the OS before
    stabilize runs -- see the module docstring.
    """
    out_dir = Path(out_dir)

    log.info("capture: loading %s", photo)
    capture = load_capture(photo)

    log.info("vlm: listing objects")
    reply = vlm.ask(LIST_OBJECTS, [capture.image])

    tree = build_tree(parse_object_lines(reply))
    log.info("scenegraph: %d objects in tree", len(tree))

    masks = segment_objects(capture.image, tree, segmenter, vlm, max_alternatives=3)
    log.info("masks: segmented %d/%d objects", len(masks), len(tree))

    # Segmentation is finished with its model, and reconstruction is about to
    # load several GB of its own. Holding both at once gets this process
    # OOM-killed (exit 137) on a 30 GB box, so the segmenter is released here
    # and the real orchestrator runs reconstruction in its own process
    # (lens.recon_worker).
    if hasattr(segmenter, "release"):
        segmenter.release()
        log.info("masks: released segmentation model")

    recon_results = recon.reconstruct(capture.image, masks)
    log.info("recon: reconstructed %d meshes", len(recon_results))

    records: list[ReconObject] = []
    for o in tree:
        r = recon_results.get(o.id)
        if r is None:
            continue  # no mask/recon for this object -- simply absent downstream
        records.append(ReconObject(
            id=o.id, category=o.category, tags=list(o.tags), parent=o.parent,
            mesh=r.mesh, confidence=float(r.confidence),
            axes=getattr(r, "obb_axes", None),
        ))

    scene_name = Path(photo).stem or "scene"
    return assemble_scene(
        records, out_dir, scene_name=scene_name,
        capture_meta={"source": capture.source, "orig_size": capture.orig_size},
        stabilize=stabilize,
    )


def _assemble_from_state(out_dir, stabilize: bool = True) -> SceneSpec:
    """Read `<out_dir>/recon_state.json` (the worker's output contract),
    load each object's mesh back off disk, and hand the result to
    `assemble_scene`. Factored out so the parent-side logic is unit-testable
    without spawning the real worker subprocess."""
    out_dir = Path(out_dir)
    state = json.loads((out_dir / "recon_state.json").read_text())

    records = [
        ReconObject(
            id=entry["id"],
            category=entry["category"],
            tags=list(entry.get("tags", [])),
            parent=entry.get("parent"),
            mesh=trimesh.load(out_dir / entry["visual_mesh"], force="mesh"),
            confidence=float(entry.get("confidence", 0.5)),
            axes=(np.asarray(entry["obb_axes"], dtype=np.float64)
                  if entry.get("obb_axes") is not None else None),
        )
        for entry in state["objects"]
    ]
    return assemble_scene(
        records, out_dir, scene_name=state["scene_name"],
        capture_meta=state["capture"], stabilize=stabilize,
    )


def run_pipeline_isolated(photo, out_dir, model: str | None = None,
                           stabilize: bool = True) -> SceneSpec:
    """The real orchestrator: spawn `lens.recon_worker` as a
    subprocess to do capture -> VLM -> tree -> segmentation -> reconstruction,
    then assemble the scene in THIS (light, model-free) process once the
    worker has exited and its model memory has been returned to the OS.

    This function reaches the worker ONLY via `subprocess.run` -- never
    `import lens.recon_worker` -- so this module never resident-loads
    torch/sam3/sam3d_objects itself.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [sys.executable, "-u", "-m", "lens.recon_worker",
            "--photo", str(photo), "--out", str(out_dir)]
    if model:
        argv += ["--model", model]

    # stdout/stderr are INHERITED, not piped: the worker's stage `log.info`
    # lines are the only visibility a monitor watching the run log has into
    # its progress, and capturing them would hide that until the process
    # exits (or bloat memory buffering it).
    result = subprocess.run(argv)
    if result.returncode != 0:
        raise RuntimeError(
            f"lens.recon_worker exited with code {result.returncode}")

    return _assemble_from_state(out_dir, stabilize=stabilize)
