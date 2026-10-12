# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Convex decomposition of a mesh into CoACD hulls, for use as MuJoCo/Isaac
collision geoms. A raw convex hull of a concave shape (e.g. a table) fills in
the space a real rigid body wouldn't occupy (under the tabletop, between
sofa cushions); CoACD splits the shape into a small set of convex parts that
hug the real geometry instead.
"""
from __future__ import annotations

# MUST import before coacd: the reverse order segfaults the interpreter with
# no traceback (both vendor native libs; coacd-first poisons open3d's loader).
import open3d as o3d

import coacd
import numpy as np
import trimesh

coacd.set_log_level("error")


# Hull budgets, set by how much concavity an object is likely to carry. More
# hulls track a concave surface more closely and cost more contact pairs at run
# time, so this is a fidelity/cost trade rather than a quality dial.
_FOLIAGE_HULLS = 32
_SLAB_HULLS = 8
_DEFAULT_HULLS = 16

# Vocabulary for the two special classes, authored for this pipeline from the
# kinds of thing its own object-listing prompt asks the model to report (see
# lens/prompts.py: furniture a robot could bump into or place things on, plus
# whatever movable objects sit on that furniture). The terms are deliberately
# broader than any one scene needs, so a new photo does not silently fall
# through to the default.

# Many thin lobes with air between them. A handful of convex parts approximates
# none of these, so they get the largest budget.
_FOLIAGE_TERMS = (
    "plant", "fern", "shrub", "palm", "tree", "hedge",
    "leaf", "frond", "flower", "blossom", "bouquet",
    "vase", "planter", "curtain", "drape",
)

# A carcass of flat panels, or a room surface: a few large convex parts already
# hug these, and spending the foliage budget on them only multiplies contact
# pairs for no gain in fidelity.
_SLAB_TERMS = (
    # seating and beds
    "sofa", "couch", "settee", "bench", "chair", "stool", "ottoman", "bed",
    # tables and work surfaces
    "table", "desk", "counter", "worktop", "nightstand",
    # storage carcasses
    "shelf", "bookcase", "cabinet", "cupboard", "dresser", "sideboard",
    "wardrobe", "drawer",
    # rectilinear containers
    "box", "crate", "carton", "tray", "bin", "pallet",
    # room surfaces
    "floor", "wall", "ceiling",
)


def hull_budget(name: str, parent: str | None = None) -> int:
    """How many convex parts this object's collider may have.

    The class is inferred from the words the scene-graph stage puts in an object
    id, foliage first so a plant standing on a table is still scored as a plant.
    Matching is on lowercase substrings, which is deliberately loose: these ids
    are generated text, not a controlled vocabulary, so `potted_plant_2` and
    `plants_on_shelf_0` should both land in the same class.

    `parent` supplies the support an id does not carry. Ids here are bare
    (`black_laptop_2_0`), and a laptop lying on a sofa wants the sofa's budget,
    so the support is passed in rather than renaming every object to suit a
    collision heuristic.
    """
    hay = name.lower() if parent is None else f"{name}_on_{parent}".lower()
    if any(k in hay for k in _FOLIAGE_TERMS):
        return _FOLIAGE_HULLS
    if any(k in hay for k in _SLAB_TERMS):
        return _SLAB_HULLS
    return _DEFAULT_HULLS


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce `mesh` to at most `target_faces` triangles. CoACD's runtime
    scales badly with input face count, so large meshes are decimated first;
    meshes already under the target (e.g. a primitive box) pass through
    unchanged."""
    if len(mesh.faces) <= target_faces:
        return mesh
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    m = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles, dtype=np.int32)
    return trimesh.Trimesh(vertices=V, faces=F)


def decompose(
    mesh: trimesh.Trimesh,
    max_hulls: int = 32,
    threshold: float = 0.05,
    coacd_faces: int = 20_000,
) -> list[trimesh.Trimesh]:
    """Decompose `mesh` into convex hulls suitable for collision geoms.

    Args:
        mesh: input triangle mesh (need not be convex).
        max_hulls: upper bound on the number of convex parts CoACD may emit.
        threshold: CoACD concavity threshold (lower = more/tighter parts).
        coacd_faces: decimation target before CoACD (see `_decimate`).

    Returns:
        A list of watertight `trimesh.Trimesh` convex hulls (1 <= len <= max_hulls).
    """
    dec = _decimate(mesh, coacd_faces)
    cmesh = coacd.Mesh(dec.vertices, dec.faces)
    parts = coacd.run_coacd(
        cmesh, threshold=threshold, max_convex_hull=max_hulls, seed=0
    )
    return [
        trimesh.Trimesh(vertices=np.asarray(pv), faces=np.asarray(pf, dtype=np.int32))
        for pv, pf in parts
    ]
