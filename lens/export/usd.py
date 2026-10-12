# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""SceneSpec -> Isaac Sim USD exporter.

Layout per object (Z-up, metersPerUnit = `spec.meters_per_unit`, default 1):
  /World/<id>            Xform  + RigidBodyAPI + MassAPI(mass=`ObjectSpec.mass`)
                          pose = `ObjectSpec.pos`/`.quat` as translate/orient
                          Xform ops (not baked into vertices)
  /World/<id>/visual      Mesh with per-vertex displayColor, no collision
  /World/<id>/col_XX      invisible Mesh, CollisionAPI, convexHull approximation
plus a static /World/floor slab at `spec.floor_z` so the scene is
self-supporting when the target sim's own ground plane is disabled.

Everything is driven by the SceneSpec: mesh paths come from
`ObjectSpec.visual_mesh`/`.collision_meshes` relative to `scene_dir`; pose is
`ObjectSpec.pos`/`.quat` applied as Xform ops; mass is `ObjectSpec.mass` via
`MassAPI.CreateMassAttr`.

Cross-environment note: `pxr` (usd-core) may not be installed in the
caller's interpreter. `write_scene` tries `import pxr` first; if that
raises `ImportError`, it serializes `spec` to a temp JSON file and re-runs
this module as `python -m lens.export.usd <spec.json> <scene_dir>` under
`PXR_PY` (an interpreter that has usd-core installed). The child
process needs `lens.scenespec` to deserialize the spec, so the subprocess's
`PYTHONPATH` is extended with this repo's root -- the `lens` package isn't
necessarily installed in the pxr-capable env. Subprocess failures raise a
`RuntimeError` that includes the child's stderr, so a broken export doesn't
disappear silently.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from lens.scenespec import SceneSpec

# The USD writer runs out-of-process in a Python that has `usd-core` (pxr)
# installed, because pxr and the rest of this package do not always coexist in
# one interpreter. Set LENS_PXR_PYTHON to that interpreter; empty means the USD
# exporter is unavailable and its tests skip.
PXR_PY = os.environ.get("LENS_PXR_PYTHON", "")


def _sanitize(name: str) -> str:
    """USD prim names must be valid identifiers (no quotes, spaces, dots,
    etc.); object ids come from a vision-model-driven naming stage, so
    non-identifier characters are plausible."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a mesh's vertices/faces/per-vertex RGB (0..1) via trimesh.
    trimesh always populates `visual.vertex_colors` (defaulting to a uniform
    mid-gray when the source has none), so the visual mesh always carries a
    displayColor primvar even for meshes authored without color."""
    mesh = trimesh.load(path, force="mesh", process=False)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32)
    C = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    return V, F, C


def _usd_out_path(spec: SceneSpec, scene_dir: Path) -> Path:
    return Path(scene_dir) / "usd" / f"{spec.name}.usdc"


def _build_stage(spec: SceneSpec, scene_dir: Path, out_path: Path) -> None:
    """Do the actual pxr work. Only called once `import pxr` is known to
    succeed in the current interpreter (either in-process, or inside the
    LENS_PXR_PYTHON subprocess via `_main`)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    def add_mesh(path, V, F, color=None, vertex_colors=None):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray((3,) * len(F)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1)))
        mesh.CreateSubdivisionSchemeAttr("none")
        if vertex_colors is not None:
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec3fArray.FromNumpy(np.clip(vertex_colors, 0, 1)))
        elif color is not None:
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        return mesh

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, spec.meters_per_unit)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    for o in spec.objects:
        safe = _sanitize(o.id)
        xf = UsdGeom.Xform.Define(stage, f"/World/{safe}")
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(o.mass))

        xformable = UsdGeom.Xformable(prim)
        px, py, pz = o.pos
        xformable.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        qw, qx, qy, qz = o.quat
        xformable.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))

        V, F, C = _load_mesh(scene_dir / o.visual_mesh)
        add_mesh(f"/World/{safe}/visual", V, F, vertex_colors=C)

        for i, cm in enumerate(o.collision_meshes):
            Vc, Fc, _ = _load_mesh(scene_dir / cm)
            m = add_mesh(f"/World/{safe}/col_{i:02d}", Vc, Fc, color=(0.8, 0.5, 0.2))
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
            mc.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            UsdGeom.Imageable(m.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    # static floor slab: collider without RigidBodyAPI
    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    fx = UsdGeom.Xformable(floor.GetPrim())
    fx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, spec.floor_z - 0.05))
    fx.AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 0.1))
    floor.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.55, 0.55, 0.53)]))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    stage.GetRootLayer().Save()


def _run_in_pxr_env(spec: SceneSpec, scene_dir: Path) -> None:
    """Re-invoke this module under `PXR_PY`, passing `spec` via a temp JSON
    file (subprocess args can't carry a dataclass). The child imports
    `lens.scenespec` to deserialize it, so `PYTHONPATH` is extended with
    this repo's root -- `lens` is installed in the caller's env, not
    necessarily in the pxr-capable one."""
    repo_root = str(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory() as td:
        spec_path = Path(td) / "spec.json"
        spec.to_json(spec_path)

        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([repo_root, existing] if existing else [repo_root])

        if not PXR_PY:
            raise RuntimeError(
                "usd-core (pxr) is not importable here and LENS_PXR_PYTHON is "
                "unset. Set LENS_PXR_PYTHON to a Python interpreter with "
                "`usd-core` installed, or install usd-core into this one "
                "(`pip install -e '.[usd]'`)."
            )
        result = subprocess.run(
            [PXR_PY, "-m", "lens.export.usd", str(spec_path), str(scene_dir)],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"USD export subprocess ({PXR_PY}) failed with exit "
                f"code {result.returncode}.\n"
                f"--- child stdout ---\n{result.stdout}\n"
                f"--- child stderr ---\n{result.stderr}"
            )


def write_scene(spec: SceneSpec, scene_dir: Path) -> Path:
    """Assemble `spec`'s objects into an Isaac Sim USD at
    `<scene_dir>/usd/<spec.name>.usdc`. Mesh paths on `ObjectSpec` are
    relative to `scene_dir`. Returns the path to the written USD.

    Tries `import pxr` in-process first; if usd-core isn't installed in the
    calling interpreter, falls back to a `LENS_PXR_PYTHON` subprocess
    (see `_run_in_pxr_env`)."""
    scene_dir = Path(scene_dir).resolve()
    out_path = _usd_out_path(spec, scene_dir)
    try:
        import pxr  # noqa: F401 -- import-only probe; real work is in _build_stage
    except ImportError:
        _run_in_pxr_env(spec, scene_dir)
        return out_path
    _build_stage(spec, scene_dir, out_path)
    return out_path


def _main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    spec_path, scene_dir = argv[0], Path(argv[1])
    spec = SceneSpec.from_json(spec_path)
    out_path = _usd_out_path(spec, scene_dir)
    _build_stage(spec, scene_dir, out_path)
    print(f"[usd] wrote {out_path}")


if __name__ == "__main__":
    _main()
