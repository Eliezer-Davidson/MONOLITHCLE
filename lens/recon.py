# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Mask -> posed 3D mesh. SAM-3D-Objects (Meta, SAM License) is the Phase-1
backend; the 12 GB low-VRAM mode depends on local patches to the backend
checkout that are not shipped (THIRD_PARTY_NOTICES.md).
Frame contract: outputs are Z-up meters (yup->zup at this boundary)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import trimesh


@dataclass
class ReconResult:
    mesh: trimesh.Trimesh
    confidence: float
    obb_axes: np.ndarray | None = None   # (3,3) model box axes in the posed scene
    #                                      frame (columns), for gravity_align's
    #                                      per-object snap. See _apply_layout.
    postopt: dict | None = None          # layout post-opt provenance
    #                                      ({iou, iou_before_optim, optim_accepted})
    #                                      or None when post-opt didn't run.


class ReconBackend(Protocol):
    def reconstruct(self, image: np.ndarray,
                    masks: dict[str, np.ndarray]) -> dict[str, ReconResult]: ...


def yup_to_zup(v: np.ndarray) -> np.ndarray:
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


class FakeRecon:
    def __init__(self, size: float = 0.2): self._s = size
    def reconstruct(self, image, masks):
        out = {}
        for i, (oid, m) in enumerate(sorted(masks.items())):
            ys, xs = np.nonzero(m)
            cx = (xs.mean() / m.shape[1] - 0.5) if len(xs) else 0.0
            box = trimesh.creation.box(extents=(self._s,) * 3)
            box.apply_translation((cx, 0.0, self._s / 2 + i * self._s))
            box.visual.vertex_colors = np.tile([120, 120, 200, 255], (len(box.vertices), 1))
            out[oid] = ReconResult(mesh=box, confidence=1.0, obb_axes=np.eye(3))
        return out


# Low-VRAM switches for the 12 GB recipe. Only PYTORCH_CUDA_ALLOC_CONF is read
# by stock PyTorch. The three LENS_* switches are read by a set of local
# patches to the SAM-3D-Objects checkout (fp16 weight casts, CPU-side sparse
# construction, embedder offload) that are NOT shipped with this repository --
# see THIRD_PARTY_NOTICES.md. Against an unpatched checkout they are inert and
# the backend needs a card larger than 12 GB.
LOW_VRAM_ENV = {
    "LENS_HALF_WEIGHTS": "1", "LENS_CPU_CONSTRUCT": "1",
    "LENS_OFFLOAD_EMBEDDER": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _run_deferred_postopt(bundle_path: str) -> dict:
    """Run one captured layout post-opt bundle through the patched
    sam-3d-objects deferred runner. Module-level seam so tests can substitute
    it; the real import is deferred because sam3d_objects is a multi-GB stack."""
    from sam3d_objects.pipeline.deferred_postopt import run_deferred_postopt
    return run_deferred_postopt(bundle_path)


_DEFER_ENV = "SAM3D_DEFER_LAYOUT_POSTOPT_DIR"


class Sam3dObjectsBackend:
    def __init__(self, skip_layout_postopt: bool | None = None, low_vram: bool = True):
        # Layout post-optimisation (deferred) is the DEFAULT since 2026-08-12;
        # running it deferred is what makes it fit in 12 GB of VRAM. LENS_SKIP_POSTOPT=1 restores
        # the old post-opt-free behaviour for A/Bs and troubleshooting.
        if skip_layout_postopt is None:
            skip_layout_postopt = os.environ.get("LENS_SKIP_POSTOPT") == "1"
        self._skip_layout = skip_layout_postopt
        if low_vram:
            for k, v in LOW_VRAM_ENV.items(): os.environ.setdefault(k, v)
        self._pipe = None

    def _load(self):
        root = os.environ.get("SAM3D_OBJECTS_ROOT")
        if not root:
            raise RuntimeError(
                "SAM3D_OBJECTS_ROOT is not set. Point it at a checkout of Meta's "
                "sam-3d-objects that carries checkpoints/hf/pipeline.yaml; see "
                "the reconstruction section of the README for how to obtain it. "
                "This stage needs a CUDA GPU and is never exercised by the "
                "default test run.")
        from sam3d_objects.pipeline.inference_pipeline_pointmap import (
            InferencePipelinePointMap)
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        cfg = OmegaConf.load(os.path.join(root, "checkpoints", "hf", "pipeline.yaml"))
        cfg.rendering_engine = "pytorch3d"
        # The shipped pipeline.yaml sets compile_model: true, but the model is
        # not Dynamo-safe: the timestep embedder calls -np.log(max_period) on a
        # traced tensor and compilation dies with "'ndarray' object has no
        # attribute 'neg'". Inference does not need the compiled graph.
        cfg.compile_model = False
        cfg.workspace_dir = os.path.join(root, "checkpoints", "hf")
        self._pipe = instantiate(cfg)

    def reconstruct(self, image, masks):
        import gc, torch
        if self._pipe is None: self._load()

        if self._skip_layout:
            # Layout post-optimisation off entirely: unchanged behaviour.
            out = {}
            for oid, mask in masks.items():
                rgba = np.dstack([image[..., :3], (mask * 255).astype(np.uint8)])
                r = self._pipe.run(rgba, None, 42, stage1_only=False,
                                   with_mesh_postprocess=False, with_texture_baking=False,
                                   with_layout_postprocess=False,
                                   use_vertex_color=True, stage1_inference_steps=None,
                                   pointmap=None)
                mesh = r["glb"]
                posed, axes = _apply_layout(mesh, r)
                out[oid] = ReconResult(mesh=posed, confidence=float(r.get("score", 0.5)),
                                       obb_axes=axes)
                del r; gc.collect(); torch.cuda.empty_cache()   # per-object prune
            return out

        # Deferred post-opt: the alignment is a pure function of per-object
        # inputs the patched library CAPTURES during recon (bundle files) and
        # we replay AFTER the ~10 GB model stack is released -- post-opt alone
        # needs ~1-2 GB, so the co-residency that OOMs a 12 GB card never
        # happens. Poses stay raw during the loop; _apply_layout runs at the
        # end with the refined pose, so the math matches in-loop post-opt.
        import shutil, tempfile
        defer_dir = tempfile.mkdtemp(prefix="lens_postopt_")
        prior = os.environ.get(_DEFER_ENV)
        raw: dict[str, dict] = {}
        try:
            os.environ[_DEFER_ENV] = defer_dir
            try:
                for oid, mask in masks.items():
                    rgba = np.dstack([image[..., :3], (mask * 255).astype(np.uint8)])
                    r = self._pipe.run(rgba, None, 42, stage1_only=False,
                                       with_mesh_postprocess=False, with_texture_baking=False,
                                       with_layout_postprocess=True,
                                       use_vertex_color=True, stage1_inference_steps=None,
                                       pointmap=None)
                    def _cpu(x):
                        return x.detach().cpu() if isinstance(x, torch.Tensor) else x
                    raw[oid] = {"glb": r["glb"],
                                "rotation": _cpu(r["rotation"]),
                                "translation": _cpu(r["translation"]),
                                "scale": _cpu(r["scale"]),
                                "score": float(r.get("score", 0.5)),
                                "bundle": r.get("postopt_deferred")}
                    del r; gc.collect(); torch.cuda.empty_cache()   # per-object prune
            finally:
                if prior is None:
                    os.environ.pop(_DEFER_ENV, None)
                else:
                    os.environ[_DEFER_ENV] = prior

            self.release()   # models gone BEFORE the first alignment

            out = {}
            for oid, d in raw.items():
                postopt = None
                if d["bundle"]:
                    # Per-object fault isolation: one bad bundle must not kill
                    # the scene's other objects. On failure the object keeps
                    # its unrefined pose, and postopt records WHY -- an
                    # "optimized" claim is never silently faked.
                    try:
                        rev = _run_deferred_postopt(d["bundle"])
                        d["rotation"] = torch.as_tensor(rev["rotation"]).reshape(1, 4)
                        d["translation"] = torch.as_tensor(rev["translation"]).reshape(1, 3)
                        d["scale"] = torch.as_tensor(rev["scale"]).reshape(1, 3)
                        postopt = {"iou": float(rev["iou"]),
                                   "iou_before_optim": float(rev["iou_before_optim"]),
                                   "optim_accepted": bool(rev["optim_accepted"])}
                    except Exception as e:  # noqa: BLE001
                        postopt = {"error": f"{type(e).__name__}: {e}"}
                posed, axes = _apply_layout(d["glb"], d)
                out[oid] = ReconResult(mesh=posed, confidence=d["score"],
                                       obb_axes=axes, postopt=postopt)
            return out
        finally:
            shutil.rmtree(defer_dir, ignore_errors=True)

    def release(self) -> None:
        """Drop the pipeline and free its memory.

        SAM-3D-Objects offloads its models GPU->CPU during decode, so after
        reconstruction ~6 GB of model weights sit in SYSTEM RAM. Stabilization
        then loads the full scene into MuJoCo and runs a CEM search on top --
        the two together OOM-kill the process (exit 137). Release the pipeline
        once reconstruction is done. Safe to call repeatedly; reconstruct()
        reloads lazily.
        """
        if self._pipe is None:
            return
        import gc
        self._pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _obb_axes_from_pose(local_vertices: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The object's box axes in the posed scene frame, computed analytically.

    Under ``v_scene = (v_local * s) @ R + t``, a local axis ``e_i`` maps to the
    scene direction ``R[i, :]`` (row i of the pose rotation), and the posed AABB
    extent along it is the local extent scaled by ``s``. Ordering the rows of
    ``R`` by the LOCAL AABB extent therefore yields the same ordered frame as
    running PCA over the posed AABB corners and sorting the eigenvectors by
    eigenvalue, without the eigen decomposition (columns = axes, right-handed).

    This is the clean axis source the gravity-align snap needs; a geometric
    trimesh OBB is too noisy and was what made every object land upside down
    when the per-object snap ran on it.
    """
    ext = local_vertices.max(axis=0) - local_vertices.min(axis=0)          # (3,)
    dirs = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)          # unit rows
    order = np.argsort(-ext)                                               # extent desc
    axes = dirs[order].T.astype(np.float64)                               # cols = axes
    if np.linalg.det(axes) < 0:                                           # right-handed
        axes[:, 2] *= -1
    return axes


# Basis change from the glTF convention (+Y up) to this pipeline's (+Z up).
# Points are row vectors throughout this module, so it right-multiplies them;
# a frame whose axes are columns takes the transpose instead.
_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
_YUP_TO_ZUP_F32 = _YUP_TO_ZUP.astype(np.float32)

# Half turns about an object's own X and Y axes. Both have determinant +1, so
# neither reverses face winding.
_NO_CORRECTION = np.eye(3, dtype=np.float32)
_HALF_TURN_X = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
_HALF_TURN_Y = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)


def _upright_correction(pose: np.ndarray) -> np.ndarray:
    """Half turn that lifts a mis-posed reconstruction's own up-axis off the floor.

    Points are row vectors, so row *i* of ``pose`` is where the object's *i*-th
    local axis ends up once the pose is applied, and row 2 is its canonical up.
    A reconstruction whose up-axis comes back below the horizon arrived
    inverted. Turning it half a revolution about whichever remaining axis still
    points away from the camera rights it while keeping the same face toward the
    viewer. Anything already the right way up gets the identity.
    """
    if pose[2, 1] >= 0.0:
        return _NO_CORRECTION
    return _HALF_TURN_Y if pose[1, 2] > 0.0 else _HALF_TURN_X


def _apply_layout(mesh: trimesh.Trimesh, r: dict) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Pose one reconstructed mesh into the scene frame.

    The decoded pose acts on the object in a +Z-up frame, so the glTF mesh is
    rotated out of +Y-up first, corrected if it came back inverted, then scaled,
    rotated and translated as ``(v * scale) @ R + t``. That leaves the object in
    the pose's own +Y-up world, which the last step rotates into the +Z-up world
    the rest of this pipeline works in -- grounding, stabilisation and both
    exporters all assume +Z. Skipping that final rotation leaves the scene +Y-up
    and the downstream ``_gravity_align`` snap then inverts every object.

    The rotation is RIGHT-multiplied: pytorch3d's Transform3d/compose_transform
    computes ``transform_points(v) == v @ R`` (verified numerically to 0.0 vs
    ``v @ R.T``). Output is already z-up meters, so ``reconstruct`` must NOT
    re-apply ``yup_to_zup``.

    Returns ``(posed_mesh, obb_axes)`` -- the box axes (see ``_obb_axes_from_pose``)
    are the clean orientation reference the downstream ``_gravity_align`` snap
    needs; without them it falls back to a noisy trimesh OBB and flips objects.
    """
    import torch
    from pytorch3d.transforms import quaternion_to_matrix

    def _np(x):
        # scale/translation/rotation come off the model as CUDA tensors;
        # np.asarray() on those calls .numpy() without .cpu() and dies with
        # "can't convert cuda:0 device type tensor to numpy".
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    vertices = mesh.vertices.astype(np.float32) @ _YUP_TO_ZUP_F32

    rot = r["rotation"]
    if not isinstance(rot, torch.Tensor):
        rot = torch.as_tensor(rot)
    R = quaternion_to_matrix(rot.detach().cpu()).squeeze(0).numpy()

    vertices = vertices @ _upright_correction(R)

    s = float(_np(r["scale"]).reshape(-1)[0])
    t = _np(r["translation"]).reshape(3)
    scene_verts = (vertices * s) @ R + t
    # `vertices` is the corrected local frame the pose acts on, so its AABB
    # extents order the box axes consistently with the posed geometry.
    axes = _obb_axes_from_pose(vertices, R)

    out = mesh.copy()
    out.vertices = scene_verts @ _YUP_TO_ZUP
    axes = _YUP_TO_ZUP.T @ axes
    return out, axes
