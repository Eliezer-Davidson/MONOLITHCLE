import os
import numpy as np
import pytest
import trimesh
from lens.recon import FakeRecon, ReconResult, _apply_layout, yup_to_zup

# torch and pytorch3d belong to the GPU reconstruction tier, which is an
# optional extra. The tests that need them skip cleanly on a plain
# `pip install -e ".[dev]"`.
try:
    import torch
except ImportError:                                   # pragma: no cover
    torch = None
try:
    import pytorch3d                                  # noqa: F401
    _HAVE_PYTORCH3D = True
except ImportError:                                   # pragma: no cover
    _HAVE_PYTORCH3D = False

requires_torch = pytest.mark.skipif(
    torch is None, reason="needs torch (pip install -e '.[cpu-recon]')")
requires_pytorch3d = pytest.mark.skipif(
    torch is None or not _HAVE_PYTORCH3D,
    reason="needs torch + pytorch3d (the GPU reconstruction tier)")


def test_fake_recon_returns_posed_zup_boxes():
    img = np.zeros((10, 10, 3), np.uint8)
    m = np.zeros((10, 10), bool); m[2:5, 2:5] = True
    out = FakeRecon().reconstruct(img, {"box_0": m})
    mesh = out["box_0"].mesh
    assert mesh.vertices.shape[1] == 3 and mesh.vertices[:, 2].min() >= -1e-6


def test_fake_recon_preserves_mask_keys_and_returns_reconresult():
    img = np.zeros((10, 10, 3), np.uint8)
    masks = {
        "a": np.zeros((10, 10), bool),
        "b": np.zeros((10, 10), bool),
    }
    masks["a"][1:3, 1:3] = True
    masks["b"][6:8, 6:8] = True
    out = FakeRecon().reconstruct(img, masks)
    assert set(out.keys()) == {"a", "b"}
    for oid, r in out.items():
        assert isinstance(r, ReconResult)
        assert r.confidence == 1.0


def test_fake_recon_stacks_boxes_by_sorted_id_without_z_overlap():
    """Boxes are stacked (non-overlapping in Z) in sorted-object-id order, so the
    result is deterministic regardless of dict insertion order."""
    img = np.zeros((10, 10, 3), np.uint8)
    m1 = np.zeros((10, 10), bool); m1[0:3, 0:3] = True
    m2 = np.zeros((10, 10), bool); m2[0:3, 5:8] = True
    # insertion order deliberately reversed vs. sort order
    out = FakeRecon(size=0.2).reconstruct(img, {"z_last": m2, "a_first": m1})
    z_first = out["a_first"].mesh.vertices[:, 2]
    z_last = out["z_last"].mesh.vertices[:, 2]
    # "a_first" sorts before "z_last", so it occupies the lower stack slot.
    assert z_first.max() <= z_last.min() + 1e-6
    assert z_first.min() >= -1e-6


def test_fake_recon_box_size_is_respected():
    img = np.zeros((10, 10, 3), np.uint8)
    m = np.zeros((10, 10), bool); m[2:5, 2:5] = True
    size = 0.5
    out = FakeRecon(size=size).reconstruct(img, {"box_0": m})
    v = out["box_0"].mesh.vertices
    extents = v.max(axis=0) - v.min(axis=0)
    np.testing.assert_allclose(extents, [size, size, size], atol=1e-6)


def test_fake_recon_sets_vertex_colors():
    img = np.zeros((10, 10, 3), np.uint8)
    m = np.zeros((10, 10), bool); m[2:5, 2:5] = True
    out = FakeRecon().reconstruct(img, {"box_0": m})
    colors = np.asarray(out["box_0"].mesh.visual.vertex_colors)
    assert colors.shape[1] == 4
    assert (colors[:, :3] == [120, 120, 200]).all()


def test_fake_recon_empty_masks_returns_empty_dict():
    img = np.zeros((10, 10, 3), np.uint8)
    assert FakeRecon().reconstruct(img, {}) == {}


def test_fake_recon_centroid_offsets_x_by_mask_position():
    """A mask centered in the left half of the image should produce a box with a
    negative x, and one in the right half a positive x (centroid -> [-0.5, 0.5])."""
    img = np.zeros((10, 20, 3), np.uint8)
    left = np.zeros((10, 20), bool); left[:, 0:2] = True
    right = np.zeros((10, 20), bool); right[:, 18:20] = True
    out = FakeRecon().reconstruct(img, {"left": left, "right": right})
    cx_left = out["left"].mesh.vertices[:, 0].mean()
    cx_right = out["right"].mesh.vertices[:, 0].mean()
    assert cx_left < 0 < cx_right


def test_yup_to_zup_maps_up_axis_to_z():
    v = np.array([[1.0, 2.0, 3.0]])
    out = yup_to_zup(v)
    # (x, y, z)_yup -> (x, -z, y)_zup
    np.testing.assert_allclose(out, [[1.0, -3.0, 2.0]])


def test_yup_to_zup_preserves_shape_for_multiple_vertices():
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = yup_to_zup(v)
    assert out.shape == (3, 3)


@requires_pytorch3d
def test_apply_layout_identity_pose_applies_both_frame_conversions():
    """With an identity layout (identity quat, scale=1, translation=0),
    `_apply_layout` reduces to the two fixed frame conversions it performs: the
    glb-local y-up->z-up BEFORE the pose, and the SAM-3D-WORLD y-up->z-up AFTER
    it. SAM-3D poses objects in a y-up world; MONOLITHCLE is z-up end to end, so
    the second conversion is what makes the frame contract hold (without it every
    object lands upside down under _gravity_align's +Z snap). A glb "up" vertex
    (0, +h, 0) therefore lands at world (0, -h, 0). Real poses are never identity
    -- this pins the composite transform, not a physical orientation."""
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    h = 0.37
    mesh.vertices = np.array([[0.0, h, 0.0]])  # single glb vertex, +Y "up"
    r = {
        "rotation": np.array([[1.0, 0.0, 0.0, 0.0]]),  # identity, wxyz
        "scale": np.array([1.0]),
        "translation": np.array([0.0, 0.0, 0.0]),
    }
    posed, _axes = _apply_layout(mesh, r)
    np.testing.assert_allclose(posed.vertices[0], [0.0, -h, 0.0], atol=1e-6)


@requires_pytorch3d
def test_apply_layout_matches_reference_compose_transform():
    """`_apply_layout` must compose the layout as: y-up->z-up convert FIRST,
    upside-down flip, THEN scale->rotate->translate with pytorch3d's `v @ R`
    right-multiply convention. The expected transform is built INDEPENDENTLY
    here from pytorch3d's own Transform3d. This fails
    under either earlier bug -- rotating as `@ R.T`, or posing the raw y-up glb
    and swapping axes last -- for any non-identity rotation."""
    import torch
    from pytorch3d.transforms import Transform3d, quaternion_to_matrix

    M_yup2zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        q = torch.tensor(rng.standard_normal((1, 4)), dtype=torch.float32)
        q = q / q.norm(dim=1, keepdim=True)           # random non-identity unit quat (wxyz)
        scale = torch.tensor([1.7])                   # uniform metric scale
        trans = torch.tensor([[0.5, -1.2, 3.0]])
        verts = rng.standard_normal((32, 3))
        mesh = trimesh.Trimesh(vertices=verts.copy(),
                               faces=np.array([[0, 1, 2]]), process=False)
        out = _apply_layout(
            mesh, {"rotation": q, "scale": scale, "translation": trans})[0].vertices

        # independent reference via pytorch3d's own transform machinery
        R = quaternion_to_matrix(q)                   # (1, 3, 3)
        Rm = R[0].numpy()
        vz = verts @ M_yup2zup
        if Rm[2, 1] < 0:                              # upside-down flip (same rule)
            if Rm[1, 2] > 0:
                vz[:, 0] *= -1.0; vz[:, 2] *= -1.0
            else:
                vz[:, 1] *= -1.0; vz[:, 2] *= -1.0
        ref = Transform3d().scale(scale).rotate(R).translate(trans).transform_points(
            torch.tensor(vz, dtype=torch.float32)[None])[0].numpy()
        # ...then the SAM-3D-world y-up -> z-up conversion _apply_layout appends
        # (MONOLITHCLE is z-up; SAM-3D poses in a y-up world). +Y->+Z, +Z->-Y.
        R_world = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        ref = ref @ R_world.T

        np.testing.assert_allclose(out, ref, atol=1e-4, err_msg=f"seed={seed}")


@pytest.mark.gpu
def test_sam3d_objects_reconstructs_posed_zup_mesh_from_office_photo():
    from PIL import Image
    from lens.recon import Sam3dObjectsBackend

    photo = os.environ.get("LENS_DEMO_PHOTO")
    if not photo:
        pytest.skip("set LENS_DEMO_PHOTO to an indoor photo to run this")
    img = np.asarray(Image.open(photo))
    mask = np.zeros(img.shape[:2], bool)
    h, w = mask.shape
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    out = Sam3dObjectsBackend().reconstruct(img, {"obj_0": mask})
    mesh = out["obj_0"].mesh
    assert mesh.vertices.shape[1] == 3 and len(mesh.vertices) > 0


# ---------------------------------------------------------------------------
# Deferred layout post-optimisation
# ---------------------------------------------------------------------------

from lens.recon import Sam3dObjectsBackend


class _FakePipe:
    """Stands in for the SAM-3D-Objects pipeline. Writes a bundle file when
    the defer env var is set and post-opt is requested, mirroring the patched
    library's capture mode."""

    def __init__(self, bundle_for=("obj_a",)):
        self.calls = []          # (oid-index, with_layout_postprocess)
        self._bundle_for = set(bundle_for)
        self._n = 0

    def run(self, rgba, mask, seed, **kw):
        self.calls.append(kw.get("with_layout_postprocess"))
        mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
        r = {
            "glb": mesh,
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),   # identity, wxyz
            "translation": torch.zeros(1, 3),
            "scale": torch.ones(1, 3),
            "score": 0.9,
        }
        d = os.environ.get("SAM3D_DEFER_LAYOUT_POSTOPT_DIR")
        if d and kw.get("with_layout_postprocess") and self._n < len(self._bundle_for):
            p = os.path.join(d, f"{self._n}.pt")
            with open(p, "wb") as f:
                f.write(b"bundle")
            r["postopt_deferred"] = p
        self._n += 1
        return r


def _backend_with(pipe):
    b = Sam3dObjectsBackend(skip_layout_postopt=False, low_vram=False)
    b._pipe = pipe
    b._load = lambda: None
    return b


def _two_masks():
    img = np.zeros((10, 10, 3), np.uint8)
    m = np.zeros((10, 10), bool); m[2:5, 2:5] = True
    return img, {"obj_a": m, "obj_b": m.copy()}


@requires_torch
def test_defer_mode_applies_refined_pose_and_records_iou(monkeypatch):
    """Objects with a captured bundle get the alignment's revised pose applied
    (here: +0.5 m along x) and carry IoU provenance; objects without a bundle
    keep the unrefined pose and postopt=None."""
    img, masks = _two_masks()
    pipe = _FakePipe(bundle_for=("obj_a",))     # only the first object bundles
    b = _backend_with(pipe)

    def fake_runner(path):
        return {"rotation": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),  # (1,1,4) as the GS fn emits
                "translation": torch.tensor([[0.5, 0.0, 0.0]]),
                "scale": torch.ones(1, 3),
                "iou": 0.91, "iou_before_optim": 0.62, "optim_accepted": True}

    monkeypatch.setattr("lens.recon._run_deferred_postopt", fake_runner)
    out = b.reconstruct(img, masks)

    ca = out["obj_a"].mesh.vertices.mean(axis=0)
    cb = out["obj_b"].mesh.vertices.mean(axis=0)
    assert ca[0] - cb[0] == pytest.approx(0.5, abs=1e-5)
    assert out["obj_a"].postopt == {"iou": 0.91, "iou_before_optim": 0.62,
                                    "optim_accepted": True}
    assert out["obj_b"].postopt is None


@requires_torch
def test_defer_mode_isolates_per_object_replay_failure(monkeypatch):
    """One bad bundle must not kill the scene (2026-08-13 preflight review):
    the failed object keeps its unrefined pose and records WHY in postopt;
    later objects are still replayed and refined."""
    img, masks = _two_masks()
    pipe = _FakePipe(bundle_for=("obj_a", "obj_b"))
    b = _backend_with(pipe)
    calls = []

    def flaky_runner(path):
        calls.append(path)
        if len(calls) == 1:                     # obj_a replays first and dies
            raise RuntimeError("corrupt bundle")
        return {"rotation": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
                "translation": torch.tensor([[0.5, 0.0, 0.0]]),
                "scale": torch.ones(1, 3),
                "iou": 0.91, "iou_before_optim": 0.62, "optim_accepted": True}

    monkeypatch.setattr("lens.recon._run_deferred_postopt", flaky_runner)
    out = b.reconstruct(img, masks)             # must NOT raise

    assert len(calls) == 2                      # obj_b's replay still ran
    assert "corrupt bundle" in out["obj_a"].postopt["error"]
    assert out["obj_b"].postopt == {"iou": 0.91, "iou_before_optim": 0.62,
                                    "optim_accepted": True}


@requires_torch
def test_defer_mode_releases_models_before_alignment(monkeypatch):
    """The whole point of deferral: the model stack is gone before the first
    alignment runs."""
    img, masks = _two_masks()
    order = []
    pipe = _FakePipe()
    b = _backend_with(pipe)
    real_release = b.release
    b.release = lambda: (order.append("release"), real_release())[1]

    def fake_runner(path):
        order.append("align")
        return {"rotation": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
                "translation": torch.zeros(1, 3), "scale": torch.ones(1, 3),
                "iou": 0.9, "iou_before_optim": 0.9, "optim_accepted": False}

    monkeypatch.setattr("lens.recon._run_deferred_postopt", fake_runner)
    b.reconstruct(img, masks)
    assert order and order[0] == "release" and "align" in order


@requires_torch
def test_skip_postopt_env_restores_old_behavior(monkeypatch):
    """LENS_SKIP_POSTOPT=1 must reproduce today's pipeline exactly: post-opt
    never requested, no bundles, no runner, postopt=None."""
    monkeypatch.setenv("LENS_SKIP_POSTOPT", "1")
    img, masks = _two_masks()
    pipe = _FakePipe()
    b = Sam3dObjectsBackend(low_vram=False)     # ctor reads the env
    b._pipe = pipe
    b._load = lambda: None

    called = []
    monkeypatch.setattr("lens.recon._run_deferred_postopt",
                        lambda p: called.append(p))
    out = b.reconstruct(img, masks)
    assert pipe.calls == [False, False]
    assert called == []
    assert all(r.postopt is None for r in out.values())


@requires_torch
def test_defer_mode_is_default_and_cleans_bundle_dir(monkeypatch):
    """Post-opt (deferred) is the default now, and the bundle tmpdir does not
    outlive reconstruct()."""
    img, masks = _two_masks()
    seen = {}

    def fake_runner(path):
        seen["dir"] = os.path.dirname(path)
        return {"rotation": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
                "translation": torch.zeros(1, 3), "scale": torch.ones(1, 3),
                "iou": 0.9, "iou_before_optim": 0.9, "optim_accepted": False}

    monkeypatch.setattr("lens.recon._run_deferred_postopt", fake_runner)
    b = Sam3dObjectsBackend(low_vram=False)     # no args: defaults
    b._pipe = _FakePipe()
    b._load = lambda: None
    b.reconstruct(img, masks)
    assert "dir" in seen, "default ctor must request+defer post-opt"
    assert not os.path.exists(seen["dir"])
    assert "SAM3D_DEFER_LAYOUT_POSTOPT_DIR" not in os.environ
