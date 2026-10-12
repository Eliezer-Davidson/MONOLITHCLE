import numpy as np
import pytest
import trimesh
from lens.depth import FakeDepth, camera_pose
from lens.recon_depthlift import (CAMERA_UP, MIN_UP_CANDIDATES, UP_CANDIDATE_COS,
                                  UP_REFINE_COS, UP_WIDE_CANDIDATE_COS,
                                  _frame_from_up, gravity_rotation)


def _scene():
    floor = trimesh.creation.box(extents=(10.0, 10.0, 0.02))
    floor.apply_translation((0.0, 0.0, -0.01))
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.5))
    box.apply_translation((0.0, 0.0, 0.25))
    return {"floor": floor, "box": box}


def _depth(scene, eye, target, size=(48, 64)):
    P = camera_pose(eye=eye, target=target)
    fd = FakeDepth(scene, P, fov_x_deg=60.0, size=size)
    dr = fd.estimate(np.zeros((size[0], size[1], 3), np.uint8))
    return P, fd, dr


def _true_up_cam(P):
    """World +Z expressed in camera coordinates."""
    return P[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def _err_deg(R, up_cam):
    return float(np.degrees(np.arccos(np.clip((R @ up_cam)[2], -1.0, 1.0))))


def _assert_proper_rotation(R, atol=1e-12):
    assert R.dtype == np.float64
    assert np.allclose(R @ R.T, np.eye(3), atol=atol)
    assert abs(float(np.linalg.det(R)) - 1.0) < atol


def _clusters(specs):
    """(points, normals, valid) arrays carrying, for each (normal, depth,
    count) spec, that many pixels with that exact normal at that depth — lets
    a test name the exact vote it wants, with no rendering involved."""
    ns, ps = [], []
    for normal, depth, count in specs:
        v = np.asarray(normal, np.float32)
        v = v / np.linalg.norm(v)
        ns.append(np.repeat(v[None], count, axis=0))
        ps.append(np.tile(np.array([0.0, 0.0, depth], np.float32), (count, 1)))
    n = np.concatenate(ns)
    p = np.concatenate(ps)
    side = int(np.ceil(np.sqrt(len(n))))
    pts = np.zeros((side, side, 3), np.float32)
    nrm = np.zeros((side, side, 3), np.float32)
    valid = np.zeros((side, side), bool)
    pts.reshape(-1, 3)[: len(n)] = p
    nrm.reshape(-1, 3)[: len(n)] = n
    valid.reshape(-1)[: len(n)] = True
    return pts, nrm, valid


def test_gravity_rotation_recovers_up_from_tilted_camera():
    P, fd, dr = _depth(_scene(), eye=(-2.0, 0.3, 1.2), target=(0.0, 0.0, 0.25))
    R, source = gravity_rotation(dr.points, dr.normals, dr.valid)
    assert source == "normals"
    assert _err_deg(R, _true_up_cam(P)) < 1.0
    fwd_world = R @ np.array([0.0, 0.0, 1.0])              # camera forward, in world coords
    assert fwd_world[0] > 0.8 and abs(fwd_world[1]) < 0.3  # X = horizontal projection of forward
    # the recovered pitch itself, which the loose bound above does not pin
    assert abs(fwd_world[2] + 0.4252) < 0.02
    _assert_proper_rotation(R)


def test_gravity_rotation_falls_back_to_camera_up_without_horizontal_surfaces():
    wall = trimesh.creation.box(extents=(0.05, 6.0, 4.0))
    wall.apply_translation((3.0, 0.0, 1.5))                 # fronto-parallel wall, no floor
    P, fd, dr = _depth({"wall": wall}, eye=(0.0, 0.0, 1.5), target=(3.0, 0.0, 1.5))
    R, source = gravity_rotation(dr.points, dr.normals, dr.valid)
    assert source == "camera_up_fallback"
    assert np.allclose(R @ CAMERA_UP, [0.0, 0.0, 1.0], atol=1e-12)
    _assert_proper_rotation(R)


def test_gravity_rotation_falls_back_when_no_pixel_is_valid():
    P, fd, dr = _depth(_scene(), eye=(-2.0, 0.0, 1.2), target=(0.0, 0.0, 0.25))
    R, source = gravity_rotation(dr.points, dr.normals, np.zeros_like(dr.valid))
    assert source == "camera_up_fallback"
    _assert_proper_rotation(R)


def test_gravity_rotation_survives_a_bimodal_vote_that_empties_the_refine_set():
    """Two up-facing clusters 30 deg either side of camera-up: both clear the
    candidate cone, their mean is true up, and BOTH are outside the 15 deg
    refine cone — so the refine set is empty. The first draft returned a
    singular matrix here, labelled up_source="normals"."""
    c, s = float(np.cos(np.radians(30.0))), float(np.sin(np.radians(30.0)))
    pts, nrm, valid = _clusters([([0.0, -c, -s], 3.0, 400), ([0.0, -c, s], 3.0, 400)])
    R, source = gravity_rotation(pts, nrm, valid)
    _assert_proper_rotation(R)                              # NOT det 0
    assert source == "normals"
    assert _err_deg(R, CAMERA_UP) < 1.0                     # the symmetric mean IS camera-up
    # and the refine cone really was empty for this input
    n = nrm[valid].astype(np.float64)
    assert int(((n @ CAMERA_UP) > UP_REFINE_COS).sum()) == 0


def test_gravity_rotation_recovers_up_from_a_steeply_pitched_camera():
    """A tabletop capture pitched ~60 deg down. Every floor normal is ~60 deg
    from camera-up, so the 45 deg cone finds nothing; the wide cone must."""
    P, fd, dr = _depth(_scene(), eye=(-0.9, 0.0, 1.8), target=(0.0, 0.0, 0.0), size=(64, 80))
    pitch = np.degrees(np.arccos(np.clip(_true_up_cam(P) @ CAMERA_UP, -1.0, 1.0)))
    assert pitch > 50.0                                     # the scene really is steep
    n = dr.normals[dr.valid].astype(np.float64)
    assert int(((n @ CAMERA_UP) > UP_CANDIDATE_COS).sum()) == 0   # the 45 deg gate found nothing
    R, source = gravity_rotation(dr.points, dr.normals, dr.valid)
    assert source == "normals"
    assert _err_deg(R, _true_up_cam(P)) < 1.5
    _assert_proper_rotation(R)


def test_a_visible_wall_never_enters_the_candidate_cone_even_on_the_wide_gate():
    """The wide 75 deg cone's one real risk is a vertical surface sneaking into
    the vote. It cannot: a wall face visible to the camera has its normal
    opposed to camera-up's horizontal component, at every pitch. Measured here
    at ~52 deg, where the wide gate IS the gate in use.
    (An earlier version of this test ran at 21.8 deg pitch, so the wide branch
    it was meant to guard was never even entered.)"""
    scene = _scene()
    scene["wall"] = trimesh.creation.box(extents=(0.05, 8.0, 3.0))
    scene["wall"].apply_translation((1.0, 0.0, 1.5))
    P, fd, dr = _depth(scene, eye=(-1.0, 0.0, 2.4), target=(0.8, 0.0, 0.1), size=(72, 88))
    pitch = np.degrees(np.arccos(np.clip(_true_up_cam(P) @ CAMERA_UP, -1.0, 1.0)))
    assert pitch > 50.0
    n = dr.normals[dr.valid].astype(np.float64)
    assert int(((n @ CAMERA_UP) > UP_CANDIDATE_COS).sum()) == 0   # so the wide gate is in use
    wall = (fd.labels == fd.names.index("wall")) & dr.valid
    assert int(wall.sum()) > 500                                  # the wall really is in frame
    wall_scores = dr.normals[wall].astype(np.float64) @ CAMERA_UP
    assert wall_scores.max() < 0.0                                # opposed to camera-up …
    assert wall_scores.max() < UP_WIDE_CANDIDATE_COS              # … so none can enter the cone
    R, source = gravity_rotation(dr.points, dr.normals, dr.valid)
    assert source == "normals"
    assert _err_deg(R, _true_up_cam(P)) < 1.0
    _assert_proper_rotation(R)


def test_the_refine_pass_rejects_a_competing_tilted_cluster():
    """Exercises the refine loop for real: 30% of the VALID pixels are given a
    normal 35 deg off true up — inside the 45 deg candidate cone, outside the
    15 deg refine cone. The first-pass mean is dragged ~10.7 deg off true up;
    the refine pass is what brings it back.
    (An earlier version injected the distractor into image rows that held no
    valid pixels at all, so the line was dead code and proved nothing.)"""
    rng = np.random.default_rng(0)
    P, fd, dr = _depth(_scene(), eye=(-2.0, 0.0, 1.3), target=(0.0, 0.0, 0.3), size=(64, 80))
    n = dr.normals.copy()
    n = n + rng.normal(scale=np.radians(3.0), size=n.shape).astype(np.float32)
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(norms > 1e-6, n / np.maximum(norms, 1e-6), n)
    up_cam = _true_up_cam(P).astype(np.float32)
    axis = np.cross(up_cam, np.array([0.0, 0.0, 1.0], np.float32))
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], np.float32)
    tilt = np.radians(35.0)
    Rt = np.eye(3, dtype=np.float32) + np.sin(tilt) * K + (1 - np.cos(tilt)) * (K @ K)
    distractor = (Rt @ up_cam).astype(np.float32)
    idx = np.argwhere(dr.valid)
    chosen = idx[rng.permutation(len(idx))[: int(0.3 * len(idx))]]
    assert len(chosen) > 500                                 # the injection is not a no-op
    n[chosen[:, 0], chosen[:, 1]] = distractor
    R, source = gravity_rotation(dr.points, n, dr.valid)
    assert source == "normals"
    assert _err_deg(R, _true_up_cam(P)) < 1.5
    _assert_proper_rotation(R)


def test_depth_squared_weighting_prefers_the_far_surface_over_a_near_tilted_one():
    """A pixel covers a fixed solid angle, so a far surface's pixels each stand
    for more real area. Here a NEAR tilted patch carries 4x the pixel count of
    a FAR floor; only depth-squared weighting gets the floor right — uniform
    weighting lands 20 deg out."""
    tilt = np.radians(20.0)
    tilted = [0.0, -float(np.cos(tilt)), -float(np.sin(tilt))]
    pts, nrm, valid = _clusters([(tilted, 1.0, 4000), (list(CAMERA_UP), 5.0, 1000)])
    R, source = gravity_rotation(pts, nrm, valid)
    assert source == "normals"
    assert _err_deg(R, CAMERA_UP) < 1.5
    _assert_proper_rotation(R)


def test_frame_from_up_renormalizes_a_non_unit_direction():
    """The orthonormality guard, covered directly. A float32 up direction is
    unit only to ~1e-7, and the first draft stacked it into R unnormalised — which
    is why its determinant measured 0.999999998. `_frame_from_up` renormalizes
    in float64, so R is orthonormal whatever the caller hands it."""
    up = (np.array([0.0, -1.0, 0.0], np.float32) * np.float32(1.0000001)).astype(np.float32)
    assert abs(float(np.linalg.norm(up.astype(np.float64))) - 1.0) > 1e-8   # genuinely non-unit
    _assert_proper_rotation(_frame_from_up(up))
    _assert_proper_rotation(_frame_from_up(np.array([0.0, -3.0, 0.0])))     # any length
    with pytest.raises(ValueError):
        _frame_from_up(np.zeros(3))


def test_candidate_gate_is_relative_to_the_valid_pixel_count():
    """A handful of up-facing pixels in a big image must not pass the gate."""
    up = [0.0, -1.0, 0.0]
    side = [0.0, 0.0, -1.0]
    n_up, n_side = MIN_UP_CANDIDATES + 20, 40_000
    pts, nrm, valid = _clusters([(up, 3.0, n_up)])
    R, source = gravity_rotation(pts, nrm, valid)
    assert source == "normals"                               # 100% of a small image: fine
    pts2, nrm2, val2 = _clusters([(up, 3.0, n_up), (side, 3.0, n_side)])
    R2, source2 = gravity_rotation(pts2, nrm2, val2)
    assert source2 == "camera_up_fallback"                   # 0.3% of a big image: not enough
    _assert_proper_rotation(R2)
