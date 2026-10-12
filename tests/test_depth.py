import numpy as np
import trimesh
from lens.depth import (DepthResult, FakeDepth, camera_pose, face_camera,
                        normals_from_points)


def _scene():
    floor = trimesh.creation.box(extents=(10.0, 10.0, 0.02))
    floor.apply_translation((0.0, 0.0, -0.01))            # top face at z=0
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.5))
    box.apply_translation((0.0, 0.0, 0.25))                # sits on the floor
    return {"floor": floor, "box": box}


def test_camera_pose_is_opencv_right_down_forward():
    P = camera_pose(eye=(-2.0, 0.0, 1.2), target=(0.0, 0.0, 0.25))
    Rwc = P[:3, :3]
    fwd = Rwc[:, 2]                                        # camera +z in world
    assert fwd[0] > 0.8 and fwd[2] < 0                     # looks toward +x and down
    assert np.allclose(np.cross(Rwc[:, 0], Rwc[:, 1]), Rwc[:, 2], atol=1e-9)
    assert Rwc[:, 1][2] < 0                                # camera +y (down) points to −Z world


def test_fake_depth_floor_pixels_land_on_z0_with_up_normals():
    P = camera_pose(eye=(-2.0, 0.0, 1.2), target=(0.0, 0.0, 0.25))
    fd = FakeDepth(_scene(), P, fov_x_deg=60.0, size=(48, 64))
    dr = fd.estimate(np.zeros((48, 64, 3), np.uint8))
    assert isinstance(dr, DepthResult) and dr.points.shape == (48, 64, 3)
    assert dr.valid.mean() > 0.5
    floor = (fd.labels == fd.names.index("floor")) & dr.valid
    assert floor.sum() > 100
    Rwc, t = P[:3, :3], P[:3, 3]
    world = dr.points[floor] @ Rwc.T + t
    assert np.allclose(world[:, 2], 0.0, atol=1e-6)
    up_cam = Rwc.T @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(dr.normals[floor], up_cam, atol=1e-6)
    assert np.allclose(dr.intrinsics[0, 2], 0.5) and np.allclose(dr.intrinsics[1, 2], 0.5)


def test_normals_from_points_floor_is_camera_up_and_wall_faces_camera():
    H, W = 32, 32
    u = (np.arange(W) + 0.5 - W / 2) / (W / 2)
    v = np.linspace(0.25, 1.0, H)                       # rays below the horizon only
    uu, vv = np.meshgrid(u, v)
    # floor 1 m below the camera: the ray (u, v, 1) hits y = 1 at z = 1/v, x = u/v
    floor = np.stack([uu / vv, np.ones_like(uu), 1.0 / vv], -1).astype(np.float32)
    n = normals_from_points(floor, np.ones((H, W), bool))
    inner = n[4:-4, 4:-4].reshape(-1, 3)
    assert np.allclose(inner, [0.0, -1.0, 0.0], atol=1e-3)         # up is −y
    wall = np.stack([uu * 3, vv * 3, np.full_like(uu, 3.0)], -1).astype(np.float32)
    n = normals_from_points(wall, np.ones((H, W), bool))
    assert np.allclose(n[4:-4, 4:-4].reshape(-1, 3), [0.0, 0.0, -1.0], atol=1e-3)


def test_face_camera_flips_normals_pointing_away():
    pts = np.array([[[0.0, 0.0, 2.0]]], np.float32)
    n = np.array([[[0.0, 0.0, 1.0]]], np.float32)             # away from the camera
    assert np.allclose(face_camera(n, pts), [[[0.0, 0.0, -1.0]]])
    n2 = np.array([[[0.0, 0.0, -1.0]]], np.float32)
    assert np.allclose(face_camera(n2, pts), n2)
