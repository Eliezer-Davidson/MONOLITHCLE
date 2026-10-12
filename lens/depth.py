# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Monocular metric depth for the CPU tier (§4 of
docs/design/cpu-depthlift-design.md). Frame contract: OpenCV camera — x right,
y down, z forward; camera-up is −y. Normals point toward free space, i.e.
toward the camera for every visible surface."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import trimesh

log = logging.getLogger(__name__)


@dataclass
class DepthResult:
    points: np.ndarray       # (H, W, 3) float32, metric, camera frame
    valid: np.ndarray        # (H, W) bool
    normals: np.ndarray      # (H, W, 3) float32 unit, camera frame, toward the camera
    intrinsics: np.ndarray   # (3, 3) normalized: fx/W, fy/H, cx=cy=0.5


class DepthEstimator(Protocol):
    def estimate(self, image: np.ndarray) -> DepthResult: ...
    def release(self) -> None: ...


def normals_from_points(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unit normals by central differences of the point map. With x right and
    y down, (d/dx) × (d/dy) points AWAY from the camera, so the sign is flipped
    to point toward it. Zero where any neighbour is invalid."""
    H, W, _ = points.shape
    dx = np.zeros_like(points); dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1, :] = points[2:, :] - points[:-2, :]
    ok = np.zeros((H, W), bool)
    ok[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
                      & valid[2:, 1:-1] & valid[:-2, 1:-1])
    n = -np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok &= norm[..., 0] > 1e-12
    n = np.where(ok[..., None], n / np.where(norm > 0, norm, 1.0), 0.0)
    return n.astype(np.float32)


def face_camera(normals: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Flip any normal with a positive component along its own view ray so
    every normal points toward the camera (n · p < 0)."""
    away = (normals * points).sum(-1, keepdims=True) > 0
    return np.where(away, -normals, normals).astype(np.float32)


def camera_pose(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """(4, 4) world-from-camera for an OpenCV camera at `eye` looking at `target`."""
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)          # right
    y = np.cross(z, x)                                   # down
    P = np.eye(4); P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, y, z, eye
    return P


class FakeDepth:
    """Exact depth by ray casting named meshes from a pinhole camera. For tests
    only: ignores the image content, uses its own scene at the requested size."""
    def __init__(self, meshes: dict[str, trimesh.Trimesh], pose: np.ndarray,
                 fov_x_deg: float = 60.0, size: tuple[int, int] = (64, 64)):
        self.names = list(meshes.keys())
        parts = [meshes[n] for n in self.names]
        self._face_owner = np.concatenate(
            [np.full(len(m.faces), i, np.int32) for i, m in enumerate(parts)])
        # Merge by hand with process=False: trimesh's default processing merges
        # vertices and can drop faces, which would break face -> owner indexing.
        verts, faces, off = [], [], 0
        for m in parts:
            verts.append(np.asarray(m.vertices, float)); faces.append(np.asarray(m.faces) + off)
            off += len(m.vertices)
        self._mesh = trimesh.Trimesh(vertices=np.vstack(verts), faces=np.vstack(faces), process=False)
        assert len(self._mesh.faces) == len(self._face_owner)
        self._pose = np.asarray(pose, float)
        self._fov_x = float(fov_x_deg)
        self._size = tuple(size)
        self.labels: np.ndarray | None = None

    def estimate(self, image: np.ndarray) -> DepthResult:
        H, W = self._size
        fx = (W / 2.0) / np.tan(np.radians(self._fov_x) / 2.0); fy = fx
        u = (np.arange(W) + 0.5 - W / 2.0) / fx
        v = (np.arange(H) + 0.5 - H / 2.0) / fy
        uu, vv = np.meshgrid(u, v)
        d_cam = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3)
        Rwc, t = self._pose[:3, :3], self._pose[:3, 3]
        d_world = d_cam @ Rwc.T
        origins = np.tile(t, (len(d_world), 1))
        loc, ray_idx, tri = self._mesh.ray.intersects_location(
            origins, d_world, multiple_hits=False)
        points = np.zeros((H * W, 3), np.float32); valid = np.zeros(H * W, bool)
        normals = np.zeros((H * W, 3), np.float32); labels = np.full(H * W, -1, np.int32)
        if len(ray_idx):
            points[ray_idx] = ((loc - t) @ Rwc).astype(np.float32)     # world -> camera
            valid[ray_idx] = True
            n_world = self._mesh.face_normals[tri]
            normals[ray_idx] = (n_world @ Rwc).astype(np.float32)
            labels[ray_idx] = self._face_owner[tri]
        points = points.reshape(H, W, 3); valid = valid.reshape(H, W)
        normals = face_camera(normals.reshape(H, W, 3), points)
        self.labels = labels.reshape(H, W)
        K = np.array([[fx / W, 0.0, 0.5], [0.0, fy / H, 0.5], [0.0, 0.0, 1.0]], np.float32)
        return DepthResult(points=points, valid=valid, normals=normals, intrinsics=K)

    def release(self) -> None:
        pass
