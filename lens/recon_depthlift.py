# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Gravity-frame recovery for the CPU reconstruction tier (§4 of
docs/design/cpu-depthlift-design.md): given a metric point map and its surface
normals, vote for world-up and return the camera→world rotation. This is the
one stage of that tier that is built; the depth model, per-object visible-point
extraction and the shape providers are specified in the design document and not
yet written.
Frames: camera is OpenCV (x right, y down, z forward, up = −y); world is Z up
with X = horizontal projection of the camera's forward and the camera at the
origin (grounding downstream puts the floor at z = 0)."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

CAMERA_UP = np.array([0.0, -1.0, 0.0])
# The angle between camera-up and world-up IS the camera's downward pitch, so a
# cone anchored to camera-up is a precondition on pitch. The 45° cone handles
# ordinary captures; the 75° cone is tried only when that finds too little, so a
# steeply pitched tabletop capture still recovers gravity instead of silently
# returning the camera frame.
UP_CANDIDATE_COS = float(np.cos(np.radians(45.0)))
UP_WIDE_CANDIDATE_COS = float(np.cos(np.radians(75.0)))
UP_REFINE_COS = float(np.cos(np.radians(15.0)))   # each refine pass keeps normals within 15° of the running estimate
UP_REFINE_PASSES = 3
MIN_UP_CANDIDATES = 100          # absolute floor, meaningful on a small render
MIN_UP_CANDIDATE_FRAC = 0.01     # …and 1% of the valid pixels, meaningful on a photo


def _weighted_mean_dir(n: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """Unit mean direction, or None when the vote cancels to nothing. float64
    throughout: the caller's normals are float32, and a frame built from them
    is orthonormal only to ~1e-7."""
    if len(n) == 0:
        return None
    v = (n.astype(np.float64) * w.astype(np.float64)[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return v / norm


def _frame_from_up(up: np.ndarray) -> np.ndarray:
    """Rows = world X, Y, Z in camera coordinates, so `R @ p_cam` is a world
    point. X is the camera's forward (+z) with its up-component removed; if the
    camera looks straight up or down that projection vanishes and the camera's
    right (+x) is used instead. Raises on a degenerate direction rather than
    manufacturing one — a silently non-orthonormal R corrupts every scene."""
    up = np.asarray(up, np.float64)
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"_frame_from_up needs a non-degenerate up direction, got {up!r}")
    up = up / norm
    fwd = np.array([0.0, 0.0, 1.0])
    x = fwd - up * float(fwd @ up)
    if np.linalg.norm(x) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
        x = right - up * float(right @ up)
    x = x / np.linalg.norm(x)
    y = np.cross(up, x)
    return np.stack([x, y, up])


def gravity_rotation(points: np.ndarray, normals: np.ndarray,
                     valid: np.ndarray) -> tuple[np.ndarray, str]:
    """Estimate the up direction from the depth map's own horizontal surfaces
    and return (R, up_source). Each pixel's vote is weighted by depth² — a
    pixel covers a fixed solid angle, so far surfaces would otherwise be
    under-counted. up_source is "normals" or "camera_up_fallback"; R is always
    a proper rotation."""
    n = normals[valid]
    p = points[valid]
    if len(n) == 0:
        log.warning("gravity: no valid depth pixels; using camera-up")
        return _frame_from_up(CAMERA_UP), "camera_up_fallback"
    w = np.maximum(p[:, 2].astype(np.float64), 1e-3) ** 2
    need = max(MIN_UP_CANDIDATES, int(MIN_UP_CANDIDATE_FRAC * len(n)))
    scores = n.astype(np.float64) @ CAMERA_UP
    cand = scores > UP_CANDIDATE_COS
    if int(cand.sum()) < need:
        wide = scores > UP_WIDE_CANDIDATE_COS
        if int(wide.sum()) >= need:
            log.info("gravity: %d up-facing pixels within 45°, %d within 75° — "
                     "steeply pitched capture, using the wide cone",
                     int(cand.sum()), int(wide.sum()))
            cand = wide
    if int(cand.sum()) < need:
        log.warning("gravity: %d up-facing pixels of %d valid (need %d); using camera-up",
                    int(cand.sum()), len(n), need)
        return _frame_from_up(CAMERA_UP), "camera_up_fallback"
    up = _weighted_mean_dir(n[cand], w[cand])
    if up is None:
        log.warning("gravity: the candidate normals cancelled out; using camera-up")
        return _frame_from_up(CAMERA_UP), "camera_up_fallback"
    for _ in range(UP_REFINE_PASSES):
        keep = (n.astype(np.float64) @ up) > UP_REFINE_COS
        refined = _weighted_mean_dir(n[keep], w[keep])
        if refined is None:
            # A bimodal vote (two clusters either side of true up) empties the
            # refine cone. The running estimate is already their mean — keep it.
            break
        converged = float(refined @ up) > 1.0 - 1e-12
        up = refined
        if converged:
            break
    return _frame_from_up(up), "normals"
