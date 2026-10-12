# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Compare two exported MJCF scenes by re-simulating both and rendering them.

    python tools/settle_compare.py A.xml B.xml --photo p.jpg --out-prefix out/compare
                                   [--label-a "scene A"] [--label-b "scene B"]

Two things this exists to get right, each of which has already produced a false
conclusion when done the obvious way:

* **COM, not body-frame origin.** Our MJCFs bake geometry in world coordinates
  and put every body origin at (0,0,0), so the origin sits metres from the mesh
  it belongs to. Displacing that origin measures the lever arm, not the object:
  a 90 deg tip of an object 2 m out reads as ~2.8 m of "drift". `data.xipos` is
  the centre of mass and is what actually moved.
* **One camera, pinned across every panel.** MuJoCo's default free camera
  auto-frames per model, so two different scenes both fill the frame and look
  alike. The camera here is derived once from scene A and reused unchanged.

Run under a display: xvfb-run -a -s "-screen 0 1400x1000x24" env MUJOCO_GL=glx ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

W, H = 700, 500


def com_drift(xml: str, seconds: float) -> dict[str, float]:
    """Per-body centre-of-mass displacement, in metres, after `seconds` of sim."""
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com0 = data.xipos[1:].copy()
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(1, model.nbody)]
    return dict(zip(names, np.linalg.norm(data.xipos[1:] - com0, axis=1)))


def _scene_bounds(model, data):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        p, r = data.geom_xpos[g], model.geom_rbound[g]
        lo = np.minimum(lo, p - r)
        hi = np.maximum(hi, p + r)
    return lo, hi


def _camera(xml: str, azimuth: float, elevation: float):
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lo, hi = _scene_bounds(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (lo + hi) / 2.0
    cam.distance = float(np.linalg.norm(hi - lo)) * 1.15 + 0.5
    cam.azimuth, cam.elevation = azimuth, elevation
    return cam


def render_pair(xml: str, label: str, cam, settle: float) -> list[Image.Image]:
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1  # visual geoms only; group 3 is the collision hulls
    ren = mujoco.Renderer(model, height=H, width=W)
    out = []
    for caption, seconds in [("t=0 (as delivered)", 0.0), (f"t={settle}s settled", settle)]:
        for _ in range(int(round(seconds / model.opt.timestep))):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        ren.update_scene(data, camera=cam, scene_option=opt)
        img = Image.fromarray(ren.render())
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 28], fill=(0, 0, 0))
        d.text((8, 6), f"{label}  |  {caption}", fill=(255, 255, 255))
        out.append(img)
    ren.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xml_a")
    ap.add_argument("xml_b")
    ap.add_argument("--photo", required=True, help="Input photo, shown as the left column.")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--label-a", default="scene A")
    ap.add_argument("--label-b", default="scene B")
    ap.add_argument("--settle", type=float, default=1.5, help="Settle time rendered (s).")
    ap.add_argument("--also-report", type=float, default=3.0,
                    help="Second settle time reported in the table only (s).")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    args = ap.parse_args()

    cam = _camera(args.xml_a, args.azimuth, args.elevation)
    print(f"[cam] lookat={np.round(cam.lookat, 3).tolist()} dist={cam.distance:.2f} "
          f"(pinned across ALL panels)")

    for label, xml in [(args.label_a, args.xml_a), (args.label_b, args.xml_b)]:
        for sec in (args.settle, args.also_report):
            print(f"=== {label}: COM displacement over {sec}s ===")
            for name, val in sorted(com_drift(xml, sec).items(), key=lambda kv: -kv[1]):
                print(f"  {name:48s} {val * 100:8.2f} cm")

    photo = Image.open(args.photo).convert("RGB")
    photo.thumbnail((W, H))
    panel = Image.new("RGB", (W, H), (20, 20, 20))
    panel.paste(photo, ((W - photo.width) // 2, (H - photo.height) // 2))
    d = ImageDraw.Draw(panel)
    d.rectangle([0, 0, W, 28], fill=(0, 0, 0))
    d.text((8, 6), "input photo", fill=(255, 255, 255))

    rows = [render_pair(args.xml_a, args.label_a, cam, args.settle),
            render_pair(args.xml_b, args.label_b, cam, args.settle)]
    grid = Image.new("RGB", (W * 3, H * 2))
    for r, pair in enumerate(rows):
        grid.paste(panel, (0, H * r))
        grid.paste(pair[0], (W, H * r))
        grid.paste(pair[1], (W * 2, H * r))

    out_png = Path(f"{args.out_prefix}_montage.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)
    print(f"[montage] {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
