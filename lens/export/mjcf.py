# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""SceneSpec -> MuJoCo MJCF exporter.

Per-object conventions:
  - visual geoms: group 2, contype/conaffinity 0, mass 0 (no physics
    contribution — the mesh is for rendering only).
  - collision geoms: group 3, CoACD convex hulls (see `lens.export.collision`),
    friction "<ObjectSpec.friction> 0.005 0.0001" (sliding component from the
    spec; torsional/rolling components fixed).
  - one freejoint per body (objects are free rigid bodies, not welded).
  - a ground-plane floor at `spec.floor_z` (fixed friction, independent of
    any object).
  - `<visual><global offwidth="1920" offheight="1080"/></visual>` so
    offscreen rendering above MuJoCo's 640px default works.

Asset naming: every asset name (and every copied/linked asset filename) is
derived from the owning `ObjectSpec.id`, not from the mesh file's basename.
SceneSpec only guarantees unique object ids — two objects' mesh files can
share a basename (e.g. `objA/mesh.obj` and `objB/mesh.obj`), and naming
assets from the bare stem would collide (`<mesh name="mesh">` twice, which
MuJoCo rejects, plus a silently-reused asset file).

Mass handling: `ObjectSpec.mass` is authoritative. MuJoCo computes body mass
from geom density * geom volume when no explicit geom mass is given, so each
body's collision geoms all get the same density:

    density = mass / sum(volume of that body's collision hulls)

which makes the compiled body mass equal `ObjectSpec.mass` (visual geoms
contribute 0 mass, so they don't skew this). This holds regardless of how
many collision hulls a body has.

XML values interpolated from spec data (object ids, the scene name,
filenames) are escaped via `xml.sax.saxutils.quoteattr` — these come from a
vision-model-driven naming stage, so characters like `"`, `<`, `>`, `&` are
plausible and must not produce malformed XML.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from xml.sax.saxutils import quoteattr

import trimesh

from lens.scenespec import ObjectSpec, SceneSpec


def _q(value: object) -> str:
    """Return a fully-quoted (quotes included) XML attribute value for
    `value`, escaping `"`, `<`, `>`, `&` as needed. Use in place of a raw
    f-string `"{value}"` for anything interpolated from spec data."""
    return quoteattr(str(value))


def _link_asset(src: Path, assets_dir: Path, dst_name: str) -> Path:
    """Make `src` available at `assets_dir / dst_name`, preferring a relative
    symlink (keeps the XML relocatable) and falling back to a copy on
    filesystems that refuse symlinks. `dst_name` is derived from the owning
    object's id (see module docstring) rather than `src`'s basename, so two
    objects whose mesh files share a basename never collide on the
    destination filename.

    Always re-points at `src`. Skipping an existing destination looks like a
    harmless optimization and is not: assets are normally symlinks, which stay
    live, but `shutil.copytree` DEREFERENCES symlinks into real files, so any
    workflow that seeds an output directory from a previous run inherits that
    run's geometry and never refreshes it. One 80-minute CEM run optimized
    against two-day-old colliders from a different hull budget -- 213-vertex
    pieces on disk, 93-vertex pieces actually loaded -- before this was caught.
    `dst` is a derived artifact, always reproducible from `src`, so replacing it
    unconditionally is safe and cheap."""
    dst = assets_dir / dst_name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, assets_dir), dst)
    except OSError:
        shutil.copy(src, dst)
    return dst


def _mean_vertex_rgb(path: Path) -> tuple[float, float, float] | None:
    """Mean per-vertex color from an OBJ's `v x y z r g b` lines (0-1 floats),
    or None when the file carries no vertex colors. MuJoCo does not render OBJ
    vertex colors (verified empirically on the held-out exports), so the exporter bakes
    the mean into the visual geom's rgba (a black camera comes out at roughly
    rgba="0.05 0.05 0.05"). A text scan, not a trimesh.load: visual meshes run to millions of vertices and
    only the color columns are needed."""
    if path.suffix.lower() != ".obj":
        return None
    n = 0
    acc = [0.0, 0.0, 0.0]
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 7:
                        acc[0] += float(parts[4])
                        acc[1] += float(parts[5])
                        acc[2] += float(parts[6])
                        n += 1
    except (OSError, ValueError):
        return None
    if n == 0:
        return None
    return tuple(min(1.0, max(0.0, c / n)) for c in acc)


def _hull_density(scene_dir: Path, o: ObjectSpec) -> tuple[list[Path], float]:
    """Resolve `o`'s collision mesh paths and compute the uniform per-geom
    density that makes the summed geom mass equal `o.mass`."""
    col_paths = [scene_dir / cm for cm in o.collision_meshes]
    total_volume = sum(abs(trimesh.load(p, force="mesh").volume) for p in col_paths)
    density = o.mass / total_volume if total_volume > 0 else 0.0
    return col_paths, density


def write_scene(spec: SceneSpec, scene_dir: Path) -> Path:
    """Assemble `spec`'s objects into a MuJoCo model at
    `<scene_dir>/mujoco/<spec.name>.xml`, with mesh assets linked/copied into
    `<scene_dir>/mujoco/assets/`. Mesh paths on `ObjectSpec` are relative to
    `scene_dir`. Returns the path to the written XML."""
    scene_dir = Path(scene_dir)
    mj_dir = scene_dir / "mujoco"
    assets_dir = mj_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # bodies: (ObjectSpec, visual_mesh_name, visual_dst,
    #          [(collision_mesh_name, collision_dst), ...], density, mean_rgb)
    bodies = []
    for o in spec.objects:
        vis_src = scene_dir / o.visual_mesh
        vis_name = f"{o.id}_visual"
        vis_dst = _link_asset(vis_src, assets_dir, vis_name + vis_src.suffix)

        col_paths, density = _hull_density(scene_dir, o)
        col_entries = []
        for i, p in enumerate(col_paths):
            col_name = f"{o.id}_col{i:02d}"
            col_dst = _link_asset(p, assets_dir, col_name + p.suffix)
            col_entries.append((col_name, col_dst))

        bodies.append((o, vis_name, vis_dst, col_entries, density,
                       _mean_vertex_rgb(vis_src)))

    L = []
    L.append(f'<mujoco model={_q(spec.name)}>')
    L.append('  <compiler meshdir="assets" angle="radian"/>')
    L.append('  <option timestep="0.002"/>')
    L.append('  <visual><global offwidth="1920" offheight="1080"/></visual>')
    L.append('  <default>')
    L.append('    <default class="visual">')
    L.append('      <geom type="mesh" contype="0" conaffinity="0" group="2" mass="0" rgba="1 1 1 1"/>')
    L.append('    </default>')
    L.append('    <default class="collision">')
    L.append('      <geom type="mesh" group="3" rgba="0.8 0.5 0.2 0.35"/>')
    L.append('    </default>')
    L.append('  </default>')
    L.append('  <asset>')
    L.append('    <texture type="skybox" builtin="gradient" rgb1="0.9 0.9 0.95" '
              'rgb2="0.6 0.65 0.75" width="64" height="64"/>')
    L.append('    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.62 0.62 0.60" '
              'rgb2="0.55 0.55 0.53" width="128" height="128"/>')
    L.append('    <material name="floor_mat" texture="floor_tex" texrepeat="8 8"/>')
    for o, vis_name, vis_dst, col_entries, _density, _rgb in bodies:
        L.append(f'    <mesh name={_q(vis_name)} file={_q(vis_dst.name)}/>')
        for col_name, col_dst in col_entries:
            L.append(f'    <mesh name={_q(col_name)} file={_q(col_dst.name)}/>')
    L.append('  </asset>')
    L.append('  <worldbody>')
    L.append('    <light pos="0 -1 3" dir="0 0.3 -1" directional="true"/>')
    L.append(f'    <geom name="floor" type="plane" pos="0 0 {spec.floor_z:.6f}" '
              'size="10 10 0.1" material="floor_mat" friction="0.9 0.005 0.0001"/>')
    for o, vis_name, vis_dst, col_entries, density, mean_rgb in bodies:
        px, py, pz = o.pos
        qw, qx, qy, qz = o.quat
        L.append(f'    <body name={_q(o.id)} pos="{px:.6f} {py:.6f} {pz:.6f}" '
                  f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">')
        L.append('      <freejoint/>')
        # Mean vertex color baked as the geom rgba; a
        # colorless mesh keeps the class default (white).
        if mean_rgb is not None:
            r, g, b = mean_rgb
            L.append(f'      <geom class="visual" mesh={_q(vis_name)} '
                      f'rgba="{r:.4f} {g:.4f} {b:.4f} 1"/>')
        else:
            L.append(f'      <geom class="visual" mesh={_q(vis_name)}/>')
        for col_name, _col_dst in col_entries:
            L.append(f'      <geom class="collision" mesh={_q(col_name)} '
                      f'density="{density:.6f}" friction="{o.friction:.6f} 0.005 0.0001"/>')
        L.append('    </body>')
    L.append('  </worldbody>')
    L.append('</mujoco>')

    xml_path = mj_dir / f"{spec.name}.xml"
    xml_path.write_text("\n".join(L))
    return xml_path
