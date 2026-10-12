import trimesh
import mujoco
from pathlib import Path
from lens.export import collision, mjcf
from tests.test_scenespec import make_obj
from lens.scenespec import SceneSpec


def test_decompose_box_returns_hulls():
    hulls = collision.decompose(trimesh.creation.box(extents=(1, 1, 1)))
    assert 1 <= len(hulls) <= 32 and all(h.is_volume for h in hulls)


def test_write_scene_loads_in_mujoco(tmp_path: Path):
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    spec = SceneSpec(name="t", objects=[make_obj()], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.nbody - 1 == 1
    body = m.body("table_0")
    assert abs(m.body_mass[body.id] - 8.0) < 1e-3     # ObjectSpec.mass honored


def test_write_scene_creates_files_in_expected_layout(tmp_path: Path):
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    spec = SceneSpec(name="my_scene", objects=[make_obj()], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    assert xml == tmp_path / "mujoco" / "my_scene.xml"
    assert xml.exists()
    assert (tmp_path / "mujoco" / "assets").is_dir()


def test_write_scene_visual_and_collision_groups(tmp_path: Path):
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    spec = SceneSpec(name="t", objects=[make_obj()], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))

    groups = sorted(m.geom_group[m.geom_bodyid == m.body("table_0").id])
    assert groups == [2, 3]

    for gid in range(m.ngeom):
        if m.geom_group[gid] == 2:
            assert m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0
        if m.geom_group[gid] == 3:
            # friction "0.9 0.005 0.0001" is the exporter's fixed torsional/rolling convention
            assert abs(m.geom_friction[gid][0] - 0.9) < 1e-6
            assert abs(m.geom_friction[gid][1] - 0.005) < 1e-6
            assert abs(m.geom_friction[gid][2] - 0.0001) < 1e-6


def test_write_scene_has_freejoint_and_floor(tmp_path: Path):
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    spec = SceneSpec(name="t", objects=[make_obj()], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))

    assert m.body("table_0").jntnum[0] == 1
    assert m.jnt_type[m.body("table_0").jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE
    floor = m.geom("floor")
    assert m.geom_type[floor.id] == mujoco.mjtGeom.mjGEOM_PLANE


def test_write_scene_sets_offscreen_render_buffer(tmp_path: Path):
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    spec = SceneSpec(name="t", objects=[make_obj()], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.vis.global_.offwidth == 1920
    assert m.vis.global_.offheight == 1080


def test_write_scene_density_scales_with_mass(tmp_path: Path):
    """Doubling ObjectSpec.mass (same geometry) must double the resulting
    per-geom density so the body mass tracks the spec exactly."""
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")

    spec_light = SceneSpec(name="light", objects=[make_obj(mass=8.0)], capture={})
    xml_light = mjcf.write_scene(spec_light, tmp_path)
    m_light = mujoco.MjModel.from_xml_path(str(xml_light))
    assert abs(m_light.body_mass[m_light.body("table_0").id] - 8.0) < 1e-3

    spec_heavy = SceneSpec(name="heavy", objects=[make_obj(mass=16.0)], capture={})
    xml_heavy = mjcf.write_scene(spec_heavy, tmp_path)
    m_heavy = mujoco.MjModel.from_xml_path(str(xml_heavy))
    assert abs(m_heavy.body_mass[m_heavy.body("table_0").id] - 16.0) < 1e-3


def test_write_scene_multiple_objects_distinct_bodies(tmp_path: Path):
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")

    chair = trimesh.creation.box(extents=(0.4, 0.4, 0.8))
    chair.apply_translation((1.0, 0.0, 0.4))
    chair.export(tmp_path / "meshes/chair_1.obj")
    collision.decompose(chair)[0].export(tmp_path / "meshes/chair_1_col00.obj")

    obj2 = make_obj(id="chair_1", visual_mesh="meshes/chair_1.obj",
                     collision_meshes=["meshes/chair_1_col00.obj"],
                     pos=(1.0, 0.0, 0.0), mass=5.0)
    spec = SceneSpec(name="two", objects=[make_obj(), obj2], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.nbody - 1 == 2
    assert abs(m.body_mass[m.body("table_0").id] - 8.0) < 1e-3
    assert abs(m.body_mass[m.body("chair_1").id] - 5.0) < 1e-3


def test_write_scene_colliding_mesh_basenames(tmp_path: Path):
    """Regression: two objects whose mesh files share a basename in different
    directories (objA/mesh.obj vs objB/mesh.obj, and objA/mesh_col.obj vs
    objB/mesh_col.obj) must not collide on generated asset name or on-disk
    asset filename. Before the fix, asset names/filenames were derived from
    the bare file stem, so both objects produced a `<mesh name="mesh">`
    entry -> MuJoCo rejects the model with a repeated-name error, and the
    second object's asset-copy silently reused the first object's file."""
    box_a = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box_a.apply_translation((0, 0, 0.225))
    box_b = trimesh.creation.box(extents=(0.4, 0.4, 0.8))
    box_b.apply_translation((0, 0, 0.4))

    (tmp_path / "objA").mkdir()
    (tmp_path / "objB").mkdir()
    box_a.export(tmp_path / "objA/mesh.obj")
    box_b.export(tmp_path / "objB/mesh.obj")
    collision.decompose(box_a)[0].export(tmp_path / "objA/mesh_col.obj")
    collision.decompose(box_b)[0].export(tmp_path / "objB/mesh_col.obj")

    obj_a = make_obj(id="objA", visual_mesh="objA/mesh.obj",
                      collision_meshes=["objA/mesh_col.obj"], mass=8.0)
    obj_b = make_obj(id="objB", visual_mesh="objB/mesh.obj",
                      collision_meshes=["objB/mesh_col.obj"], mass=5.0,
                      pos=(1.0, 0.0, 0.0))
    spec = SceneSpec(name="collide", objects=[obj_a, obj_b], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))  # must not raise a repeated-name error

    assert m.nbody - 1 == 2
    # Correct mass per body proves each body got its OWN mesh geometry/volume,
    # not a reused/aliased asset from the other object.
    assert abs(m.body_mass[m.body("objA").id] - 8.0) < 1e-3
    assert abs(m.body_mass[m.body("objB").id] - 5.0) < 1e-3


def test_write_scene_escapes_special_characters_in_id(tmp_path: Path):
    """Regression: an object id containing a `"` must not break the generated
    XML. Names come from a vision-model-driven naming stage, so odd
    characters are plausible; the exporter must escape interpolated
    attribute values instead of relying on raw f-strings."""
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")

    obj = make_obj(id='bad"id', visual_mesh="meshes/table_0.obj",
                    collision_meshes=["meshes/table_0_col00.obj"])
    spec = SceneSpec(name="t", objects=[obj], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))  # must not raise an XML parse error

    assert m.nbody - 1 == 1
    body = m.body('bad"id')
    assert abs(m.body_mass[body.id] - 8.0) < 1e-3


def test_write_scene_object_friction_reaches_collision_geom(tmp_path: Path):
    """Regression: ObjectSpec.friction must reach the compiled model as the
    sliding-friction component of collision geoms, instead of being
    hardcoded to 0.9 regardless of the spec."""
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")

    obj = make_obj(friction=0.4)
    spec = SceneSpec(name="t", objects=[obj], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))

    body = m.body("table_0")
    collision_gids = [gid for gid in range(m.ngeom)
                       if m.geom_bodyid[gid] == body.id and m.geom_group[gid] == 3]
    assert collision_gids, "expected at least one collision geom"
    for gid in collision_gids:
        assert abs(m.geom_friction[gid][0] - 0.4) < 1e-6
        # torsional/rolling components unchanged
        assert abs(m.geom_friction[gid][1] - 0.005) < 1e-6
        assert abs(m.geom_friction[gid][2] - 0.0001) < 1e-6


def test_write_scene_multi_hull_mass_correct(tmp_path: Path):
    """Regression: an L-shaped concave object that genuinely decomposes into
    >=2 CoACD hulls must still produce a compiled body mass equal to
    ObjectSpec.mass. Every other test in this file uses a box, which CoACD
    returns as a single hull -- this is the only test that exercises the
    density formula (mass / sum of hull volumes) across multiple hulls."""
    base = trimesh.creation.box(extents=(1.0, 1.0, 0.3))
    base.apply_translation((0, 0, 0.15))
    tower = trimesh.creation.box(extents=(0.3, 0.3, 1.0))
    tower.apply_translation((0.35, 0.35, 0.5))  # overlaps one corner of `base`, rises above it
    l_shape = trimesh.util.concatenate([base, tower])
    assert l_shape.is_watertight

    hulls = collision.decompose(l_shape)
    assert len(hulls) >= 2, f"expected a genuine multi-hull decomposition, got {len(hulls)}"

    (tmp_path / "meshes").mkdir()
    l_shape.export(tmp_path / "meshes/lshape_0.obj")
    col_names = []
    for i, h in enumerate(hulls):
        name = f"meshes/lshape_0_col{i:02d}.obj"
        h.export(tmp_path / name)
        col_names.append(name)

    obj = make_obj(id="lshape_0", category="lshape",
                    visual_mesh="meshes/lshape_0.obj",
                    collision_meshes=col_names,
                    dims=(1.0, 1.0, 1.3), mass=12.0)
    spec = SceneSpec(name="lshape_scene", objects=[obj], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert abs(m.body_mass[m.body("lshape_0").id] - 12.0) < 1e-3


def test_write_scene_refreshes_a_stale_asset(tmp_path: Path):
    """`_link_asset` skipped any destination that already existed, so re-running
    a scene into a directory that already held `mujoco/assets/` silently kept
    the OLD geometry and the MJCF described a scene that no longer existed.

    Assets are normally symlinks, which stay live -- the bug needs them to be
    REAL FILES, which is exactly what `shutil.copytree` produces because it
    dereferences symlinks by default. `mkrun.py` builds a re-runnable scene that
    way, so an 80-minute CEM run optimized against collision meshes two days old
    and from a different hull budget (213-vertex pieces on disk, 93-vertex
    pieces actually loaded). Every settle figure it produced described the wrong
    scene.
    """
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    spec = SceneSpec(name="t", objects=[make_obj()], capture={})
    mjcf.write_scene(spec, tmp_path)

    asset = tmp_path / "mujoco/assets/table_0_col00.obj"
    stale = b"# stale geometry from a previous run\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    asset.unlink()                       # copytree turns the symlink into a file
    asset.write_bytes(stale)

    mjcf.write_scene(spec, tmp_path)

    live = (tmp_path / "meshes/table_0_col00.obj").read_bytes()
    assert asset.resolve().read_bytes() == live, (
        "asset still holds the previous run's geometry -- the MJCF describes a "
        "scene that is not the one on disk"
    )


def test_write_scene_bakes_mean_vertex_color_into_visual_rgba(tmp_path: Path):
    """MuJoCo does not render OBJ vertex colors, so every exported
    scene rendered white. The fix is per-geom rgba baked from the mesh's mean
    vertex color. A
    uniformly (51,102,204)-colored box must yield rgba ≈ (0.2, 0.4, 0.8, 1)
    on its VISUAL geom."""
    import numpy as np
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    box.visual.vertex_colors = np.tile([51, 102, 204, 255], (len(box.vertices), 1))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    spec = SceneSpec(name="t", objects=[make_obj()], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))

    body_id = m.body("table_0").id
    vis = [g for g in range(m.ngeom)
           if m.geom_bodyid[g] == body_id and m.geom_group[g] == 2]
    assert len(vis) == 1
    rgba = m.geom_rgba[vis[0]]
    np.testing.assert_allclose(rgba, [0.2, 0.4, 0.8, 1.0], atol=0.02)


def test_write_scene_colorless_visual_mesh_stays_white(tmp_path: Path):
    """An OBJ with no vertex colors keeps the class default rgba (white) —
    no invented color."""
    import numpy as np
    box = trimesh.creation.box(extents=(0.6, 0.6, 0.45))
    box.apply_translation((0, 0, 0.225))
    (tmp_path / "meshes").mkdir()
    # export WITHOUT colors: plain geometry
    trimesh.Trimesh(vertices=box.vertices, faces=box.faces, process=False
                    ).export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    spec = SceneSpec(name="t", objects=[make_obj()], capture={})
    xml = mjcf.write_scene(spec, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))
    body_id = m.body("table_0").id
    vis = [g for g in range(m.ngeom)
           if m.geom_bodyid[g] == body_id and m.geom_group[g] == 2]
    np.testing.assert_allclose(m.geom_rgba[vis[0]], [1, 1, 1, 1], atol=1e-6)
