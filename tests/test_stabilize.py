import mujoco
import numpy as np
import pytest
import trimesh
from pathlib import Path
from lens.export import collision, mjcf
from lens.scenespec import SceneSpec
from lens.stabilize import FaithfulCEM, _qpos_adr, settle, stabilize_scene
from tests.test_scenespec import make_obj


def _scene(tmp_path, z0):
    """A crate whose GEOMETRY is baked with its bottom face at `z0`, mirroring
    the real pipeline convention: `lens.pipeline.assemble_scene` always writes
    `ObjectSpec.pos=(0.0, 0.0, 0.0)` and bakes world position into the mesh
    vertices. Tests that instead put the height in `pos` exercise the inverse
    of what the pipeline produces and can pass while the shipped scene is
    broken."""
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    box.apply_translation((0, 0, 0.1 + z0))          # bottom face at z0
    (tmp_path / "meshes").mkdir(exist_ok=True)
    box.export(tmp_path / "meshes/crate_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/crate_0_col00.obj")
    o = make_obj(id="crate_0", visual_mesh="meshes/crate_0.obj",
                 collision_meshes=["meshes/crate_0_col00.obj"],
                 pos=(0.0, 0.0, 0.0),
                 dims=(0.2, 0.2, 0.2), mass=1.0)
    return SceneSpec(name="t", objects=[o], capture={})


# --- restored regression tests (deleted in the 2026-07-29 CEM rewrite) -------

def test_settle_reports_fall_of_floating_box(tmp_path: Path):
    xml = mjcf.write_scene(_scene(tmp_path, z0=0.30), tmp_path)
    rep = settle(xml, seconds=1.5)
    assert 0.25 < rep.drift["crate_0"] < 0.35        # fell ~30 cm onto the floor


def test_settle_measures_penetration_after_physics_not_before(tmp_path: Path):
    """Regression test for the penetration-timing bug: the box starts embedded
    5 cm below the floor (deep initial overlap purely from geometry -- no
    physics has run yet), which physics resolves within the settle window. If
    max_penetration were measured before physics ran it would report ~0.05 m;
    measured after settling it must be a small residual."""
    xml = mjcf.write_scene(_scene(tmp_path, z0=-0.05), tmp_path)
    rep = settle(xml, seconds=1.5)
    assert rep.max_penetration < 0.01, (
        f"max_penetration={rep.max_penetration} m -- looks like the initial "
        "~0.05 m overlap, not the post-settle residual"
    )


# --- the two bugs being fixed ------------------------------------------------

def test_base_dz_is_a_2mm_clearance_not_half_the_tallest_object(tmp_path: Path):
    """The lift is a 2 mm clearance above the scene's true mesh minimum
    (`-scene_min + 0.002` from mesh AABB minima).

    The regressed formula was `floor_z - min(pos[2] - dims[2]/2)`. Because
    `assemble_scene` always writes `pos=(0,0,0)`, that reduces to
    `max(dims_z)/2` -- half the tallest object -- and measured +0.43 m on
    office_lounge, teleporting the entire scene into the air before every
    CEM rollout."""
    spec = _scene(tmp_path, z0=0.0)                  # geometry already on the floor
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    cem = FaithfulCEM(spec, model, tmp_path)
    assert cem.base_dz == pytest.approx(0.002, abs=1e-4), (
        f"base_dz={cem.base_dz} m for a scene already resting on the floor; "
        "expected the 2 mm clearance. ~0.1 m here (half the 0.2 m crate) means "
        "the pos/dims formula is back."
    )


def test_settle_window_is_one_second_of_physical_time():
    """The settle window is 1.0 s of simulated time (60 steps at dt=1/60) with
    velocity read at 0.25 s. This implementation's MJCF runs dt=0.002, so a literal 60 steps
    is 0.12 s -- free fall covers only 7 cm in that window, so no sample is
    ever scored in a settled state. The durations are what must match, not the
    step counts."""
    from lens import stabilize
    assert stabilize.SETTLE_SECONDS == pytest.approx(1.0)
    assert stabilize.VEL_READ_SECONDS == pytest.approx(0.25)


def test_stabilize_lands_a_floating_box_on_the_floor(tmp_path: Path):
    """End-to-end consequence of both bugs: a crate baked 30 cm in the air must
    come to rest ON the floor. With base_dz=+0.1 and a 0.12 s window it cannot
    -- it is scored mid-fall and baked there."""
    spec = _scene(tmp_path, z0=0.30)
    out = stabilize_scene(spec, tmp_path)
    xml = mjcf.write_scene(out, tmp_path)
    rep = settle(xml, seconds=1.5)
    assert rep.drift["crate_0"] < 0.02, (
        f"crate drifted {rep.drift['crate_0']} m when re-simulated -- the "
        "stabilized scene is not at rest"
    )
    assert rep.max_penetration < 0.005


def test_stabilize_marks_provenance(tmp_path: Path):
    out = stabilize_scene(_scene(tmp_path, z0=0.30), tmp_path)
    assert out.objects[0].provenance["stabilize"] == "cem+settle(mujoco)"


def test_identical_samples_score_identically(tmp_path: Path):
    """The CEM objective must be a pure function of the action. `_compute_cost`
    reuses one MjData across the whole population and resets only qpos between
    samples, so qvel/qacc_warmstart/time carry over and each rollout starts
    with its predecessor's terminal velocities. Measured on office_lounge,
    four IDENTICAL all-zero samples scored 46, 145, 771 and 1805 -- ranking a
    population by a history-dependent objective is not optimizing anything.
    Velocities are zeroed on every pose write."""
    from lens.stabilize import FaithfulCEM, _compute_cost, _qpos_adr

    # A crate resting on a plinth: two bodies, one group, real contact.
    (tmp_path / "meshes").mkdir(exist_ok=True)
    specs = []
    for name, extents, z in (("plinth_0", (0.6, 0.6, 0.4), 0.2),
                             ("crate_0", (0.2, 0.2, 0.2), 0.5)):
        m = trimesh.creation.box(extents=extents)
        m.apply_translation((0, 0, z))
        m.export(tmp_path / f"meshes/{name}.obj")
        collision.decompose(m)[0].export(tmp_path / f"meshes/{name}_col00.obj")
        specs.append(make_obj(id=name, visual_mesh=f"meshes/{name}.obj",
                              collision_meshes=[f"meshes/{name}_col00.obj"],
                              pos=(0.0, 0.0, 0.0), dims=extents, mass=1.0,
                              provenance={} if name == "plinth_0"
                              else {"parent": "plinth_0"}))
    spec = SceneSpec(name="t", objects=specs, capture={})

    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.gravity[:] = [0, 0, -9.8]
    cem = FaithfulCEM(spec, model, tmp_path)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    orig_quat = {}
    for o in spec.objects:
        adr = _qpos_adr(model, o.id)
        if adr != -1:
            data.qpos[adr + 2] += cem.base_dz
            orig_quat[o.id] = data.qpos[adr + 3:adr + 7].copy()
    mujoco.mj_forward(model, data)

    samples = np.zeros((4, 6))                      # four identical actions
    costs = _compute_cost(model, cem.groups[0], samples, cem.orig_center,
                          cem.mesh_center, cem.ref_pos, orig_quat,
                          cem.hull_verts, cem.movable_set,
                          qpos0=data.qpos.copy())
    assert np.allclose(costs, costs[0], rtol=1e-6, atol=1e-6), (
        f"identical actions scored differently: {costs} -- simulation state "
        "is leaking between population samples")


def test_ref_pos_is_the_world_centroid_the_layout_term_measures(tmp_path: Path):
    """`ref_pos` is the object's world reference: where the object's
    geometry actually sits. The layout term is
    `norm(settled_world_centroid - ref_pos)`, so both sides must be world
    quantities. This crate's geometry sits at world (3, 2) while its freejoint
    qpos origin is (0, 0).

    (This test previously asserted the same value as a rotation PIVOT. There
    is no pivot -- `placed_pos = child_ref + delta` -- and the pivot machinery
    is gone; the value is still load-bearing, for the layout term instead.)"""
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    box.apply_translation((3.0, 2.0, 0.1))
    (tmp_path / "meshes").mkdir(exist_ok=True)
    box.export(tmp_path / "meshes/obj_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/obj_0_col00.obj")
    obj = make_obj(id="obj_0", visual_mesh="meshes/obj_0.obj",
                   collision_meshes=["meshes/obj_0_col00.obj"],
                   pos=(0.0, 0.0, 0.0), dims=(0.2, 0.2, 0.2), mass=1.0)
    spec = SceneSpec(name="t", objects=[obj], capture={})

    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cem = FaithfulCEM(spec, model, tmp_path)

    bid = model.body("obj_0").id
    com_before = data.xipos[bid, :2].copy()
    np.testing.assert_allclose(com_before, (3.0, 2.0), atol=1e-3)

    # `ref_pos` (centroid + ground drop) -- NOT `ref_root_pos`, which
    # is this implementation's `orig_center` and is the body-frame anchor the action adds
    # its delta to. Conflating them breaks whichever one gets the wrong value.
    ref = cem.ref_pos["obj_0"]
    np.testing.assert_allclose(ref[:2], com_before, atol=0.01, err_msg=(
        "ref_pos must track the object's world COM for the layout term to "
        f"compare like with like; got {ref[:2]} for a COM at {com_before}"))


def test_action_anchor_stays_in_the_body_frame_qpos_uses(tmp_path: Path):
    """`orig_center` is the other half of the pair and must NOT be the centroid.

    It is `ref_root_pos` / `child_ref_pos`: the anchor a sample's
    translation is added to, `placed_pos = orig_center + delta`. That result is
    written straight into qpos, which holds the BODY-FRAME position --
    (0, 0, base_dz) here, because assemble_scene bakes world position into the
    mesh vertices. Pointing `orig_center` at the world centroid instead adds a
    ~2 m constant to every placement (measured on office_lounge: median
    placement error 38.1 cm, groups translated ~0.4 m in y)."""
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    box.apply_translation((3.0, 2.0, 0.1))
    (tmp_path / "meshes").mkdir(exist_ok=True)
    box.export(tmp_path / "meshes/obj_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/obj_0_col00.obj")
    obj = make_obj(id="obj_0", visual_mesh="meshes/obj_0.obj",
                   collision_meshes=["meshes/obj_0_col00.obj"],
                   pos=(0.0, 0.0, 0.0), dims=(0.2, 0.2, 0.2), mass=1.0)
    spec = SceneSpec(name="t", objects=[obj], capture={})
    model = mujoco.MjModel.from_xml_path(str(mjcf.write_scene(spec, tmp_path)))
    cem = FaithfulCEM(spec, model, tmp_path)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adr = model.jnt_qposadr[model.body_jntadr[model.body("obj_0").id]]
    start = data.qpos[adr:adr + 3].copy()
    start[2] += cem.base_dz                      # stabilize_scene's ground drop

    np.testing.assert_allclose(cem.orig_center["obj_0"], start, atol=1e-9, err_msg=(
        "orig_center must equal the freejoint's STARTING body-frame position, "
        "so the layout term measures displacement rather than a frame mismatch"))
    assert not np.allclose(cem.orig_center["obj_0"][:2], cem.ref_pos["obj_0"][:2]), (
        "orig_center and ref_pos are distinct quantities (ref_root_pos "
        "vs ref_pos); this fixture places geometry at (3, 2) so they must differ")


def _stacked_scene(tmp_path, gap):
    """Crate on the floor with a small box floating `gap` metres above its top
    face -- the shape of every real "object on furniture" case.

    Geometry is baked into the vertices (pos stays (0,0,0)), matching
    `lens.pipeline.assemble_scene`. The scene's minimum is the crate's bottom
    at z=0, so `base_dz` is the 2 mm clearance and does NOT incidentally lower
    the box: the gap can only be closed by physics during a settle. The CEM
    cannot close it either -- INIT_STD's dz sigma is 5 mm.
    """
    (tmp_path / "meshes").mkdir(exist_ok=True)

    crate = trimesh.creation.box(extents=(0.4, 0.4, 0.2))
    crate.apply_translation((0, 0, 0.1))                    # bottom face at z=0
    crate.export(tmp_path / "meshes/crate_0.obj")
    collision.decompose(crate)[0].export(tmp_path / "meshes/crate_0_col00.obj")

    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    box.apply_translation((0, 0, 0.2 + gap + 0.05))         # bottom face `gap` above the crate
    box.export(tmp_path / "meshes/box_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/box_0_col00.obj")

    parent = make_obj(id="crate_0", category="table",
                      visual_mesh="meshes/crate_0.obj",
                      collision_meshes=["meshes/crate_0_col00.obj"],
                      pos=(0.0, 0.0, 0.0), dims=(0.4, 0.4, 0.2), mass=8.0,
                      provenance={})
    child = make_obj(id="box_0", category="box",
                     visual_mesh="meshes/box_0.obj",
                     collision_meshes=["meshes/box_0_col00.obj"],
                     pos=(0.0, 0.0, 0.0), dims=(0.1, 0.1, 0.1), mass=1.0,
                     provenance={"parent": "crate_0", "relation": "on"})
    return SceneSpec(name="t", objects=[parent, child], capture={})


def test_final_bake_settles_for_the_same_duration_the_cem_scored_with(
        tmp_path: Path, monkeypatch):
    """The baked pose must be the pose that was scored.

    Every CEM rollout settles for SETTLE_SECONDS (1.0 s) before its cost is
    read, but the final bake ran a LITERAL `range(60)`. At this model's
    dt=0.002 that is 0.12 s, so the winning action was chosen on the strength
    of a settled scene and then written out mid-fall: an object not already in
    contact drops only 0.5*9.8*0.12**2 = 7.1 cm and freezes there.

    The box starts 20 cm above the crate, which is more than that 7.1 cm, so
    the two windows give visibly different output. Re-simulating a correctly
    baked scene must show it already at rest.
    """
    monkeypatch.setattr("lens.stabilize.POP_SIZE", 16)
    monkeypatch.setattr("lens.stabilize.N_ITERS", 2)

    out = stabilize_scene(_stacked_scene(tmp_path, gap=0.20), tmp_path)
    xml = mjcf.write_scene(out, tmp_path)
    rep = settle(xml, seconds=1.5)

    assert rep.drift["box_0"] < 0.02, (
        f"box_0 moved {rep.drift['box_0']:.3f} m when re-simulated, so it was "
        "baked mid-fall rather than at rest"
    )


def test_damping_matches_the_dimensionless_scoring_coefficient(tmp_path: Path):
    """The scoring physics is specified as a dimensionless 0.3 linear and
    angular damping coefficient. Dimensionless means the engine integrates
    `dv/dt = -0.3 v`, a 3.33 s decay constant for every body regardless of its
    mass.

    MuJoCo's `dof_damping` is a force (torque) per unit velocity, so the
    equivalent is `0.3 * mass` on the three translational DoFs and
    `0.3 * principal moment` on the three rotational ones. One scalar on the
    freejoint cannot express both: 0.3 N.s/m is right for a 1 kg body's
    translation but ~20x over-damped for the rotation of a 0.2 m crate
    (I ~ 0.013 kg.m^2).

    MONOLITHCLE shipped with no damping whatsoever, which is the prime suspect
    for the QACC NaN warnings the office_lounge run emits.
    """
    from lens import stabilize

    spec = _scene(tmp_path, z0=0.0)
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.dof_damping.max() == 0.0, "fixture assumes an undamped compile"

    stabilize.configure_physics(model)

    b = model.body("crate_0").id
    adr = model.jnt_dofadr[model.body_jntadr[b]]
    assert model.dof_damping[adr:adr + 3] == pytest.approx(
        stabilize.DAMPING * model.body_mass[b]), "translational DoFs undamped"
    assert model.dof_damping[adr + 3:adr + 6] == pytest.approx(
        stabilize.DAMPING * model.body_inertia[b]), "rotational DoFs undamped"


def test_configure_physics_leaves_a_1kg_body_with_the_scoring_decay_constant(tmp_path: Path):
    """The point of the mass scaling: a damped body's velocity must decay with
    the 1/0.3 = 3.33 s time constant the scoring physics gives it, measured in the sim rather
    than asserted from the formula.

    Two things have to be held off for the measurement to mean anything, and
    both were found by watching this test fail against correct code:

    - Gravity is off and contacts are disabled. Leaving the body touching the
      floor adds friction to the decay: 0.645 resting on it, 0.125 for a
      body-centred box that consequently straddles it.
    - The geometry is centred on the body frame. The pipeline bakes world
      position into the vertices and leaves `pos=(0,0,0)`, which puts the COM
      metres from the body origin; a free joint's translational and rotational
      DoFs are then coupled through the off-diagonal mass matrix, the decay
      stops being a pure exponential, and it measures 0.808.

    That second point is a real caveat on the per-DoF damping translation,
    not just on the test: MuJoCo damps the body-frame generalised velocity,
    whereas the dimensionless coefficient is defined on the COM velocity. They
    agree exactly when the COM is on the body origin or the body
    is not rotating, and differ by the `omega x r_com` term otherwise -- small
    over a 1 s settle whose whole job is to bleed energy.
    """
    from lens import stabilize

    (tmp_path / "meshes").mkdir(exist_ok=True)
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))          # centred on origin
    box.export(tmp_path / "meshes/crate_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/crate_0_col00.obj")
    spec = SceneSpec(name="t", capture={}, objects=[make_obj(
        id="crate_0", visual_mesh="meshes/crate_0.obj",
        collision_meshes=["meshes/crate_0_col00.obj"],
        pos=(0.0, 0.0, 0.0), dims=(0.2, 0.2, 0.2), mass=1.0)])

    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    stabilize.configure_physics(model)
    model.opt.gravity[:] = [0.0, 0.0, 0.0]           # damping is the only force
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    data = mujoco.MjData(model)
    adr = model.jnt_dofadr[model.body_jntadr[model.body("crate_0").id]]
    data.qvel[adr] = 1.0
    for _ in range(_steps := int(round(1.0 / model.opt.timestep))):
        mujoco.mj_step(model, data)

    assert data.qvel[adr] == pytest.approx(np.exp(-stabilize.DAMPING), rel=0.02), (
        f"v(1s)={data.qvel[adr]:.4f}, expected exp(-0.3)={np.exp(-0.3):.4f}"
    )


def _group_scene(tmp_path):
    """A crate root with TWO children, each parked well off the body-frame
    origin so that a yaw sweeps its centroid through a visible arc.

    Geometry carries the world position and `pos` stays (0,0,0), as
    `assemble_scene` writes it -- so every body frame is at the origin and each
    object's AABB centre IS its body-frame centroid offset.
    """
    (tmp_path / "meshes").mkdir(exist_ok=True)
    specs = []
    for oid, ext, ctr, parent in [
            ("crate_0", (1.6, 1.0, 0.4), (0.0, 0.0, 0.2), None),
            ("box_a_0", (0.2, 0.2, 0.2), (-0.5, 0.0, 0.5), "crate_0"),
            ("box_b_0", (0.2, 0.2, 0.2), (0.5, 0.0, 0.5), "crate_0")]:
        m = trimesh.creation.box(extents=ext)
        m.apply_translation(ctr)
        m.export(tmp_path / f"meshes/{oid}.obj")
        collision.decompose(m)[0].export(tmp_path / f"meshes/{oid}_col00.obj")
        specs.append(make_obj(
            id=oid, category="box", visual_mesh=f"meshes/{oid}.obj",
            collision_meshes=[f"meshes/{oid}_col00.obj"],
            pos=(0.0, 0.0, 0.0), dims=ext, mass=1.0,
            provenance={} if parent is None
            else {"parent": parent, "relation": "on"}))
    return SceneSpec(name="t", objects=specs, capture={})


def test_local_action_gives_each_child_its_own_six_numbers(tmp_path: Path):
    """The sample is viewed as (batch, child, 6) and each child is posed
    from its own slice: that child's own reference position plus the sampled
    translation delta. MONOLITHCLE drew ONE 6-vector and applied it to every child in
    the group, so no sample could move the backpack without moving the laptop
    by exactly the same amount -- the optimizer could only ever pick a
    compromise wrong for both.
    """
    from lens import stabilize

    spec = _group_scene(tmp_path)
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    cem = FaithfulCEM(spec, model, tmp_path)
    data = mujoco.MjData(model)
    orig_quat = {o.id: np.array([1.0, 0.0, 0.0, 0.0]) for o in spec.objects}

    children = ["box_a_0", "box_b_0"]
    action = np.zeros(12)
    action[0] = 0.5                                   # dx of box_a only

    stabilize._place_children(model, data, children, action,
                              cem.orig_center, orig_quat)

    a = _qpos_adr(model, "box_a_0")
    b = _qpos_adr(model, "box_b_0")
    assert data.qpos[a:a + 3] == pytest.approx(
        cem.orig_center["box_a_0"] + np.array([0.5, 0.0, 0.0]))
    assert data.qpos[b:b + 3] == pytest.approx(cem.orig_center["box_b_0"]), (
        "box_b moved too -- the action is still shared across the group"
    )


def test_child_pose_is_reference_plus_delta_with_no_pivot(tmp_path: Path):
    """A child is posed as its reference position plus the sampled
    translation delta, and its reference orientation composed with the sampled
    rotation delta. Position is NEVER rotated about a pivot: the delta rotation
    changes orientation only.

    MONOLITHCLE rotated the body-frame position about `ref_pos` first, which for
    a body frame at the origin and geometry metres away swings the object
    bodily across the scene.
    """
    from lens import stabilize

    spec = _group_scene(tmp_path)
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    cem = FaithfulCEM(spec, model, tmp_path)
    data = mujoco.MjData(model)
    orig_quat = {o.id: np.array([1.0, 0.0, 0.0, 0.0]) for o in spec.objects}

    action = np.zeros(12)
    action[5] = 0.7                                   # pure yaw on box_a
    stabilize._place_children(model, data, ["box_a_0", "box_b_0"], action,
                              cem.orig_center, orig_quat)

    a = _qpos_adr(model, "box_a_0")
    assert data.qpos[a:a + 3] == pytest.approx(cem.orig_center["box_a_0"]), (
        "a pure rotation displaced the body frame -- pivot machinery is back"
    )
    assert data.qpos[a + 3:a + 7] == pytest.approx(
        stabilize._axis_angle_to_quat(np.array([0.0, 0.0, 0.7])))


def test_world_centroid_tracks_the_geometry_not_the_body_frame(tmp_path: Path):
    """`_world_centroid` is `placed_root + R @ centroid_ofs`. Yawing a
    body whose geometry sits 0.5 m off its origin must sweep that centroid
    through the corresponding arc."""
    from lens import stabilize

    ofs = np.array([0.5, 0.0, 0.0])
    q = stabilize._axis_angle_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    got = stabilize._world_centroid(np.zeros(3), q, ofs)
    assert got == pytest.approx(np.array([0.0, 0.5, 0.0]), abs=1e-9)


def test_a_pure_yaw_costs_more_than_doing_nothing(tmp_path: Path):
    """The consequence of measuring the layout term in world space. Under the
    body-frame metric a pure rotation leaves `qpos[:3]` untouched, so yaw is
    free and the CEM spends it freely -- the office_lounge backpack came out
    yawed 44 deg with a 1.18 m compensating translation. Scored on world
    centroids, the same yaw drags the centroid through an arc and is paid for.
    """
    from lens import stabilize

    spec = _group_scene(tmp_path)
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    stabilize.configure_physics(model)
    cem = FaithfulCEM(spec, model, tmp_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos0 = data.qpos.copy()
    for o in spec.objects:
        adr = _qpos_adr(model, o.id)
        if adr != -1:
            qpos0[adr + 2] += cem.base_dz
    orig_quat = {o.id: np.array([1.0, 0.0, 0.0, 0.0]) for o in spec.objects}

    samples = np.zeros((2, 12))
    samples[1, 5] = 0.6                               # box_a yaws 34 deg

    costs = stabilize._compute_cost(
        model, ["crate_0", "box_a_0", "box_b_0"], samples,
        cem.orig_center, cem.mesh_center, cem.ref_pos, orig_quat,
        cem.hull_verts, cem.movable_set, qpos0=qpos0)

    assert costs[1] > costs[0] + 0.5, (
        f"yaw cost {costs[1]:.4f} vs rest {costs[0]:.4f} -- rotation is still "
        "nearly free, so the layout term is not measuring world centroids"
    )


def test_global_entity_pose_is_reference_plus_delta_with_absolute_quat(tmp_path: Path):
    """The global pass composes exactly like the local one but with an ABSOLUTE
    orientation: `placed_quat_all = axis_angle_to_quat(delta_aa)` is written
    directly, not `ref_quat * delta`, because a group root
    was pinned at identity for the whole local pass.

    Position is again `ref_root_pos + delta` with no pivot.
    """
    from lens import stabilize

    spec = _group_scene(tmp_path)
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    cem = FaithfulCEM(spec, model, tmp_path)
    data = mujoco.MjData(model)

    action = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.3])
    stabilize._place_entities(model, data, ["crate_0"], action,
                              cem.orig_center, {})

    adr = _qpos_adr(model, "crate_0")
    assert data.qpos[adr:adr + 3] == pytest.approx(
        cem.orig_center["crate_0"] + np.array([0.1, 0.2, 0.0]))
    assert data.qpos[adr + 3:adr + 7] == pytest.approx(
        stabilize._axis_angle_to_quat(np.array([0.0, 0.0, 0.3])))


def test_global_riders_follow_their_root_rigidly(tmp_path: Path):
    """Children do not get their own global action -- they ride on the root
    through the offsets the local pass left them at, `p_root + R_root @ ofs`
    and `q_root * q_rel` (`place_all_descendants`).

    The rigid-body invariant is the check: the root-to-child vector must come
    out rotated by the root's new orientation and otherwise unchanged in
    length.
    """
    from lens import stabilize

    spec = _group_scene(tmp_path)
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    cem = FaithfulCEM(spec, model, tmp_path)
    data = mujoco.MjData(model)

    ofs = np.array([0.4, 0.0, 0.1])
    q_rel = np.array([1.0, 0.0, 0.0, 0.0])
    riders = {"crate_0": [("box_a_0", ofs, q_rel)]}

    yaw = 0.5
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, yaw])
    stabilize._place_entities(model, data, ["crate_0"], action,
                              cem.orig_center, riders)

    r = _qpos_adr(model, "crate_0")
    c = _qpos_adr(model, "box_a_0")
    root_p = np.array(data.qpos[r:r + 3])
    child_p = np.array(data.qpos[c:c + 3])
    R = stabilize._quat_to_rotmat(stabilize._axis_angle_to_quat(
        np.array([0.0, 0.0, yaw])))
    assert child_p - root_p == pytest.approx(R @ ofs), (
        "the child did not follow the root's rotation rigidly"
    )
    assert np.linalg.norm(child_p - root_p) == pytest.approx(
        np.linalg.norm(ofs)), "rigid attachment changed the separation"


def test_isolate_removes_other_bodies_from_collision_but_keeps_the_floor(tmp_path: Path):
    """Each pass gets a FRESH env with the actor filter set to just
    that pass's actors, and the filter gates both the fixed
    and the movable lists. Objects outside the filter are not
    slow or pinned -- they are absent.

    MuJoCo cannot drop bodies from a compiled model, so the equivalent is to
    clear contype/conaffinity on their geoms. The floor lives on the worldbody
    and must survive, or the group falls out of the universe.
    """
    from lens import stabilize

    spec = _group_scene(tmp_path)
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))

    def collidable():
        return {model.body(model.geom_bodyid[g]).name
                for g in range(model.ngeom)
                if model.geom_contype[g] or model.geom_conaffinity[g]}

    before = collidable()
    assert {"crate_0", "box_a_0", "box_b_0"} <= before

    with stabilize._isolate(model, {"crate_0", "box_a_0"}):
        during = collidable()
        assert "box_b_0" not in during, "excluded body still collides"
        assert {"crate_0", "box_a_0"} <= during
        assert "world" in during, "the floor plane was disabled too"

    assert collidable() == before, "isolation did not restore the model"


def test_an_object_outside_the_group_neither_falls_nor_blocks(tmp_path: Path):
    """The behavioural consequence. `box_b` is parked in mid-air and belongs to
    no group being optimized; during another group's local pass it must stay
    exactly where it is.

    With every body in the sim -- what MONOLITHCLE did -- it falls under gravity
    for the whole 1 s settle of every one of the ~7680 rollouts, lands on
    whatever is beneath it, and that contact perturbs the very children being
    scored. On office_lounge all six objects were live in every pass, starting
    each rollout with 36 mutually penetrating contacts.
    """
    from lens import stabilize

    spec = _group_scene(tmp_path)
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    stabilize.configure_physics(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    b = _qpos_adr(model, "box_b_0")
    data.qpos[b + 2] += 0.5                       # park it well clear
    mujoco.mj_forward(model, data)
    before = np.array(data.qpos[b:b + 3])

    active = {"crate_0", "box_a_0"}
    inactive = stabilize._inactive_pins(model, active, data)
    with stabilize._isolate(model, active):
        stabilize._step_pinned(model, data, _steps := 500, inactive)

    assert np.array(data.qpos[b:b + 3]) == pytest.approx(before, abs=1e-9), (
        f"box_b moved {np.linalg.norm(np.array(data.qpos[b:b+3]) - before):.4f} m "
        "during a pass it is not part of"
    )


def test_configure_physics_gives_every_body_the_scoring_uniform_mass(tmp_path: Path):
    """The scoring physics gives every body a uniform 1 kg mass --
    one kilogram for a sofa and for a tumbler alike. Mass is a
    stabilization-time choice there, not a physical claim, and it only holds
    while the CEM runs.

    MONOLITHCLE derives mass from density x hull volume, which gave the
    scene A bench 26.9 kg. Inertia is rescaled by the same factor so the body
    keeps the shape-correct distribution for its new mass (a uniform-mass
    override that leaves the inertia shape alone).
    """
    from lens import stabilize

    spec = _group_scene(tmp_path)
    for o in spec.objects:                         # deliberately heavy and uneven
        o.mass = 40.0 if o.id == "crate_0" else 0.05
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))

    b = model.body("crate_0").id
    heavy_before = float(model.body_mass[b])
    inertia_before = np.array(model.body_inertia[b])
    assert heavy_before > 30.0, "fixture did not compile a heavy body"

    stabilize.configure_physics(model)

    for name in ("crate_0", "box_a_0", "box_b_0"):
        i = model.body(name).id
        assert model.body_mass[i] == pytest.approx(stabilize.STABILIZE_MASS), name
    # same shape, 1/40th the mass -> 1/40th the inertia, directions unchanged
    assert np.array(model.body_inertia[b]) == pytest.approx(
        inertia_before * (stabilize.STABILIZE_MASS / heavy_before), rel=1e-6)


def test_a_heavy_body_does_not_sink_through_the_floor(tmp_path: Path):
    """The failure this causes. MuJoCo's contacts are compliant, so resting
    penetration grows with weight: at MONOLITHCLE's density-derived 26.9 kg the
    office_lounge bench settled 13 cm BELOW the floor plane over a 1 s window,
    dragging every object riding on it down with it (measured -0.1296 m min-z
    where the expected clearance is +0.0100 m).
    """
    from lens import stabilize

    spec = _scene(tmp_path, z0=0.0)                # crate resting on the floor
    spec.objects[0].mass = 40.0
    xml = mjcf.write_scene(spec, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    stabilize.configure_physics(model)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adr = _qpos_adr(model, "crate_0")
    z0 = float(data.qpos[adr + 2])
    _run_steps = _steps_for = None                 # use the module's own helper
    for _ in range(stabilize._steps_for(model, 1.0)):
        mujoco.mj_step(model, data)
    sink = z0 - float(data.qpos[adr + 2])

    assert sink < 0.005, (
        f"body sank {100*sink:.2f} cm into the floor over 1 s -- contact is "
        "carrying more weight than the scoring physics' 1 kg bodies ever put on it"
    )


# ---------------------------------------------------------------------------
# Collision-only scoring model + RAM-capped workers
# Collision-only CEM scoring
# ---------------------------------------------------------------------------

def _two_body_xml(tmp_path):
    (tmp_path / "meshes").mkdir(exist_ok=True)
    specs = []
    for name, extents, z in (("plinth_0", (0.6, 0.6, 0.4), 0.2),
                             ("crate_0", (0.2, 0.2, 0.2), 0.5)):
        m = trimesh.creation.box(extents=extents)
        m.apply_translation((0, 0, z))
        m.export(tmp_path / f"meshes/{name}.obj")
        collision.decompose(m)[0].export(tmp_path / f"meshes/{name}_col00.obj")
        specs.append(make_obj(id=name, visual_mesh=f"meshes/{name}.obj",
                              collision_meshes=[f"meshes/{name}_col00.obj"],
                              pos=(0.0, 0.0, 0.0), dims=extents, mass=1.0,
                              provenance={} if name == "plinth_0"
                              else {"parent": "plinth_0"}))
    spec = SceneSpec(name="t", objects=specs, capture={})
    return mjcf.write_scene(spec, tmp_path), spec


def test_scoring_xml_strips_visuals_and_keeps_identical_dynamics(tmp_path: Path):
    """The CEM workers' scoring model must drop every visual geom AND its mesh
    asset (the 255 MB/worker dead weight) while stepping bit-identically to
    the full model: visual geoms are contype/conaffinity 0 and mass 0, so
    removing them cannot change dynamics."""
    from lens.stabilize import _scoring_xml

    xml, _ = _two_body_xml(tmp_path)
    sxml = _scoring_xml(xml)
    assert str(sxml) != str(xml)

    full = mujoco.MjModel.from_xml_path(str(xml))
    slim = mujoco.MjModel.from_xml_path(str(sxml))

    assert (slim.geom_group != 2).all(), "visual geoms must be gone"
    assert slim.nmesh < full.nmesh, "visual mesh assets must be gone"
    assert slim.nbody == full.nbody
    np.testing.assert_array_equal(slim.body_mass, full.body_mass)

    df, ds = mujoco.MjData(full), mujoco.MjData(slim)
    for _ in range(200):
        mujoco.mj_step(full, df)
        mujoco.mj_step(slim, ds)
    np.testing.assert_allclose(ds.qpos, df.qpos, atol=1e-12)


def test_scoring_xml_scores_identically_to_full_model(tmp_path: Path):
    """End-relevant equivalence: a population chunk evaluated on the scoring
    model must produce the exact costs the full model produces."""
    from lens.stabilize import FaithfulCEM, _eval_local_chunk, _scoring_xml, _steps_for

    xml, spec = _two_body_xml(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    cem = FaithfulCEM(spec, model, tmp_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos0 = data.qpos.copy()
    orig_quat = {o.id: qpos0[_qpos_adr(model, o.id) + 3:_qpos_adr(model, o.id) + 7].copy()
                 for o in spec.objects}

    rng = np.random.default_rng(7)
    samples = rng.normal(size=(4, 6)) * 0.02
    group = ["plinth_0", "crate_0"]
    args = lambda x: (str(x), group, samples, cem.orig_center, cem.mesh_center,
                      cem.ref_pos, orig_quat, cem.hull_verts, cem.movable_set,
                      qpos0, _steps_for(model, 0.25), _steps_for(model, 0.25))
    np.testing.assert_allclose(_eval_local_chunk(args(_scoring_xml(xml))),
                               _eval_local_chunk(args(xml)), rtol=0, atol=1e-10)


def test_worker_cap_is_ram_aware_never_bare_cpu_count(monkeypatch):
    """cpu_count()=24 workers x 3.3 GB was the OOM. The cap must respect
    available RAM (50% headroom), keep 4 cores for the system, floor at 1,
    and honour an explicit LENS_CEM_WORKERS override."""
    from lens.stabilize import _worker_cap

    assert _worker_cap(cpus=24, avail_bytes=8 << 30, per_worker_bytes=1 << 30) == 4
    assert _worker_cap(cpus=24, avail_bytes=0, per_worker_bytes=1 << 30) == 1
    assert _worker_cap(cpus=24, avail_bytes=1 << 40, per_worker_bytes=1 << 30) == 20
    assert _worker_cap(cpus=4, avail_bytes=1 << 40, per_worker_bytes=1 << 30) == 1
    monkeypatch.setenv("LENS_CEM_WORKERS", "2")
    assert _worker_cap(cpus=24, avail_bytes=1 << 40, per_worker_bytes=1 << 30) == 2


def test_cem_budget_env_overrides(monkeypatch):
    """Population and iteration budget must be settable per run (the campaign
    budget is 2048x15x2 vs the shipped 512x15) without editing code."""
    from lens import stabilize

    assert stabilize._budget() == (stabilize.POP_SIZE, stabilize.N_ITERS)
    monkeypatch.setenv("LENS_CEM_POP", "2048")
    monkeypatch.setenv("LENS_CEM_ITERS", "20")
    assert stabilize._budget() == (2048, 20)
