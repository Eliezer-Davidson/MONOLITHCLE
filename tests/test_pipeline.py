# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""End-to-end pipeline orchestration tests, entirely on fake backends
(no GPU, no network)."""
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from lens.pipeline import ReconObject, assemble_scene, run_pipeline
from lens.recon import FakeRecon
from lens.scenegraph.masks import FakeSegmenter
from lens.vlm import FakeVLM


def test_pipeline_end_to_end_fakes(tmp_path):
    Image.new("RGB", (64, 64), (90, 90, 90)).save(tmp_path / "p.jpg")
    m = np.zeros((64, 64), bool); m[10:30, 10:30] = True
    vlm = FakeVLM(["grey crate | box | floor", "YES"])
    spec = run_pipeline(tmp_path / "p.jpg", tmp_path / "out",
                        vlm=vlm, segmenter=FakeSegmenter({"grey crate": [m]}),
                        recon=FakeRecon(), stabilize=True)
    assert spec.validate() == []
    assert (tmp_path / "out/scene.json").exists()
    assert (tmp_path / "out/mujoco").glob("*.xml")
    o = spec.objects[0]
    assert o.id == "grey_crate_0" and o.mass > 0 and o.dims[2] > 0.1


def test_pipeline_actually_writes_mjcf_xml_and_meshes(tmp_path):
    """A bare `.glob("*.xml")` truthiness assertion is true for any Path object
    (a generator is always truthy) even if the directory doesn't exist or
    is empty -- so it doesn't actually prove an XML file was written. This
    test checks the real artifacts on disk directly."""
    Image.new("RGB", (64, 64), (90, 90, 90)).save(tmp_path / "p.jpg")
    m = np.zeros((64, 64), bool); m[10:30, 10:30] = True
    vlm = FakeVLM(["grey crate | box | floor", "YES"])
    spec = run_pipeline(tmp_path / "p.jpg", tmp_path / "out",
                        vlm=vlm, segmenter=FakeSegmenter({"grey crate": [m]}),
                        recon=FakeRecon(), stabilize=False)
    xml_files = list((tmp_path / "out/mujoco").glob("*.xml"))
    assert len(xml_files) == 1
    assert (tmp_path / "out/meshes/grey_crate_0.obj").exists()
    assert list((tmp_path / "out/meshes").glob("grey_crate_0_col*.obj"))
    assert spec.objects[0].provenance["parent"] is None


def test_pipeline_drops_objects_the_segmenter_found_no_mask_for(tmp_path):
    """Two objects are listed by the VLM but the segmenter only has a mask
    for one of them -- the other must simply be absent from the final
    scene, without breaking id bookkeeping for the one that *was* found."""
    Image.new("RGB", (64, 64), (90, 90, 90)).save(tmp_path / "p.jpg")
    m = np.zeros((64, 64), bool); m[10:30, 10:30] = True
    vlm = FakeVLM(["grey crate | box | floor\nghost lamp | lamp | floor", "YES"])
    spec = run_pipeline(tmp_path / "p.jpg", tmp_path / "out",
                        vlm=vlm, segmenter=FakeSegmenter({"grey crate": [m]}),
                        recon=FakeRecon(), stabilize=False)
    assert spec.validate() == []
    assert [o.id for o in spec.objects] == ["grey_crate_0"]


def test_pipeline_survives_usd_export_failure(tmp_path, monkeypatch, caplog):
    """If the USD exporter's dependency is missing/broken, the pipeline must
    still finish with a valid MJCF + scene.json, only logging a warning."""
    import lens.pipeline as pipeline_mod

    def _boom(spec, scene_dir):
        raise RuntimeError("pxr not available on this machine")
    monkeypatch.setattr(pipeline_mod.usd_export, "write_scene", _boom)

    Image.new("RGB", (64, 64), (90, 90, 90)).save(tmp_path / "p.jpg")
    m = np.zeros((64, 64), bool); m[10:30, 10:30] = True
    vlm = FakeVLM(["grey crate | box | floor", "YES"])
    with caplog.at_level("WARNING"):
        spec = run_pipeline(tmp_path / "p.jpg", tmp_path / "out",
                            vlm=vlm, segmenter=FakeSegmenter({"grey crate": [m]}),
                            recon=FakeRecon(), stabilize=False)
    assert spec.validate() == []
    assert (tmp_path / "out/scene.json").exists()
    assert list((tmp_path / "out/mujoco").glob("*.xml"))
    assert not (tmp_path / "out/usd").exists()
    assert any("usd" in r.message.lower() for r in caplog.records)


def test_pipeline_dims_are_pca_aligned_not_axis_aligned_bbox(tmp_path):
    """A box reconstructed rotated 45 degrees about Z in the xy-plane should
    still report its true (short) side lengths via PCA, not the larger
    axis-aligned bounding box diagonal."""
    import trimesh
    from lens.recon import ReconResult

    class RotatedBoxRecon:
        def reconstruct(self, image, masks):
            out = {}
            for oid in masks:
                box = trimesh.creation.box(extents=(0.4, 0.2, 0.3))
                box.apply_transform(trimesh.transformations.rotation_matrix(
                    np.pi / 4, [0, 0, 1]))
                box.apply_translation((0, 0, 0.3 / 2 + 0.02))
                out[oid] = ReconResult(mesh=box, confidence=0.9)
            return out

    Image.new("RGB", (64, 64), (90, 90, 90)).save(tmp_path / "p.jpg")
    m = np.zeros((64, 64), bool); m[10:30, 10:30] = True
    vlm = FakeVLM(["grey crate | box | floor", "YES"])
    spec = run_pipeline(tmp_path / "p.jpg", tmp_path / "out",
                        vlm=vlm, segmenter=FakeSegmenter({"grey crate": [m]}),
                        recon=RotatedBoxRecon(), stabilize=False)
    o = spec.objects[0]
    # axis-aligned bbox of a 45-degree-rotated 0.4x0.2 rectangle would be
    # close to (0.424, 0.424) -- PCA must recover the true (0.4, 0.2).
    assert abs(o.dims[0] - 0.4) < 1e-3
    assert abs(o.dims[1] - 0.2) < 1e-3
    assert abs(o.dims[2] - 0.3) < 1e-3


def _box(extents, z):
    b = trimesh.creation.box(extents=extents)
    b.apply_translation((0.0, 0.0, z))
    return b


def _sample_records():
    """A tiny model-free scene tree: a floor-rooted table with a cup resting
    on it, plus an independent floor-rooted lamp -- exercises both the
    parent-provenance path and the no-parent path."""
    return [
        ReconObject(id="table_0", category="table", tags=["surface"],
                    parent=None, mesh=_box((0.6, 0.4, 0.4), z=0.2),
                    confidence=0.9),
        ReconObject(id="cup_0", category="cup", tags=["graspable"],
                    parent="table_0", mesh=_box((0.08, 0.08, 0.1), z=0.45),
                    confidence=0.8),
        ReconObject(id="lamp_0", category="lamp", tags=["graspable"],
                    parent=None, mesh=_box((0.1, 0.1, 0.3), z=0.15),
                    confidence=0.7),
    ]


def test_assemble_scene_is_model_free_no_stabilize(tmp_path):
    """`assemble_scene` takes already-reconstructed ReconObject records --
    no recon/segmenter object ever enters this call -- and must still
    produce a fully valid, on-disk scene."""
    records = _sample_records()
    spec = assemble_scene(records, tmp_path / "out", scene_name="mock_scene",
                           capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                           stabilize=False)
    assert spec.validate() == []
    assert [o.id for o in spec.objects] == ["table_0", "cup_0", "lamp_0"]
    assert (tmp_path / "out/meshes/table_0.obj").exists()
    assert (tmp_path / "out/meshes/cup_0.obj").exists()
    assert (tmp_path / "out/meshes/lamp_0.obj").exists()
    assert (tmp_path / "out/scene.json").exists()
    cup = next(o for o in spec.objects if o.id == "cup_0")
    assert cup.provenance["parent"] == "table_0"
    assert cup.confidence == pytest.approx(0.8)


def test_assemble_scene_is_model_free_with_stabilize(tmp_path):
    records = _sample_records()
    spec = assemble_scene(records, tmp_path / "out", scene_name="mock_scene",
                           capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                           stabilize=True)
    assert spec.validate() == []
    assert [o.id for o in spec.objects] == ["table_0", "cup_0", "lamp_0"]
    assert (tmp_path / "out/scene.json").exists()
    assert list((tmp_path / "out/mujoco").glob("*.xml"))


def _box_at(extents, center):
    """Like `_box`, but places the box's center anywhere in xyz (not just z)
    so grounding tests can prove x/y survive untouched."""
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


def test_assemble_scene_grounds_floor_rooted_objects_to_zero(tmp_path):
    """Two independent floor-rooted objects straddle z=0 on construction
    (example: z[-0.5,+0.1] and z[-0.3,+0.4]); after
    assemble_scene the lowest floor-rooted vertex must sit at z=0 (within
    1e-6), on the exported mesh on disk (not just the in-memory object),
    and SceneSpec.floor_z must be explicitly 0.0."""
    records = [
        ReconObject(id="crate_0", category="crate", tags=[], parent=None,
                    mesh=_box((0.4, 0.4, 0.6), z=-0.2), confidence=0.9),  # z in [-0.5, +0.1]
        ReconObject(id="stool_0", category="stool", tags=[], parent=None,
                    mesh=_box((0.3, 0.3, 0.7), z=0.05), confidence=0.9),  # z in [-0.3, +0.4]
    ]
    spec = assemble_scene(records, tmp_path / "out", scene_name="ground_test",
                           capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                           stabilize=False)

    assert spec.floor_z == 0.0
    mins = [float(r.mesh.vertices[:, 2].min()) for r in records]
    assert min(mins) == pytest.approx(0.0, abs=1e-6)

    # exported artifact on disk must reflect the same grounded geometry --
    # every downstream consumer (scene.json, MJCF, USD) reads from these
    # files, not from the in-memory ReconObject.
    exported = trimesh.load(tmp_path / "out/meshes/crate_0.obj", force="mesh")
    assert float(exported.vertices[:, 2].min()) == pytest.approx(0.0, abs=1e-6)


def test_assemble_scene_grounds_each_floor_object_independently(tmp_path):
    """Regression: TWO floor-rooted objects at DIFFERENT heights must
    EACH land with min-z == 0 -- not just the lowest. The old single-global-
    offset grounding dropped only the lowest floor object to z=0 and left the
    taller one floating by its (arbitrary, per-object-reconstructed) relative-z
    -- exactly the ~0.6 m 'sofa floats above the table' bug. (A child is
    re-seated onto its parent's top; see the dedicated re-seat test below.)"""
    sofa = ReconObject(id="sofa_0", category="sofa", tags=[], parent=None,
                       mesh=_box((1.2, 0.9, 0.8), z=0.0), confidence=0.9)    # z in [-0.4, +0.4]
    table = ReconObject(id="table_0", category="table", tags=[], parent=None,
                        mesh=_box((0.6, 0.6, 0.5), z=1.0), confidence=0.9)   # z in [+0.75, +1.25]
    cup = ReconObject(id="cup_0", category="cup", tags=[], parent="table_0",
                      mesh=_box((0.08, 0.08, 0.1), z=1.30), confidence=0.8)  # z in [+1.25, +1.35]
    records = [sofa, table, cup]
    pre_cup_gap = float(cup.mesh.vertices[:, 2].min()) - float(table.mesh.vertices[:, 2].max())

    spec = assemble_scene(records, tmp_path / "out", scene_name="perobj_test",
                           capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                           stabilize=False)

    # BOTH floor-rooted objects land on the floor -- the crux of the fix
    # (this FAILS under the old single-offset code: table would stay at +1.15).
    assert float(sofa.mesh.vertices[:, 2].min()) == pytest.approx(0.0, abs=1e-6)
    assert float(table.mesh.vertices[:, 2].min()) == pytest.approx(0.0, abs=1e-6)
    # their per-object offsets DIFFER (old code applied one shared offset)
    sofa_off = next(o for o in spec.objects if o.id == "sofa_0").provenance["ground_offset_z"]
    table_off = next(o for o in spec.objects if o.id == "table_0").provenance["ground_offset_z"]
    assert sofa_off == pytest.approx(0.4, abs=1e-6)
    assert table_off == pytest.approx(-0.75, abs=1e-6)
    # the child is RE-SEATED onto its parent's top (CHILD_GAP above), not dragged
    from lens.pipeline import CHILD_GAP
    post_cup_gap = float(cup.mesh.vertices[:, 2].min()) - float(table.mesh.vertices[:, 2].max())
    assert post_cup_gap == pytest.approx(CHILD_GAP, abs=1e-6)
    assert float(cup.mesh.vertices[:, 2].min()) > 0.4


def test_ground_records_reseats_child_onto_parent_top(tmp_path):
    """Regression: a child reconstructed BELOW its parent's base (the
    unreliable recon relative-z that detonated the settle as a contact explosion
    -- measured 143 mm under the floor on office_lounge) must be RE-SEATED onto
    its parent's top surface, ABOVE the floor, not dragged along below it.
    Fails on the old rigid-subtree grounding."""
    from lens.pipeline import CHILD_GAP
    sofa = ReconObject(id="sofa_0", category="sofa", tags=[], parent=None,
                       mesh=_box((1.2, 0.9, 0.8), z=0.0), confidence=0.9)     # z in [-0.4, +0.4]
    laptop = ReconObject(id="laptop_0", category="laptop", tags=[], parent="sofa_0",
                         mesh=_box((0.3, 0.2, 0.02), z=-0.7), confidence=0.8)  # z in [-0.71, -0.69]
    records = [sofa, laptop]
    assemble_scene(records, tmp_path / "out", scene_name="reseat_test",
                    capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                    stabilize=False)
    sofa_top = float(sofa.mesh.vertices[:, 2].max())
    laptop_bottom = float(laptop.mesh.vertices[:, 2].min())
    assert laptop_bottom == pytest.approx(sofa_top + CHILD_GAP, abs=1e-6)   # on the sofa top
    assert laptop_bottom > 0.0                                              # ABOVE the floor


def test_assemble_scene_grounding_preserves_xy_and_relative_gaps(tmp_path):
    """Grounding preserves every object's x/y centroid (a pure vertical shift),
    floor objects drop to z=0, and a child is re-seated CHILD_GAP above its
    parent's top. (Here both floor objects share a min-z, so their drops coincide.)"""
    table = ReconObject(id="table_0", category="table", tags=[], parent=None,
                         mesh=_box_at((0.6, 0.4, 0.4), (1.0, -0.5, 1.2)), confidence=0.9)
    cup = ReconObject(id="cup_0", category="cup", tags=[], parent="table_0",
                       mesh=_box_at((0.08, 0.08, 0.1), (1.0, -0.5, 1.45)), confidence=0.8)
    lamp = ReconObject(id="lamp_0", category="lamp", tags=[], parent=None,
                        mesh=_box_at((0.1, 0.1, 0.3), (-0.3, 0.7, 1.15)), confidence=0.7)
    records = [table, cup, lamp]

    pre_xy = {r.id: tuple(r.mesh.vertices[:, :2].mean(axis=0)) for r in records}
    pre_table_top = float(table.mesh.vertices[:, 2].max())
    pre_cup_bottom = float(cup.mesh.vertices[:, 2].min())
    pre_gap = pre_cup_bottom - pre_table_top

    spec = assemble_scene(records, tmp_path / "out", scene_name="layout_test",
                           capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                           stabilize=False)

    for r in records:
        xy = tuple(r.mesh.vertices[:, :2].mean(axis=0))
        assert xy[0] == pytest.approx(pre_xy[r.id][0], abs=1e-9)
        assert xy[1] == pytest.approx(pre_xy[r.id][1], abs=1e-9)

    from lens.pipeline import CHILD_GAP
    post_gap = float(cup.mesh.vertices[:, 2].min()) - float(table.mesh.vertices[:, 2].max())
    assert post_gap == pytest.approx(CHILD_GAP, abs=1e-6)

    # a pure translation must not change dims (PCA in-plane extents + z-extent)
    table_dims = next(o for o in spec.objects if o.id == "table_0").dims
    assert table_dims == pytest.approx((0.6, 0.4, 0.4), abs=1e-6)


def test_assemble_scene_child_below_parent_does_not_lower_floor(tmp_path):
    """A child record that dips (reconstruction noise) well below its
    floor-rooted parent's base must NOT be used to compute floor_z -- floor
    level comes from floor-rooted (parent is None) objects only. If the
    child were wrongly included, floor_z would be -0.55 instead of 0.3 and
    the table would land at +0.55, not 0."""
    table = ReconObject(id="table_0", category="table", tags=[], parent=None,
                         mesh=_box((0.6, 0.4, 0.4), z=0.5), confidence=0.9)  # z in [0.3, 0.7]
    sunk_cup = ReconObject(id="cup_0", category="cup", tags=[], parent="table_0",
                            mesh=_box((0.08, 0.08, 0.1), z=-0.5), confidence=0.8)  # z in [-0.55,-0.45]
    records = [table, sunk_cup]

    spec = assemble_scene(records, tmp_path / "out", scene_name="sunk_test",
                           capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                           stabilize=False)

    assert float(table.mesh.vertices[:, 2].min()) == pytest.approx(0.0, abs=1e-6)
    assert spec.floor_z == 0.0
    table_spec = next(o for o in spec.objects if o.id == "table_0")
    assert table_spec.provenance["ground_offset_z"] == pytest.approx(-0.3, abs=1e-9)


def test_gravity_align_untilts_a_tilted_scene(tmp_path):
    """Gravity canonicalisation: a low slab tilted 30 deg has an
    inflated vertical (z) extent; after _gravity_align its OBB up-axis returns
    to +Z and the z-extent collapses back to the true small height. This is the
    fix for the ~40 deg camera-tilt that made grounding scramble the layout."""
    from lens.pipeline import _gravity_align, _obb_up_axis_from_axes, _trimesh_obb_axes
    slab = trimesh.creation.box(extents=(1.0, 0.8, 0.2))   # true height 0.2
    th = np.radians(30.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    slab.vertices = np.asarray(slab.vertices) @ Rx.T        # tilt about x
    tilted_h = float(slab.bounding_box.extents[2])
    assert tilted_h > 0.3                                   # tilt inflated the z-extent

    # No model axes -> exercises the trimesh-OBB fallback (exact for a clean box).
    r = ReconObject(id="slab", category="table", tags=[], parent=None,
                    mesh=slab, confidence=1.0)
    _gravity_align([r])

    up = _obb_up_axis_from_axes(_trimesh_obb_axes(r.mesh), np.array([0.0, 0.0, 1.0]))
    assert abs(float(up[2])) > 0.999                        # OBB up is now +Z
    assert float(r.mesh.bounding_box.extents[2]) == pytest.approx(0.2, abs=0.03)


def test_assemble_scene_applies_mass_floor_for_near_degenerate_geometry(tmp_path):
    """M2 regression test: a near-flat object (rug/mat/sheet the VLM may
    list) yields CoACD hulls with ~0 total volume, so
    `mass = DENSITY_DEFAULT * volume` is ~0 -- `SceneSpec.validate()` would
    flag `mass must be > 0` and MuJoCo rejects a freejoint body with
    non-positive mass, crashing stabilize/export. A paper-thin
    (0.4 x 0.3 x 1e-6 m) slab decomposes to `DENSITY_DEFAULT * volume` ~=
    2.4e-5 kg, well under the MASS_FLOOR -- this fails `o.mass >= MASS_FLOOR`
    pre-fix (unfloored mass) and passes only once the floor is applied."""
    from lens.pipeline import MASS_FLOOR

    records = [
        ReconObject(id="rug_0", category="rug", tags=[], parent=None,
                    mesh=_box((0.4, 0.3, 1e-6), z=5e-7), confidence=0.9),
    ]
    spec = assemble_scene(records, tmp_path / "out", scene_name="thin_test",
                           capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                           stabilize=False)
    o = next(o for o in spec.objects if o.id == "rug_0")
    assert o.mass >= MASS_FLOOR
    assert not any("mass" in e for e in spec.validate())


def test_assemble_scene_grounds_using_global_min_when_no_floor_rooted_object(tmp_path):
    """When every record has a parent (no floor-rooted object at all),
    grounding falls back to the global minimum z across ALL records."""
    a = ReconObject(id="shelf_item_a", category="book", tags=[], parent="shelf_0",
                    mesh=_box((0.2, 0.2, 0.3), z=1.0), confidence=0.9)  # z in [0.85, 1.15]
    b = ReconObject(id="shelf_item_b", category="vase", tags=[], parent="shelf_0",
                    mesh=_box((0.15, 0.15, 0.25), z=0.9), confidence=0.8)  # z in [0.775, 1.025]
    records = [a, b]

    spec = assemble_scene(records, tmp_path / "out", scene_name="fallback_test",
                           capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                           stabilize=False)

    assert float(b.mesh.vertices[:, 2].min()) == pytest.approx(0.0, abs=1e-6)
    assert spec.floor_z == 0.0
    b_spec = next(o for o in spec.objects if o.id == "shelf_item_b")
    assert b_spec.provenance["ground_offset_z"] == pytest.approx(-0.775, abs=1e-6)


def _sofa_mesh():
    """An L-shaped sofa: a low seat plus a tall backrest along -y.

    A single box cannot expose the defect below, because for a box the
    global max-z and the max-z under any XY footprint are the same value.
    Real furniture is not a box, and that is exactly where the bug lives.
    Seat top = 0.35, backrest top = 0.85, both after grounding to z=0.
    """
    seat = trimesh.creation.box(extents=(1.2, 0.9, 0.35))
    seat.apply_translation((0.0, 0.0, 0.175))
    back = trimesh.creation.box(extents=(1.2, 0.15, 0.85))
    back.apply_translation((0.0, -0.375, 0.425))
    return trimesh.util.concatenate([seat, back])


def test_default_seating_is_the_global_top_rule(tmp_path):
    """DEFAULT (since 2026-08-12): children seat at the parent's GLOBAL max-z,
    the configuration every campaign ran under. The laptop over the SEAT
    therefore lands at backrest height (0.85)."""
    from lens.pipeline import CHILD_GAP
    sofa = ReconObject(id="sofa_0", category="sofa", tags=[], parent=None,
                       mesh=_sofa_mesh(), confidence=0.9)
    laptop = ReconObject(id="laptop_0", category="laptop", tags=[], parent="sofa_0",
                         mesh=_box_at((0.3, 0.2, 0.02), (0.0, 0.15, 1.0)), confidence=0.8)

    assemble_scene([sofa, laptop], tmp_path / "out", scene_name="default_seat",
                   capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                   stabilize=False)

    assert float(laptop.mesh.vertices[:, 2].min()) == pytest.approx(0.85 + CHILD_GAP, abs=1e-6)


def test_child_is_seated_on_the_support_surface_under_its_own_footprint(tmp_path, monkeypatch):
    """LENS_SUPPORT_TOP=1 (the parked IMPROVEMENT, off by default): a laptop
    lying on a sofa's SEAT is seated on the seat (0.35), not lifted to the top
    of the backrest (0.85).

    `_ground_records` seats every child at the parent's GLOBAL max-z, so both
    children of a sofa land on the backrest. On office_lounge that puts the
    sofa laptop 31.7 cm too high -- the single largest remaining placement
    error on that scene, and the only object worse than 2.6 cm. The
    backpack, whose footprint really is over the backrest, is unaffected:
    for it the local and global tops agree to 0.2 mm.
    """
    monkeypatch.setenv("LENS_SUPPORT_TOP", "1")
    from lens.pipeline import CHILD_GAP
    sofa = ReconObject(id="sofa_0", category="sofa", tags=[], parent=None,
                       mesh=_sofa_mesh(), confidence=0.9)
    # footprint entirely over the seat (y > -0.30), clear of the backrest
    laptop = ReconObject(id="laptop_0", category="laptop", tags=[], parent="sofa_0",
                         mesh=_box_at((0.3, 0.2, 0.02), (0.0, 0.15, 1.0)), confidence=0.8)
    # footprint over the backrest -- must NOT regress
    bag = ReconObject(id="bag_0", category="backpack", tags=[], parent="sofa_0",
                      mesh=_box_at((0.25, 0.14, 0.3), (0.4, -0.375, 1.4)), confidence=0.8)

    assemble_scene([sofa, laptop, bag], tmp_path / "out", scene_name="support_test",
                   capture_meta={"source": "p.jpg", "orig_size": (64, 64)},
                   stabilize=False)

    assert float(laptop.mesh.vertices[:, 2].min()) == pytest.approx(0.35 + CHILD_GAP, abs=1e-6)
    assert float(bag.mesh.vertices[:, 2].min()) == pytest.approx(0.85 + CHILD_GAP, abs=1e-6)


def test_collider_budget_is_per_object_and_parent_aware(tmp_path, monkeypatch):
    """`assemble_scene` shipped a flat `max_hulls=32` for every object. The
    budget is now picked per object from the asset name, and every
    office_lounge asset takes 8 -- four times less concave than what
    MONOLITHCLE was building, which is what lets a small object slip into a
    crease of the sofa mesh instead of resting on top of it.

    The child's budget must follow its PARENT's keywords too, because a
    child is named `<child>_on_<parent>`.
    """
    sofa = ReconObject(id="grey_sofa_0", category="sofa", tags=["surface"],
                       parent=None, mesh=_sofa_mesh(), confidence=0.9)
    laptop = ReconObject(id="black_laptop_0", category="laptop", tags=[],
                         parent="grey_sofa_0",
                         mesh=_box((0.3, 0.2, 0.02), z=0.5), confidence=0.8)
    plant = ReconObject(id="potted_plant_0", category="plant", tags=[],
                        parent=None, mesh=_box((0.3, 0.3, 0.6), z=0.3),
                        confidence=0.8)
    thermos = ReconObject(id="black_thermos_0", category="thermos", tags=[],
                          parent=None, mesh=_box((0.08, 0.08, 0.2), z=0.1),
                          confidence=0.8)

    # assemble_scene decomposes in record order, so the call order identifies
    # which object each budget belongs to.
    order = ["grey_sofa_0", "black_laptop_0", "potted_plant_0", "black_thermos_0"]
    calls: list[int] = []
    from lens import pipeline as _pipeline
    real = _pipeline.collision.decompose

    def spy(mesh, max_hulls=32, **kw):
        calls.append(max_hulls)
        return real(mesh, max_hulls=max_hulls, **kw)

    monkeypatch.setattr(_pipeline.collision, "decompose", spy)

    assemble_scene([sofa, laptop, plant, thermos], tmp_path / "out",
                   scene_name="budget", capture_meta={"source": "p.jpg"},
                   stabilize=False)

    got = dict(zip(order, calls))
    assert got["grey_sofa_0"] == 8, "sofa is on the blocky list"
    assert got["black_laptop_0"] == 8, "child inherits its sofa parent's budget"
    assert got["potted_plant_0"] == 32, "organic keywords take the 32 branch"
    assert got["black_thermos_0"] == 16, "no keyword matches -> the default"

# NOTE (2026-08-12): an in-suite CEM placement test that lived here was DELETED.
# It asserted <5 cm against a LOCAL HAND-EDIT of a scene (backpack moved 0.357 m
# by hand on 2026-07-22), not pipeline output. Placement quality is measured by
# the settle harness on exported scenes, not by an in-suite CEM run.


# ---------------------------------------------------------------------------
# Ship-gate integration (overnight 2026-08-14)
# ---------------------------------------------------------------------------

def test_pipeline_records_ship_verdict_in_scene_json(tmp_path):
    """assemble_scene must run the ship-gate on the exported MJCF and record
    the verdict in scene.json, round-trippable through SceneSpec.from_json.
    The scene artifacts themselves must exist regardless of the verdict --
    the gate reports, it never alters or deletes."""
    import json

    from lens.scenespec import SceneSpec

    Image.new("RGB", (64, 64), (90, 90, 90)).save(tmp_path / "p.jpg")
    m = np.zeros((64, 64), bool); m[10:30, 10:30] = True
    vlm = FakeVLM(["grey crate | box | floor", "YES"])
    spec = run_pipeline(tmp_path / "p.jpg", tmp_path / "out",
                        vlm=vlm, segmenter=FakeSegmenter({"grey crate": [m]}),
                        recon=FakeRecon(), stabilize=False)

    assert spec.verify is not None
    assert spec.verify["verdict"] in ("PASS", "WARN", "FAIL")

    on_disk = json.loads((tmp_path / "out/scene.json").read_text())
    assert on_disk["verify"]["verdict"] == spec.verify["verdict"]
    reloaded = SceneSpec.from_json(tmp_path / "out/scene.json")
    assert reloaded.verify == spec.verify

    # gate never deletes/alters the artifacts
    assert list((tmp_path / "out/mujoco").glob("*.xml"))
    assert (tmp_path / "out/scene.json").exists()


def test_cli_exits_nonzero_when_ship_verdict_is_fail(tmp_path, monkeypatch, capsys):
    """lens run must propagate a FAIL verdict as a nonzero exit code and say
    so loudly -- without touching the scene on disk."""
    from lens import cli
    from lens.scenespec import SceneSpec

    def fake_run(photo, out, model=None, stabilize=True):
        return SceneSpec(name="s", objects=[],
                         verify={"verdict": "FAIL",
                                 "reasons": ["divergence: BADQACC at t=1.3s on body 'cup' (dof 24)"]})

    monkeypatch.setattr(cli, "run_pipeline_isolated", fake_run)
    rc = cli.main(["run", str(tmp_path / "p.jpg"), "--out", str(tmp_path / "out")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "FAIL" in err and "divergence" in err

    def fake_run_pass(photo, out, model=None, stabilize=True):
        return SceneSpec(name="s", objects=[], verify={"verdict": "PASS", "reasons": []})

    monkeypatch.setattr(cli, "run_pipeline_isolated", fake_run_pass)
    rc = cli.main(["run", str(tmp_path / "p.jpg"), "--out", str(tmp_path / "out")])
    assert rc == 0


# ---------------------------------------------------------------------------
# De-penetration pass integration (overnight 2026-08-14)
# ---------------------------------------------------------------------------

def _overlapping_box_records():
    """Two floor-rooted boxes overlapping 50 mm in x, world pose baked into
    the vertices (pipeline convention)."""
    a = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    a.apply_translation((0.0, 0.0, 0.1))
    b = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    b.apply_translation((0.15, 0.0, 0.1))
    return [
        ReconObject(id="crate_a_0", category="box", tags=[], parent=None,
                    mesh=a, confidence=1.0),
        ReconObject(id="crate_b_0", category="box", tags=[], parent=None,
                    mesh=b, confidence=1.0),
    ]


def _worst_root_pen_mm(xml_path):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0:
            continue
        worst = max(worst, -min(0.0, float(c.dist)))
    return worst * 1000


def test_depenetrate_flag_resolves_root_overlap_in_shipped_xml(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DEPENETRATE", "1")
    monkeypatch.setenv("LENS_VERIFY_SECONDS", "2")
    spec = assemble_scene(_overlapping_box_records(), tmp_path / "out",
                          scene_name="depen", capture_meta={"source": "p.jpg"},
                          stabilize=False)
    xml = tmp_path / "out/mujoco/depen.xml"
    assert _worst_root_pen_mm(xml) <= 2.1
    moved = [o for o in spec.objects if "depenetrate" in o.provenance]
    assert len(moved) == 1


def test_depenetrate_default_on_resolves_overlap(tmp_path, monkeypatch):
    """Since 2026-08-17 (after a sliding A/B on the development scenes) the
    pass is ON by default; LENS_DEPENETRATE=0 opts out."""
    monkeypatch.delenv("LENS_DEPENETRATE", raising=False)
    monkeypatch.setenv("LENS_VERIFY_SECONDS", "2")
    spec = assemble_scene(_overlapping_box_records(), tmp_path / "out",
                          scene_name="depen", capture_meta={"source": "p.jpg"},
                          stabilize=False)
    xml = tmp_path / "out/mujoco/depen.xml"
    assert _worst_root_pen_mm(xml) <= 2.1
    assert any("depenetrate" in o.provenance for o in spec.objects)


def test_depenetrate_can_be_disabled_with_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DEPENETRATE", "0")
    monkeypatch.setenv("LENS_VERIFY_SECONDS", "2")
    spec = assemble_scene(_overlapping_box_records(), tmp_path / "out",
                          scene_name="depen", capture_meta={"source": "p.jpg"},
                          stabilize=False)
    xml = tmp_path / "out/mujoco/depen.xml"
    assert _worst_root_pen_mm(xml) > 20.0
    assert all("depenetrate" not in o.provenance for o in spec.objects)
