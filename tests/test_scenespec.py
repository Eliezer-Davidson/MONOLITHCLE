from pathlib import Path
from lens.scenespec import ObjectSpec, SceneSpec

def make_obj(**kw):
    d = dict(id="table_0", category="table", visual_mesh="meshes/table_0.obj",
             collision_meshes=["meshes/table_0_col00.obj"],
             pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
             dims=(0.6, 0.6, 0.45), mass=8.0, friction=0.9,
             mass_source="default", scale_source="pointmap",
             confidence=0.8, tags=["surface"], provenance={})
    d.update(kw); return ObjectSpec(**d)

def test_roundtrip(tmp_path: Path):
    s = SceneSpec(name="t", objects=[make_obj()], capture={"src": "x.jpg"})
    p = tmp_path / "scene.json"; s.to_json(p)
    s2 = SceneSpec.from_json(p)
    assert s2.objects[0].dims == (0.6, 0.6, 0.45) and s2.up_axis == "z"

def test_validate_catches_bad_quat_and_dupe_ids(tmp_path: Path):
    s = SceneSpec(name="t", objects=[make_obj(), make_obj(quat=(0, 0, 0, 0))])
    errs = s.validate()
    assert any("quat" in e for e in errs) and any("duplicate id" in e for e in errs)

def test_validate_catches_wrong_length_quat():
    """Test that validate() rejects quaternions with wrong arity."""
    # 3-element quat with unit norm (0.6, 0.8, 0.0) has norm=1.0
    # This ensures the arity check catches it, not the unit-norm check
    s = SceneSpec(name="t", objects=[make_obj(quat=(0.6, 0.8, 0.0))])
    errs = s.validate()
    assert any("quat" in e for e in errs), f"Expected quat error, got {errs}"

def test_validate_catches_wrong_length_pos():
    """Test that validate() rejects pos with wrong arity."""
    s = SceneSpec(name="t", objects=[make_obj(pos=(0.0, 0.0))])
    errs = s.validate()
    assert any("pos" in e for e in errs), f"Expected pos error, got {errs}"

def test_validate_catches_wrong_length_dims():
    """Test that validate() rejects dims with wrong arity."""
    s = SceneSpec(name="t", objects=[make_obj(dims=(0.6, 0.6))])
    errs = s.validate()
    assert any("dims" in e for e in errs), f"Expected dims error, got {errs}"

def test_validate_catches_non_positive_meters_per_unit():
    """Test that validate() rejects non-positive meters_per_unit."""
    # Test negative
    s = SceneSpec(name="t", objects=[make_obj()], meters_per_unit=-1.0)
    errs = s.validate()
    assert any("meters_per_unit" in e for e in errs), f"Expected meters_per_unit error for -1.0, got {errs}"

    # Test zero
    s = SceneSpec(name="t", objects=[make_obj()], meters_per_unit=0.0)
    errs = s.validate()
    assert any("meters_per_unit" in e for e in errs), f"Expected meters_per_unit error for 0.0, got {errs}"
