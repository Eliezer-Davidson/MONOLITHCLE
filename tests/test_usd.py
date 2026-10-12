import os
import shutil
import subprocess
import sys

import pytest

# A Python interpreter that has `usd-core` (pxr) installed. These tests force
# the out-of-process export path, so they need it named explicitly; without it
# they skip.
PXR_PY = os.environ.get("LENS_PXR_PYTHON", "")
_HAVE_PXR_PY = bool(PXR_PY) and (bool(shutil.which(PXR_PY)) or os.path.exists(PXR_PY))
_NO_PXR_PY = "set LENS_PXR_PYTHON to a Python with usd-core installed"


@pytest.mark.skipif(not _HAVE_PXR_PY, reason=_NO_PXR_PY)
def test_usd_written_with_physics(tmp_path):
    """Runs whichever path `write_scene` actually takes in this interpreter:
    in-process pxr when usd-core is importable here, otherwise the
    LENS_PXR_PYTHON subprocess. The other tests below force the subprocess
    path explicitly, so that mechanism is exercised for real rather than by
    happenstance."""
    import trimesh
    from lens.export import collision, usd, mjcf
    from lens.scenespec import SceneSpec
    from tests.test_scenespec import make_obj

    box = trimesh.creation.box(extents=(0.2,) * 3)
    box.apply_translation((0, 0, 0.1))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    out = usd.write_scene(SceneSpec(name="t", objects=[make_obj()], capture={}), tmp_path)
    chk = subprocess.run([PXR_PY, "-c", f"""
from pxr import Usd, UsdPhysics
s = Usd.Stage.Open('{out}')
rb = sum(1 for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
co = sum(1 for p in s.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
print(rb, co)"""], capture_output=True, text=True)
    assert chk.stdout.split() == ["1", "2"]          # 1 body; 1 hull + floor


def _make_scene(tmp_path):
    import trimesh
    from lens.export import collision
    from lens.scenespec import SceneSpec
    from tests.test_scenespec import make_obj

    box = trimesh.creation.box(extents=(0.2,) * 3)
    box.apply_translation((0, 0, 0.1))
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    return SceneSpec(name="t", objects=[make_obj()], capture={})


@pytest.mark.skipif(not _HAVE_PXR_PY, reason=_NO_PXR_PY)
def test_usd_subprocess_path_actually_works(tmp_path, monkeypatch):
    """`write_scene` must fall back to a `PXR_PY` subprocess when `import pxr`
    fails in-process. On THIS machine pxr also happens to be importable in
    the reconstruction env (a `kaolin` dependency pulled in usd-core), so the test
    above exercises the in-process branch, not the subprocess branch -- that
    would leave the subprocess mechanism completely unverified. Force the
    fallback deterministically by shadowing `sys.modules['pxr']` with `None`,
    which makes a bare `import pxr` raise ImportError regardless of whether
    the real package is installed (standard CPython import semantics), then
    confirm the child process -- which must import `lens.scenespec` via
    PYTHONPATH to deserialize the spec -- produces the same physics-API
    counts as the in-process path above."""
    from lens.export import usd
    from lens.scenespec import SceneSpec

    spec = _make_scene(tmp_path)
    monkeypatch.setitem(sys.modules, "pxr", None)
    out = usd.write_scene(spec, tmp_path)
    monkeypatch.undo()  # restore before using pxr again in this process, for the check below

    assert out.exists()
    chk = subprocess.run([PXR_PY, "-c", f"""
from pxr import Usd, UsdPhysics
s = Usd.Stage.Open('{out}')
rb = sum(1 for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
co = sum(1 for p in s.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
print(rb, co)"""], capture_output=True, text=True)
    assert chk.stdout.split() == ["1", "2"]


@pytest.mark.skipif(not _HAVE_PXR_PY, reason=_NO_PXR_PY)
def test_usd_subprocess_mass_and_pose_applied_as_xform_op(tmp_path, monkeypatch):
    """Mass must reach USD via MassAPI.mass (not baked density), and pos/quat
    must be an Xform op on `/World/<id>` (not baked into mesh vertices).
    Forces the subprocess path as above so the child's output is checked."""
    from lens.export import usd
    from tests.test_scenespec import make_obj
    from lens.scenespec import SceneSpec
    import trimesh
    from lens.export import collision

    box = trimesh.creation.box(extents=(0.2,) * 3)
    (tmp_path / "meshes").mkdir()
    box.export(tmp_path / "meshes/table_0.obj")
    collision.decompose(box)[0].export(tmp_path / "meshes/table_0_col00.obj")
    obj = make_obj(pos=(1.5, -2.0, 0.3), quat=(0.7071068, 0.0, 0.0, 0.7071068), mass=12.5)
    spec = SceneSpec(name="posed", objects=[obj], capture={})

    monkeypatch.setitem(sys.modules, "pxr", None)
    out = usd.write_scene(spec, tmp_path)
    monkeypatch.undo()

    chk = subprocess.run([PXR_PY, "-c", f"""
from pxr import Usd, UsdPhysics, UsdGeom
s = Usd.Stage.Open('{out}')
p = s.GetPrimAtPath('/World/table_0')
mass = UsdPhysics.MassAPI(p).GetMassAttr().Get()
xf = UsdGeom.Xformable(p)
ops = xf.GetOrderedXformOps()
names = [o.GetOpName() for o in ops]
t = [op for op in ops if op.GetOpName() == 'xformOp:translate'][0].Get()
print(mass, list(t), names)"""], capture_output=True, text=True)
    assert chk.returncode == 0, chk.stderr
    out_line = chk.stdout.strip()
    assert "12.5" in out_line
    assert "1.5, -2.0, 0.3" in out_line
    assert "xformOp:translate" in out_line and "xformOp:orient" in out_line


@pytest.mark.skipif(not _HAVE_PXR_PY, reason=_NO_PXR_PY)
def test_usd_subprocess_failure_surfaces_stderr(tmp_path, monkeypatch):
    """If the child process exits non-zero, `write_scene` must raise with the
    child's stderr included -- not fail silently or swallow the error."""
    from lens.export import usd
    from tests.test_scenespec import make_obj
    from lens.scenespec import SceneSpec

    # Deliberately do NOT create the mesh files -- the child will hit a
    # missing-file error while loading the visual mesh and exit non-zero.
    obj = make_obj()
    spec = SceneSpec(name="broken", objects=[obj], capture={})

    monkeypatch.setitem(sys.modules, "pxr", None)
    with pytest.raises(RuntimeError) as excinfo:
        usd.write_scene(spec, tmp_path)
    monkeypatch.undo()

    msg = str(excinfo.value)
    assert "table_0" in msg or "does not exist" in msg or ".obj" in msg
