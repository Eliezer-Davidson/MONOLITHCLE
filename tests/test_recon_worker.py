# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Tests for the process-isolated reconstruction path: the
recon_state.json contract and the parent-side orchestration around the
worker subprocess. NONE of these tests import torch/sam3/sam3d_objects or
spawn the real worker -- `subprocess.run` is monkeypatched throughout."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import trimesh

from lens.pipeline import _assemble_from_state, run_pipeline_isolated


def _write_mesh_fixture(path: Path, extents=(0.2, 0.2, 0.2), z=0.1):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation((0.0, 0.0, z))
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def _write_state(out_dir: Path):
    """Hand-write a recon_state.json + two tiny .obj fixtures, exactly the
    contract the real worker is specified to produce."""
    _write_mesh_fixture(out_dir / "meshes/table_0.obj", extents=(0.6, 0.4, 0.4), z=0.2)
    _write_mesh_fixture(out_dir / "meshes/cup_0.obj", extents=(0.08, 0.08, 0.1), z=0.45)
    state = {
        "scene_name": "mock_scene",
        "capture": {"source": "p.jpg", "orig_size": [64, 64]},
        "objects": [
            {"id": "table_0", "category": "table", "tags": ["surface"],
             "parent": None, "visual_mesh": "meshes/table_0.obj", "confidence": 0.9},
            {"id": "cup_0", "category": "cup", "tags": ["graspable"],
             "parent": "table_0", "visual_mesh": "meshes/cup_0.obj", "confidence": 0.8},
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recon_state.json").write_text(json.dumps(state, indent=2))
    return state


def test_assemble_from_state_round_trip(tmp_path):
    out_dir = tmp_path / "out"
    _write_state(out_dir)

    spec = _assemble_from_state(out_dir, stabilize=False)

    assert spec.validate() == []
    assert spec.name == "mock_scene"
    assert [o.id for o in spec.objects] == ["table_0", "cup_0"]
    cats = {o.id: o.category for o in spec.objects}
    assert cats == {"table_0": "table", "cup_0": "cup"}
    cup = next(o for o in spec.objects if o.id == "cup_0")
    assert cup.provenance["parent"] == "table_0"
    assert cup.confidence == pytest.approx(0.8)
    assert cup.tags == ["graspable"]
    table = next(o for o in spec.objects if o.id == "table_0")
    assert table.provenance["parent"] is None
    assert (out_dir / "scene.json").exists()


def test_run_pipeline_isolated_spawns_worker_with_correct_argv(tmp_path, monkeypatch):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"not a real jpeg -- argv test only, never decoded")
    out_dir = tmp_path / "out"
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        _write_state(out_dir)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    spec = run_pipeline_isolated(photo, out_dir, model="gemini-2.5-flash", stabilize=False)

    argv = captured["argv"]
    assert "-m" in argv and "lens.recon_worker" in argv
    assert str(photo) in argv
    assert str(out_dir) in argv
    assert "--model" in argv and "gemini-2.5-flash" in argv
    # The worker's own stage logs are the only external visibility into
    # progress -- the parent must inherit stdout/stderr, never pipe them.
    assert captured["kwargs"].get("stdout") is None
    assert captured["kwargs"].get("stderr") is None

    assert spec.validate() == []
    assert [o.id for o in spec.objects] == ["table_0", "cup_0"]


def test_run_pipeline_isolated_no_model_omits_flag(tmp_path, monkeypatch):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"not a real jpeg -- argv test only, never decoded")
    out_dir = tmp_path / "out"
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        _write_state(out_dir)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_pipeline_isolated(photo, out_dir, stabilize=False)
    assert "--model" not in captured["argv"]


def test_run_pipeline_isolated_raises_on_worker_failure(tmp_path, monkeypatch):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"not a real jpeg -- argv test only, never decoded")
    out_dir = tmp_path / "out"

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        run_pipeline_isolated(photo, out_dir, stabilize=False)


def test_run_worker_state_carries_postopt_provenance(tmp_path, monkeypatch):
    """recon_state.json objects carry the postopt IoU provenance when the
    backend reports it (and omit nothing for objects without it)."""
    import json
    import types
    import numpy as np
    import trimesh
    import lens.recon_worker as rw
    from lens.recon import ReconResult

    img = np.zeros((8, 8, 3), np.uint8)
    cap = types.SimpleNamespace(image=img, source="x.jpg", orig_size=(8, 8))
    monkeypatch.setattr(rw, "load_capture", lambda p: cap)
    monkeypatch.setattr(rw, "GeminiVLM",
                        lambda **kw: types.SimpleNamespace(ask=lambda *a, **k: "ignored"))
    monkeypatch.setattr(rw, "Sam3Segmenter",
                        lambda: types.SimpleNamespace(release=lambda: None))

    obj = types.SimpleNamespace(id="obj_a", category="box", tags=(), parent="floor")
    monkeypatch.setattr(rw, "parse_object_lines", lambda reply: [obj])
    monkeypatch.setattr(rw, "build_tree", lambda objs: objs)
    mask = np.ones((8, 8), bool)
    monkeypatch.setattr(rw, "segment_objects", lambda *a, **k: {"obj_a": mask})

    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    res = {"obj_a": ReconResult(mesh=box, confidence=1.0, obb_axes=np.eye(3),
                                postopt={"iou": 0.88, "iou_before_optim": 0.5,
                                         "optim_accepted": True})}

    class _FakeBackend:
        def reconstruct(self, image, masks): return res
        def release(self): pass

    import lens.recon as recon_mod
    monkeypatch.setattr(recon_mod, "Sam3dObjectsBackend", lambda **kw: _FakeBackend())

    rw.run_worker(tmp_path / "x.jpg", tmp_path)
    state = json.loads((tmp_path / "recon_state.json").read_text())
    assert state["objects"][0]["postopt"] == {"iou": 0.88, "iou_before_optim": 0.5,
                                              "optim_accepted": True}
