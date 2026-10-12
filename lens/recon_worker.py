# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Model-heavy worker subprocess: capture -> VLM object list ->
scene tree -> segmentation -> reconstruction, then exit.

Run standalone: `python -m lens.recon_worker --photo PATH --out DIR [--model MODEL]`.

Why a subprocess: SAM3 + SAM-3D-Objects together hold ~16 GB of resident
memory once loaded. Calling `.release()` on them frees the Python objects,
but glibc/pymalloc do not hand that heap back to the OS -- the process's
RSS stays high, and the stabilize stage (MuJoCo + CEM) loading on top of it
is what OOM-killed the machine on the first real end-to-end run. Running
this half of the pipeline in its own process and letting it EXIT is the
only way that memory is actually returned to the OS before stabilize runs.
The parent (`lens.pipeline.run_pipeline_isolated`) never imports this
module -- it only shells out to it by name, so it can stay light.

Output contract with the parent: `<out>/meshes/<id>.obj` per reconstructed
object, plus `<out>/recon_state.json` (schema: see `lens.pipeline._assemble_from_state`). The
parent loads the meshes back off disk and does everything from there
(`lens.pipeline.assemble_scene` / `_assemble_from_state`).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lens.capture import load_capture
from lens.prompts import LIST_OBJECTS
from lens.scenegraph.masks import Sam3Segmenter, segment_objects
from lens.scenegraph.tree import build_tree, parse_object_lines
from lens.vlm import GeminiVLM

log = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lens.recon_worker")
    p.add_argument("--photo", required=True, type=Path, help="Path to the input photo.")
    p.add_argument("--out", required=True, type=Path, help="Output directory (shared with the parent).")
    p.add_argument("--model", default=None, help="Override the Gemini model.")
    return p


def run_worker(photo: Path, out_dir: Path, model: str | None = None) -> None:
    """Do the model-heavy half of the pipeline and write its contract to
    disk. Builds the REAL backends itself -- this is the only place in the
    codebase that resident-loads SAM3/SAM-3D-Objects."""
    # Deferred: Sam3dObjectsBackend's own import chain is heavy, and every
    # other symbol this module needs is cheap at module scope -- keeping
    # this one deferred makes it easy to see, from the top of the file,
    # exactly which import pulls the multi-GB model stack.
    from lens.recon import Sam3dObjectsBackend

    out_dir = Path(out_dir)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    vlm = GeminiVLM(model=model) if model else GeminiVLM()
    segmenter = Sam3Segmenter()
    recon = Sam3dObjectsBackend()

    log.info("capture: loading %s", photo)
    capture = load_capture(photo)

    log.info("vlm: listing objects")
    reply = vlm.ask(LIST_OBJECTS, [capture.image])

    tree = build_tree(parse_object_lines(reply))
    log.info("scenegraph: %d objects in tree", len(tree))

    masks = segment_objects(capture.image, tree, segmenter, vlm, max_alternatives=3)
    log.info("masks: segmented %d/%d objects", len(masks), len(tree))

    segmenter.release()
    log.info("masks: released segmentation model")

    recon_results = recon.reconstruct(capture.image, masks)
    log.info("recon: reconstructed %d meshes", len(recon_results))
    recon.release()

    objects = []
    for o in tree:
        r = recon_results.get(o.id)
        if r is None:
            continue  # no mask/recon for this object -- simply absent downstream
        vis_rel = f"meshes/{o.id}.obj"
        r.mesh.export(out_dir / vis_rel)
        objects.append({
            "id": o.id,
            "category": o.category,
            "tags": list(o.tags),
            "parent": o.parent,
            "visual_mesh": vis_rel,
            "confidence": float(r.confidence),
            # Clean model box axes for the parent's gravity_align per-object snap;
            # the raw posed mesh alone only yields a noisy trimesh OBB (which
            # flips objects). None-safe: an older state simply falls back.
            "obb_axes": r.obb_axes.tolist() if r.obb_axes is not None else None,
            # Layout post-opt provenance (iou before/after, accepted) or None
            # when post-opt didn't run for this object.
            "postopt": r.postopt,
        })

    state = {
        "scene_name": Path(photo).stem or "scene",
        "capture": {"source": capture.source, "orig_size": capture.orig_size},
        "objects": objects,
    }
    (out_dir / "recon_state.json").write_text(json.dumps(state, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        run_worker(args.photo, args.out, args.model)
    except Exception:
        log.exception("recon_worker: failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
