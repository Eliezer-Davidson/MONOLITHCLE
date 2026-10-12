# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""`lens` console script: photo -> physics-valid sim scene, end to end.

    lens run <photo> --out <dir> [--no-stabilize] [--vlm gemini|fake]

`--vlm fake` is refused: Fake* backends exist only so tests can run without
GPU/network, and always produce placeholder boxes -- letting someone build
a real scene from them by accident would be a silent correctness trap.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lens.pipeline import run_pipeline_isolated

_FAKE_VLM_ERROR = (
    "lens run --vlm fake: fake backends (FakeVLM/FakeSegmenter/FakeRecon) "
    "exist only for tests -- they produce placeholder boxes, not a real "
    "scene. Use --vlm gemini for a real run."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lens", description="Photo -> physics-valid sim scene.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full photo -> scene pipeline.")
    run_p.add_argument("photo", type=Path, help="Path to the input photo.")
    run_p.add_argument("--out", type=Path, required=True,
                        help="Output directory for meshes/exports/scene.json.")
    run_p.add_argument("--no-stabilize", action="store_true",
                        help="Skip the CEM + MuJoCo settle stabilization pass.")
    run_p.add_argument("--vlm", choices=["gemini", "fake"], default="gemini",
                        help="VLM/segmenter/recon backend stack to use.")
    run_p.add_argument("--model", default=None,
                        help="Override the Gemini model (e.g. gemini-2.5-flash). "
                             "Useful when the default tier is returning 503s.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "run":
        if args.vlm == "fake":
            print(_FAKE_VLM_ERROR, file=sys.stderr)
            return 2
        # The model-heavy half runs in a worker subprocess (lens.recon_worker)
        # that exits before stabilize runs -- see run_pipeline_isolated's
        # docstring for why an in-process release() alone
        # was not enough to avoid the OOM-kill at the stabilize stage.
        spec = run_pipeline_isolated(
            args.photo, args.out, model=args.model,
            stabilize=not args.no_stabilize,
        )
        errs = spec.validate()
        if errs:
            print(f"warning: scene has validation issues: {errs}", file=sys.stderr)
        print(f"wrote scene to {args.out}")
        # Ship-gate FAIL propagates as a nonzero exit. The scene stays on disk
        # untouched -- the gate reports, it never deletes.
        if spec.verify and spec.verify.get("verdict") == "FAIL":
            print("SHIP-GATE FAIL — scene kept on disk for inspection:",
                  file=sys.stderr)
            for r in spec.verify.get("reasons", []):
                print(f"  - {r}", file=sys.stderr)
            return 3
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
