#!/usr/bin/env python3
# MONOLITHCLE (package `lens`). CLI for the scene ship-gate (overnight 2026-08-14).
"""Verify a shipped MJCF scene: guarded settle + ship verdict.

Usage: verify_scene.py <scene.xml> [--seconds 30] [--pen-mm 20] [--escape-m 5]
                       [--topple-deg 30] [--jitter 1.0] [--json out.json]

Exit code 0 only on PASS (1 = FAIL, 2 = WARN).  A usage error exits 64
(EX_USAGE) so a malformed invocation can never be misread as a verdict.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lens.verify_scene import (ESCAPE_M, JITTER_THRESHOLD, PEN_THRESHOLD_MM,
                               TOPPLE_DEG, verify_scene)


EX_USAGE = 64  # argparse's own 2 would be indistinguishable from WARN


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(EX_USAGE, f"{self.prog}: error: {message}\n")


def main() -> int:
    ap = _Parser()
    ap.add_argument("xml")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--pen-mm", type=float, default=PEN_THRESHOLD_MM)
    ap.add_argument("--escape-m", type=float, default=ESCAPE_M)
    ap.add_argument("--topple-deg", type=float, default=TOPPLE_DEG)
    ap.add_argument("--jitter", type=float, default=JITTER_THRESHOLD)
    ap.add_argument("--json", help="write the full verdict as JSON here")
    args = ap.parse_args()

    v = verify_scene(args.xml, seconds=args.seconds, pen_threshold_mm=args.pen_mm,
                     escape_m=args.escape_m, topple_deg=args.topple_deg,
                     jitter_threshold=args.jitter)

    print(f"{v.verdict}: {args.xml} (settled {v.settled_s}s, "
          f"worst pen {v.worst_penetration_mm:.1f} mm)")
    for r in v.reasons:
        print(f"  - {r}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dataclasses.asdict(v), f, indent=1)

    return {"PASS": 0, "FAIL": 1, "WARN": 2}[v.verdict]


if __name__ == "__main__":
    sys.exit(main())
