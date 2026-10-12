#!/usr/bin/env python3
# MONOLITHCLE (package `lens`). CLI for the settle-measurement harness (overnight 2026-08-14).
"""Measure a shipped MJCF's delivered-physics settle: per-body drift AND tilt
at each window, divergence-guarded (the fixes over the old settle_delivered.py:
30 s reach, rotation, clock stops at the first warning, escape flags).

Usage: measure_scene.py <scene.xml> --label <name> --out <out.json>
                        [--windows 1.5,3,5,10,30] [--escape-m 5]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lens.measure import ESCAPE_M, WINDOWS, measure_scene


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default=",".join(str(w) for w in WINDOWS),
                    help="comma-separated seconds, e.g. 1.5,3,5,10,30")
    ap.add_argument("--escape-m", type=float, default=ESCAPE_M)
    args = ap.parse_args()

    windows = tuple(float(w) for w in args.windows.split(","))
    res = measure_scene(args.xml, windows=windows, escape_m=args.escape_m)
    res["label"] = args.label
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"\n=== {args.label} ===")
    print(f"  dt={res['timestep']}  nbody={res['nbody']}  "
          f"free={len(res['free_bodies'])}  diverged={res['diverged']}")
    if res["escaped"]:
        print(f"  ESCAPED: {res['escaped']}")
    last_valid = None
    for w in reversed(res["windows_s"]):
        if res["measurements"][str(w)]["valid"]:
            last_valid = str(w)
            break
    if last_valid is None:
        print("  no valid window (diverged before the first)")
        return 0
    bodies = res["measurements"][last_valid]["bodies"]
    print(f"  {'body':<42}{'drift_cm@' + last_valid + 's':>16}{'tilt_deg':>10}")
    for nm in sorted(bodies, key=lambda n: -bodies[n]["drift_cm"]):
        print(f"  {nm:<42}{bodies[nm]['drift_cm']:>16.2f}{bodies[nm]['tilt_deg']:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
