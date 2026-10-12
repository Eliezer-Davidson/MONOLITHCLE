"""Replay one captured layout post-opt bundle TWICE and require bit-identical
results — the runtime proof that the deferred replay is a pure function of the
bundle.

Usage: probe_determinism.py <bundle.pt>   (needs the GPU reconstruction env with
the patched SAM-3D-Objects checkout that provides `deferred_postopt` -- see
THIRD_PARTY_NOTICES.md -- plus the CUDA toolchain vars). Exit 0 = PASS.
"""
import sys

import torch

from sam3d_objects.pipeline.deferred_postopt import run_deferred_postopt


def main() -> int:
    bundle = sys.argv[1]
    print(f"probe: replaying {bundle} twice")
    r1 = run_deferred_postopt(bundle)
    r2 = run_deferred_postopt(bundle)
    ok = True
    for k in ("rotation", "translation", "scale"):
        t1, t2 = torch.as_tensor(r1[k]), torch.as_tensor(r2[k])
        same = t1.shape == t2.shape and torch.equal(t1, t2)
        print(f"  {k:12s}: {'bit-identical' if same else 'DIFFERS'}")
        ok &= same
    for k in ("iou", "iou_before_optim", "optim_accepted"):
        same = r1[k] == r2[k]
        print(f"  {k:16s}: {r1[k]} vs {r2[k]} -> {'same' if same else 'DIFFERS'}")
        ok &= same
    print(f"DETERMINISM PROBE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
