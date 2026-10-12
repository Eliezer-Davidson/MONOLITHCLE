# CEM scoring-vs-delivered physics mismatch — spike findings (2026-08-06)

**Status:** findings only. No fix implemented. All spike instrumentation reverted
(nothing left in `lens/`). This doc records what a controlled spike proved, so the
fix can be designed rather than guessed.

**Scenes.** Two development scenes are used throughout, and they certify the
machinery, never capability. *Scene A* is a photographed office lounge; *scene B*
is a drawn interior. Neither source image is redistributed here (scene A's
photograph is a private capture; scene B's drawing is third-party artwork), so
no render, mesh or photograph of either scene ships with this document — only
the measurements. The two scenes
are referred to as A and B throughout; the JSON files beside this document were
renamed to match, and their `label` and `xml` string fields carry the same
renaming. Nothing else in them was altered.

**Caveat on the JSON `integrity` field.** Each run records `STALE [...]` listing
visual meshes that were older than the run. The spike deliberately reused the
stored collision hulls instead of regenerating them (see §7), which is what makes
the physics comparable across the four regimes; the visual meshes play no part in
the measurement (`mass=0`, `contype/conaffinity=0`). The flag is honest bookkeeping,
not a warning about the numbers below.

**Caveat on the collider budget.** On 2026-09-23 the per-object convex-hull
budget rule was re-authored and one class changed: a bench now
takes 8 hulls where it previously took 16. These runs predate that change.

**Grounding configuration.** These eight runs were produced with the
non-default `_support_top` seating rule **ON** — a child is seated on its
parent's surface under its own footprint rather than at the parent's global
highest vertex. That is not the shipped default and it is not what the held-out
campaign of 2026-08-13 ran under (`LENS_SUPPORT_TOP` unset there). It is what
makes scene A's backpack stable at 1.1 cm under R0 — see §6(a).

Raw evidence: the eight JSON runs beside this file
(`docs/evidence/cem-physics-spike-2026-08-06/scene{A,B}_R{0,1,2,3}.json`).
Every drift number quoted below can be read straight out of them.

---

## 1. Where this started

The task was to prove and land a 2026-08-05 change (multiprocess CEM,
`_support_top` seating, vertex colours, COM-drift metric, coarse-to-fine window).
Its own note claimed the coarse-to-fine window *lengthened the scoring window and
fixed scene B's floor-lamp topple*. Reading the code contradicted that:
`FINE_SETTLE_SECONDS = SETTLE_SECONDS = 1.0` — the window was made coarse-to-fine
(cheap short settles early) but **not lengthened**, and the note's "fixed" claim was
never measured. That triggered the investigation below.

## 2. The lamp geometry is feasible

A tempting framing: scene B is a *drawing*, so maybe the lamp just isn't a
physically feasible object and toppling is honest. What was measured:

- The 2026-08-01 export toppled the floor lamp 36 / 84 / 82 cm at 1.5 / 3 / 5 s
  under delivered physics (recorded; that export is not shipped — the 3 s value
  reappears as 83.8 cm in §3).
- The lamp's own geometry is 1.03 m tall, base ⌀0.29 m, COM at 0.63 m
  (recorded from the 08-01 mesh) — tall, but not so top-heavy that toppling is
  inevitable.
- Under the R1 scoring regime the same lamp settles at 7.1 cm COM drift and
  stays flat across every window (§4, `sceneB_R1.json`): the geometry can stand.

So the object is stabilizable; the 08-01 run certified a bad pose.

## 3. Root cause: the CEM scores a different physical world than it ships

`stabilize.configure_physics` puts the model on the scoring
footing — **uniform 1 kg mass + per-DoF damping (0.3)** — for both scoring *and*
the final bake. The exported MJCF ships **real density-derived mass and zero
damping** (visual geoms `mass=0`). A pose
that reads stable under
the scoring physics can topple under the delivered physics. Measured on the 08-01
lamp pose:

| window | scoring physics (1 kg + damp) | delivered physics (2.98 kg, no damp) |
|---|---|---|
| 1.0 s (fine window end) | 2.4 cm — "looks stable" | 33.0 cm — already toppling |
| 3.0 s | 29.5 cm | 83.8 cm |

The optimizer is blind to the topple because it measures a system the product
doesn't run.

## 4. The controlled A/B (the decisive result)

Method: re-stabilize each scene's existing recon on current code, then measure
**per-body COM drift under delivered physics** (`data.xipos`, the plain exported
MJCF). Four scoring regimes, one variable at a time (golden-baseline law):

- **R0** `1 kg + damp` — the shipped baseline
- **R2** `real mass + damp`
- **R3** `1 kg + no damp`
- **R1** `real mass + no damp` — fully delivered-physics scoring

**Scene B (drawn interior) — COM drift @3 s (cm), from `sceneB_R*.json`:**

| body | R0 | R2 | R3 | R1 |
|---|---|---|---|---|
| floor lamp | **59.4** | 17.3 | 10.2 | **7.1** |
| table lamp | 51.7 | 0.6 | 1.9 | 0.6 |
| yellow book (explodes) | 852.2 | 0.5 | 0.2 | 0.5 |
| red book | 34.1 | 0.6 | 1.7 | 0.6 |

**Scene A (photographed lounge) — COM drift @3 s (cm), from `sceneA_R*.json`:**

| body | R0 | R2 | R3 | R1 |
|---|---|---|---|---|
| **backpack** | **1.1** | 47.7 | 45.4 | 47.9 |
| everything else | ≤1.2 | ≤0.4 | ≤1.1 | ≤0.4 |

Full windows confirm behaviour type: scene A's backpack R0 = 1.15/1.13/1.11
(flat, stable); R1 = 47.7/47.9/47.9 (fast topple). Scene B's floor lamp R0 =
61/59/61 (toppled); R1 = 7.06/7.12/7.16 (flat — **settled**, not toppling).

## 5. The design-decisive conclusion

**There is no single global scoring-physics setting that is best for every object.**

- Realistic scoring (R1/R2/R3) robustly fixes scene B — the floor lamp settles
  (59→7 cm), and the table lamp, the exploding book and the red book all recover.
- But scene A's backpack is stable **only** at the exact `1 kg + damp` baseline;
  **any** deviation topples it (real-mass alone 48, no-damp alone 45, both 48). So
  it is *not* specifically the missing damping — either variable breaks it. Its
  baseline stability is a **fragile narrow basin** the `1 kg + damp` scoring happened
  to find, not a robust result.

Therefore the naive fix — "score under delivered physics globally" — is **ruled
out**: it trades scene B's four bad objects for scene A's backpack, because CEM
*convergence* is sensitive to scoring physics in object-specific ways.

The result points toward a **smarter design**: score and validate under
delivered physics but **keep the best pose per object across regimes** — an
ensemble or acceptance gate that re-optimises only the objects that fail their
delivered-physics check — rather than one global physics switch.

## 6. Secondary findings (both real, both actionable)

**(a) The 08-05 change did NOT regress scene A — it improved it.** With exact
physics, the current code produces a *stable* backpack (1.1 cm) where the 07-31
export tipped it 46 cm: `_support_top` seats the sofa's backpack on the seat
instead of on the backrest ledge. (An earlier alarming "backpack 73 / laptop 133
cm" reading was a measurement confound — see §7 — not a code regression.)

**(b) The multiprocess CEM exhausts memory on a 24-core machine.** It spawns one
worker per core, each loading the full model with ~255 MB of full-resolution
visual meshes (~3.3 GB resident per worker) → out of memory. Those visual meshes
are **dead weight for scoring** (`mass=0`, `contype/conaffinity=0`). The fix is to
give the CEM a **collision-only / decimated-visual scoring model** and to cap
workers by available RAM; until then the multiprocess CEM is not usable at that
worker count.

## 7. Methodology & caveats (so these results are trustworthy)

- **A decimation confound was found and fixed.** To fit memory I first decimated
  visual meshes — but the assembler *regenerates* collision hulls from the mesh,
  so decimation silently changed hull shape **and** mass (that
  polluted the first A/B: bouncing/inflated drift). Fixed by **reusing the stored
  original hulls** instead of regenerating — verified physics-exact (0 hulls differ
  byte-for-byte, all body masses match to 5 decimals). Every number in §4 is from
  the physics-exact runs. Lesson: "visual decimation is physics-neutral" is only
  true if collision is *not* regenerated from the decimated mesh.
- The spike used temporary env-gated instrumentation in `stabilize.py`
  (mass/damping/window/worker-cap) and `pipeline.py` (collision-reuse). **All
  reverted; nothing committed.** The env gates were inert unless set, so the
  baseline was unchanged.
- The harness needed an `if __name__ == "__main__"` guard — without it, spawned
  CEM workers re-execute the whole pipeline (recursive spawn storm).

## 8. Open items

1. **North star** (still unanswered): optimise for stability per object, or keep
   one global scoring physics? The spike shows the two can conflict — fixing the
   lamp perturbs the backpack.
2. **Design the real fix** (§5): most likely a delivered-physics acceptance gate +
   per-object re-optimisation, or a multi-regime ensemble keep-best. Brainstorm →
   spec → plan → TDD.
3. **Collision-only CEM scoring model** (§6b) — needed to make the multiprocess
   CEM viable, independent of the physics fix.
