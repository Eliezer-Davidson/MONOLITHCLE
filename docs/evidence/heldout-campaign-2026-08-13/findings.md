# Held-out campaign — 2026-08-13/14

An unattended overnight run put **five held-out photographs** through the
pipeline. Each exported scene was then measured with a divergence-guarded
10 s settle under the delivered physics (`tools/forensic_probe.py`).

The five photographs are not redistributed. What ships is this project's own
measurements of its own scenes, plus renders of the reconstructed geometry
(`docs/media/heldout_endstates.png`, from the 30 s gate re-run of 2026-08-17 in
`../heldout-gate-2026-08-17/`).

**Data in this directory**

| File | What it is |
|---|---|
| `forensic_lens_5{26,27,28,30,34}.json` | one per scene: body census, spawn interpenetration, guarded 10 s settle. Its `drift_cm` is body-origin displacement (see §3's caveat); `tilt_deg`, penetration and divergence fields are the ones to read |
| `summary.json` | per-scene run record (completion, wall-clock where recorded, gate verdict) |
| `findings.md` | this document |

## Caveats that come before any conclusion

1. **Development scenes certify machinery, never capability.** Capability claims
   come only from held-out photographs. A held-out failure is *data*; it is
   never hot-fixed, because fixing code against a held-out result burns that
   photograph as evidence. (These five photographs were burned on 2026-08-17,
   when de-penetration was changed in response to their failures; every later
   measurement on them is a reproducer result, not a capability claim.)
2. **Configuration for the night:** deferred layout post-optimisation on, the
   global-top seating rule (`LENS_SUPPORT_TOP` unset), CEM population 2048, the
   12 GB low-VRAM recipe (`THIRD_PARTY_NOTICES.md`).
3. **Measurement windows contaminate themselves after divergence.** Where a
   scene goes NaN, every number for that scene after the divergence time is
   meaningless. The times are recorded per scene below.
4. **The collider-budget rule changed after these runs.** On 2026-09-23 the
   per-object convex-hull budget was re-authored and one class changed: a bench
   now takes the slab budget of 8 hulls where it previously fell through to the
   default of 16. Every number in this document was produced under the earlier
   budget. A rerun on today's code would not reproduce them bit for bit.

## Run outcome

**5 of 5** scenes completed and were measured unattended. Scene 528 took
45.7 minutes end to end (recorded from the run log; the log is not shipped).

## The forensic table

Guarded 10 s settles plus a geometry census. Every cell is read from the JSON
files in this directory (`bodies` = `nbody - 1`, excluding MuJoCo's world body).

| scene | bodies | thin | worst spawn penetration | diverged | late jitter @10 s | toppled >30° |
|---|---:|---:|---:|---|---:|---:|
| 526 | 14 | 13 | 114.3 mm | yes — thermos bottle @ 9.45 s | (pre-NaN) | 8 |
| 527 | 10 | 10 | 44.2 mm | yes — toy robot @ 0.46 s | — | 0 |
| 528 | 7 | 7 | **0.2 mm** | **no** | **0.022** | 0 |
| 530 | 9 | 8 | 20.6 mm | yes — thermos cup @ 1.33 s | — | 1 |
| 534 | 11 | 10 | 8.7 mm | **no** | 0.913 | 0 |

## Findings

### 1. Spawn interpenetration, not mesh thinness, predicts divergence

Ordering the five scenes by worst interpenetration at t = 0 puts the three
that diverged at the top: 114.3 mm → NaN, 44.2 mm → NaN, 20.6 mm → NaN, 8.7 mm
→ clean, 0.2 mm → clean — n = 5, and a random ordering gives this split one
time in ten.

Thinness does **not** predict it. The cleanest scene is 7 of 7 thin bodies; the
worst is 13 of 14. The drawn scene (534) has some of the thinnest geometry in the
whole set — median minimum hull half-thickness 0.13–0.24 cm — and it holds.

What separates them is scene **composition**, not the drawn-versus-photographed
distinction that first suggested itself. Real photographs have tight nested
layouts — a chair tucked under a desk, a mug inside a cubby, piled clutter — and
monocular depth error in a tight layout becomes overlap. The drawn scene's
objects stand apart on open floor, so the same error lands harmlessly. This is
the working explanation over n = 5; the causal chain is not instrumented.

The control case: scene 528 is a **real photograph** with a cartoon-like
composition (large separated sofas, no clutter), and it scored the cleanest
forensics of the five — 0.2 mm penetration, the lowest late jitter, zero
topples.

### 2. The three diverging bodies

Two of the three are cylinders (a thermos bottle and a thermos cup); the third,
in scene 527, is `yellow_toy_robot_0` — a toy robot, not a cylinder. The
plausible mechanism — cylinder line-contact plus spawn penetration injecting
energy — remained a **hypothesis** at this point; the toy robot is a
counter-example to the simple version of it. The root cause was later
established on scene 530's cup (`../overnight-2026-08-14/phase5/`): a
near-massless thin-shell rollable at the end of a mass-ratio chain.

### 3. The measurement window flattered everyone

Scene 526 diverges at 9.45 s, invisible to the night's original 5 s measurement.

Translation-only drift is the other blind spot. A leaning lamp moves its centre
of mass by a couple of centimetres, so a scene that reads stable can be full of
fallen objects. The forensic probe records `tilt_deg` per body for exactly this
reason: scene 534's brass desk lamp is at **18.3°** at 10 s
(`forensic_lens_534.json` → `final`) while the 30 s centre-of-mass measure of
the same export reads only 2.16 cm of drift at every window
(`../heldout-gate-2026-08-17/measure_534_as_shipped.json`).

**Caveat on the probe's own `drift_cm`.** The forensic probe measures
displacement of the *body origin* (`d.xpos`), and exported bodies carry their
origin at the world origin with geometry baked in world coordinates, so for a
leaning body that number is mostly lever arm — the same lamp reads 33.3 cm
there. Read the probe's `tilt_deg`, and take drift from the centre-of-mass
harness (`lens/measure.py`, `d.xipos`). (The successor harness defaults to 30 s
windows with per-body tilt, and the ship gate `lens/verify_scene.py` fails on
divergence.)

### 4. Root-versus-root interpenetration ships unresolved

Chair-into-desk overlap in scene 526 survives the whole pipeline (114 mm at
spawn → 115 mm settled). The cause is a design blind spot rather than a bug: the
plausibility optimiser freezes all root bodies and only optimises children on
surfaces, so furniture-versus-furniture overlap is never examined.

Fix options recorded at the time, in preference order:

- **(A) Detect it first, whichever fix follows.** A ship gate must measure
  residual interpenetration at settle and fail loudly above a threshold. Cheap
  and engine-native. *Built on 2026-08-15: `lens/verify_scene.py`.*
- **(B) Minimal-displacement de-penetration** after layout, before the gate:
  push overlapping root pairs apart, minimising total displacement. The
  photograph legitimately has the chair tucked *under* the desk, so the pass
  must remove interpenetration only, never "move to free space". *Built on
  2026-08-15: `lens/depenetrate.py`; 114.3 → 0.0 mm on this scene.*
- **(C) Extend the CEM with a final root-layout pass** — roots become movable,
  translation-only, small bounds. Not built.
- **(D) Cross-object scale consistency.** Much of the overlap *is* relative
  scale error between independently reconstructed objects. Not built.

### 5. Walls are the largest missing capability

The pipeline skips wall-attached objects at the prompt (`lens/prompts.py`), so
framed photographs that are listed anyway fall flat (~5 cm, then stable lying
down — the wall they lean on was never built). Wall-bracket folding desks came
out with **hallucinated floor legs**: the single-image generative prior completes
unseen geometry from its training prior, and "desk" implies legs. The invented
legs accidentally rescue stability here, which will not generalise.

### 6. Small rollable objects

A blue ball settled at 1.8 cm while a green ball rolled 7.7 cm at 3 s, 10.2 cm at
5 s, and kept going. Spawn stabilisation for rollable objects is an open item.

### 7. Exported scenes render grey

MuJoCo ignores OBJ vertex colours, and this exporter wrote no per-geom rgba, so
every body rendered white. The `_visual.obj` files already carry full
per-vertex colour, so this is a small export feature — and a *review-accuracy*
feature, not a cosmetic one: a grey laptop lying flat on a grey checkerboard
floor is nearly invisible, and a reviewer reads it as a missing object when the
model has it at 0.0 cm drift. *Built on 2026-08-15: colour baking in
`lens/export/mjcf.py`.*

## Open items this campaign produced

1. A ship gate that fails loudly on residual interpenetration or on divergence — built.
2. Root-versus-root penetration resolution (option A, then B above) — built.
3. Wall reconstruction and static mounting for wall-parented objects — designed only.
4. Spawn stabilisation for rollable objects — root-caused, not fixed.
5. Longer settle windows (30 s+) with rotation in the metric — built.
6. Baking mean vertex colour into geom `rgba` on export — built.
