# Evidence index

Every claim on the front page, and where to check it. **measured** means the
number comes from a file in this repository or from the command shown;
**recorded** means it was taken on a development run whose artefact is not
shipped, and is stated as such.

Paths inside the shipped JSON files were sanitised for publication: the
original absolute paths were replaced by `<repo>/`, `<run dir>/`, `<photo dir>/`,
`<scratch>/` and `<evidence>/` placeholders. No measurement field was altered.

## Results

### §1 Four fresh photographs (2026-08-18/19)

| Claim | Label | Evidence |
|---|---|---|
| Verdicts PASS / PASS / FAIL / WARN; worst residual penetration 0.1 / 0.2 / 21.4 / 1.7 mm; topples 0 / 0 / 8 / 7; photo 03 diverges at 0.598 s on `yellow_sign_0` | measured | `docs/evidence/fresh-photos-2026-08-18/gate_0{1,2,3,4}_*.json` → `verdict`, `worst_penetration_mm`, `topples`, `diverged`. Gate files 01, 02 and 04 were written by the unattended run; `gate_03_*.json` is a hand re-run of `tools/verify_scene.py --seconds 30` the following morning on that run's scene, because `lens run` exited 3 on its internal gate and the driver recorded no verdict for 03 — it reproduced the in-run gate exactly (0.598 s, 21.4 mm, 8 topples) |
| Bodies in scene 2 / 4 / 18 / 16; scored recall 0.50 / 0.50 / 0.529 / 0.769 | measured | `docs/evidence/fresh-photos-2026-08-18/recall.json` → `n_bodies`, `recall` |
| Recall floors: photo 01 true recall 2 of 2 (cushion present as `yellow_velvet_pillow_0`), photo 04 12 of 13 (`small_framed_photo_0`, `grey_framed_artwork_0`), photo 02's 0.50 genuine | measured | same file → `missed` and `extra` lists |
| Wall-clock 12.6 / 17.1 / 67.8 / 49.3 min | recorded | the driver's timing, carried into `docs/evidence/fresh-photos-2026-08-18/summary.json`; the driver record itself is not shipped |
| Photo 02: the vision model listed the table and all six chairs (the papers were never listed) and three chair masks were dropped as coincident (IoU ≥ 0.80); photo 03: 25 model-listed, seven lost at the mask stage, five of them glass with no mask; photo 04: 16 listed, 16 segmented, so its one genuine miss (the floor lamp) was never listed | recorded | run log, not shipped; the surviving counts (4, 18 and 16 bodies) and the ground-truth denominators (8, 17 and 13) are in `recall.json`, and photo 04's `n_bodies = 16` with `floor lamp` in `missed` and no lamp in `extra` is measured there |
| Seven photographs chosen; the fifth exited in the reconstruction worker after a network outage exhausted its retries, the sixth was stopped mid-run, the seventh never started | recorded | driver log, not shipped; `summary.json` lists the four that completed |
| Photo 04's four wall-art topples: 88.7° / 65.9° / 65.2° / 60.7° | measured | `gate_04_meeting_book_stack.json` → `topples` |
| Photo 03: cup stacks and grinder topple (39.3°, 43.6°, 51.5°), 21.4 mm residual `espresso_machine_0 \| stack_of_paper_cups_2_0` | measured | `gate_03_cafe_counter_cups.json` |
| The ground-truth lists were written before the run | recorded | `docs/evidence/fresh-photos-2026-08-18/photos/*.objects.txt` (object lines unmodified; only the comment header was rewritten for publication) |
| The four photographs are Pexels-licensed | recorded | `THIRD_PARTY_NOTICES.md` |

### §2 Spawn interpenetration predicts divergence (2026-08-13 held-out campaign)

| Claim | Label | Evidence |
|---|---|---|
| Worst spawn interpenetration 114.3 / 44.2 / 20.6 / 8.7 / 0.2 mm: the three deepest are the three that diverged; divergence time (9.45 / 0.46 / 1.33 s) is not monotonic in overlap | measured | `docs/evidence/heldout-campaign-2026-08-13/forensic_lens_{526,527,528,530,534}.json` → `spawn_worst_pen_mm`, `diverged` |
| Divergence times and bodies: thermos bottle @ 9.45 s, toy robot @ 0.46 s, thermos cup @ 1.33 s | measured | same files → `diverged.time_s`, `diverged.body` |
| Thinness does not predict divergence — cleanest scene 7/7 thin, worst 13/14; bodies = `nbody − 1` | measured | fields `n_thin_bodies`, `nbody` |
| Late jitter 0.022 (528) and 0.913 (534); topples 8 / 0 / 0 / 1 / 0 | measured | fields `late_jitter_sum_qvel`, `n_toppled_gt30deg` |
| The `composition` column | recorded | descriptions of photographs that are not redistributed |
| Tight layouts turn monocular depth error into overlap | recorded (working explanation) | ordering measured over n = 5; the causal chain is not instrumented — `findings.md` §1 |
| 30 s gate on the same exports: 528, 534 PASS; 526, 527 FAIL on divergence; 530 FAIL by escape, cup 60.82 m from spawn at settled_s 1.0 | measured | `docs/evidence/heldout-gate-2026-08-17/gate_{526,527,528,530,534}_as_shipped.json` |
| End-state renders at each scene's own `settled_s` (526 9.0 / 527 0.0 / 528 30 / 530 1.0 s); 534 (`settled_s` 30) is not rendered | measured | `docs/media/heldout_endstates.png`; `settled_s` in the gate files above; 534's render is withheld because its source is a drawing that is not this project's |

### §3 Scoring physics vs delivered physics (2026-08-06)

| Claim | Label | Evidence |
|---|---|---|
| Scene B at 3 s: yellow book 852.2 → 0.5 cm, floor lamp 59.4 → 7.1, table lamp 51.7 → 0.6, red book 34.1 → 0.6 (R0 → R1); R2 0.5 / 17.3 / 0.6 / 0.6, R3 0.2 / 10.2 / 1.9 / 1.7 (same body order) | measured | `docs/evidence/cem-physics-spike-2026-08-06/sceneB_R{0,1,2,3}.json` → `drift_cm["3.0"]` |
| Scene A backpack: 1.1 → 47.9 cm; 47.7 with real mass alone, 45.4 with no damping alone | measured | `sceneA_R{0,1,2,3}.json` → `drift_cm["3.0"]["black_backpack_0"]` |
| No single global scoring physics wins | measured | the eight runs together; reasoning in `docs/evidence/cem-physics-spike-2026-08-06/findings.md` §5 |
| Those eight runs used the non-default `_support_top` rule | recorded | `findings.md` header and §6(a); the JSON `regime` field records only the spike's own env vars |
| Scene 527: scoring and shipped colliders identical (120 geoms); shipped masses 37.16 kg (shelving) to 5 g (paper packet), undamped; scoring runs every body at 1 kg with 0.3 damping | measured | `docs/evidence/overnight-2026-08-14/phase6/cubby_findings.json` → `b_gate_vs_ship` |

### §4 The ship gate and AUTORESET (2026-08-15)

| Claim | Label | Evidence |
|---|---|---|
| Gate thresholds: FAIL on divergence, escape > 5 m, residual penetration > 20 mm; WARN on topple > 30° or late jitter > 1.0; 30 s default | measured | `lens/verify_scene.py` constants and `verify_scene()` signature |
| `lens run` exits 3 on FAIL and never deletes the scene; the verdict is in `scene.json` | measured | `lens/cli.py`; `tests/test_pipeline.py` ship-gate integration tests |
| Usage errors exit 64, never the WARN code 2; nine mistyped gate runs once read as WARN | measured (the contract) / recorded (the episode) | `tests/test_verify_scene.py::test_cli_usage_error_does_not_collide_with_the_warn_exit_code` |
| Office lounge as shipped on 08-13: WARN — thermos 31.2°, late jitter 2.171 | measured | `docs/evidence/overnight-2026-08-14/phase1/dev_office_lounge.json` |
| Held-out 528 PASS at 0.2 mm; 530 FAIL by escape (60.82 m) at t = 1 s | measured | `phase1/repro_lens_528.json`, `phase1/repro_lens_530.json` |
| The second development scene diverges at 0.948 s where the old 5 s harness reported 3.21 cm | recorded | that scene's files are not shipped (third-party artwork); mechanism: `mjDSBL_AUTORESET` was never disabled in the old harness |
| Office lounge thermos: 1.63 cm @ 5 s → 2.10 cm @ 30 s with tilt 31.2°; table 0.26 → 2.35 cm | measured | `phase2/office_lounge_new30s.json` → `measurements["5.0"]`, `["30.0"]` |
| End-to-end regeneration with everything on: PASS, 0.5 mm worst penetration, no divergence over 30 s; `black_laptop_2_0` still drifts 32.5 cm | measured | `phase8/office_lounge_regen_verdict.json`, `phase8/office_lounge_regen_measure.json` → `measurements["30.0"]` |
| The regeneration's de-penetration pass had moved that laptop ~12 cm | recorded | the regeneration's de-penetration report is not shipped (phase 4's report, which is, covers the earlier 2.6 cm move) |

### §5 De-penetration (2026-08-15 and 08-17)

| Claim | Label | Evidence |
|---|---|---|
| Office lounge 15.0 → 0.0 mm in 22 iterations (laptop 2.6 cm, thermos 1.0 cm); 526 114.3 → 0.0 mm in 99 iterations (desk 29.8 cm, chair 13.5 cm) | measured | `docs/evidence/overnight-2026-08-14/phase4/office_lounge_depen_report.json`, `phase4/lens_526_depen_report.json` (displacement norms from `displacements_cm`) |
| 526 chair-desk overlap 114.3 mm at spawn → 115.0 mm settled | measured | `heldout-campaign-2026-08-13/forensic_lens_526.json` → `spawn_penetrations_mm`; `heldout-gate-2026-08-17/gate_526_as_shipped.json` → `residual_penetrations_mm` |
| De-penetrated office lounge: WARN → PASS; thermos 0.1° / 0.08 cm; the moved laptop drifts 30.0 cm vs 7.2 cm | measured | `phase4/office_lounge_depen_verdict.json`, `phase4/office_lounge_depen_measure.json` vs `phase2/office_lounge_new30s.json` |
| The floor bug: 530's computer mouse 29.3 mm into the floor after the old pass | measured | `phase5/lens_530_depen_verdict.json` → `residual_penetrations_mm["black_computer_mouse_0 \| world"]` |
| 526 and 527 floor penetration 0.1 → 30.8 mm and 4.8 → 45.2 mm under the old pass | recorded | spawn-time measurement from the 08-17 sweep; the old-pass XMLs are not shipped |
| Fixed pass: 526 114.3 → 1.5 mm in 117 iterations; 527 44.2 → 1.4; 530 20.6 → 1.2; 534 8.7 → 0.0; 528 nothing to do | measured | `heldout-gate-2026-08-17/depenetration_{526,527,528,530,534}_report.json` |
| With the fix 526 stops diverging: FAIL → WARN (1.8 mm, 30 s); 527 still diverges at 0.70 s; 530's cup still escapes (36.4 m) | measured | `heldout-gate-2026-08-17/gate_{526,527,530}_depenetrated.json` |
| The floor test | measured | `tests/test_depenetrate.py::test_depenetration_never_pushes_a_body_into_the_floor` |
| Fresh-photo worst residual overlaps at the gate 0.1 / 0.2 / 1.7 / 21.4 mm | measured | `fresh-photos-2026-08-18/gate_*.json` → `worst_penetration_mm` (residual at settle, not spawn — the held-out figures are spawn overlaps, so the comparison is indicative only) |
| On by default since 2026-08-17 | measured | `lens/pipeline.py` (`LENS_DEPENETRATE`), `tests/test_pipeline.py::test_depenetrate_default_on_resolves_overlap` |

### §6 Rollables (2026-08-15)

| Claim | Label | Evidence |
|---|---|---|
| Bisection: cup + static desk no; cup + free desk no; cup + free desk + free chair diverges 1.136 s; full minus cup no; full scene 6.698 s | measured | `docs/evidence/overnight-2026-08-14/phase5/rollable_bisect.json` |
| Masses: cup 21.7 g, desk 0.759 kg, chair 5.21 kg | measured | same file → `masses` |
| Spawn penetration 0–25 mm into a flat box: no divergence, 2.9–4.3 cm drift | measured | `phase5/rollable_sweep.json` → `sweep == "a_penetration"` |
| 50 g dies 1.17 s, 100 g 0.93 s, 150 g 5.40 s, 300 g survives; convex-hull floor 42 g | measured | `phase5/rollable_threshold.json` |
| dt = 0.001 survives; the de-penetrated 530 still diverges (0.41 s) | measured | `phase5/rollable_fixprobe.json` (`f4_dt_0.001`), `phase5/lens_530_depen_verdict.json` |

### §7 Colour baking and seating (2026-08-15)

| Claim | Label | Evidence |
|---|---|---|
| Mean vertex colour baked into geom `rgba`; colourless meshes stay white | measured | `lens/export/mjcf.py::_mean_vertex_rgb`; `tests/test_mjcf.py::test_write_scene_bakes_mean_vertex_color_into_visual_rgba` |
| Saturation 15.57 → 23.24 | recorded | the A/B renders ship in `docs/media/colour_baking_before_after.png`; the saturation script does not |
| All five cubby occupants seated at 92.8 cm (roof 92.3 + 5 mm), ground offsets 77–124 cm | measured | `phase6/cubby_findings.json` → `c_support_rule` |
| Hull overhang ≤ 0.9 mm on the cubby occupants, ≤ 1.7 mm overall (largest cardboard box) | measured | `phase6/cubby_findings.json` → `a_hull_inflation` |
| The backpack tips off the sofa backrest onto the seat | measured | `docs/media/global_top_seating.png` (2026-08-01 export, one pinned camera) |

### §8 The test suite

| Claim | Label | Evidence |
|---|---|---|
| 186 test functions (169 module-level, 17 in test classes) across 19 modules; 200 collected cases, so 14 come from parametrisation | measured | `grep -oE "^\s*def test_\w*" tests/*.py \| wc -l` (186); `grep -c "^def test_" tests/*.py` (169); `python -m pytest --collect-only -q -p no:cacheprovider` (198 collected + 2 deselected); `ls tests/test_*.py` (19) |
| 194 passed / 4 skipped / 2 deselected in 206 s with the torch stack (2026-10-12); 7 more skips without it | measured | `python -m pytest -q -p no:cacheprovider` |

## Engineering highlights

| # | Claim | Label | Evidence |
|---|---|---|---|
| 1 | The replaced GJK-style routine cycled on 12 % of hull pairs and missed 31 of 111 genuine overlaps; a 16× cap changed nothing but made it 9× slower | recorded | development measurement, 2026-07/08; restated in `lens/collide.py`; `tests/test_collide.py::test_terminates_on_the_inputs_that_made_the_old_gjk_cycle` pins termination |
| 2 | A 30°-tilted slab's inflated z-extent collapses back to 0.2 m | measured | `tests/test_pipeline.py::test_gravity_align_untilts_a_tilted_scene` |
| 3 | "bench" returned zero masks where "sofa" returned one; every "laptop" phrase grounded to the same laptop at IoU 0.998 | recorded | development measurements, 2026-07/08; `tests/test_masks.py` pins the retry and recovery behaviour with fixtures |
| 4 | Body-origin drift reads ~2.8 m for a 90° tip two metres out; the default free camera auto-frames per model | recorded | worked example and fix in `tools/settle_compare.py`'s docstring; `lens/measure.py`, `lens/verify_scene.py` and `tools/settle_compare.py` read `data.xipos` |
| 4 | Held-out 534's brass lamp: 33.3 cm body-origin drift and 18.3° tilt in the probe, 2.16 cm COM drift in the 30 s measure | measured | `heldout-campaign-2026-08-13/forensic_lens_534.json` → `final` (`tools/forensic_probe.py` uses `d.xpos`); `heldout-gate-2026-08-17/measure_534_as_shipped.json` → `measurements[*].bodies["brass_desk_lamp_0"].drift_cm` (2.16 at every window) |
| 5 | One 80-minute run optimised against two-day-old collision geometry; the numbers were voided | recorded | `tests/test_mjcf.py` holds the regression |
| 6 | Gate and harness depend on `mujoco` + `numpy` only | measured | imports of `lens/verify_scene.py`, `lens/measure.py` |
| 7 | Tucked chair preserved; lateral escape; no step increases floor penetration | measured | `tests/test_depenetrate.py` |
| 8 | Gravity frame from depth-map normals: 45° cone, 15° refine, 75° widening; 10 tests; the rest of the tier is design only | measured | `lens/recon_depthlift.py` (`UP_CANDIDATE_COS`, `UP_REFINE_COS`, `UP_WIDE_CANDIDATE_COS`, 109 lines); `tests/test_recon_depthlift.py`; `docs/design/cpu-depthlift-design.md` |

## Status claims

| Claim | Label | Evidence |
|---|---|---|
| No walls; wall-leaning objects fall | measured | the skip is in `lens/prompts.py`; the falls in `fresh-photos-2026-08-18/gate_04_meeting_book_stack.json` |
| A scene can PASS while missing objects | measured | `gate_02_conference_six_chairs.json` (PASS) with `recall.json` (4 bodies of 8 listed) |
| Stabilisation took 106 min (10 objects) / 72 min (6 objects) single-threaded | recorded | development run logs, not shipped |
| `settle_isaac` raises `NotImplementedError`; `lens/isaac_settle.py` has not been run for any shipped evidence | measured | `lens/stabilize.py::settle_isaac`; no evidence file references `isaac_settle.py` |
| The 12 GB mode needs unshipped patches | measured | `lens/recon.py` (`LOW_VRAM_ENV`, `_run_deferred_postopt`); `THIRD_PARTY_NOTICES.md` |

## What is deliberately absent

- **No held-out or development photograph.** The five 2026-08-13 photographs
  and the office-lounge photograph are not redistributed; renders of this
  project's reconstructed geometry and the measurements are.
- **No photograph, mesh, MJCF or render of the second development scene**
  (third-party artwork). Its spike drift JSON (`sceneB_R*.json`, §3) ships and
  is measured; its AUTORESET numbers (0.948 s, 3.21 cm) are recorded.
- **No render of held-out scene 534.** Its source is a photograph of a drawing
  that is not this project's; the scene's measurements ship, the render of its
  reconstruction does not.
- **No model weights.** SAM 3 and SAM-3D-Objects are obtained by you under
  Meta's terms.
- **No sample scene.** The exported scene directories (MJCF, USD and the
  meshes they reference) are not shipped and were not available when this
  export was assembled, so no example scene is included. The front-page
  renders were made from those directories while they were still available.
