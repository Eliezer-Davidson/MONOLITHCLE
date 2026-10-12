# CPU / 2 GB reconstruction tier for MONOLITHCLE — design

**Date:** 2026-09-22 · **Status:** design approved; the gravity frame (§4) is implemented
and unit-tested, and the remainder — the depth model, per-object visible-point extraction
and the shape providers — is specified here and not yet built.

## 1. Goal

Run the whole photo → physics-valid scene pipeline on a 16-thread laptop CPU with 15 GB of
RAM and a 2 GB GPU — i.e. on the CPU, since no reconstruction model fits in 2 GB of VRAM. Output must be **physics-grade**: correct object count, metric placement and
size, support relations, and stable collision geometry. Visual shape is a proxy or a
single-image mesh, not SAM-3D-Objects fidelity.

Decisions already taken (2026-09-22): fidelity = physics-grade; time budget unconstrained
(overnight acceptable) but the design runs in minutes; approach **B** — a depth-lift backend
behind a shape-provider seam, with the TripoSR shape provider designed in from the start.

## 2. Non-goals

- A CPU port of SAM-3D-Objects (13 GB gated weights, debug-only single-threaded sparse conv,
  two fp32 generators of 6.7 and 4.9 GB against ~7 GB free RAM, hours per object). Tried and
  abandoned; not built here.
- VLM size priors, ICP refinement of fitted meshes, textures, Isaac-side changes.
- Any change to the stages after reconstruction (gravity align, grounding, CEM settle and
  MJCF/USD export are consumed unchanged).

## 3. Architecture and data flow

Everything new sits behind MONOLITHCLE's existing seams: `Segmenter`, `ReconBackend`, and the
`ReconResult` / `recon_state.json` contract the parent process assembles from.

Worker process (`lens.recon_worker`), all on CPU:

1. Capture → Gemini `LIST_OBJECTS` → scene tree → **SAM3 masks** (unchanged, SAM3 loaded on
   the CPU; the image is encoded once per photo, not once per phrase).
2. **Depth once per scene.** `MogeDepth.estimate(image)` → metric point map in the camera
   frame, validity mask, per-pixel normals, intrinsics. Cached to `<out>/depth.npz`.
3. **Gravity frame.** `gravity_rotation(points, normals, valid)` → `R` (3×3) mapping the
   camera frame to a Z-up world. Up = area-weighted mean of normals that face camera-up,
   refined once within 15°. No floor mask is needed; `_gravity_align` downstream still runs
   and can only nudge.
4. **Per object.** `visible_points` = point map under the object's mask ∩ valid, rotated by
   `R`, depth outliers trimmed (median absolute deviation, k = 3). `footprint_axes` gives the
   clean box axes (`obb_axes`) from PCA of the XY footprint; `view_extents` gives width,
   depth and height in the view frame (side, toward-camera, up).
5. **Shape provider.** `shapes.shape_for(crop, visible)` → watertight mesh. `ProxyShapes`
   extrudes the visible surface away from the camera and hulls it. `TriposrShapes` runs the
   masked crop through TripoSR and fits the canonical mesh to the visible points. Any
   TripoSR failure or too-small crop falls back to the proxy for that object.
6. Each object → `ReconResult(mesh Z-up metres, confidence, obb_axes, postopt=None)`.

Parent process: unchanged (`_assemble_from_state` → gravity align → ground → settle → export).

## 4. Components and interfaces

### `lens/depth.py` (new)

```python
@dataclass
class DepthResult:
    points: np.ndarray            # (H, W, 3) float32 metric, OpenCV camera frame (x right, y down, z forward)
    valid: np.ndarray             # (H, W) bool
    normals: np.ndarray           # (H, W, 3) unit, camera frame (from the model, else finite differences)
    intrinsics: np.ndarray        # (3, 3) normalized (MoGe convention)

class DepthEstimator(Protocol):
    def estimate(self, image: np.ndarray) -> DepthResult: ...
    def release(self) -> None: ...

class MogeDepth:      # model="Ruicheng/moge-2-vitl-normal", device="cpu", use_fp16=False on CPU
class FakeDepth:      # ray-casts a trimesh scene from a given camera pose/fov -> exact DepthResult
```

### `lens/shapes.py` (new)

```python
@dataclass
class ObjectCrop:
    rgb: np.ndarray               # (h, w, 3) uint8, photo cropped to the mask bbox (+margin)
    alpha: np.ndarray             # (h, w) bool, the mask crop

@dataclass
class VisiblePoints:
    xyz: np.ndarray               # (N, 3) world Z-up metres
    axes: np.ndarray              # (3, 3) columns: footprint major, minor, +Z  (-> obb_axes)
    view_dir: np.ndarray          # (3,) unit horizontal, from object toward camera
    extents: tuple[float, float, float]   # width (side), depth (along view), height (Z)

class ShapeProvider(Protocol):
    def shape_for(self, crop: ObjectCrop, visible: VisiblePoints) -> trimesh.Trimesh | None: ...
    def release(self) -> None: ...

class ProxyShapes:    # thickness = max(observed depth spread, PROXY_MIN_THICKNESS_FRAC * min(width, height)); hull
class TriposrShapes:  # repo_dir, device="cpu", mc_resolution=128, min_mask_px=1024, foreground_ratio=0.85
class FakeShapes:     # returns a box of the visible extents
```

`PROXY_MIN_THICKNESS_FRAC = 0.5` is a tunable, to be measured in the office_lounge A/B.

### `lens/recon_depthlift.py` (new)

```python
class DepthLiftBackend:           # implements ReconBackend
    def __init__(self, depth: DepthEstimator, shapes: ShapeProvider, cache_dir=None,
                 min_valid_px=200, min_valid_frac=0.2): ...
    def reconstruct(self, image, masks) -> dict[str, ReconResult]: ...
    def release(self) -> None: ...
    dropped: list[dict]           # [{"id", "reason"}] for the worker's provenance

def gravity_rotation(points, normals, valid) -> tuple[np.ndarray, str]   # (R, up_source)
def visible_points(points, valid, mask, R, mad_k=3.0) -> np.ndarray
def footprint_axes(xyz) -> np.ndarray
def view_extents(xyz, view_dir) -> tuple[float, float, float]
def crop_for(image, mask, margin=0.1) -> ObjectCrop
def prepare_triposr_input(crop: ObjectCrop, foreground_ratio=0.85) -> PIL.Image   # TripoSR's recipe, mask as alpha
def fit_shape_to_visible(mesh_canon: trimesh.Trimesh, visible: VisiblePoints) -> trimesh.Trimesh
```

`fit_shape_to_visible`: rotate so mesh +x → `view_dir`, +z → +Z, +y → Z × view_dir; scale
anisotropically — y by observed width / mesh width, z by observed height / mesh height, x by
the same factor as y (the model's depth/width ratio is kept, since depth is unobserved);
translate so the mesh's front-most point (max along `view_dir`) and lowest point (min Z) meet
the observed ones, and its lateral centre matches the observed centre.

### Existing code touched

- `lens/scenegraph/masks.py` — `Sam3Segmenter(device=None)`: `"cuda"` if available else
  `"cpu"`, passed to `Sam3Processor(..., device=...)`; on CPU set `USE_PERFLIB=0` before
  importing `sam3`; cache the `set_image` state per image (content hash) and hand each
  `set_text_prompt` a shallow copy, so the encoder runs once per photo.
- `lens/recon_worker.py` — `--recon {auto,sam3d,depthlift}` `--shapes {proxy,triposr}`;
  `auto` = `depthlift` when CUDA is absent. Writes the new provenance fields.
- `lens/pipeline.py` — `run_pipeline_isolated(..., recon=, shapes=)` passes the flags through.
- `lens/cli.py` — the two flags; env defaults `LENS_RECON`, `LENS_SHAPES`.
- `THIRD_PARTY_NOTICES.md` — MoGe (MIT), TripoSR (MIT, vendored commit), unchanged SAM3 entry.
- `README.md` — the laptop recipe.

### Coordinate conventions (pinned)

| Frame | Axes |
|---|---|
| MoGe / OpenCV camera | x right, y down, z forward; camera-up is −y |
| World (ours) | Z up; X = horizontal projection of the camera's forward; Y = Z × X |
| TripoSR canonical mesh | right-handed, z up, +x faces the camera (its azimuth-0 camera sits on +x) |
| `ReconResult` | world, metres, Z up (same as today) |

## 5. Configuration and environment

Flags / env: `--recon` (`LENS_RECON`), `--shapes` (`LENS_SHAPES`), `LENS_DEPTH_MODEL`
(default `Ruicheng/moge-2-vitl-normal`), `LENS_TRIPOSR_DIR` (default `third_party/TripoSR`),
`HF_TOKEN` (SAM3 is gated), `GEMINI_API_KEY`.

Install, into the existing `.venv` (py3.12, CPU torch 2.13, numpy 2.5):

| Piece | How | Licence | Weights |
|---|---|---|---|
| MoGe-2 | `pip install --no-deps git+https://github.com/microsoft/MoGe.git` + `utils3d_moge` (pinned commit) + `huggingface_hub`; its Triton GEMM dep is only used by MoGe-3 | MIT | 1.3 GB |
| TripoSR | clone to `third_party/TripoSR` at a pinned commit; `transformers`, `einops`; `torchmcubes` built from source for CPU (gcc 13 / cmake 3.28 present) | MIT | 1.7 GB |
| SAM3 | `pip install --no-deps git+…/sam3.git@46957e4` + `timm ftfy regex iopath tqdm typing_extensions`; **numpy stays 2.x** (SAM3 pins `<2`, MoGe `>=2`) | SAM licence, gated | 3.4 GB |

Total ≈ 6.4 GB of weights over a 1–3 MB/s Wi-Fi: 40–100 min, done once into the HF cache.

## 6. Error handling and provenance

- Depth model failure → exception → worker exits non-zero → scene fails loudly (today's contract).
  No fallback to placeholder geometry; the CLI already refuses fake backends.
- Object with `< min_valid_px` or `< min_valid_frac × mask` valid depth pixels → dropped, with
  `{"id", "reason"}` in `recon_state.json["dropped"]` and a log line. Never silent: an object
  never created makes a scene *more* likely to pass the ship gate.
- No upward-facing surface → up = camera-up, `up_source = "camera_up_fallback"`, warning.
- TripoSR failure / empty / non-watertight / degenerate mesh, or crop below `min_mask_px` →
  `ProxyShapes` for that object, `shape_source = "proxy_fallback"`, reason recorded.
- Missing weights or `HF_TOKEN` → one error naming the variable and the gated model page.
- RAM: SAM3, MoGe and TripoSR are never co-resident; each `release()`s before the next loads.

`recon_state.json` additions — top level: `depth: {model, cache, up_source, intrinsics}`,
`dropped: [...]`; per object: `recon_backend`, `shape_source`, `shape_reason`, `depth_px`,
`mask_px`, `fit: {scale: [sx, sy, sz], yaw_deg}`.

## 7. Testing (test first; existing suite stays green)

New marker `weights` (skip when weights are absent), beside `gpu` and `api`.

Unit, no models:
- `test_gravity_rotation_recovers_up_from_tilted_plane` — FakeDepth, plane + box, camera
  pitched 20°: up within 1°, `up_source == "normals"`.
- `test_gravity_rotation_falls_back_to_camera_up_without_horizontal_surfaces`.
- `test_visible_points_trims_silhouette_outliers` — injected far-depth rim pixels removed.
- `test_footprint_axes_and_view_extents_recover_box_yaw_and_size` — 30°-yawed box: yaw ±2°,
  width/height ±3%.
- `test_proxy_shape_is_watertight_and_encloses_visible_points`.
- `test_fit_shape_to_visible_places_unit_box_on_observed_front_bottom_and_extents`.
- `test_prepare_triposr_input_matches_reference_recipe` — pixel-exact vs TripoSR's
  crop/pad/gray-composite on a synthetic mask.
- `test_depthlift_backend_returns_zup_reconresults_with_axes_and_drops_thin_masks`.
- `test_pipeline_runs_with_depthlift_fake_depth_fake_shapes` — full `run_pipeline` → valid
  `SceneSpec`.
- `test_sam3_state_cache_gives_identical_masks_to_uncached_path` (marked `weights`).

`weights`-marked, real models on `demo/office_lounge.jpg`:
- MoGe table height within ±30 % of the GPU-tier office_lounge table.
- TripoSR returns a watertight mesh for one crop on CPU.
- SAM3 on CPU returns ≥ 1 mask for "sofa".

## 8. Evidence plan (project laws: one change at a time, visual proof, never SSIM)

1. `depthlift` + `proxy` on office_lounge vs the GPU-tier scene of the same photo: the
   existing centroid-distance tool, settle montage, coverage count beside every number.
2. `depthlift` + `triposr` vs `proxy` on the same photo: one component changed; centroid,
   per-object extents, montage.
3. Wall-clock and peak RSS per stage on the target 2 GB-GPU laptop, recorded in the run report.

## 9. Risks and fallbacks (ranked)

| # | Risk | Fallback |
|---|---|---|
| 1 | SAM3 (`numpy<2`) vs MoGe (`numpy>=2`) in one venv | smoke-test SAM3 on numpy 2 first; else SAM3 in its own subprocess + interpreter |
| 2 | SAM3 CPU speed unmeasured (~860 M params at 1008 px) | encode-once cache (designed in); else a Gemini polygon-mask `Segmenter` |
| 3 | TripoSR on occluded / cluttered crops | per-object proxy fallback; the A/B decides the default |
| 4 | `torchmcubes` CPU build | PyMCubes shim behind `MarchingCubeHelper.mc_func` |
| 5 | MoGe metric scale error (10–20 %) | recorded; VLM size prior only if the A/B shows drift |
| 6 | TripoSR's old `transformers==4.35` pin | try current; pin only if its DINO tokenizer breaks |
| 7 | Three models in 15 GB RAM | never co-resident; `release()` before the next loads |

