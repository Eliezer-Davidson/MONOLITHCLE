# Third-party notices

MONOLITHCLE vendors no third-party code, weights or meshes. Everything below is
installed from a package index at your own choosing, obtained separately by you
under its owner's terms, or reached over the network. The only third-party
material carried in this repository is the SAM License text
(`third_party_licenses/SAM-LICENSE.txt`) and the four Pexels photographs listed
at the end.

## External models — obtained by you, under their own licences

| Component | Owner | Licence | How to obtain |
|---|---|---|---|
| **SAM 3** (instance segmentation) | Meta Platforms, Inc. | SAM License — `third_party_licenses/SAM-LICENSE.txt` | Gated on Hugging Face (`facebook/sam3`). Accept the licence on the model page, then set `HF_TOKEN`. |
| **SAM-3D-Objects** (GPU reconstruction backend) | Meta Platforms, Inc. | SAM License | Obtain Meta's release separately and set `SAM3D_OBJECTS_ROOT` to the checkout containing `checkpoints/hf/pipeline.yaml`. Not resolvable from PyPI. |

Neither is vendored. `lens/scenegraph/masks.py` and `lens/recon.py` import them
as external packages. The wrapper code in this repository was written for this
project and is not part of Meta's release.

### The low-VRAM mode needs a patched checkout that is not shipped

`lens/recon.py` sets three environment variables before loading the
reconstruction backend — `LENS_HALF_WEIGHTS`, `LENS_CPU_CONSTRUCT` and
`LENS_OFFLOAD_EMBEDDER` — and, when the deferred layout post-optimisation is on,
imports `sam3d_objects.pipeline.deferred_postopt`. Stock SAM-3D-Objects reads
none of those variables and has no such module. They are consumed by a set of
local patches to the SAM-3D-Objects checkout (fp16 weight casts, CPU-side
sparse-structure construction, offloading the image embedder, and a
capture-then-replay runner for the layout post-optimisation) that made the
backend fit a 12 GB card. **Those patches are not part of this repository.**
Against an unpatched checkout the three switches are inert, the deferred
post-optimisation import fails, and the backend needs more than 12 GB of VRAM.
Every reconstruction result in `docs/evidence/` was produced with the patched
checkout. `tools/probe_determinism.py` also depends on it.

### Named in the CPU-tier design, not used by the code today

The CPU reconstruction tier is designed against **MoGe-2** (Microsoft, MIT —
<https://github.com/microsoft/MoGe>, weights `Ruicheng/moge-2-vitl-normal`) and
**TripoSR** (MIT). Neither is vendored, installed or invoked by the code in this
repository today; only the gravity-frame stage of that tier is built. See
`docs/design/cpu-depthlift-design.md`, which carries their licences and sizes
inline.

## Network services

| Service | Owner | Terms |
|---|---|---|
| **Gemini API** | Google | Google's API terms. The object-listing and mask-verdict stages **upload your photograph** to it. Read Google's terms before sending photos you would not otherwise share. Set `GEMINI_API_KEY`. |

## Installed dependencies

Installed from PyPI by `pip install -e .`, each under its own licence:

| Package | Licence |
|---|---|
| MuJoCo | Apache-2.0 |
| NumPy | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| trimesh | MIT |
| Rtree | MIT |
| Open3D | MIT |
| CoACD | MIT |
| Pillow | HPND (MIT-CMU) |
| pytest (dev extra) | MIT |
| usd-core (usd extra) | Apache-2.0 (with NVIDIA/Pixar terms) |
| google-genai (gemini extra) | Apache-2.0 |
| PyTorch (cpu-recon extra) | BSD-3-Clause |
| huggingface_hub (cpu-recon extra) | Apache-2.0 |

Licence texts are distributed with each package; only the SAM License text is
carried here, in `third_party_licenses/`, because it governs components you
must obtain separately rather than through a package index.

`lens/isaac_settle.py` imports NVIDIA Isaac Sim (`isaacsim`, `omni.isaac.core`),
which is obtained and licensed from NVIDIA separately and is not a declared
dependency of this package.

## Photographs

The four input photographs under `docs/evidence/fresh-photos-2026-08-18/photos/`
and reproduced (scaled) in `docs/media/hero_fresh_photos.png` and
`docs/media/fresh_photos_fail_warn.png` are stock photographs from Pexels,
distributed under the [Pexels License](https://www.pexels.com/license/) (free
for commercial use, no attribution required, modification permitted). They
contain incidental third-party signage and framed artwork that is part of the
stock image.

- `01_armchair_corner.jpg`
- `02_conference_six_chairs.jpg`
- `03_cafe_counter_cups.jpg`
- `04_meeting_book_stack.jpg`

They are carried here unmodified, as the inputs of the results they sit beside.
No other input photograph is redistributed: the five held-out photographs of the
2026-08-13 campaign and the photograph behind the office-lounge development
scene are not in this repository — only renders of this project's reconstructed
geometry and the measurements taken from them.

## This repository

All rights reserved; no licence is granted. See `LICENSE`.
