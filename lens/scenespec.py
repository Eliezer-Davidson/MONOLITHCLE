# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""SceneSpec: the single contract between pipeline stages and consumers.
Frame: Z-up, right-handed, meters, gravity -Z. Quaternions wxyz."""
from __future__ import annotations
import json, math
from dataclasses import dataclass, field, asdict
from pathlib import Path

MASS_SOURCES = {"measured", "category-prior", "default"}
SCALE_SOURCES = {"pointmap", "prior-corrected", "measured", "fiducial"}

@dataclass
class ObjectSpec:
    id: str
    category: str
    visual_mesh: str
    collision_meshes: list[str]
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]   # wxyz
    dims: tuple[float, float, float]          # L, D, H (m)
    mass: float
    friction: float
    mass_source: str
    scale_source: str
    confidence: float
    tags: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

@dataclass
class SceneSpec:
    name: str
    objects: list[ObjectSpec] = field(default_factory=list)
    capture: dict = field(default_factory=dict)
    up_axis: str = "z"
    meters_per_unit: float = 1.0
    floor_z: float = 0.0
    spawn_polygon: list[tuple[float, float]] | None = None
    version: str = "0.1.0"
    # Ship-gate verdict for the exported MJCF (lens.verify_scene), recorded by
    # assemble_scene. None on specs that predate the gate or were never exported.
    verify: dict | None = None

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "SceneSpec":
        d = json.loads(Path(path).read_text())
        objs = [ObjectSpec(**{**o,
                "pos": tuple(o["pos"]), "quat": tuple(o["quat"]),
                "dims": tuple(o["dims"])}) for o in d.pop("objects")]
        if d.get("spawn_polygon") is not None:
            d["spawn_polygon"] = [tuple(p) for p in d["spawn_polygon"]]
        return cls(objects=objs, **d)

    def validate(self) -> list[str]:
        errs, seen = [], set()
        if self.up_axis != "z":
            errs.append("up_axis must be 'z'")
        if self.meters_per_unit <= 0:
            errs.append("meters_per_unit must be > 0")
        for o in self.objects:
            if o.id in seen: errs.append(f"duplicate id {o.id}")
            seen.add(o.id)
            if len(o.pos) != 3: errs.append(f"{o.id}: pos must have exactly 3 elements")
            if len(o.quat) != 4: errs.append(f"{o.id}: quat must have exactly 4 elements")
            if len(o.dims) != 3: errs.append(f"{o.id}: dims must have exactly 3 elements")
            n = math.sqrt(sum(c * c for c in o.quat))
            if abs(n - 1.0) > 1e-3: errs.append(f"{o.id}: quat not unit-norm")
            if o.mass <= 0: errs.append(f"{o.id}: mass must be > 0")
            if o.mass_source not in MASS_SOURCES: errs.append(f"{o.id}: bad mass_source")
            if o.scale_source not in SCALE_SOURCES: errs.append(f"{o.id}: bad scale_source")
            if any(d <= 0 for d in o.dims): errs.append(f"{o.id}: non-positive dim")
        return errs
