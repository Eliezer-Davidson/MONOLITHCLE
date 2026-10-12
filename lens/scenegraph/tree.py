# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""VLM reply -> validated support tree. The VLM is untrusted input: unknown
parents and cycles are repaired to floor, ids are slugified unique."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

SURFACE_CATEGORIES = {"sofa", "table", "bench", "shelf", "chair", "desk", "cabinet"}

# Leading list-numbering markers ("1.", "1)", "-", "*") that VLM replies
# sometimes prefix names with; stripped before slugifying so they don't leak
# into ids.
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")

@dataclass
class RawObject:
    name: str
    category: str
    parent: str | None          # id after build_tree; None == floor
    id: str = ""
    tags: list[str] = field(default_factory=list)

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def _strip_list_marker(s: str) -> str:
    return _LIST_MARKER_RE.sub("", s, count=1)

def parse_object_lines(text: str) -> list[RawObject]:
    out = []
    for line in text.strip().splitlines():
        # Drop empty fields so a trailing "|" (markdown-table artifact) or an
        # extra stray "|" doesn't shift/duplicate fields; tolerate more than
        # 3 fields by keeping only the first 3 non-empty ones.
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 3:
            continue
        name, category, parent = parts[:3]
        out.append(RawObject(name=name, category=_slug(category), parent=parent))
    return out

def build_tree(objs: list[RawObject]) -> list[RawObject]:
    counts: dict[str, int] = {}
    by_name: dict[str, RawObject] = {}
    for o in objs:
        base = _slug(_strip_list_marker(o.name))
        o.id = f"{base}_{counts.get(base, 0)}"; counts[base] = counts.get(base, 0) + 1
        by_name[base] = o
    ids = {o.id: o for o in objs}
    for o in objs:                                   # resolve parents by slug
        if o.parent is not None and o.parent in ids:
            continue                                 # already a resolved id: safe to call build_tree again
        p = _slug(_strip_list_marker(o.parent or "floor"))
        o.parent = None if p == "floor" or p not in by_name else by_name[p].id
    for o in objs:                                   # break cycles -> floor, at the node where the repeat closes
        seen, cur = set(), o
        while cur.parent is not None:
            if cur.parent in seen:
                ids[cur.parent].parent = None
                break
            seen.add(cur.id); cur = ids[cur.parent]
    parents = {o.parent for o in objs if o.parent}
    for o in objs:
        o.tags = []                                  # reset so re-running build_tree stays idempotent
        if o.id in parents or o.category in SURFACE_CATEGORIES:
            o.tags.append("surface")
        else:
            o.tags.append("graspable")

    return objs
