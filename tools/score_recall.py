#!/usr/bin/env python3
# MONOLITHCLE (package `lens`). Post-run scoring, not pipeline code.
"""Score a shipped scene's bodies against a human-written ground-truth list.

Without this, a missing object is indistinguishable from a destroyed one: if a
mug is absent from the shipped scene, we cannot tell whether the VLM never
listed it, SAM-3D failed on it, or physics threw it out of the world. This
separates 'never found' (recall) from 'found and broken' (the ship-gate).

Ground truth is one plain-text file per photo, beside the photo:

    <photo_stem>.objects.txt      one object per line; '#' comments ignored

Usage:
    score_recall.py --summary out/_<tag>/summary.json [--json report.json]
    score_recall.py --xml scene.xml --gt photo.objects.txt

Matching is by HEAD NOUN. Recall asks "did the pipeline find this object", not
"did it label the colour correctly", so ground truth "red book" matching a body
called `yellow_book_0` is the right answer -- the disagreement is visible in the
report for a human to judge.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Words that carry no identity. Kept deliberately tiny: an aggressive stoplist
# silently deletes real head nouns ("plant" from "potted plant" etc.).
_STOP = {"the", "a", "an", "of", "with", "and"}

# A locative phrase says WHERE an object is, not what it is. "bottles on the
# shelf" is a bottle claim; crediting it to a shelf body would award recall the
# pipeline never earned. "of" is deliberately NOT here -- in "stack of cups"
# both words carry the identity and the pipeline may name the body either way.
_LOCATIVE = {"on", "in", "at", "under", "beside", "against", "behind",
             "near", "by", "atop", "inside", "above", "below", "over"}


def _raw_tokens(s: str) -> list[str]:
    """Lowercase alphanumeric tokens, structure words kept, bare indices dropped.

    Trailing digits are MuJoCo/pipeline instance suffixes (`mug_0`), never
    part of the object's identity."""
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and not t.isdigit()]


def _singular(t: str) -> str:
    """Crude English de-pluralisation. Ground truth is written by a person in
    natural plural ("paper cups", "cardboard boxes") while bodies are named in
    the singular, so without this every plural entry reads as a miss."""
    if len(t) > 3 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and t[-3] in "sxzh":
        return t[:-2]
    if len(t) > 2 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _content(tokens: list[str]) -> set[str]:
    """Identity-bearing tokens, singularised: what the overlap score sees."""
    return {_singular(t) for t in tokens
            if t not in _STOP and t not in _LOCATIVE}


def _heads(tokens: list[str]) -> list[str]:
    """Candidate head nouns for a ground-truth phrase.

    Everything from the first locative preposition onward describes location,
    not identity, and is discarded. Of what remains the last word is the head;
    when the phrase is an "X of Y" construction the Y is offered as a second
    candidate, because a pipeline may name `paper_cup_stack_0` or `paper_cups_0`
    for the very same object."""
    for i, t in enumerate(tokens):
        if t in _LOCATIVE:
            tokens = tokens[:i]
            break
    core = [t for t in tokens if t not in _STOP or t == "of"]
    if not core:
        return []
    heads = [core[-1]]
    if "of" in core:
        j = core.index("of")
        if j > 0:
            heads.append(core[j - 1])
    return [_singular(h) for h in heads if h != "of"]


@dataclass
class Match:
    matched: list[tuple[str, str]] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        n = len(self.matched) + len(self.missed)
        return len(self.matched) / n if n else 0.0


def match_objects(ground_truth: list[str], bodies: list[str]) -> Match:
    """Assign each ground-truth object at most one body, best overlap first."""
    gt_raw = [_raw_tokens(g) for g in ground_truth]
    gt_heads = [_heads(t) for t in gt_raw]
    gt_content = [_content(t) for t in gt_raw]
    body_tok = [_content(_raw_tokens(b)) for b in bodies]

    # A candidate pair requires one of the ground-truth HEAD NOUNS to be present
    # in the body name; among those, more shared modifiers is a better fit. Ties
    # break on input order so the report is reproducible.
    cands = []
    for i, heads in enumerate(gt_heads):
        if not heads:
            continue
        for j, bt in enumerate(body_tok):
            if any(h in bt for h in heads):
                cands.append((-len(gt_content[i] & bt), i, j))
    cands.sort()

    used_gt: set[int] = set()
    used_body: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in cands:
        if i in used_gt or j in used_body:
            continue
        used_gt.add(i)
        used_body.add(j)
        pairs.append((i, j))

    pairs.sort()
    return Match(
        matched=[(ground_truth[i], bodies[j]) for i, j in pairs],
        missed=[g for i, g in enumerate(ground_truth) if i not in used_gt],
        extra=[b for j, b in enumerate(bodies) if j not in used_body],
    )


def read_ground_truth(path: str | Path) -> list[str]:
    """One object per line. A missing file is 'unscored', never an error."""
    try:
        text = Path(path).read_text()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def free_bodies(xml: str | Path) -> list[str]:
    """Names of the scene's free (simulated) bodies, in model order."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    names = []
    for b in range(1, m.nbody):
        adr = m.body_jntadr[b]
        if adr >= 0 and m.jnt_type[adr] == mujoco.mjtJoint.mjJNT_FREE:
            names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}")
    return names


def _report_row(label: str, xml: Path, gt: list[str]) -> dict:
    bodies = free_bodies(xml)
    m = match_objects(gt, bodies)
    return {"label": label, "xml": str(xml), "n_ground_truth": len(gt),
            "n_bodies": len(bodies), "recall": round(m.recall, 3),
            "matched": m.matched, "missed": m.missed, "extra": m.extra}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", help="out/_<tag>/summary.json from overnight_eval")
    ap.add_argument("--xml", help="score a single MJCF instead")
    ap.add_argument("--gt", help="ground-truth file for --xml")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.xml:
        rows.append(_report_row(Path(args.xml).stem, Path(args.xml),
                                read_ground_truth(args.gt) if args.gt else []))
    elif args.summary:
        summary = json.loads(Path(args.summary).read_text())
        for stem, entry in sorted(summary.get("scenes", {}).items()):
            gt = read_ground_truth(Path(entry["photo"]).with_suffix("").as_posix()
                                   + ".objects.txt")
            for side, sub in sorted(entry.items()):
                xml = sub.get("xml") if isinstance(sub, dict) else None
                if xml and Path(xml).exists():
                    rows.append(_report_row(f"{stem}/{side}", Path(xml), gt))
    else:
        print("need --summary or --xml", file=sys.stderr)
        return 64

    for r in rows:
        print(f"\n{r['label']}: recall {r['recall']:.2f} "
              f"({len(r['matched'])}/{r['n_ground_truth']} ground truth, "
              f"{r['n_bodies']} bodies)")
        if r["missed"]:
            print(f"  NOT FOUND: {', '.join(r['missed'])}")
        if r["extra"]:
            print(f"  unlisted bodies: {', '.join(r['extra'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
