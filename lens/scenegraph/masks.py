# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Phrase -> mask via SAM3, adjudicated by the VLM. SAM3 is Meta code used as
an external library (SAM License); this wrapper is this repository's own code."""
from __future__ import annotations
from typing import Protocol
import contextlib
import logging
import re
import numpy as np
from lens.prompts import render_verdict_prompt

log = logging.getLogger(__name__)

DEDUP_IOU_THRESHOLD = 0.8   # I2: coincident-mask de-duplication threshold

# Fix #1 (vocabulary gap): SAM3 grounds on a phrase, and the VLM's chosen word
# for an object is not always in SAM3's effective vocabulary. Measured on
# office_lounge: "grey and yellow bench" -> 0 masks, "bench" -> 0, but "sofa" ->
# 1 correct mask. When the VLM's name yields nothing we retry with the phrase
# stripped of colour/material/size qualifiers, then with these category
# synonyms, before giving up on the object.
_QUALIFIER_WORDS = frozenset({
    # colours
    "black", "white", "grey", "gray", "brown", "beige", "tan", "cream", "red",
    "orange", "yellow", "green", "blue", "navy", "purple", "pink", "gold",
    "golden", "silver", "dark", "light",
    # materials / finish
    "wooden", "wood", "metal", "metallic", "plastic", "glass", "leather",
    "fabric", "cloth", "steel", "chrome", "matte", "glossy",
    # size / shape / state adjectives
    "small", "large", "big", "little", "tall", "short", "round", "square",
    "rectangular", "circular", "oval", "open", "closed", "folded",
    # filler
    "and", "the", "a", "an", "of", "with",
})
_SYNONYMS = {
    "bench": ["sofa", "couch", "seat", "loveseat"],
    "sofa": ["couch", "settee", "loveseat"],
    "couch": ["sofa", "settee"],
    "settee": ["sofa", "couch", "bench"],
    "loveseat": ["sofa", "couch"],
    "ottoman": ["footstool", "stool", "pouf"],
    "pouf": ["ottoman", "footstool"],
    "stool": ["chair", "seat"],
    "armchair": ["chair", "sofa"],
    "recliner": ["chair", "sofa"],
    "coffee table": ["table"],
    "side table": ["table"],
    "end table": ["table"],
    "nightstand": ["table", "cabinet"],
    "desk": ["table"],
    "cupboard": ["cabinet"],
    "wardrobe": ["cabinet", "closet"],
    "dresser": ["cabinet", "drawers"],
    "bookshelf": ["shelf", "bookcase"],
    "bookcase": ["shelf", "bookshelf"],
    "tv": ["television", "monitor", "screen"],
    "television": ["tv", "monitor", "screen"],
    "monitor": ["screen", "tv"],
    "rug": ["carpet", "mat"],
    "carpet": ["rug", "mat"],
    "thermos": ["bottle", "flask", "tumbler", "cup"],
    "flask": ["bottle", "thermos", "cup"],
    "tumbler": ["cup", "glass", "mug"],
    "mug": ["cup"],
}


def _strip_qualifiers(name: str) -> str:
    """Drop leading colour/material/size adjectives and instance numbers, leaving
    the head noun(s): 'grey and yellow bench' -> 'bench', 'black laptop 1' ->
    'laptop', 'white round table' -> 'table'."""
    words = [w for w in name.lower().split()
             if w not in _QUALIFIER_WORDS and not w.isdigit()]
    return " ".join(words).strip()


# Categories/nouns too generic to ground to a single instance -- a SAM3 query
# for "furniture"/"object" returns an arbitrary blob (or a merge), which is worse
# than dropping. Never issued as a fallback phrase.
_GENERIC_NOUNS = frozenset({
    "furniture", "object", "item", "items", "thing", "things", "stuff", "decor",
    "appliance", "appliances", "electronic", "electronics", "fixture",
})


def _fallback_phrases(name: str, category: str) -> list[str]:
    """SAM3 phrases to try, in order, when `name` yields no mask: the
    qualifier-stripped noun FIRST, then specific synonyms of the noun and of the
    category, then the bare category as a last resort. Specific words beat the
    generic category ('bench'->'sofa' is precise; 'furniture' grabs a blob), and
    generic catch-alls (`_GENERIC_NOUNS`) are skipped entirely. De-duplicated and
    never re-issuing `name` itself."""
    seen = {name.lower().strip()}
    out: list[str] = []
    stripped = _strip_qualifiers(name)
    cat = (category or "").replace("_", " ").strip()
    for p in [stripped, *_SYNONYMS.get(stripped, []), *_SYNONYMS.get(cat, []), cat]:
        p = (p or "").strip()
        if p and p not in seen and p not in _GENERIC_NOUNS:
            seen.add(p)
            out.append(p)
    return out

def _parse_verdict(reply: str) -> bool:
    """Parse VLM verdict reply robustly.

    Returns True if reply contains the standalone word "yes" (case-insensitive),
    False if it contains "no", or False if unparseable/empty.

    If both "yes" and "no" appear, the first one encountered wins.
    Word boundaries ensure "yesterday" or "eyes" are not matched.
    """
    if not reply or not reply.strip():
        return False

    # Find first occurrence of "yes" or "no" as standalone words
    yes_match = re.search(r'\byes\b', reply, re.IGNORECASE)
    no_match = re.search(r'\bno\b', reply, re.IGNORECASE)

    # If neither found, not affirmative
    if yes_match is None and no_match is None:
        return False

    # If only one found, return accordingly
    if yes_match is None:
        return False
    if no_match is None:
        return True

    # Both found; first one wins
    return yes_match.start() < no_match.start()

class Segmenter(Protocol):
    def masks_for(self, image: np.ndarray, phrase: str) -> list[np.ndarray]: ...

class FakeSegmenter:
    def __init__(self, mask_map: dict[str, list[np.ndarray]]):
        self._m = mask_map
    def masks_for(self, image, phrase):
        return list(self._m.get(phrase, []))

class Sam3Segmenter:
    """Lazy: model loads on first call (~4.7 GB VRAM).

    Real API (verified against the installed `sam3` package, see
    task-7-report.md): there is no single `process_image_text` call. SAM3's
    image-grounding path is stateful — `Sam3Processor.set_image(image)`
    returns an `inference_state` dict, which `set_text_prompt(prompt, state)`
    mutates in place (and also returns) after running the grounding forward
    pass. The populated state holds `state["masks"]`, a boolean tensor of
    shape (N, 1, H, W) already filtered by `confidence_threshold`, and
    `state["scores"]`, a length-N float tensor in the *same, unsorted* query
    order -- candidates are not returned best-first, so we sort by score
    ourselves to honour this class's contract.
    """
    def __init__(self, confidence: float = 0.5):
        self._proc = None; self._confidence = confidence
    def _load(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self._proc = Sam3Processor(build_sam3_image_model(),
                                   confidence_threshold=self._confidence)
    def release(self) -> None:
        """Drop the model and free its memory.

        The pipeline runs segmentation and reconstruction in ONE process, so
        without this SAM3's weights stay resident while SAM-3D-Objects loads
        its own multi-GB stack, and the OS OOM-killer takes the process
        (exit 137). Safe to call more than once; masks_for() reloads lazily.
        """
        if self._proc is None:
            return
        import gc
        self._proc = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def masks_for(self, image: np.ndarray, phrase: str) -> list[np.ndarray]:
        if self._proc is None: self._load()
        import torch
        from PIL import Image as PILImage
        # SAM3's own predictor runs inference inside bf16 autocast
        # (sam3/model/sam3_base_predictor.py). Sam3Processor does NOT, so
        # calling it directly hands bf16 weights fp32 activations and the
        # backbone dies with "mat1 and mat2 must have the same dtype, but got
        # BFloat16 and Float". Supply the context the model expects.
        _amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available() else contextlib.nullcontext())
        with _amp:
            state = self._proc.set_image(PILImage.fromarray(image))
            state = self._proc.set_text_prompt(phrase, state)
        masks, scores = state.get("masks"), state.get("scores")
        if masks is None or masks.shape[0] == 0:
            return []
        # Under bf16 autocast these come back as BFloat16, which numpy cannot
        # represent at all ("Got unsupported ScalarType BFloat16"). Cast to
        # float32/bool on the torch side before crossing into numpy.
        order = np.argsort(-scores.detach().float().cpu().numpy())
        return [masks[i, 0].detach().bool().cpu().numpy() for i in order]

def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[mask] = (0.45 * out[mask] + np.array([140, 0, 0])).clip(0, 255).astype(np.uint8)
    return out

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks. A zero union (both
    masks empty) is defined as IoU 0, not a division-by-zero NaN."""
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)

def _verdict(vlm, name: str, image, mask: np.ndarray) -> bool:
    """VLM adjudication: does `mask` cover exactly the object `name`?"""
    return _parse_verdict(
        vlm.ask(render_verdict_prompt(name), [image, _overlay(image, mask)]))


def _collides(mask: np.ndarray, accepted: list[np.ndarray]) -> bool:
    return any(_mask_iou(mask, prev) >= DEDUP_IOU_THRESHOLD for prev in accepted)


def _candidates(image, o, seg: Segmenter, max_alternatives: int) -> list[np.ndarray]:
    """SAM3 candidates for `o`, with the Fix #1 vocabulary fallback: try the
    VLM's exact name first, then qualifier-stripped / synonym phrases, so a
    word SAM3 doesn't ground ('bench') doesn't silently lose the object when a
    synonym ('sofa') would have found it."""
    cands = seg.masks_for(image, o.name)
    if not cands:
        for alt in _fallback_phrases(o.name, o.category):
            cands = seg.masks_for(image, alt)
            if cands:
                log.info("masks: '%s' gave no mask; fell back to '%s'", o.name, alt)
                break
    return cands[:max_alternatives]


def _select_mask(image, o, cands, vlm, accepted):
    """Pick the first VLM-approved candidate that is spatially DISTINCT from
    every already-accepted mask. A candidate the VLM approves but that collides
    with an accepted mask is a coincident duplicate -- skipped here so Fix #2
    recovery can look for a distinct alternative before we give up. If nothing
    is approved at all, fall back to the top candidate only when it too is
    distinct (mirrors the old default-to-cands[0] without resurrecting a dup)."""
    saw_approved = False
    for m in cands:
        if _verdict(vlm, o.name, image, m):
            saw_approved = True
            if not _collides(m, accepted):
                return m
    if not saw_approved and cands and not _collides(cands[0], accepted):
        return cands[0]
    return None


def _recover_mask(o, objs, cand_map, accepted):
    """Fix #2 (instance recovery): the object's own candidates all collided with
    accepted masks -- SAM3 returned the same salient instance for two objects
    (measured laptop-vs-laptop IoU 0.998). Take a spatially DISTINCT, unused
    candidate from a SAME-CATEGORY sibling's pool as another instance of that
    category.

    Deliberately NO VLM re-verification: both SAM3 and the VLM are instance-blind
    (they cannot tell 'laptop 1' from 'laptop 2'), so asking the VLM 'is this
    exactly <name>?' just rejects the valid recovery and drops the second
    instance -- which is exactly what happened on the real office_lounge run
    before this change. A distinct, unused, same-category SAM3 mask is trusted as
    the genuine second instance."""
    for x in objs:
        if x.id == o.id or x.category != o.category:
            continue
        for m in cand_map.get(x.id, []):
            if not _collides(m, accepted):
                log.info("masks: recovered a distinct mask for %s from %s's "
                         "pool (same category %r)", o.id, x.id, o.category)
                return m
    return None


def segment_objects(image, objs, seg: Segmenter, vlm, max_alternatives: int = 3):
    """`o.name`-keyed mask selection with two robustness passes:

    * Fix #1 -- vocabulary fallback (`_candidates`): the VLM's word for an object
      is not always in SAM3's grounding vocabulary, so a zero-mask name retries
      with qualifier-stripped / synonym phrases before the object is dropped.
    * I2 dedup + Fix #2 recovery: when the VLM lists two objects that both
      ground to the SAME region (IoU >= DEDUP_IOU_THRESHOLD), keeping both
      reconstructs one instance twice while the other's real region is lost.
      The later object is not accepted at the coincident mask; instead recovery
      looks for a DISTINCT, unused candidate in the pool (a second instance SAM3
      ranked lower elsewhere) and only drops the object if none exists.

    Objects are processed in traversal order; masks are computed once per object
    up front (SAM3 only -- no VLM calls) so recovery can borrow across pools."""
    out: dict[str, np.ndarray] = {}
    accepted: list[np.ndarray] = []
    cand_map = {o.id: _candidates(image, o, seg, max_alternatives) for o in objs}
    for o in objs:
        cands = cand_map[o.id]
        if not cands:
            log.info("masks: no mask for %s (name %r) even after fallback -- dropped",
                     o.id, o.name)
            continue
        chosen = _select_mask(image, o, cands, vlm, accepted)
        if chosen is None:
            chosen = _recover_mask(o, objs, cand_map, accepted)
        if chosen is None:
            log.info("masks: dropping %s -- coincident with an already-accepted "
                     "mask and no distinct alternative (IoU >= %.2f)",
                     o.id, DEDUP_IOU_THRESHOLD)
            continue
        accepted.append(chosen)
        out[o.id] = chosen
    return out
