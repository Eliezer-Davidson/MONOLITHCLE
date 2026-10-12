import os
import numpy as np
import pytest
from lens.scenegraph.masks import FakeSegmenter, segment_objects, _parse_verdict
from lens.scenegraph.tree import RawObject
from lens.vlm import FakeVLM

IMG = np.zeros((8, 8, 3), np.uint8)
A = np.zeros((8, 8), bool); A[:4] = True
B = np.zeros((8, 8), bool); B[4:] = True

def _obj(name):
    return RawObject(name=name, category="box", parent=None, id=name)

def test_vlm_verdict_picks_second_candidate():
    seg = FakeSegmenter({"crate": [A, B]})
    vlm = FakeVLM(["NO", "YES"])
    out = segment_objects(IMG, [_obj("crate")], seg, vlm)
    assert (out["crate"] == B).all()

def test_no_candidates_drops_object():
    out = segment_objects(IMG, [_obj("ghost")], FakeSegmenter({}), FakeVLM([]))
    assert out == {}

def test_parse_verdict_affirmative_cases():
    """Test that _parse_verdict correctly identifies affirmative replies."""
    affirmatives = [
        "YES",
        "yes",
        "Yes",
        "Yes.",
        "Answer: Yes",
        "The answer is yes.",
        "I think yes",
        "Yes, this covers the whole crate.",
    ]
    for reply in affirmatives:
        assert _parse_verdict(reply) is True, f"Failed for: {reply}"

def test_parse_verdict_negative_cases():
    """Test that _parse_verdict correctly identifies negative replies."""
    negatives = [
        "NO",
        "no",
        "No",
        "No.",
        "Answer: No",
        "No, it covers two objects.",
    ]
    for reply in negatives:
        assert _parse_verdict(reply) is False, f"Failed for: {reply}"

def test_parse_verdict_false_positives():
    """Test that _parse_verdict rejects non-standalone yes/no words."""
    false_positives = [
        "yesterday",
        "eyes",
        "Yesterday I saw it.",
        "I like the eyes.",
    ]
    for reply in false_positives:
        assert _parse_verdict(reply) is False, f"Failed for: {reply}"

def test_parse_verdict_empty():
    """Test that _parse_verdict treats empty/unparseable as NOT affirmative."""
    assert _parse_verdict("") is False
    assert _parse_verdict("   ") is False
    assert _parse_verdict("maybe") is False
    assert _parse_verdict("unclear") is False

def test_parse_verdict_first_wins():
    """Test that when both yes and no appear, first one wins."""
    assert _parse_verdict("Yes, no wait") is True
    assert _parse_verdict("No, actually yes") is False

def test_vlm_verdict_accepts_answer_yes_format():
    """Integration test: 'Answer: Yes' format should be accepted on first candidate."""
    seg = FakeSegmenter({"crate": [A, B]})
    vlm = FakeVLM(["Answer: Yes"])
    out = segment_objects(IMG, [_obj("crate")], seg, vlm)
    # First candidate should be selected because "Answer: Yes" is affirmative
    assert (out["crate"] == A).all()

def test_segment_objects_drops_coincident_duplicate_mask():
    """I2 regression test: two objects the VLM listed under different names
    that resolve to the SAME physical region (IoU >= 0.8) must not both
    survive -- keeping both reconstructs one instance twice while the other
    object's real region is never captured. `chair_a` is processed first and
    is accepted; `chair_b`'s chosen mask is identical (IoU 1.0 >= 0.8) so it
    must be dropped. Fails under the pre-fix code, which has no spatial
    de-duplication and would return both."""
    seg = FakeSegmenter({"chair_a": [A], "chair_b": [A]})
    vlm = FakeVLM(["YES", "YES"])
    out = segment_objects(IMG, [_obj("chair_a"), _obj("chair_b")], seg, vlm)
    assert set(out.keys()) == {"chair_a"}


def test_segment_objects_keeps_disjoint_masks():
    """Companion safety test: two objects whose chosen masks don't overlap at
    all (IoU 0) are genuinely distinct instances and must both survive the
    de-duplication pass added for I2."""
    seg = FakeSegmenter({"chair_a": [A], "table_b": [B]})
    vlm = FakeVLM(["YES", "YES"])
    out = segment_objects(IMG, [_obj("chair_a"), _obj("table_b")], seg, vlm)
    assert set(out.keys()) == {"chair_a", "table_b"}


def test_segment_objects_falls_back_to_synonym_when_name_yields_no_mask():
    """Fix #1 (vocabulary gap): SAM3 returns 0 masks for the VLM's word 'bench'
    but does segment the same object under 'sofa' (measured on office_lounge:
    'bench'->0, 'sofa'->1). When `o.name` yields nothing, `segment_objects` must
    fall back to qualifier-stripped / synonym phrases instead of silently
    dropping the object. Here the bench is only findable as 'sofa'."""
    seg = FakeSegmenter({"grey and yellow bench": [], "sofa": [A]})
    obj = RawObject(name="grey and yellow bench", category="bench", parent=None, id="bench_0")
    out = segment_objects(IMG, [obj], seg, FakeVLM(["YES"]))
    assert "bench_0" in out and (out["bench_0"] == A).all()


def test_segment_objects_recovers_second_instance_from_extra_candidate():
    """Fix #2 (instance recovery): SAM3 grounds both laptop phrases to the SAME
    salient laptop (measured IoU 0.998), so the coincident-mask dedup drops the
    duplicate and the real second laptop is lost -- even though SAM3 returned it
    as a lower-ranked candidate on the first query. On a dedup collision,
    `segment_objects` falls through to a distinct, unused SAME-CATEGORY candidate
    from the pool (here `laptop_a`'s second mask `B`) instead of dropping.

    Crucially the recovery makes NO VLM verdict call: the VLM is as instance-blind
    as SAM3, so asking 'is this exactly laptop b?' rejects the valid recovery
    (this is what dropped `black_laptop_2_0` on the real office_lounge run). Only
    the two _select calls hit the VLM -> exactly 2 calls."""
    seg = FakeSegmenter({"laptop a": [A, B], "laptop b": [A]})
    a = RawObject(name="laptop a", category="laptop", parent=None, id="laptop_a")
    b = RawObject(name="laptop b", category="laptop", parent=None, id="laptop_b")
    vlm = FakeVLM(["YES", "YES"])
    out = segment_objects(IMG, [a, b], seg, vlm)
    assert set(out) == {"laptop_a", "laptop_b"}
    assert (out["laptop_a"] == A).all() and (out["laptop_b"] == B).all()
    assert len(vlm.calls) == 2                 # recovery did NOT re-verify with the VLM


@pytest.mark.gpu
def test_sam3_finds_table_in_office():
    from PIL import Image
    from lens.scenegraph.masks import Sam3Segmenter
    photo = os.environ.get("LENS_DEMO_PHOTO")
    if not photo:
        pytest.skip("set LENS_DEMO_PHOTO to an indoor photo to run this")
    img = np.asarray(Image.open(photo))
    m = Sam3Segmenter().masks_for(img, "white round table")
    assert m and m[0].mean() > 0.01
