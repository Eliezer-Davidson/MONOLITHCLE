# MONOLITHCLE (package `lens`). Tests for ground-truth object recall scoring.
"""Right now a missing object is indistinguishable from a destroyed one: if a
mug is absent from the shipped scene we cannot say whether the VLM never
listed it, SAM-3D failed to reconstruct it, or physics threw it out of the
world.  Scoring the shipped body list against a human-written ground truth
separates 'never found' from 'found and broken'.

Matching is by HEAD NOUN, deliberately.  Recall asks 'did the pipeline find
this object', not 'did it label the colour correctly' -- so ground truth
"red book" matching a body named `yellow_book_0` is the right answer, and the
colour disagreement is for a human to read off the report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.score_recall import match_objects, read_ground_truth


class TestMatching:
    def test_head_noun_match_survives_extra_modifiers(self):
        m = match_objects(["blue mug"], ["blue_ceramic_mug_0"])
        assert m.matched == [("blue mug", "blue_ceramic_mug_0")]
        assert m.missed == []

    def test_a_bare_head_noun_matches_a_decorated_body(self):
        m = match_objects(["laptop"], ["black_laptop_2_0"])
        assert m.matched == [("laptop", "black_laptop_2_0")]

    def test_a_different_head_noun_is_a_miss(self):
        m = match_objects(["mug"], ["yellow_book_0"])
        assert m.missed == ["mug"]
        assert m.extra == ["yellow_book_0"]

    def test_a_colour_disagreement_still_counts_as_found(self):
        """Recall is about finding the object, not labelling it."""
        m = match_objects(["red book"], ["yellow_book_0"])
        assert m.matched == [("red book", "yellow_book_0")]

    def test_one_body_cannot_satisfy_two_ground_truth_objects(self):
        m = match_objects(["mug", "mug"], ["mug_0"])
        assert len(m.matched) == 1
        assert m.missed == ["mug"]

    def test_modifiers_break_ties_so_pairs_are_not_crossed(self):
        """Two books, two bodies: each must claim its own colour."""
        m = match_objects(["red book", "blue book"], ["blue_book_0", "red_book_0"])
        assert dict(m.matched) == {"red book": "red_book_0",
                                   "blue book": "blue_book_0"}

    def test_bodies_with_no_ground_truth_are_reported_as_extra(self):
        m = match_objects(["mug"], ["mug_0", "hallucinated_canopy_0"])
        assert m.extra == ["hallucinated_canopy_0"]

    def test_recall_is_matched_over_ground_truth_count(self):
        m = match_objects(["mug", "book", "lamp"], ["mug_0", "book_0"])
        assert m.recall == 2 / 3

    def test_recall_of_an_empty_ground_truth_is_not_a_crash(self):
        assert match_objects([], ["mug_0"]).recall == 0.0


class TestGroundTruthFile:
    def test_one_object_per_line_with_comments_and_blanks_ignored(self, tmp_path):
        f = tmp_path / "photo_01.objects.txt"
        f.write_text("# kitchen counter, morning light\n"
                     "blue mug\n"
                     "\n"
                     "wooden cutting board\n"
                     "  stainless kettle  \n")
        assert read_ground_truth(f) == ["blue mug", "wooden cutting board",
                                        "stainless kettle"]

    def test_a_missing_ground_truth_file_reads_as_empty_not_an_error(self, tmp_path):
        """A photo without ground truth is unscored, never a crashed report."""
        assert read_ground_truth(tmp_path / "nope.objects.txt") == []


class TestPluralsAndPhrases:
    """Real ground-truth entries written for the 2026-08-18 run. Every case here
    is a phrase a person actually wrote in the fresh-photo ground-truth files
    (`docs/evidence/fresh-photos-2026-08-18/photos/*.objects.txt`)."""

    def test_a_plural_entry_matches_a_singular_body_name(self):
        m = match_objects(["juice bottles on the shelf"], ["juice_bottle_2"])
        assert m.matched == [("juice bottles on the shelf", "juice_bottle_2")]

    def test_es_plurals_are_normalised(self):
        m = match_objects(["cardboard boxes on the racks"], ["cardboard_box_0"])
        assert m.missed == []

    def test_a_stack_phrase_matches_a_body_named_for_the_stack(self):
        """"stack of paper cups" -- the head noun is the STACK, not the cups."""
        m = match_objects(["tall stack of paper cups"], ["white_paper_cup_stack_0"])
        assert m.matched == [("tall stack of paper cups", "white_paper_cup_stack_0")]

    def test_a_stack_phrase_also_matches_a_body_named_for_its_contents(self):
        """The pipeline may name the same object either way; both are the object."""
        m = match_objects(["tall stack of paper cups"], ["paper_cups_0"])
        assert m.missed == []

    def test_a_locative_object_is_never_the_head_noun(self):
        """"bottles ON THE SHELF" is a bottle claim, not a shelf claim. Matching
        the shelf would silently credit recall the pipeline never earned."""
        m = match_objects(["juice bottles on the shelf"], ["wooden_shelf_0"])
        assert m.missed == ["juice bottles on the shelf"]
        assert m.extra == ["wooden_shelf_0"]

    def test_the_of_object_still_counts_toward_tie_breaking(self):
        """Two stacks, two bodies: the contents must decide which is which."""
        m = match_objects(["stack of books", "stack of plates"],
                          ["plate_stack_0", "book_stack_0"])
        assert dict(m.matched) == {"stack of books": "book_stack_0",
                                   "stack of plates": "plate_stack_0"}
