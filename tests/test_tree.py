# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Scene tree builder tests: VLM reply parsing and repair."""
from lens.scenegraph.tree import parse_object_lines, build_tree

REPLY = """yellow sofa | sofa | floor
black laptop | laptop | yellow sofa
red mug | mug | glass desk
"""

def test_parse_and_repair():
    objs = build_tree(parse_object_lines(REPLY))
    by = {o.id: o for o in objs}
    assert by["yellow_sofa_0"].parent is None                 # floor-rooted
    assert by["black_laptop_0"].parent == "yellow_sofa_0"
    assert by["red_mug_0"].parent is None                     # unknown parent -> floor
    assert "surface" in by["yellow_sofa_0"].tags and "graspable" in by["red_mug_0"].tags

def test_cycle_breaks_to_floor():
    objs = build_tree(parse_object_lines("a | box | b\nb | box | a\n"))
    assert any(o.parent is None for o in objs)


def test_cycle_break_does_not_orphan_bystander():
    # x -> a -> b -> c -> a: x merely rests on a member of the cycle, it is
    # not itself part of it. The repair must break the cycle at the node
    # where the repetition is detected (a), not at the walk's starting
    # object -- x must keep resting on a, and only one cycle member may be
    # floor-rooted.
    objs = build_tree(parse_object_lines(
        "x | box | a\na | box | b\nb | box | c\nc | box | a\n"
    ))
    by = {o.id: o for o in objs}
    assert by["x_0"].parent == "a_0"          # bystander keeps its parent
    assert by["a_0"].parent is None           # cycle broken at the repeat node
    assert by["b_0"].parent == "c_0"          # rest of the chain intact
    assert by["c_0"].parent == "a_0"
    floor_rooted = [o.id for o in objs if o.id in {"a_0", "b_0", "c_0"} and o.parent is None]
    assert floor_rooted == ["a_0"]            # exactly one cycle member orphaned


def test_build_tree_is_idempotent():
    objs = parse_object_lines(REPLY)
    once = build_tree(objs)
    snapshot = [(o.id, o.parent, tuple(o.tags)) for o in once]
    twice = build_tree(once)                  # calling again on the same objects
    snapshot2 = [(o.id, o.parent, tuple(o.tags)) for o in twice]
    assert snapshot == snapshot2


def test_trailing_pipe_still_parses():
    objs = parse_object_lines("trailing pipe | table | floor |")
    assert len(objs) == 1
    assert objs[0].name == "trailing pipe"
    assert objs[0].category == "table"
    assert objs[0].parent == "floor"


def test_extra_pipes_still_parses():
    objs = parse_object_lines("too | many | pipes | here")
    assert len(objs) == 1
    assert objs[0].name == "too"
    assert objs[0].category == "many"
    assert objs[0].parent == "pipes"


def test_genuinely_malformed_line_still_skipped():
    assert parse_object_lines("just two | fields") == []


def test_list_marker_stripped_from_id():
    objs = build_tree(parse_object_lines("1. black laptop | laptop | floor"))
    assert objs[0].id == "black_laptop_0"


def test_dash_prefix_still_works():
    objs = build_tree(parse_object_lines("- yellow sofa | sofa | floor"))
    assert objs[0].id == "yellow_sofa_0"
