"""Convex-decomposition budget rule.

An earlier revision gave every object a flat 32 hulls -- four times as concave a
collider as a sofa needs -- which let small objects slip into the creases
instead of resting on top. The budget is now chosen per object by a keyword
rule (foliage 32, slabs 8, default 16), and these tests pin that rule so that a
change to any class is deliberate.
"""
import pytest

from lens.export import collision


@pytest.mark.parametrize("name,expected", [
    ("potted_plant_000", 32),
    ("glass_vase_000", 32),
    ("grey_and_yellow_sofa_000", 8),
    ("white_round_table_000", 8),
    ("black_backpack_000", 16),
    ("black_laptop_000", 16),
])
def test_hull_budget_follows_the_keyword_rule(name, expected):
    assert collision.hull_budget(name) == expected


def test_foliage_terms_win_over_slab_ones():
    """Foliage is tested first, so a name carrying both classes resolves to 32.
    Testing slabs first would silently coarsen a plant on a table to 8 hulls."""
    assert collision.hull_budget("potted_plant_on_table_000") == 32


def test_hull_budget_is_case_insensitive():
    assert collision.hull_budget("Grey_And_Yellow_SOFA_000") == 8


def test_office_lounge_names_all_resolve_to_eight():
    """Ground truth. Every office_lounge asset takes a budget of 8 hulls --
    including the backpack and both laptops, which only score as slabs because
    each child id carries its support (`black_laptop_on_sofa_000`). Any rule
    that gives those 16 breaks the convention the campaigns were built on."""
    for name in ("grey_and_yellow_sofa_000", "white_round_table_000",
                 "black_backpack_on_sofa_000", "black_laptop_on_sofa_000",
                 "black_laptop_on_white_table_000",
                 "black_tumbler_on_white_table_000"):
        assert collision.hull_budget(name) == 8, name


def test_second_development_scene_names_all_resolve_to_eight():
    """The same holds on the second development scene, so the budgets follow a
    convention rather than an accident of office_lounge."""
    for name in ("orange_sofa_000", "purple_side_table_000",
                 "table_lamp_on_purple_side_table_000",
                 "teal_telephone_on_purple_side_table_000",
                 "floor_lamp_with_red_pleated_shade_000"):
        assert collision.hull_budget(name) == 8, name


def test_child_budget_accounts_for_its_support():
    """Ids here do not embed the support (`black_laptop_2_0`), so the bare rule
    would give a laptop resting on furniture the generic budget. The support is
    supplied from the scene tree rather than by renaming every object.

    A laptop on its own is generic; the same laptop on a bench, a sofa or a
    table inherits the slab budget, because what the collider has to rest
    against is the furniture, not the laptop.
    """
    assert collision.hull_budget("black_laptop_2_0") == 16
    assert collision.hull_budget("black_laptop_2_0",
                                 parent="grey_and_yellow_bench_0") == 8
    assert collision.hull_budget("black_laptop_2_0",
                                 parent="grey_and_yellow_sofa_0") == 8
    assert collision.hull_budget("black_thermos_0",
                                 parent="white_round_table_0") == 8
    # a support that is not furniture leaves the child generic
    assert collision.hull_budget("black_thermos_0",
                                 parent="black_backpack_0") == 16
