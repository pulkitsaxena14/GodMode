from __future__ import annotations

import pytest
from godmode.resources import DEFAULT_CONFIGS, ResourceConfig, ResourceTile, ResourceType


def make_food_tile(amount: float = 5.0) -> ResourceTile:
    cfg = DEFAULT_CONFIGS[ResourceType.FOOD]
    return ResourceTile(resource_type=ResourceType.FOOD, config=cfg, amount=amount)


def make_wood_tile(amount: float = 8.0) -> ResourceTile:
    cfg = DEFAULT_CONFIGS[ResourceType.WOOD]
    return ResourceTile(resource_type=ResourceType.WOOD, config=cfg, amount=amount)


def make_custom_tile(
    starting: float = 5.0,
    max_val: float = 10.0,
    growth: float = 1.0,
    depleted_growth: float = 0.5,
    depletion_duration: int = 3,
) -> ResourceTile:
    cfg = ResourceConfig(
        resource_type=ResourceType.FOOD,
        starting_value=starting,
        max_value=max_val,
        growth_rate=growth,
        depleted_growth_rate=depleted_growth,
        depletion_duration=depletion_duration,
    )
    return ResourceTile(resource_type=ResourceType.FOOD, config=cfg, amount=starting)


# --- tick tests ---

def test_tick_normal_growth():
    tile = make_custom_tile(starting=5.0, growth=1.0, max_val=10.0)
    tile.tick()
    assert tile.amount == pytest.approx(6.0)


def test_tick_capped_at_max():
    tile = make_custom_tile(starting=9.8, growth=1.0, max_val=10.0)
    tile.tick()
    assert tile.amount == pytest.approx(10.0)


def test_tick_at_max_stays():
    tile = make_custom_tile(starting=10.0, growth=1.0, max_val=10.0)
    tile.tick()
    assert tile.amount == pytest.approx(10.0)


def test_tick_depleted_uses_slow_rate():
    tile = make_custom_tile(starting=5.0, growth=1.0, depleted_growth=0.5, depletion_duration=3)
    tile.depleted_ticks_remaining = 3
    tile.amount = 0.0
    tile.tick()
    assert tile.amount == pytest.approx(0.5)
    assert tile.depleted_ticks_remaining == 2


def test_tick_depleted_count_decrements_to_zero():
    tile = make_custom_tile(starting=0.0, growth=1.0, depleted_growth=0.5, depletion_duration=1)
    tile.depleted_ticks_remaining = 1
    tile.amount = 0.0
    tile.tick()
    assert tile.depleted_ticks_remaining == 0
    assert not tile.is_depleted


# --- harvest tests ---

def test_harvest_partial():
    tile = make_custom_tile(starting=5.0)
    got = tile.harvest(2.0)
    assert got == pytest.approx(2.0)
    assert tile.amount == pytest.approx(3.0)
    assert not tile.is_depleted


def test_harvest_all_triggers_depletion():
    tile = make_custom_tile(starting=5.0, depletion_duration=3)
    got = tile.harvest(5.0)
    assert got == pytest.approx(5.0)
    assert tile.amount == pytest.approx(0.0)
    assert tile.is_depleted
    assert tile.depleted_ticks_remaining == 3


def test_harvest_more_than_available_clamped():
    tile = make_custom_tile(starting=3.0, depletion_duration=3)
    got = tile.harvest(100.0)
    assert got == pytest.approx(3.0)
    assert tile.amount == pytest.approx(0.0)
    assert tile.is_depleted


def test_harvest_zero_no_depletion():
    tile = make_custom_tile(starting=5.0)
    got = tile.harvest(0.0)
    assert got == pytest.approx(0.0)
    assert tile.amount == pytest.approx(5.0)
    assert not tile.is_depleted


# --- depletion recovery cycle ---

def test_depletion_full_recovery_cycle():
    # depletion_duration=2, depleted_growth=0.5, normal_growth=1.0
    tile = make_custom_tile(starting=5.0, growth=1.0, depleted_growth=0.5, depletion_duration=2)
    tile.harvest(5.0)  # fully deplete
    assert tile.is_depleted

    tile.tick()  # tick 1 — depleted rate, ticks_remaining goes 2→1
    assert tile.amount == pytest.approx(0.5)
    assert tile.depleted_ticks_remaining == 1

    tile.tick()  # tick 2 — still depleted rate, ticks_remaining goes 1→0
    assert tile.amount == pytest.approx(1.0)
    assert tile.depleted_ticks_remaining == 0
    assert not tile.is_depleted

    tile.tick()  # tick 3 — normal rate resumes
    assert tile.amount == pytest.approx(2.0)


# --- is_depleted property ---

def test_is_depleted_false_initially():
    tile = make_food_tile()
    assert not tile.is_depleted


def test_is_depleted_true_after_exhaustion():
    tile = make_custom_tile(starting=1.0)
    tile.harvest(1.0)
    assert tile.is_depleted


def test_is_depleted_false_after_recovery():
    tile = make_custom_tile(starting=1.0, depletion_duration=1)
    tile.harvest(1.0)
    tile.tick()  # counts down and applies growth
    assert not tile.is_depleted
