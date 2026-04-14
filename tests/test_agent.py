from __future__ import annotations

import pytest
from godmode.agent import (
    CARRY_LIMIT,
    REST_SATIATION_DRAIN,
    Agent,
    ActionResult,
    execute_action,
    get_tips,
    _apply_satiation_drain,
    _apply_starvation,
)
from godmode.brain import ScriptedBrain
from godmode.resources import DEFAULT_CONFIGS, ResourceConfig, ResourceTile, ResourceType
from godmode.world import World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(x: int = 0, y: int = 0, satiation: float = 100.0, health: float = 100.0, brain=None, **kw) -> Agent:
    if brain is None:
        brain = ScriptedBrain([])
    return Agent(name="test", birth_tick=0, x=x, y=y, brain=brain, satiation=satiation, health=health, **kw)


def make_world_food_at(x: int, y: int, amount: float = 8.0) -> World:
    """3x3 empty world with a food tile planted at (x, y)."""
    world = World(width=3, height=3, resource_density=0.0, seed=0)
    cfg = DEFAULT_CONFIGS[ResourceType.FOOD]
    world.grid[y][x] = ResourceTile(resource_type=ResourceType.FOOD, config=cfg, amount=amount)
    return world


def make_empty_world() -> World:
    return World(width=5, height=5, resource_density=0.0, seed=0)


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def test_move_updates_position():
    world = make_empty_world()
    agent = make_agent(x=2, y=2)
    result = execute_action(agent, world, {"action": "move", "dx": 1, "dy": 0})
    assert agent.x == 3
    assert agent.y == 2
    assert result.action == "move"


def test_move_updates_position_negative():
    world = make_empty_world()
    agent = make_agent(x=2, y=2)
    execute_action(agent, world, {"action": "move", "dx": -1, "dy": -1})
    assert agent.x == 1
    assert agent.y == 1


def test_move_clamped_at_west_boundary():
    world = make_empty_world()
    agent = make_agent(x=0, y=2)
    result = execute_action(agent, world, {"action": "move", "dx": -1, "dy": 0})
    assert agent.x == 0  # stayed
    assert "boundary" in result.detail


def test_move_clamped_at_east_boundary():
    world = make_empty_world()
    agent = make_agent(x=4, y=2)
    execute_action(agent, world, {"action": "move", "dx": 1, "dy": 0})
    assert agent.x == 4


def test_move_clamped_at_north_boundary():
    world = make_empty_world()
    agent = make_agent(x=2, y=0)
    execute_action(agent, world, {"action": "move", "dx": 0, "dy": -1})
    assert agent.y == 0


def test_move_clamped_at_south_boundary():
    world = make_empty_world()
    agent = make_agent(x=2, y=4)
    execute_action(agent, world, {"action": "move", "dx": 0, "dy": 1})
    assert agent.y == 4


def test_move_dx_dy_clamped_to_one():
    world = make_empty_world()
    agent = make_agent(x=2, y=2)
    execute_action(agent, world, {"action": "move", "dx": 99, "dy": -99})
    assert agent.x == 3
    assert agent.y == 1


def test_move_costs_3_hunger():
    world = make_empty_world()
    agent = make_agent(x=2, y=2, satiation=50.0)
    agent.tick.__func__  # just check it exists
    # manually execute + drain
    execute_action(agent, world, {"action": "move", "dx": 1, "dy": 0})
    _apply_satiation_drain(agent, "move")
    assert agent.satiation == pytest.approx(47.0)


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------

def test_harvest_adds_to_inventory():
    world = make_world_food_at(1, 1, amount=8.0)
    agent = make_agent(x=1, y=1)
    execute_action(agent, world, {"action": "harvest"})
    assert agent.inventory.get(ResourceType.FOOD, 0.0) > 0


def test_harvest_capped_by_yield():
    world = make_world_food_at(1, 1, amount=10.0)
    agent = make_agent(x=1, y=1)
    execute_action(agent, world, {"action": "harvest"})
    # DEFAULT_CONFIGS FOOD harvest_yield = 5.0
    assert agent.inventory.get(ResourceType.FOOD, 0.0) == pytest.approx(5.0)


def test_harvest_capped_by_carry_limit():
    world = make_world_food_at(1, 1, amount=10.0)
    agent = make_agent(x=1, y=1)
    # Pre-fill inventory to leave only 2 units of carry room
    agent.inventory[ResourceType.WOOD] = CARRY_LIMIT - 2.0
    execute_action(agent, world, {"action": "harvest"})
    assert agent.inventory.get(ResourceType.FOOD, 0.0) == pytest.approx(2.0)


def test_harvest_empty_tile_yields_zero():
    world = make_empty_world()
    agent = make_agent(x=2, y=2)
    result = execute_action(agent, world, {"action": "harvest"})
    assert agent.carry_total == pytest.approx(0.0)
    assert "nothing" in result.detail


def test_harvest_at_carry_limit_yields_zero():
    world = make_world_food_at(1, 1, amount=10.0)
    agent = make_agent(x=1, y=1)
    agent.inventory[ResourceType.WOOD] = CARRY_LIMIT
    result = execute_action(agent, world, {"action": "harvest"})
    assert agent.inventory.get(ResourceType.FOOD, 0.0) == pytest.approx(0.0)
    assert "too much" in result.detail


def test_harvest_costs_4_hunger():
    world = make_world_food_at(1, 1)
    agent = make_agent(x=1, y=1, satiation=50.0)
    execute_action(agent, world, {"action": "harvest"})
    _apply_satiation_drain(agent, "harvest")
    assert agent.satiation == pytest.approx(46.0)


# ---------------------------------------------------------------------------
# Eat
# ---------------------------------------------------------------------------

def test_eat_restores_hunger():
    world = make_empty_world()
    agent = make_agent(satiation=40.0)
    agent.inventory[ResourceType.FOOD] = 5.0
    execute_action(agent, world, {"action": "eat", "amount": 1.0})
    assert agent.satiation == pytest.approx(48.0)  # +8 per unit


def test_eat_clamped_at_100():
    world = make_empty_world()
    agent = make_agent(satiation=98.0)
    agent.inventory[ResourceType.FOOD] = 5.0
    execute_action(agent, world, {"action": "eat", "amount": 5.0})
    assert agent.satiation == pytest.approx(100.0)


def test_eat_more_than_inventory_consumes_only_available():
    world = make_empty_world()
    agent = make_agent(satiation=20.0)
    agent.inventory[ResourceType.FOOD] = 2.0
    execute_action(agent, world, {"action": "eat", "amount": 10.0})
    assert agent.inventory.get(ResourceType.FOOD, 0.0) == pytest.approx(0.0)
    assert agent.satiation == pytest.approx(36.0)  # 20 + 2*8


def test_eat_costs_2_hunger():
    world = make_empty_world()
    agent = make_agent(satiation=50.0)
    agent.inventory[ResourceType.FOOD] = 5.0
    execute_action(agent, world, {"action": "eat", "amount": 1.0})
    _apply_satiation_drain(agent, "eat")
    # +8 then -2
    assert agent.satiation == pytest.approx(56.0)


def test_eat_zero_food_no_crash():
    world = make_empty_world()
    agent = make_agent(satiation=50.0)
    result = execute_action(agent, world, {"action": "eat", "amount": 5.0})
    assert "no food" in result.detail
    assert agent.satiation == pytest.approx(50.0)


def test_eat_zero_amount_no_crash():
    world = make_empty_world()
    agent = make_agent(satiation=50.0)
    agent.inventory[ResourceType.FOOD] = 5.0
    result = execute_action(agent, world, {"action": "eat", "amount": 0.0})
    assert "nothing" in result.detail


# ---------------------------------------------------------------------------
# Rest
# ---------------------------------------------------------------------------

def test_rest_costs_minimal_hunger():
    world = make_empty_world()
    agent = make_agent(satiation=50.0)
    execute_action(agent, world, {"action": "rest"})
    _apply_satiation_drain(agent, "rest")
    assert agent.satiation == pytest.approx(50.0 - REST_SATIATION_DRAIN)


def test_rest_changes_nothing_else():
    world = make_empty_world()
    agent = make_agent(x=2, y=2, satiation=50.0, health=80.0)
    execute_action(agent, world, {"action": "rest"})
    assert agent.x == 2
    assert agent.y == 2
    assert agent.health == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Hunger / health mechanics
# ---------------------------------------------------------------------------

def test_hunger_floors_at_zero():
    world = make_empty_world()
    agent = make_agent(satiation=1.0)
    _apply_satiation_drain(agent, "harvest")  # -4
    assert agent.satiation == pytest.approx(0.0)


def test_starvation_drains_health():
    agent = make_agent(satiation=0.0, health=100.0)
    _apply_starvation(agent)
    assert agent.health == pytest.approx(95.0)


def test_health_never_recovers():
    agent = make_agent(satiation=0.0, health=50.0)
    _apply_starvation(agent)
    assert agent.health == pytest.approx(45.0)
    # Even if we somehow restore hunger, health stays at 45
    agent.satiation = 100.0
    _apply_starvation(agent)
    assert agent.health == pytest.approx(45.0)


def test_health_floors_at_zero():
    agent = make_agent(satiation=0.0, health=3.0)
    _apply_starvation(agent)
    assert agent.health == pytest.approx(0.0)


def test_agent_alive_false_at_zero_health():
    agent = make_agent(health=0.0)
    assert not agent.alive


def test_agent_alive_true_above_zero_health():
    agent = make_agent(health=0.1)
    assert agent.alive


def test_dead_agent_cannot_act():
    world = make_empty_world()
    brain = ScriptedBrain([{"action": "move", "dx": 1, "dy": 0}])
    agent = make_agent(x=2, y=2, health=0.0, brain=brain)
    result = agent.tick(world)
    assert result.action == "dead"
    assert agent.x == 2  # didn't move


# ---------------------------------------------------------------------------
# Invalid / unknown action
# ---------------------------------------------------------------------------

def test_invalid_action_falls_back_to_rest_cost():
    world = make_empty_world()
    agent = make_agent(satiation=50.0)
    result = execute_action(agent, world, {"action": "fly"})
    assert result.action == "invalid"
    _apply_satiation_drain(agent, result.action)
    assert agent.satiation == pytest.approx(48.0)  # base cost


def test_missing_action_key_falls_back():
    world = make_empty_world()
    agent = make_agent()
    result = execute_action(agent, world, {})
    assert result.action == "invalid"


def test_extra_keys_ignored():
    world = make_empty_world()
    agent = make_agent(x=2, y=2)
    result = execute_action(agent, world, {"action": "move", "dx": 1, "dy": 0, "reason": "exploring"})
    assert agent.x == 3


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def test_carry_total_sums_types():
    agent = make_agent()
    agent.inventory[ResourceType.FOOD] = 7.5
    agent.inventory[ResourceType.WOOD] = 4.0
    assert agent.carry_total == pytest.approx(11.5)


def test_carry_total_empty():
    agent = make_agent()
    assert agent.carry_total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tips
# ---------------------------------------------------------------------------

def test_tips_healthy_agent_empty():
    agent = make_agent(satiation=100.0, health=100.0)
    assert get_tips(agent) == []


def test_tips_hunger_faint():
    agent = make_agent(satiation=39.0)
    tips = get_tips(agent)
    assert any("faint hunger" in t for t in tips)


def test_tips_hunger_25():
    agent = make_agent(satiation=24.0)
    tips = get_tips(agent)
    assert any("becoming hard to ignore" in t for t in tips)


def test_tips_hunger_15_no_food():
    agent = make_agent(satiation=14.0)
    tips = get_tips(agent)
    assert any("nothing to eat" in t for t in tips)


def test_tips_hunger_15_has_food():
    agent = make_agent(satiation=14.0)
    agent.inventory[ResourceType.FOOD] = 3.0
    tips = get_tips(agent)
    assert any("consider eating" in t for t in tips)


def test_tips_hunger_zero():
    agent = make_agent(satiation=0.0)
    tips = get_tips(agent)
    assert any("starving" in t for t in tips)


def test_tips_health_low():
    agent = make_agent(satiation=80.0, health=49.0)
    tips = get_tips(agent)
    assert any("weakened" in t for t in tips)


def test_tips_carry_heavy():
    agent = make_agent()
    agent.inventory[ResourceType.WOOD] = 17.0
    tips = get_tips(agent)
    assert any("heavy load" in t for t in tips)


def test_tips_multiple_simultaneous():
    agent = make_agent(satiation=0.0, health=40.0)
    agent.inventory[ResourceType.WOOD] = 17.0
    tips = get_tips(agent)
    assert len(tips) >= 3  # starving + weakened + heavy


def test_tips_exact_boundary_hunger_40_not_triggered():
    agent = make_agent(satiation=40.0)
    tips = get_tips(agent)
    assert tips == []


def test_tips_exact_boundary_carry_16_not_triggered():
    agent = make_agent()
    agent.inventory[ResourceType.WOOD] = 16.0
    tips = get_tips(agent)
    assert not any("heavy load" in t for t in tips)


# ---------------------------------------------------------------------------
# Last actions log trimming
# ---------------------------------------------------------------------------

def test_last_actions_capped_at_max():
    world = make_empty_world()
    brain = ScriptedBrain([{"action": "rest"}] * 10)
    agent = make_agent(x=2, y=2, brain=brain)
    for _ in range(6):
        agent.tick(world)
    assert len(agent.last_actions) == agent._max_last_actions


# ---------------------------------------------------------------------------
# Warmth drain
# ---------------------------------------------------------------------------

from godmode.agent import (
    ACTIVE_WARMTH_DRAIN,
    REST_WARMTH_DRAIN,
    MOVE_WARMTH_EXTRA,
    BURN_RESTORE_PER_UNIT,
    FREEZING_HEALTH_DRAIN,
    _apply_warmth_drain,
    _apply_freezing,
)


def test_warmth_starts_at_100():
    agent = make_agent()
    assert agent.warmth == pytest.approx(100.0)


def test_warmth_drain_rest_minimal():
    agent = make_agent()
    _apply_warmth_drain(agent, "rest")
    assert agent.warmth == pytest.approx(100.0 - REST_WARMTH_DRAIN)


def test_warmth_drain_move():
    agent = make_agent()
    _apply_warmth_drain(agent, "move")
    assert agent.warmth == pytest.approx(100.0 - ACTIVE_WARMTH_DRAIN - MOVE_WARMTH_EXTRA)


def test_warmth_drain_harvest():
    agent = make_agent()
    _apply_warmth_drain(agent, "harvest")
    assert agent.warmth == pytest.approx(100.0 - ACTIVE_WARMTH_DRAIN)


def test_warmth_drain_eat():
    agent = make_agent()
    _apply_warmth_drain(agent, "eat")
    assert agent.warmth == pytest.approx(100.0 - ACTIVE_WARMTH_DRAIN)


def test_warmth_floors_at_zero():
    agent = make_agent(warmth=0.5)
    _apply_warmth_drain(agent, "move")
    assert agent.warmth == pytest.approx(0.0)


def test_freezing_drains_health():
    agent = make_agent(warmth=0.0)
    _apply_freezing(agent)
    assert agent.health == pytest.approx(100.0 - FREEZING_HEALTH_DRAIN)


def test_freezing_no_effect_above_zero():
    agent = make_agent(warmth=1.0)
    _apply_freezing(agent)
    assert agent.health == pytest.approx(100.0)


def test_starvation_and_freezing_stack():
    agent = make_agent(satiation=0.0, warmth=0.0)
    from godmode.agent import STARVATION_HEALTH_DRAIN
    _apply_starvation(agent)
    _apply_freezing(agent)
    assert agent.health == pytest.approx(100.0 - STARVATION_HEALTH_DRAIN - FREEZING_HEALTH_DRAIN)


# ---------------------------------------------------------------------------
# Burn action
# ---------------------------------------------------------------------------

def test_burn_restores_warmth():
    agent = make_agent(warmth=50.0)
    agent.inventory[ResourceType.WOOD] = 3.0
    result = execute_action(agent, make_empty_world(), {"action": "burn", "amount": 2.0})
    assert result.action == "burn"
    assert agent.warmth == pytest.approx(50.0 + 2.0 * BURN_RESTORE_PER_UNIT)


def test_burn_capped_at_100():
    agent = make_agent(warmth=95.0)
    agent.inventory[ResourceType.WOOD] = 5.0
    execute_action(agent, make_empty_world(), {"action": "burn", "amount": 5.0})
    assert agent.warmth == pytest.approx(100.0)


def test_burn_no_wood_fails():
    agent = make_agent()
    result = execute_action(agent, make_empty_world(), {"action": "burn", "amount": 2.0})
    assert "no wood" in result.detail


def test_burn_zero_amount():
    agent = make_agent(warmth=80.0)
    agent.inventory[ResourceType.WOOD] = 3.0
    result = execute_action(agent, make_empty_world(), {"action": "burn", "amount": 0.0})
    assert "burned nothing" in result.detail
    assert agent.warmth == pytest.approx(80.0)


def test_burn_consumes_wood():
    agent = make_agent()
    agent.inventory[ResourceType.WOOD] = 5.0
    execute_action(agent, make_empty_world(), {"action": "burn", "amount": 3.0})
    assert agent.inventory[ResourceType.WOOD] == pytest.approx(2.0)


def test_burn_more_than_available_partial():
    agent = make_agent(warmth=50.0)
    agent.inventory[ResourceType.WOOD] = 1.5
    execute_action(agent, make_empty_world(), {"action": "burn", "amount": 5.0})
    assert agent.warmth == pytest.approx(50.0 + 1.5 * BURN_RESTORE_PER_UNIT)
    assert agent.inventory[ResourceType.WOOD] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Warmth tips
# ---------------------------------------------------------------------------

def test_tips_cold_chill():
    agent = make_agent(warmth=39.0)
    tips = get_tips(agent)
    assert any("chill" in t for t in tips)


def test_tips_cold_shivering():
    agent = make_agent(warmth=24.0)
    tips = get_tips(agent)
    assert any("shivering" in t for t in tips)


def test_tips_cold_freezing():
    agent = make_agent(warmth=14.0)
    tips = get_tips(agent)
    assert any("freezing" in t.lower() for t in tips)


def test_tips_cold_freezing_to_death():
    agent = make_agent(warmth=0.0)
    tips = get_tips(agent)
    assert any("freezing to death" in t for t in tips)


def test_tips_cold_and_hunger_simultaneous():
    agent = make_agent(satiation=10.0, warmth=10.0)
    tips = get_tips(agent)
    assert len(tips) >= 2


def test_tips_warmth_exact_boundary_40_not_triggered():
    agent = make_agent(warmth=40.0)
    tips = get_tips(agent)
    assert not any("chill" in t for t in tips)


# ---------------------------------------------------------------------------
# Visited tiles (memory)
# ---------------------------------------------------------------------------

def test_visited_tiles_empty_initially():
    agent = make_agent()
    assert agent.visited_tiles == []


def test_visited_tiles_records_after_tick():
    world = make_empty_world()
    brain = ScriptedBrain([{"action": "rest"}])
    agent = make_agent(x=2, y=2, brain=brain)
    agent.tick(world)
    assert len(agent.visited_tiles) == 1


def test_visited_tiles_capped_at_5():
    world = make_empty_world()
    brain = ScriptedBrain([{"action": "rest"}] * 10)
    agent = make_agent(x=2, y=2, brain=brain)
    for _ in range(8):
        agent.tick(world)
    assert len(agent.visited_tiles) <= agent._max_visited_tiles


def test_visited_tiles_correct_position():
    world = make_empty_world()
    brain = ScriptedBrain([{"action": "rest"}])
    agent = make_agent(x=1, y=3, brain=brain)
    agent.tick(world)
    assert agent.visited_tiles[-1]["x"] == 1
    assert agent.visited_tiles[-1]["y"] == 3


def test_visited_tiles_describes_tile_type():
    world = make_empty_world()
    cfg = DEFAULT_CONFIGS[ResourceType.FOOD]
    world.grid[2][2] = ResourceTile(resource_type=ResourceType.FOOD, config=cfg, amount=5.0)
    brain = ScriptedBrain([{"action": "rest"}])
    agent = make_agent(x=2, y=2, brain=brain)
    agent.tick(world)
    assert "Food" in agent.visited_tiles[-1]["tile"] or "F5" in agent.visited_tiles[-1]["tile"]


def test_visited_tiles_describes_empty():
    world = make_empty_world()
    brain = ScriptedBrain([{"action": "rest"}])
    agent = make_agent(x=2, y=2, brain=brain)
    agent.tick(world)
    assert agent.visited_tiles[-1]["tile"] == "empty"


def test_dead_agent_no_memory_recorded():
    world = make_empty_world()
    brain = ScriptedBrain([{"action": "rest"}])
    agent = make_agent(x=2, y=2, brain=brain, health=0.0)
    agent.tick(world)
    assert agent.visited_tiles == []


def test_tips_carry_full():
    agent = make_agent()
    agent.inventory[ResourceType.FOOD] = 20.0
    tips = get_tips(agent)
    assert any("hands are full" in t for t in tips)


def test_tips_carry_full_overrides_heavy_load():
    # at exactly CARRY_LIMIT, only the "full" tip fires, not the "heavy load" tip
    from godmode.agent import CARRY_LIMIT
    agent = make_agent()
    agent.inventory[ResourceType.FOOD] = CARRY_LIMIT
    tips = get_tips(agent)
    assert any("hands are full" in t for t in tips)
    assert not any("heavy load" in t for t in tips)


def test_tips_depleted_tile():
    from godmode.resources import DEFAULT_CONFIGS, ResourceTile, ResourceType
    world = make_empty_world()
    cfg = DEFAULT_CONFIGS[ResourceType.FOOD]
    tile = ResourceTile(resource_type=ResourceType.FOOD, config=cfg, amount=5.0)
    tile.harvest(tile.amount)  # deplete it
    world.grid[2][2] = tile
    agent = make_agent(x=2, y=2)
    tips = get_tips(agent, world)
    assert any("exhausted" in t for t in tips)


def test_tips_no_depleted_tip_on_healthy_tile():
    from godmode.resources import DEFAULT_CONFIGS, ResourceTile, ResourceType
    world = make_empty_world()
    cfg = DEFAULT_CONFIGS[ResourceType.FOOD]
    world.grid[2][2] = ResourceTile(resource_type=ResourceType.FOOD, config=cfg, amount=5.0)
    agent = make_agent(x=2, y=2)
    tips = get_tips(agent, world)
    assert not any("exhausted" in t for t in tips)


def test_tips_no_depleted_tip_on_empty_tile():
    world = make_empty_world()
    agent = make_agent(x=2, y=2)
    tips = get_tips(agent, world)
    assert not any("exhausted" in t for t in tips)


def test_tips_no_world_still_works():
    agent = make_agent()
    tips = get_tips(agent)  # no world passed — should not raise
    assert isinstance(tips, list)


# ---------------------------------------------------------------------------
# Relationship / update_relationship
# ---------------------------------------------------------------------------

def test_relationships_empty_initially():
    agent = make_agent()
    assert agent.relationships == {}


def test_update_relationship_creates_entry():
    from godmode.agent import Relationship, update_relationship
    agent = make_agent()
    update_relationship(agent, "Bo", 0.8, "generous", 1)
    assert "Bo" in agent.relationships
    assert agent.relationships["Bo"].score == pytest.approx(0.8)
    assert agent.relationships["Bo"].note == "generous"
    assert agent.relationships["Bo"].count == 1
    assert agent.relationships["Bo"].last_tick == 1


def test_update_relationship_weighted_average():
    from godmode.agent import update_relationship
    agent = make_agent()
    update_relationship(agent, "Bo", 1.0, "great", 1)
    update_relationship(agent, "Bo", -1.0, "terrible", 2)
    # 0.7 * 1.0 + 0.3 * -1.0 = 0.4
    assert agent.relationships["Bo"].score == pytest.approx(0.4)
    assert agent.relationships["Bo"].count == 2


def test_update_relationship_clamped_to_bounds():
    from godmode.agent import update_relationship
    agent = make_agent()
    update_relationship(agent, "Bo", 2.0, "over", 1)
    assert agent.relationships["Bo"].score <= 1.0
    update_relationship(agent, "Bo", -5.0, "under", 2)
    assert agent.relationships["Bo"].score >= -1.0


def test_update_relationship_multiple_agents():
    from godmode.agent import update_relationship
    agent = make_agent()
    update_relationship(agent, "Bo", 0.5, "ok", 1)
    update_relationship(agent, "Cal", -0.3, "rude", 1)
    assert len(agent.relationships) == 2
    assert agent.relationships["Bo"].score == pytest.approx(0.5)
    assert agent.relationships["Cal"].score == pytest.approx(-0.3)
