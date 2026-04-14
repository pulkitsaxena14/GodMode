from __future__ import annotations

import pytest
from godmode.agent import Agent
from godmode.brain import ScriptedBrain
from godmode.resources import DEFAULT_CONFIGS, ResourceConfig, ResourceTile, ResourceType
from godmode.world import World


def reveal_all(world: World) -> None:
    """Reveal every tile — used in tests that need a fully-populated grid."""
    cx, cy = world.width // 2, world.height // 2
    world.reveal_around(cx, cy, radius=max(world.width, world.height))


# --- generation ---

def test_generation_deterministic():
    w1 = World(seed=42)
    w2 = World(seed=42)
    reveal_all(w1)
    reveal_all(w2)
    for y in range(w1.height):
        for x in range(w1.width):
            t1 = w1.grid[y][x]
            t2 = w2.grid[y][x]
            if t1 is None:
                assert t2 is None
            else:
                assert t2 is not None
                assert t1.resource_type == t2.resource_type
                assert t1.amount == pytest.approx(t2.amount)


def test_generation_different_seeds_differ():
    w1 = World(seed=1)
    w2 = World(seed=2)
    reveal_all(w1)
    reveal_all(w2)
    grids_differ = any(
        (w1.grid[y][x] is None) != (w2.grid[y][x] is None)
        for y in range(w1.height)
        for x in range(w1.width)
    )
    assert grids_differ


def test_generation_density_approximate():
    world = World(width=20, height=20, resource_density=0.3, seed=0)
    reveal_all(world)
    filled = sum(1 for y in range(20) for x in range(20) if world.grid[y][x] is not None)
    # Allow generous margin: density 0.3 → expect 20-200 tiles out of 400
    assert 20 <= filled <= 200


def test_generation_empty_world():
    world = World(resource_density=0.0, seed=0)
    reveal_all(world)
    for row in world.grid:
        for tile in row:
            assert tile is None


def test_generation_full_world():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    for row in world.grid:
        for tile in row:
            assert tile is not None


def test_generation_uses_starting_value():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    for row in world.grid:
        for tile in row:
            expected = tile.config.starting_value
            assert tile.amount == pytest.approx(expected)


# --- tick ---

def test_tick_advances_resource_amounts():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    before = {
        (y, x): world.grid[y][x].amount
        for y in range(world.height)
        for x in range(world.width)
    }
    world.tick()
    for y in range(world.height):
        for x in range(world.width):
            tile = world.grid[y][x]
            prev = before[(y, x)]
            assert tile.amount >= prev  # can't decrease from tick alone


def test_tick_count_increments():
    world = World(seed=0)
    assert world.tick_count == 0
    world.tick()
    assert world.tick_count == 1
    world.tick()
    assert world.tick_count == 2


def test_world_time_property():
    world = World(seed=0)
    assert world.time.hour == 0
    assert world.time.day == 1
    assert world.time.year == 1
    world.tick()
    assert world.time.hour == 1


def test_tick_empty_world_no_crash():
    world = World(resource_density=0.0, seed=0)
    world.tick()
    assert world.tick_count == 1


# --- harvest ---

def test_harvest_valid_tile():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    # find first food tile
    for y in range(world.height):
        for x in range(world.width):
            tile = world.grid[y][x]
            if tile and tile.resource_type == ResourceType.FOOD:
                before = tile.amount
                got = world.harvest(x, y, 2.0)
                assert got == pytest.approx(min(2.0, before))
                assert tile.amount == pytest.approx(before - got)
                return
    pytest.skip("No food tile found — increase density or change seed")


def test_harvest_empty_tile():
    world = World(resource_density=0.0, seed=0)  # all empty
    got = world.harvest(0, 0, 5.0)
    assert got == pytest.approx(0.0)


def test_harvest_out_of_bounds_negative():
    world = World(seed=0)
    assert world.harvest(-1, 0, 5.0) == pytest.approx(0.0)
    assert world.harvest(0, -1, 5.0) == pytest.approx(0.0)


def test_harvest_out_of_bounds_too_large():
    world = World(width=5, height=5, seed=0)
    assert world.harvest(5, 0, 5.0) == pytest.approx(0.0)
    assert world.harvest(0, 5, 5.0) == pytest.approx(0.0)


# --- get_tile ---

def test_get_tile_in_bounds_returns_tile_or_none():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    tile = world.get_tile(0, 0)
    assert isinstance(tile, ResourceTile)


def test_get_tile_out_of_bounds_returns_none():
    world = World(width=5, height=5, seed=0)
    assert world.get_tile(-1, 0) is None
    assert world.get_tile(5, 0) is None
    assert world.get_tile(0, -1) is None
    assert world.get_tile(0, 5) is None


# --- custom configs ---

# --- agents ---

def test_add_agent_valid_position():
    world = World(width=5, height=5, seed=0)
    agent = Agent(name="ada", birth_tick=0, x=2, y=2, brain=ScriptedBrain([]))
    world.add_agent(agent)
    assert agent in world.agents


def test_add_agent_out_of_bounds_raises():
    world = World(width=5, height=5, seed=0)
    agent = Agent(name="ada", birth_tick=0, x=10, y=0, brain=ScriptedBrain([]))
    with pytest.raises(ValueError):
        world.add_agent(agent)


def test_world_tick_calls_agent_tick():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    brain = ScriptedBrain([{"action": "move", "dx": 1, "dy": 0}])
    agent = Agent(name="ada", birth_tick=0, x=2, y=2, brain=brain)
    world.add_agent(agent)
    world.tick()
    assert agent.x == 3  # agent moved


def test_dead_agent_tick_does_not_crash():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    agent = Agent(name="ada", birth_tick=0, x=2, y=2, brain=ScriptedBrain([]), health=0.0)
    world.add_agent(agent)
    world.tick()  # should not raise
    assert world.tick_count == 1


# --- fog of war / lazy generation ---

def test_world_starts_unrevealed():
    world = World(seed=0)
    assert len(world._revealed) == 0


def test_reveal_around_marks_tiles():
    world = World(seed=0)
    world.reveal_around(2, 2, radius=1)
    # 3x3 area = 9 tiles
    assert len(world._revealed) == 9


def test_reveal_around_density_creates_resources():
    world = World(resource_density=1.0, seed=42)
    world.reveal_around(2, 2, radius=1)
    revealed_tiles = [world.grid[y][x] for (x, y) in world._revealed]
    assert all(t is not None for t in revealed_tiles)


def test_reveal_around_idempotent():
    world = World(resource_density=1.0, seed=42)
    world.reveal_around(2, 2, radius=1)
    tile_before = world.grid[2][2]
    world.reveal_around(2, 2, radius=1)  # second call — no re-roll
    assert world.grid[2][2] is tile_before
    assert len(world._revealed) == 9


def test_add_agent_reveals_spawn_area():
    world = World(width=5, height=5, seed=0)
    agent = Agent(name="ada", birth_tick=0, x=2, y=2, brain=ScriptedBrain([]))
    world.add_agent(agent)
    # 3x3 around (2,2) should be revealed
    assert world.is_revealed(2, 2)
    assert world.is_revealed(1, 1)
    assert world.is_revealed(3, 3)


def test_tick_reveals_around_agent_after_move():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    brain = ScriptedBrain([{"action": "move", "dx": 1, "dy": 0}])
    agent = Agent(name="ada", birth_tick=0, x=2, y=2, brain=brain)
    world.add_agent(agent)
    assert not world.is_revealed(4, 2)  # not yet visible
    world.tick()
    assert world.is_revealed(4, 2)  # now revealed after moving to (3,2)


def test_is_revealed_out_of_bounds():
    world = World(seed=0)
    assert not world.is_revealed(-1, 0)
    assert not world.is_revealed(0, -1)
    assert not world.is_revealed(999, 999)


def test_reveal_deterministic_with_seed():
    w1 = World(resource_density=0.5, seed=7)
    w2 = World(resource_density=0.5, seed=7)
    w1.reveal_around(2, 2, radius=1)
    w2.reveal_around(2, 2, radius=1)
    for (x, y) in w1._revealed:
        assert (w1.grid[y][x] is None) == (w2.grid[y][x] is None)


def test_custom_resource_configs():
    custom = {
        ResourceType.FOOD: ResourceConfig(
            resource_type=ResourceType.FOOD,
            starting_value=1.0,
            max_value=2.0,
            growth_rate=0.1,
            depleted_growth_rate=0.05,
            depletion_duration=2,
        )
    }
    world = World(resource_density=1.0, seed=0, resource_configs=custom)
    reveal_all(world)
    # Only food should exist (only one type in configs)
    for row in world.grid:
        for tile in row:
            if tile is not None:
                assert tile.resource_type == ResourceType.FOOD
                assert tile.amount == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Interaction system
# ---------------------------------------------------------------------------

def _make_agent_at(name, x, y):
    return Agent(name=name, birth_tick=0, x=x, y=y, brain=ScriptedBrain([]))


def test_interact_action_resolves_to_rest_when_target_unknown():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    agent = _make_agent_at("Ada", 3, 3)
    world.add_agent(agent)
    # ScriptedBrain returns interact; target doesn't exist → replaced with rest
    from godmode.brain import ScriptedBrain
    agent.brain = ScriptedBrain([{"action": "interact", "target": "Ghost", "message": "hello"}])
    world.tick()
    assert world._last_interactions == []


def test_interact_too_far_resolves_to_rest():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    ada = _make_agent_at("Ada", 0, 0)
    bo = _make_agent_at("Bo", 4, 4)
    world.add_agent(ada)
    world.add_agent(bo)
    from godmode.brain import ScriptedBrain
    ada.brain = ScriptedBrain([{"action": "interact", "target": "Bo", "message": "hello"}])
    world.tick()
    assert world._last_interactions == []


def test_interact_within_range_logged():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    ada = _make_agent_at("Ada", 2, 2)
    bo = _make_agent_at("Bo", 3, 2)
    world.add_agent(ada)
    world.add_agent(bo)
    from godmode.brain import ScriptedBrain
    ada.brain = ScriptedBrain([{"action": "interact", "target": "Bo", "message": "hello"}])
    world.tick()
    assert len(world._last_interactions) == 1
    ix = world._last_interactions[0]
    assert ix["initiator"] == "Ada"
    assert ix["target"] == "Bo"
    assert ix["message"] == "hello"


def test_interact_updates_both_relationships():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    ada = _make_agent_at("Ada", 2, 2)
    bo = _make_agent_at("Bo", 2, 2)
    world.add_agent(ada)
    world.add_agent(bo)
    from godmode.brain import ScriptedBrain
    ada.brain = ScriptedBrain([{"action": "interact", "target": "Bo", "message": "hi"}])
    world.tick()
    assert "Bo" in ada.relationships
    assert "Ada" in bo.relationships


def test_interact_empty_message_no_interaction():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    ada = _make_agent_at("Ada", 2, 2)
    bo = _make_agent_at("Bo", 2, 2)
    world.add_agent(ada)
    world.add_agent(bo)
    from godmode.brain import ScriptedBrain
    ada.brain = ScriptedBrain([{"action": "interact", "target": "Bo", "message": ""}])
    world.tick()
    assert world._last_interactions == []


def test_last_interactions_cleared_each_tick():
    world = World(width=5, height=5, resource_density=0.0, seed=0)
    ada = _make_agent_at("Ada", 2, 2)
    bo = _make_agent_at("Bo", 2, 2)
    world.add_agent(ada)
    world.add_agent(bo)
    from godmode.brain import ScriptedBrain
    ada.brain = ScriptedBrain([{"action": "interact", "target": "Bo", "message": "hi"}, {"action": "rest"}])
    world.tick()
    assert len(world._last_interactions) == 1
    world.tick()
    assert len(world._last_interactions) == 0
