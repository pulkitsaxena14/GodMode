from __future__ import annotations

import io
import sys

import pytest
from godmode.agent import Agent
from godmode.brain import ScriptedBrain
from godmode.display import print_tile_detail, print_world, render_world
from godmode.resources import DEFAULT_CONFIGS, ResourceTile, ResourceType
from godmode.world import World


def reveal_all(world: World) -> None:
    """Reveal every tile — used in tests that need a fully-populated grid."""
    cx, cy = world.width // 2, world.height // 2
    world.reveal_around(cx, cy, radius=max(world.width, world.height))


def capture_print_world(world: World) -> str:
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_world(world)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def capture_print_tile_detail(world: World, x: int, y: int) -> str:
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_tile_detail(world, x, y)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


# --- print_world ---

def test_print_world_runs_without_error():
    world = World(seed=42)
    capture_print_world(world)  # no exception


def test_print_world_contains_tick_count():
    world = World(seed=42)
    world.tick()
    output = capture_print_world(world)
    assert "tick=1" in output


def test_print_world_contains_calendar_time():
    world = World(seed=42)
    output = capture_print_world(world)
    assert "Y1 M01 D01 H00" in output


def test_print_world_shows_food_marker():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    output = capture_print_world(world)
    assert "F" in output


def test_print_world_shows_wood_marker():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    output = capture_print_world(world)
    assert "W" in output


def test_print_world_shows_empty_marker():
    world = World(resource_density=0.0, seed=0)
    reveal_all(world)
    output = capture_print_world(world)
    assert "." in output


def test_print_world_shows_depleted_marker():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    # Deplete a tile manually
    for y in range(world.height):
        for x in range(world.width):
            tile = world.grid[y][x]
            if tile is not None:
                tile.harvest(tile.amount)  # exhaust it
                break
        else:
            continue
        break
    output = capture_print_world(world)
    assert "!" in output


def test_print_world_correct_dimensions():
    world = World(width=5, height=3, seed=0)
    output = capture_print_world(world)
    lines = [l for l in output.splitlines() if l.strip()]
    # Header line + column label line + 3 data rows = 5 lines minimum
    assert len(lines) >= 5


# --- print_tile_detail ---

def test_print_tile_detail_resource_tile():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    output = capture_print_tile_detail(world, 0, 0)
    assert "amount=" in output


def test_print_tile_detail_empty_tile():
    world = World(resource_density=0.0, seed=0)
    output = capture_print_tile_detail(world, 0, 0)
    assert "empty" in output


def test_print_tile_detail_out_of_bounds():
    world = World(seed=0)
    output = capture_print_tile_detail(world, -1, 0)
    assert "empty" in output


def test_print_world_shows_fog_marker():
    world = World(resource_density=0.0, seed=0)
    # No agent added, no tiles revealed → everything should be ?
    output = capture_print_world(world)
    assert "?" in output


def test_print_world_shows_agent_marker():
    world = World(resource_density=0.0, seed=0)
    agent = Agent(name="ada", birth_tick=0, x=3, y=3, brain=ScriptedBrain([]))
    world.add_agent(agent)
    output = capture_print_world(world)
    assert "@" in output


def test_print_world_agent_overlays_tile():
    world = World(resource_density=1.0, seed=0)
    agent = Agent(name="ada", birth_tick=0, x=0, y=0, brain=ScriptedBrain([]))
    world.add_agent(agent)
    output = capture_print_world(world)
    # Agent marker should appear; tile content at (0,0) is replaced
    assert "@" in output


def test_print_tile_detail_shows_depleted_flag():
    world = World(resource_density=1.0, seed=0)
    reveal_all(world)
    tile = world.grid[0][0]
    if tile is not None:
        tile.harvest(tile.amount)
    output = capture_print_tile_detail(world, 0, 0)
    assert "depleted=True" in output


# --- render_world ---

def test_render_world_returns_string():
    world = World(seed=42)
    result = render_world(world)
    assert isinstance(result, str)


def test_render_world_contains_same_content_as_print_world():
    world = World(seed=42)
    result = render_world(world)
    buf = io.StringIO()
    import sys
    old = sys.stdout
    sys.stdout = buf
    try:
        print_world(world)
    finally:
        sys.stdout = old
    assert result in buf.getvalue()
