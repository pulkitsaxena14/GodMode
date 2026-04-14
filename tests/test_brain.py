from __future__ import annotations

import json

import pytest
from godmode.agent import Agent
from godmode.brain import (
    SYSTEM_PROMPT,
    ScriptedBrain,
    _parse_response,
    build_surroundings,
    build_tick_context,
)
from godmode.resources import DEFAULT_CONFIGS, ResourceTile, ResourceType
from godmode.world import World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(x: int = 2, y: int = 2, satiation: float = 100.0, health: float = 100.0) -> Agent:
    return Agent(name="ada", birth_tick=0, x=x, y=y, brain=ScriptedBrain([]), satiation=satiation, health=health)


def make_world(width: int = 5, height: int = 5) -> World:
    return World(width=width, height=height, resource_density=0.0, seed=0)


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT
# ---------------------------------------------------------------------------

def test_system_prompt_non_empty():
    assert len(SYSTEM_PROMPT) > 0


def test_system_prompt_contains_actions():
    assert "harvest" in SYSTEM_PROMPT
    assert "move" in SYSTEM_PROMPT
    assert "eat" in SYSTEM_PROMPT
    assert "rest" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# build_tick_context
# ---------------------------------------------------------------------------

def test_build_tick_context_contains_vitals():
    world = make_world()
    agent = make_agent(satiation=55.5, health=80.0)
    ctx = build_tick_context(agent, world)
    assert "55.5" in ctx
    assert "80.0" in ctx


def test_build_tick_context_contains_position():
    world = make_world()
    agent = make_agent(x=2, y=3)
    ctx = build_tick_context(agent, world)
    assert "(2,3)" in ctx


def test_build_tick_context_contains_time():
    world = make_world()
    agent = make_agent()
    ctx = build_tick_context(agent, world)
    assert "Y1" in ctx


def test_build_tick_context_contains_tips_when_hungry():
    world = make_world()
    agent = make_agent(satiation=20.0)
    ctx = build_tick_context(agent, world)
    assert "How you feel" in ctx


def test_build_tick_context_no_tips_when_healthy():
    world = make_world()
    agent = make_agent(satiation=100.0, health=100.0)
    ctx = build_tick_context(agent, world)
    assert "How you feel" not in ctx


def test_build_tick_context_contains_surroundings():
    world = make_world()
    agent = make_agent()
    ctx = build_tick_context(agent, world)
    assert "Surroundings" in ctx


def test_build_tick_context_shows_last_actions():
    from godmode.agent import ActionResult
    world = make_world()
    agent = make_agent()
    agent.last_actions.append(ActionResult("move", "moved to (3,2)"))
    ctx = build_tick_context(agent, world)
    assert "Last actions" in ctx
    assert "moved to (3,2)" in ctx


def test_build_tick_context_tile_description_empty():
    world = make_world()
    agent = make_agent(x=2, y=2)
    ctx = build_tick_context(agent, world)
    assert "empty" in ctx


def test_build_tick_context_tile_description_food():
    world = make_world()
    cfg = DEFAULT_CONFIGS[ResourceType.FOOD]
    world.grid[2][2] = ResourceTile(resource_type=ResourceType.FOOD, config=cfg, amount=7.0)
    agent = make_agent(x=2, y=2)
    ctx = build_tick_context(agent, world)
    assert "Food" in ctx or "food" in ctx


# ---------------------------------------------------------------------------
# build_surroundings
# ---------------------------------------------------------------------------

def test_build_surroundings_contains_agent_marker():
    world = make_world()
    agent = make_agent(x=2, y=2)
    s = build_surroundings(agent, world)
    assert "[" in s  # agent cell shows [ ] or [F5] etc.


def test_build_surroundings_center_all_empty():
    world = make_world()
    agent = make_agent(x=2, y=2)
    world.reveal_around(2, 2, radius=1, agent=agent)  # reveal so tiles show . not ?
    s = build_surroundings(agent, world)
    assert "." in s


def test_build_surroundings_edge_shows_out_of_bounds():
    world = make_world()
    agent = make_agent(x=0, y=0)
    s = build_surroundings(agent, world)
    assert "~" in s


def test_build_surroundings_shows_food_tile():
    world = make_world()
    agent = make_agent(x=2, y=2)
    cfg = DEFAULT_CONFIGS[ResourceType.FOOD]
    world.grid[2][3] = ResourceTile(resource_type=ResourceType.FOOD, config=cfg, amount=9.0)
    world._revealed.add((3, 2))   # generate the tile
    agent.revealed.add((3, 2))    # mark visible to this agent
    s = build_surroundings(agent, world)
    assert "F9" in s


def test_build_surroundings_corner_agent():
    world = make_world()
    agent = make_agent(x=0, y=0)
    s = build_surroundings(agent, world)
    assert "[" in s   # agent marker present
    assert "~" in s   # out-of-bounds marker present


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, content: str):
        self.content = content

class FakeResponse:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


def test_parse_response_valid():
    resp = FakeResponse(json.dumps({"action": "move", "dx": 1, "dy": 0}))
    result = _parse_response(resp)
    assert result["action"] == "move"


def test_parse_response_invalid_json_falls_back():
    resp = FakeResponse("not json at all")
    result = _parse_response(resp)
    assert result == {"action": "rest"}


def test_parse_response_missing_action_falls_back():
    resp = FakeResponse(json.dumps({"dx": 1, "dy": 0}))
    result = _parse_response(resp)
    assert result == {"action": "rest"}


def test_parse_response_preserves_extra_keys():
    resp = FakeResponse(json.dumps({"action": "eat", "amount": 3.0}))
    result = _parse_response(resp)
    assert result["amount"] == pytest.approx(3.0)


def test_parse_response_strips_markdown_fence():
    content = '```json\n{"action": "move", "dx": 1, "dy": 0}\n```'
    resp = FakeResponse(content)
    result = _parse_response(resp)
    assert result["action"] == "move"


def test_parse_response_strips_fence_without_language_tag():
    content = '```\n{"action": "harvest"}\n```'
    resp = FakeResponse(content)
    result = _parse_response(resp)
    assert result["action"] == "harvest"


# ---------------------------------------------------------------------------
# ScriptedBrain
# ---------------------------------------------------------------------------

def test_scripted_brain_sequences_actions():
    world = make_world()
    agent = make_agent()
    brain = ScriptedBrain([{"action": "move", "dx": 1, "dy": 0}, {"action": "harvest"}])
    assert brain.decide(agent, world) == {"action": "move", "dx": 1, "dy": 0}
    assert brain.decide(agent, world) == {"action": "harvest"}


def test_scripted_brain_rests_after_exhausted():
    world = make_world()
    agent = make_agent()
    brain = ScriptedBrain([{"action": "rest"}])
    brain.decide(agent, world)  # consume the one action
    result = brain.decide(agent, world)
    assert result == {"action": "rest"}


# ---------------------------------------------------------------------------
# Warmth in vitals / memory / burn in SYSTEM_PROMPT
# ---------------------------------------------------------------------------

def test_system_prompt_contains_burn():
    assert "burn" in SYSTEM_PROMPT


def test_system_prompt_contains_warmth():
    assert "warmth" in SYSTEM_PROMPT.lower()


def test_build_tick_context_contains_warmth():
    world = make_world()
    agent = make_agent()
    ctx = build_tick_context(agent, world)
    assert "Warmth" in ctx


def test_build_tick_context_contains_memory():
    world = make_world()
    agent = make_agent(x=2, y=2)
    agent.memory.record("harvested 3.0 food near the river", tick=1, importance=4.0)
    ctx = build_tick_context(agent, world)
    assert "Memories" in ctx
    assert "harvested" in ctx


def test_build_tick_context_no_memory_when_empty():
    world = make_world()
    agent = make_agent()
    ctx = build_tick_context(agent, world)
    assert "Memories" not in ctx


def test_build_surroundings_unrevealed_shows_question_mark():
    world = make_world()
    agent = make_agent(x=2, y=2)
    # Don't reveal any tiles — all neighbours should be ?
    s = build_surroundings(agent, world)
    assert "?" in s
