from __future__ import annotations

import json
import logging
from typing import Optional

from godmode.agent import ActionResult, Agent, Relationship
from godmode.brain import OllamaBrain
from godmode.memory import Memory, MemoryStore
from godmode.resources import DEFAULT_CONFIGS, ResourceTile, ResourceType
from godmode.world import World

log = logging.getLogger(__name__)

SAVE_PATH = "godmode.save"
SAVE_VERSION = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_state(path: str, world: World, total_spawned: int, last_spawn_tick: int) -> None:
    """Persist full simulation state to *path* as JSON."""
    data = {
        "version": SAVE_VERSION,
        "total_spawned": total_spawned,
        "last_spawn_tick": last_spawn_tick,
        "world": _serialize_world(world),
        "agents": [_serialize_agent(a) for a in world.agents],
    }
    with open(path, "w") as fh:
        json.dump(data, fh)
    log.debug("state saved to %s (tick=%d)", path, world.tick_count)


def load_state(path: str) -> tuple[World, int, int]:
    """Load state from *path*.  Returns (world, total_spawned, last_spawn_tick).
    Agents are already attached to world.agents."""
    with open(path) as fh:
        data = json.load(fh)
    if data.get("version") != SAVE_VERSION:
        raise ValueError(f"Save version {data.get('version')} != expected {SAVE_VERSION}")
    world = _deserialize_world(data["world"])
    for raw_agent in data["agents"]:
        world.agents.append(_deserialize_agent(raw_agent))
    log.info("state loaded from %s (tick=%d, agents=%d)", path, world.tick_count, len(world.agents))
    return world, data["total_spawned"], data["last_spawn_tick"]


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

def _serialize_world(world: World) -> dict:
    rng = world._rng.getstate()   # (int, tuple[int, ...], float|None)
    return {
        "width": world.width,
        "height": world.height,
        "resource_density": world.resource_density,
        "tick_count": world.tick_count,
        "rng_state": [rng[0], list(rng[1]), rng[2]],
        "revealed": [list(pos) for pos in world._revealed],
        "grid": [
            [_serialize_tile(cell) for cell in row]
            for row in world.grid
        ],
    }


def _deserialize_world(data: dict) -> World:
    world = World(
        width=data["width"],
        height=data["height"],
        resource_density=data["resource_density"],
        seed=None,  # RNG state restored directly below
    )
    world.tick_count = data["tick_count"]
    rng_raw = data["rng_state"]
    world._rng.setstate((rng_raw[0], tuple(rng_raw[1]), rng_raw[2]))
    world._revealed = {tuple(pos) for pos in data["revealed"]}
    world.grid = [
        [_deserialize_tile(cell) for cell in row]
        for row in data["grid"]
    ]
    return world


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------

def _serialize_tile(tile: Optional[ResourceTile]) -> Optional[dict]:
    if tile is None:
        return None
    return {
        "type": tile.resource_type.value,
        "amount": tile.amount,
        "depleted_ticks_remaining": tile.depleted_ticks_remaining,
    }


def _deserialize_tile(data: Optional[dict]) -> Optional[ResourceTile]:
    if data is None:
        return None
    rtype = ResourceType(data["type"])
    return ResourceTile(
        resource_type=rtype,
        config=DEFAULT_CONFIGS[rtype],
        amount=data["amount"],
        depleted_ticks_remaining=data["depleted_ticks_remaining"],
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def _serialize_agent(agent: Agent) -> dict:
    brain_history: list = []
    if isinstance(agent.brain, OllamaBrain):
        brain_history = list(agent.brain._history)
    return {
        "name": agent.name,
        "birth_tick": agent.birth_tick,
        "x": agent.x,
        "y": agent.y,
        "satiation": agent.satiation,
        "health": agent.health,
        "warmth": agent.warmth,
        "inventory": {k.value: v for k, v in agent.inventory.items()},
        "last_actions": [{"action": r.action, "detail": r.detail} for r in agent.last_actions],
        "visited_tiles": list(agent.visited_tiles),
        "revealed": [list(pos) for pos in agent.revealed],
        "relationships": {
            name: {
                "score": r.score,
                "count": r.count,
                "note": r.note,
                "last_tick": r.last_tick,
            }
            for name, r in agent.relationships.items()
        },
        "brain_history": brain_history,
        "memory": _serialize_memory_store(agent.memory),
    }


def _deserialize_agent(data: dict) -> Agent:
    brain = OllamaBrain()
    brain._history = data.get("brain_history", [])
    agent = Agent(
        name=data["name"],
        birth_tick=data["birth_tick"],
        x=data["x"],
        y=data["y"],
        brain=brain,
        satiation=data["satiation"],
        health=data["health"],
        warmth=data["warmth"],
    )
    agent.inventory = {ResourceType(k): v for k, v in data["inventory"].items()}
    agent.last_actions = [
        ActionResult(r["action"], r["detail"]) for r in data["last_actions"]
    ]
    agent.visited_tiles = list(data["visited_tiles"])
    agent.relationships = {
        name: Relationship(
            score=r["score"],
            count=r["count"],
            note=r["note"],
            last_tick=r["last_tick"],
        )
        for name, r in data["relationships"].items()
    }
    if "memory" in data:
        agent.memory = _deserialize_memory_store(data["memory"])
    agent.revealed = {tuple(pos) for pos in data.get("revealed", [])}
    return agent


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _serialize_memory_store(store: MemoryStore) -> dict:
    return {
        "vst": [_serialize_memory(m) for m in store.vst],
        "st":  [_serialize_memory(m) for m in store.st],
        "lt":  [_serialize_memory(m) for m in store.lt],
    }


def _serialize_memory(m: Memory) -> dict:
    return {
        "content": m.content,
        "tick": m.tick,
        "last_access": m.last_access,
        "importance": m.importance,
        "tier": m.tier,
    }


def _deserialize_memory_store(data: dict) -> MemoryStore:
    store = MemoryStore()
    store.vst = [_deserialize_memory(m) for m in data.get("vst", [])]
    store.st  = [_deserialize_memory(m) for m in data.get("st",  [])]
    store.lt  = [_deserialize_memory(m) for m in data.get("lt",  [])]
    return store


def _deserialize_memory(data: dict) -> Memory:
    return Memory(
        content=data["content"],
        tick=data["tick"],
        last_access=data["last_access"],
        importance=data["importance"],
        tier=data["tier"],
    )
