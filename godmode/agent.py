from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from godmode.memory import MemoryStore, score_importance
from godmode.resources import ResourceType

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from godmode.world import World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVE_SATIATION_DRAIN: float = 2.0
REST_SATIATION_DRAIN: float = 0.5
MOVE_SATIATION_EXTRA: float = 1.0
HARVEST_SATIATION_EXTRA: float = 2.0
EAT_RESTORE_PER_UNIT: float = 8.0
STARVATION_HEALTH_DRAIN: float = 5.0
CARRY_LIMIT: float = 20.0

ACTIVE_WARMTH_DRAIN: float = 1.5
REST_WARMTH_DRAIN: float = 0.5
MOVE_WARMTH_EXTRA: float = 0.5
BURN_RESTORE_PER_UNIT: float = 10.0
FREEZING_HEALTH_DRAIN: float = 3.0

# ---------------------------------------------------------------------------
# Brain protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Brain(Protocol):
    def decide(self, agent: Agent, world: World) -> dict: ...
    def respond_to_message(self, agent: Agent, world: World, sender_name: str, message: str) -> dict: ...
    def respond_to_trade(self, agent: Agent, world: World, sender_name: str, offer_give: list[dict], offer_take: list[dict], can_counter: bool) -> dict: ...
    def score_interaction(self, agent: Agent, world: World, other_name: str, sent: str, received: str | None) -> tuple[float, str, str]: ...
    def compress_memories(self, agent: Agent, memories: list, compression_type: str) -> list: ...

# ---------------------------------------------------------------------------
# Action result
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    action: str    # "move"|"harvest"|"eat"|"burn"|"rest"|"interact"|"invalid"|"dead"
    detail: str    # human-readable summary

# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

@dataclass
class Relationship:
    score: float = 0.0    # -1.0 (hostile) to 1.0 (friendly)
    count: int = 0
    note: str = ""        # short phrase from last scoring
    last_tick: int = 0

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    name: str
    birth_tick: int
    x: int
    y: int
    brain: Brain
    satiation: float = 100.0
    health: float = 100.0
    warmth: float = 100.0
    inventory: dict[ResourceType, float] = field(default_factory=dict)
    last_actions: list[ActionResult] = field(default_factory=list)
    visited_tiles: list[dict] = field(default_factory=list)
    relationships: dict[str, Relationship] = field(default_factory=dict)
    memory: MemoryStore = field(default_factory=MemoryStore)
    revealed: set = field(default_factory=set)  # tiles this agent has personally explored
    _max_last_actions: int = field(default=3, repr=False)
    _max_visited_tiles: int = field(default=5, repr=False)

    @property
    def alive(self) -> bool:
        return self.health > 0

    @property
    def carry_total(self) -> float:
        return sum(self.inventory.values())

    def decide(self, world: World) -> dict:
        """Ask the brain for an action. Safe to call in parallel across agents."""
        if not self.alive:
            return {"action": "dead"}
        return self.brain.decide(self, world)

    def apply(self, world: World, raw: dict) -> ActionResult:
        """Execute a pre-decided action and apply all stat changes."""
        if not self.alive:
            return ActionResult("dead", "agent is dead")
        result = execute_action(self, world, raw)
        _apply_satiation_drain(self, result.action)
        _apply_starvation(self)
        _apply_warmth_drain(self, result.action)
        _apply_freezing(self)
        log.info(
            "[%s] %s: %s | satiation=%.1f warmth=%.1f health=%.1f",
            self.name, result.action, result.detail, self.satiation, self.warmth, self.health,
        )
        if self.health <= 0:
            log.warning("[%s] has died at tick %d", self.name, world.tick_count)

        # Record to memory — interactions are recorded by World (full exchange context)
        if result.action != "interact":
            importance = score_importance(
                result.action, result.detail, self.satiation, self.warmth, self.health
            )
            # Only store what the LLM explicitly chose to remember
            llm_memory = str(raw.get("memory", "")).strip()[:100]
            if llm_memory:
                self.memory.record(llm_memory, world.tick_count, importance)
        # Record visited tile
        tile = world.get_tile(self.x, self.y)
        if tile is not None:
            tile_desc = f"{tile.resource_type.value.capitalize()} ({tile.resource_type.value[0].upper()}{tile.amount:.0f})"
        else:
            tile_desc = "empty"
        self.visited_tiles.append({"x": self.x, "y": self.y, "tile": tile_desc, "tick": world.tick_count})
        if len(self.visited_tiles) > self._max_visited_tiles:
            self.visited_tiles.pop(0)
        self.last_actions.append(result)
        if len(self.last_actions) > self._max_last_actions:
            self.last_actions.pop(0)
        return result

    def tick(self, world: World) -> ActionResult:
        """Convenience wrapper: decide + apply in one call."""
        return self.apply(world, self.decide(world))

# ---------------------------------------------------------------------------
# Relationship helpers
# ---------------------------------------------------------------------------

def update_relationship(agent: Agent, other_name: str, score: float, note: str, tick: int) -> None:
    """Create or update a relationship using a weighted moving average (70/30)."""
    rel = agent.relationships.get(other_name, Relationship())
    if rel.count == 0:
        rel.score = score
    else:
        rel.score = 0.7 * rel.score + 0.3 * score
    rel.score = max(-1.0, min(1.0, rel.score))
    rel.count += 1
    rel.note = note
    rel.last_tick = tick
    agent.relationships[other_name] = rel

# ---------------------------------------------------------------------------
# Action execution (pure functions)
# ---------------------------------------------------------------------------

def execute_action(agent: Agent, world: World, raw: dict) -> ActionResult:
    action = raw.get("action", "")
    if action == "move":
        dx = int(raw.get("dx", 0))
        dy = int(raw.get("dy", 0))
        return _do_move(agent, world, dx, dy)
    elif action == "harvest":
        return _do_harvest(agent, world)
    elif action == "eat":
        amount = float(raw.get("amount", 0.0))
        return _do_eat(agent, amount)
    elif action == "burn":
        amount = float(raw.get("amount", 0.0))
        return _do_burn(agent, amount)
    elif action == "rest":
        return _do_rest(agent)
    else:
        return ActionResult("invalid", f"unknown action '{action}' — resting instead")


def _do_move(agent: Agent, world: World, dx: int, dy: int) -> ActionResult:
    dx = max(-1, min(1, dx))
    dy = max(-1, min(1, dy))
    new_x = agent.x + dx
    new_y = agent.y + dy
    if new_x < 0 or new_x >= world.width or new_y < 0 or new_y >= world.height:
        return ActionResult("move", f"blocked at boundary, stayed at ({agent.x},{agent.y})")
    agent.x = new_x
    agent.y = new_y
    return ActionResult("move", f"moved to ({agent.x},{agent.y})")


def _do_harvest(agent: Agent, world: World) -> ActionResult:
    tile = world.get_tile(agent.x, agent.y)
    if tile is None:
        return ActionResult("harvest", "nothing to harvest here")
    remaining_carry = CARRY_LIMIT - agent.carry_total
    if remaining_carry <= 0:
        return ActionResult("harvest", "carrying too much to harvest")
    yield_cap = tile.config.harvest_yield if tile.config.harvest_yield > 0 else float("inf")
    amount = min(yield_cap, tile.amount, remaining_carry)
    got = world.harvest(agent.x, agent.y, amount)
    if got > 0:
        agent.inventory[tile.resource_type] = agent.inventory.get(tile.resource_type, 0.0) + got
    return ActionResult("harvest", f"harvested {got:.1f} {tile.resource_type.value}")


def _do_eat(agent: Agent, amount: float) -> ActionResult:
    if amount <= 0:
        return ActionResult("eat", "ate nothing")
    available = agent.inventory.get(ResourceType.FOOD, 0.0)
    consumed = min(amount, available)
    if consumed <= 0:
        return ActionResult("eat", "no food to eat")
    agent.inventory[ResourceType.FOOD] = available - consumed
    restored = consumed * EAT_RESTORE_PER_UNIT
    agent.satiation = min(100.0, agent.satiation + restored)
    return ActionResult("eat", f"ate {consumed:.1f} food, satiation now {agent.satiation:.1f}")


def _do_burn(agent: Agent, amount: float) -> ActionResult:
    if amount <= 0:
        return ActionResult("burn", "burned nothing")
    available = agent.inventory.get(ResourceType.WOOD, 0.0)
    consumed = min(amount, available)
    if consumed <= 0:
        return ActionResult("burn", "no wood to burn")
    agent.inventory[ResourceType.WOOD] = available - consumed
    restored = consumed * BURN_RESTORE_PER_UNIT
    agent.warmth = min(100.0, agent.warmth + restored)
    return ActionResult("burn", f"burned {consumed:.1f} wood, warmth now {agent.warmth:.1f}")


def _do_rest(_agent: Agent) -> ActionResult:
    return ActionResult("rest", "rested")

# ---------------------------------------------------------------------------
# Hunger / health mechanics
# ---------------------------------------------------------------------------

def _apply_satiation_drain(agent: Agent, action: str) -> None:
    if action == "rest":
        drain = REST_SATIATION_DRAIN
    elif action == "move":
        drain = ACTIVE_SATIATION_DRAIN + MOVE_SATIATION_EXTRA
    elif action == "harvest":
        drain = ACTIVE_SATIATION_DRAIN + HARVEST_SATIATION_EXTRA
    else:
        # eat, burn, invalid, dead — base cost only
        drain = ACTIVE_SATIATION_DRAIN
    agent.satiation = max(0.0, agent.satiation - drain)


def _apply_starvation(agent: Agent) -> None:
    if agent.satiation <= 0:
        agent.health = max(0.0, agent.health - STARVATION_HEALTH_DRAIN)

# ---------------------------------------------------------------------------
# Warmth / freezing mechanics
# ---------------------------------------------------------------------------

def _apply_warmth_drain(agent: Agent, action: str) -> None:
    if action == "rest":
        drain = REST_WARMTH_DRAIN
    elif action == "move":
        drain = ACTIVE_WARMTH_DRAIN + MOVE_WARMTH_EXTRA
    else:
        # harvest, eat, burn, invalid, dead — base cost only
        drain = ACTIVE_WARMTH_DRAIN
    agent.warmth = max(0.0, agent.warmth - drain)


def _apply_freezing(agent: Agent) -> None:
    if agent.warmth <= 0:
        agent.health = max(0.0, agent.health - FREEZING_HEALTH_DRAIN)

# ---------------------------------------------------------------------------
# Tips (embodied sensations)
# ---------------------------------------------------------------------------

def get_tips(agent: Agent, world: World | None = None) -> list[str]:
    tips: list[str] = []
    has_food = agent.inventory.get(ResourceType.FOOD, 0.0) > 0

    # Hunger tips
    if agent.satiation <= 0:
        tips.append("You are starving. Your body is consuming itself.")
    elif agent.satiation < 15 and not has_food:
        tips.append("You are very hungry and carrying nothing to eat. Find food soon.")
    elif agent.satiation < 15 and has_food:
        tips.append("You are very hungry. You have food — consider eating.")
    elif agent.satiation < 25:
        tips.append("You are hungry. It's becoming hard to ignore.")
    elif agent.satiation < 40:
        tips.append("You feel a faint hunger. Your body wants food.")

    # Cold tips
    if agent.warmth <= 0:
        tips.append("You are freezing to death. Your body is shutting down.")
    elif agent.warmth < 15:
        tips.append("You are freezing. Your limbs are going numb.")
    elif agent.warmth < 25:
        tips.append("You are shivering uncontrollably. You need warmth.")
    elif agent.warmth < 40:
        tips.append("You feel a chill. The cold is creeping in.")

    if agent.health < 50:
        tips.append("You feel weakened. The cold and hunger have left their mark.")

    # Carry tips
    if agent.carry_total >= CARRY_LIMIT:
        tips.append("Your hands are full. You cannot carry anything more — eat food or burn wood to free space.")
    elif agent.carry_total > 16:
        tips.append("You are carrying a heavy load. You won't be able to pick up much more.")

    # Boundary bouncing tip — escalate on consecutive blocks
    consecutive_blocks = 0
    for r in reversed(agent.last_actions):
        if "blocked at boundary" in r.detail:
            consecutive_blocks += 1
        else:
            break
    if consecutive_blocks >= 2:
        tips.append("You keep hitting the edge of the world. There is nothing that way — turn around and go somewhere completely different.")
    elif consecutive_blocks == 1:
        tips.append("There is nothing beyond here. Other directions may have more to offer.")

    # Tile tips (require world context)
    if world is not None:
        tile = world.get_tile(agent.x, agent.y)
        if tile is not None and tile.is_depleted:
            tips.append("This ground is exhausted. Harvesting here yields nothing — move on and let it recover.")
        elif tile is not None and 0 < tile.amount < tile.config.harvest_yield:
            tips.append("This ground is nearly exhausted. One more harvest may strip it bare and slow its recovery.")

    # Exploration nudge when hungry or cold and inventory is low
    if agent.satiation < 40 or agent.warmth < 40:
        tips.append("Exploring new areas may reveal resources you haven't found yet.")

    return tips
