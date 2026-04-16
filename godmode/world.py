from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

from godmode.agent import Agent, update_relationship
from godmode.resources import DEFAULT_CONFIGS, ResourceConfig, ResourceTile, ResourceType
from godmode.time import WorldTime

if TYPE_CHECKING:
    from godmode.agent import Agent


# ---------------------------------------------------------------------------
# Trade helpers
# ---------------------------------------------------------------------------

def _clean_trade_items(raw: list, agent: Agent | None, resource_configs: dict) -> list[dict]:
    """Validate and normalise a list of trade items.

    Each item must be a dict with 'resource' (str matching a ResourceType value)
    and 'qty' (positive float). Items with zero/negative qty, unknown resource types,
    or non-tradeable resources are silently dropped. If *agent* is given, qty is also
    capped at the agent's current inventory so they can never promise more than they hold.
    Returns cleaned items with an internal '_rtype' key for execution use.
    """
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        resource_str = str(item.get("resource", "")).lower().strip()
        try:
            qty = float(item.get("qty", 0.0))
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        try:
            rtype = ResourceType(resource_str)
        except ValueError:
            log.warning("_clean_trade_items: unknown resource %r — skipping", resource_str)
            continue
        config = resource_configs.get(rtype)
        if config is None or not config.tradeable:
            log.warning("_clean_trade_items: resource %r is not tradeable — skipping", resource_str)
            continue
        if agent is not None:
            available = agent.inventory.get(rtype, 0.0)
            qty = min(qty, available)
            if qty <= 0:
                continue
        result.append({"resource": resource_str, "qty": qty, "_rtype": rtype})
    return result


def _can_fulfill(agent: Agent, items: list[dict]) -> bool:
    """Return True if *agent* currently holds enough of each resource to give away *items*."""
    needed: dict[ResourceType, float] = {}
    for item in items:
        rtype: ResourceType = item["_rtype"]
        needed[rtype] = needed.get(rtype, 0.0) + item["qty"]
    for rtype, qty in needed.items():
        if agent.inventory.get(rtype, 0.0) < qty - 1e-9:
            return False
    return True


def _execute_trade(giver_a: Agent, a_gives: list[dict], giver_b: Agent, b_gives: list[dict]) -> None:
    """Transfer resources: a_gives moves from giver_a to giver_b; b_gives moves from giver_b to giver_a."""
    for item in a_gives:
        rtype: ResourceType = item["_rtype"]
        qty = item["qty"]
        giver_a.inventory[rtype] = max(0.0, giver_a.inventory.get(rtype, 0.0) - qty)
        giver_b.inventory[rtype] = giver_b.inventory.get(rtype, 0.0) + qty
    for item in b_gives:
        rtype: ResourceType = item["_rtype"]
        qty = item["qty"]
        giver_b.inventory[rtype] = max(0.0, giver_b.inventory.get(rtype, 0.0) - qty)
        giver_a.inventory[rtype] = giver_a.inventory.get(rtype, 0.0) + qty


def _fmt_items(items: list[dict]) -> str:
    """Compact string representation of trade items for logging."""
    return ", ".join(f"{i['qty']:.1f} {i['resource']}" for i in items) or "nothing"


class World:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        resource_configs: Optional[dict] = None,
        resource_density: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.resource_configs: dict[ResourceType, ResourceConfig] = (
            resource_configs if resource_configs is not None else dict(DEFAULT_CONFIGS)
        )
        self.resource_density = resource_density
        self.tick_count = 0
        self._rng = random.Random(seed)
        self._revealed: Set[Tuple[int, int]] = set()
        self.grid: List[List[Optional[ResourceTile]]] = self._generate()
        self.agents: List[Agent] = []
        self._last_interactions: List[dict] = []

    def _generate(self) -> List[List[Optional[ResourceTile]]]:
        """Create an empty grid. Tiles are generated lazily via reveal_around()."""
        return [[None] * self.width for _ in range(self.height)]

    def is_revealed(self, x: int, y: int) -> bool:
        return (x, y) in self._revealed

    def reveal_around(self, x: int, y: int, radius: int = 1, agent=None) -> None:
        """Generate tiles within radius of (x, y). If agent is given, also marks
        those tiles as visible to that agent (per-agent fog of war)."""
        resource_types = list(self.resource_configs.keys())
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                if (nx, ny) not in self._revealed:
                    self._revealed.add((nx, ny))
                    if self._rng.random() < self.resource_density:
                        rtype = self._rng.choice(resource_types)
                        config = self.resource_configs[rtype]
                        self.grid[ny][nx] = ResourceTile(
                            resource_type=rtype,
                            config=config,
                            amount=config.starting_value,
                        )
                        log.debug("generated (%d,%d): %s", nx, ny, rtype.value)
                    else:
                        log.debug("generated (%d,%d): empty", nx, ny)
                if agent is not None:
                    agent.revealed.add((nx, ny))

    def add_agent(self, agent: Agent) -> None:
        if agent.x < 0 or agent.x >= self.width or agent.y < 0 or agent.y >= self.height:
            raise ValueError(
                f"Agent position ({agent.x},{agent.y}) is out of world bounds "
                f"({self.width}x{self.height})"
            )
        self.agents.append(agent)
        self.reveal_around(agent.x, agent.y, agent=agent)
        log.info("agent '%s' added at (%d,%d)", agent.name, agent.x, agent.y)

    def tick(self) -> None:
        log.debug("tick %d starting", self.tick_count)
        for row in self.grid:
            for tile in row:
                if tile is not None:
                    tile.tick()

        alive_agents = [a for a in self.agents if a.alive]

        # Phase 1: all agents decide in parallel — each LLM call is independent.
        # Keyed by id(agent) since Agent is a mutable dataclass (not hashable).
        decisions: dict[int, dict] = {}
        if alive_agents:
            with ThreadPoolExecutor(max_workers=len(alive_agents)) as executor:
                futures = {executor.submit(a.decide, self): a for a in alive_agents}
                for future in as_completed(futures):
                    agent = futures[future]
                    try:
                        decisions[id(agent)] = future.result()
                    except Exception as exc:
                        log.warning("[%s] decide() raised %s — defaulting to rest", agent.name, exc)
                        decisions[id(agent)] = {"action": "rest"}

        # Phase 1.5: resolve interact decisions before applying anything
        self._last_interactions = []
        processed_pairs: set[frozenset] = set()
        for agent in alive_agents:
            decision = decisions.get(id(agent), {"action": "rest"})
            if decision.get("action") != "interact":
                continue
            target_name = decision.get("target", "")
            message = str(decision.get("message", "")).strip()
            if not message or not target_name:
                decisions[id(agent)] = {"action": "rest"}
                continue
            target = next((a for a in self.agents if a.alive and a.name == target_name), None)
            if target is None:
                log.info("[%s] tried to interact with unknown agent '%s'", agent.name, target_name)
                decisions[id(agent)] = {"action": "rest"}
                continue
            if abs(agent.x - target.x) > 1 or abs(agent.y - target.y) > 1:
                log.info("[%s] tried to interact with %s but too far away", agent.name, target.name)
                decisions[id(agent)] = {"action": "rest"}
                continue
            pair: frozenset = frozenset({id(agent), id(target)})
            if pair in processed_pairs:
                decisions[id(agent)] = {"action": "rest"}
                continue
            processed_pairs.add(pair)

            # Ask target to respond — separate LLM call outside the parallel phase
            reply_dict = target.brain.respond_to_message(target, self, agent.name, message)

            if reply_dict.get("action") == "trade" and reply_dict.get("target") == agent.name:
                # Target wants to trade in response to the message — handle inline
                t_give = _clean_trade_items(reply_dict.get("give", []), target, self.resource_configs)
                t_take = _clean_trade_items(reply_dict.get("take", []), None, self.resource_configs)

                if t_give or t_take:
                    a_response = agent.brain.respond_to_trade(agent, self, target.name, t_give, t_take, can_counter=False)
                    if a_response.get("action") == "trade_accept":
                        if _can_fulfill(target, t_give) and _can_fulfill(agent, t_take):
                            _execute_trade(target, t_give, agent, t_take)
                            outcome = "accepted"
                            log.info("[%s↔%s] trade (via message) accepted: %s gives %s; %s gives %s",
                                     target.name, agent.name, target.name, _fmt_items(t_give),
                                     agent.name, _fmt_items(t_take))
                        else:
                            outcome = "failed (inventory changed)"
                            log.info("[%s↔%s] trade (via message) accepted but inventory changed — cancelled",
                                     target.name, agent.name)
                    else:
                        outcome = "rejected"
                        log.info("[%s↔%s] trade (via message) rejected by %s",
                                 target.name, agent.name, agent.name)

                    trade_desc = f"[trade via message] give={_fmt_items(t_give)} take={_fmt_items(t_take)} — {outcome}"
                    i_score, i_note, i_memory = agent.brain.score_interaction(agent, self, target.name, trade_desc, None)
                    t_score, t_note, t_memory = target.brain.score_interaction(target, self, agent.name, trade_desc, None)

                    update_relationship(agent, target.name, i_score, i_note, self.tick_count)
                    update_relationship(target, agent.name, t_score, t_note, self.tick_count)

                    self._last_interactions.append({
                        "type": "trade",
                        "initiator": target.name, "target": agent.name,
                        "give": [{"resource": i["resource"], "qty": i["qty"]} for i in t_give],
                        "take": [{"resource": i["resource"], "qty": i["qty"]} for i in t_take],
                        "outcome": outcome,
                        "initiator_score": t_score, "target_score": i_score,
                    })

                    if i_memory:
                        agent.memory.record(i_memory, self.tick_count, 8.0)
                    if t_memory:
                        target.memory.record(t_memory, self.tick_count, 8.0)

                    decisions[id(agent)] = {"action": "rest"}
                    continue

                # Empty trade items — fall through and treat as ignored message
                reply: str | None = None
            elif (
                reply_dict.get("action") == "interact"
                and reply_dict.get("target") == agent.name
                and reply_dict.get("message")
            ):
                reply = str(reply_dict["message"]).strip() or None
            else:
                reply = None

            # Score from both perspectives — each agent also authors their own memory
            i_score, i_note, i_memory = agent.brain.score_interaction(agent, self, target.name, message, reply)
            t_score, t_note, t_memory = target.brain.score_interaction(target, self, agent.name, message, reply)

            update_relationship(agent, target.name, i_score, i_note, self.tick_count)
            update_relationship(target, agent.name, t_score, t_note, self.tick_count)

            self._last_interactions.append({
                "type": "message",
                "initiator": agent.name, "target": target.name,
                "message": message, "reply": reply,
                "initiator_score": i_score, "target_score": t_score,
            })

            if reply:
                log.info(
                    "[%s→%s]: \"%s\" | [%s→%s]: \"%s\" | scores: %.2f / %.2f",
                    agent.name, target.name, message, target.name, agent.name, reply, i_score, t_score,
                )
            else:
                log.info(
                    "[%s→%s]: \"%s\" | %s ignored | score: %.2f",
                    agent.name, target.name, message, target.name, i_score,
                )

            # Store agent-authored memories — only if the agent chose to record something
            if i_memory:
                agent.memory.record(i_memory, self.tick_count, 8.0)
            if t_memory:
                target.memory.record(t_memory, self.tick_count, 8.0)

            decisions[id(agent)] = {"action": "rest"}

        for agent in alive_agents:
            decision = decisions.get(id(agent), {"action": "rest"})
            if decision.get("action") != "trade":
                continue
            target_name = decision.get("target", "")
            target = next((a for a in self.agents if a.alive and a.name == target_name), None)
            if target is None:
                log.info("[%s] tried to trade with unknown agent '%s'", agent.name, target_name)
                decisions[id(agent)] = {"action": "rest"}
                continue
            if abs(agent.x - target.x) > 1 or abs(agent.y - target.y) > 1:
                log.info("[%s] tried to trade with %s but too far away", agent.name, target.name)
                decisions[id(agent)] = {"action": "rest"}
                continue
            pair: frozenset = frozenset({id(agent), id(target)})
            if pair in processed_pairs:
                decisions[id(agent)] = {"action": "rest"}
                continue
            processed_pairs.add(pair)

            # Validate and cap initiator's offer against their inventory
            give = _clean_trade_items(decision.get("give", []), agent, self.resource_configs)
            take = _clean_trade_items(decision.get("take", []), None, self.resource_configs)
            if not give and not take:
                log.info("[%s] trade offer to %s is empty after validation — skipping", agent.name, target.name)
                decisions[id(agent)] = {"action": "rest"}
                continue

            # Ask target to respond
            t_response = target.brain.respond_to_trade(target, self, agent.name, give, take, can_counter=True)
            t_action = t_response.get("action")

            outcome: str
            if t_action == "trade_accept":
                if _can_fulfill(agent, give) and _can_fulfill(target, take):
                    _execute_trade(agent, give, target, take)
                    outcome = "accepted"
                    log.info("[%s↔%s] trade accepted: %s gives %s; %s gives %s",
                             agent.name, target.name, agent.name, _fmt_items(give),
                             target.name, _fmt_items(take))
                else:
                    outcome = "failed (inventory changed)"
                    log.info("[%s↔%s] trade accepted but inventory changed — cancelled", agent.name, target.name)

            elif t_action == "trade_counter":
                c_give = _clean_trade_items(t_response.get("give", []), target, self.resource_configs)
                c_take = _clean_trade_items(t_response.get("take", []), None, self.resource_configs)
                if (not c_give and not c_take) or not _can_fulfill(target, c_give):
                    outcome = "counter invalid — treated as reject"
                    log.info("[%s↔%s] counter-offer invalid", agent.name, target.name)
                else:
                    # Ask initiator to accept or reject the counter (no further counters)
                    a_response = agent.brain.respond_to_trade(
                        agent, self, target.name, c_give, c_take, can_counter=False
                    )
                    if a_response.get("action") == "trade_accept":
                        if _can_fulfill(target, c_give) and _can_fulfill(agent, c_take):
                            _execute_trade(target, c_give, agent, c_take)
                            outcome = "counter accepted"
                            log.info("[%s↔%s] counter accepted: %s gives %s; %s gives %s",
                                     agent.name, target.name, target.name, _fmt_items(c_give),
                                     agent.name, _fmt_items(c_take))
                        else:
                            outcome = "counter failed (inventory changed)"
                            log.info("[%s↔%s] counter accepted but inventory changed — cancelled", agent.name, target.name)
                    else:
                        outcome = "counter rejected"
                        log.info("[%s↔%s] counter-offer rejected by %s", agent.name, target.name, agent.name)

            else:
                outcome = "rejected"
                log.info("[%s↔%s] trade rejected by %s", agent.name, target.name, target.name)

            # Score from both perspectives using the same mechanism as messaging
            trade_desc = f"[trade] offered give={_fmt_items(give)} take={_fmt_items(take)} — {outcome}"
            i_score, i_note, i_memory = agent.brain.score_interaction(agent, self, target.name, trade_desc, None)
            t_score, t_note, t_memory = target.brain.score_interaction(target, self, agent.name, trade_desc, None)

            update_relationship(agent, target.name, i_score, i_note, self.tick_count)
            update_relationship(target, agent.name, t_score, t_note, self.tick_count)

            self._last_interactions.append({
                "type": "trade",
                "initiator": agent.name, "target": target.name,
                "give": [{"resource": i["resource"], "qty": i["qty"]} for i in give],
                "take": [{"resource": i["resource"], "qty": i["qty"]} for i in take],
                "outcome": outcome,
                "initiator_score": i_score, "target_score": t_score,
            })

            if i_memory:
                agent.memory.record(i_memory, self.tick_count, 8.0)
            if t_memory:
                target.memory.record(t_memory, self.tick_count, 8.0)

            decisions[id(agent)] = {"action": "rest"}

        # Phase 2: apply decisions sequentially against the shared world state
        for agent in self.agents:
            if agent.alive:
                agent.apply(self, decisions.get(id(agent), {"action": "rest"}))
                if agent.alive:
                    self.reveal_around(agent.x, agent.y, agent=agent)

        self.tick_count += 1

        # Memory compression — end of day (VST→ST) and end of week (ST→LT)
        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.memory.needs_week_compression(self.tick_count):
                # End of week: compress today's VST first, then the whole week's ST
                if agent.memory.needs_day_compression(self.tick_count):
                    summaries = agent.brain.compress_memories(agent, agent.memory.vst, "day")
                    agent.memory.promote_vst_to_st(summaries)
                summaries = agent.brain.compress_memories(agent, agent.memory.st, "week")
                agent.memory.promote_st_to_lt(summaries)
            elif agent.memory.needs_day_compression(self.tick_count):
                summaries = agent.brain.compress_memories(agent, agent.memory.vst, "day")
                agent.memory.promote_vst_to_st(summaries)

    @property
    def time(self) -> WorldTime:
        return WorldTime(self.tick_count)

    def get_tile(self, x: int, y: int) -> Optional[ResourceTile]:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None
        return self.grid[y][x]

    def harvest(self, x: int, y: int, amount: float) -> float:
        tile = self.get_tile(x, y)
        if tile is None:
            return 0.0
        return tile.harvest(amount)
