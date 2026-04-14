from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import ollama

log = logging.getLogger(__name__)

from godmode.agent import Agent, Relationship, get_tips
from godmode.memory import Memory, MemoryStore
from godmode.resources import ResourceType

if TYPE_CHECKING:
    from godmode.world import World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "gemma4:e4b-it-q8_0"

OLLAMA_OPTIONS = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "num_ctx": 4096,
    "num_predict": 1024,
    "repeat_penalty": 1.2,
}

SYSTEM_PROMPT = """\
You are a being in a living world. You have no assigned purpose.

## Your body
You feel hunger. It starts as background noise and grows louder over time
until it becomes the only thing you can think about. Food quiets it.
If you go long enough without eating, you grow weaker.

You also feel warmth leaving your body over time. The cold is slow but relentless.
Burning wood keeps you warm. If you let it fall to zero, your body starts to fail.

## The world
Time passes in hours, days, months, years.
The land around you holds resources — food that grows back,
wood from trees that regrow slowly.
You can only see a short distance. The rest is unknown until you explore it.
Unexplored tiles appear as ?.

## What you can do (pick exactly one per tick, reply as JSON)
Always include a brief "reasoning" field first, then your chosen action.
{"reasoning": "I see food to my east, I should move there", "action": "move", "dx": 1, "dy": 0}
{"reasoning": "I am on a food tile and have carry space", "action": "harvest"}
{"reasoning": "I have food and satiation is low", "action": "eat", "amount": 3.0}
{"reasoning": "I have wood and warmth is dropping", "action": "burn", "amount": 2.0}
{"reasoning": "I am safe and well-fed, conserving energy", "action": "rest"}
{"reasoning": "Cal is nearby and might have food to trade", "action": "interact", "target": "Ada", "message": "want to share food?"}
{"reasoning": "Ada has wood and I have surplus food", "action": "trade", "target": "Ada", "give": [{"resource": "food", "qty": 3.0}], "take": [{"resource": "wood", "qty": 2.0}]}

dx and dy must each be -1, 0, or 1.
target must be the exact name of a nearby agent (within 1 tile).
For interact: "message" is the key for what you say — not "action", not "text", not "what".
For trade: give is what you offer; take is what you want in return.
  Either give or take may be [] (gift or request), but not both simultaneously.
  qty must be positive and cannot exceed what you currently carry.
  Tradeable resources: food, wood.
  The other agent may accept, reject, or make a counter-offer (once).

## Memory
You may add an optional "memory" field (max 100 chars) to record something worth
remembering — a resource location, a decision, a key event. Only include it when
something genuinely new or useful happened. Leave it out for routine actions.
Examples:
{"action": "move", "dx": 1, "dy": 0, "memory": "food at (5,2), heading there while hungry"}
{"action": "harvest", "memory": "wood cluster at (4,3), revisit when cold"}
{"action": "rest"}  ← no memory needed

## Social interactions
When you see nearby agents, you may send them a message.
They may reply or ignore you. After the exchange, you will each form an
impression of the other — positive, negative, or neutral. These feelings
persist and colour how you see each other in future encounters.
You can see your current relationships under "Relationships" if any exist.
Trust or distrust based on experience.\

## Each tick you will be told
Your position, what you can see, what you are carrying, and how you feel.
Your memory of recent tiles is also shown — use it to navigate.
What you do next is up to you.\
"""

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _ollama_chat(model: str, messages: list, format: str, options: dict, think: bool, max_retries: int = 3):
    """ollama.chat with exponential-backoff retry on transient 5xx errors."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return ollama.chat(model=model, messages=messages, format=format, options=options, think=think)
        except Exception as exc:
            if attempt < max_retries - 1:
                log.warning("ollama.chat failed (attempt %d/%d): %s — retrying in %.1fs", attempt + 1, max_retries, exc, delay)
                time.sleep(delay)
                delay *= 2
            else:
                raise

# ---------------------------------------------------------------------------
# Scripted brain (for tests and demos)
# ---------------------------------------------------------------------------

class ScriptedBrain:
    """Returns pre-scripted actions in sequence, then rests indefinitely."""

    def __init__(self, actions: list[dict]) -> None:
        self._actions = list(actions)
        self._index = 0

    def decide(self, agent: Agent, world: World) -> dict:
        if self._index >= len(self._actions):
            return {"action": "rest"}
        action = self._actions[self._index]
        self._index += 1
        return action

    def respond_to_message(self, agent: Agent, world: World, sender_name: str, message: str) -> dict:
        return {"action": "rest"}  # scripted brain ignores all incoming messages

    def respond_to_trade(self, agent: Agent, world: World, sender_name: str, offer_give: list[dict], offer_take: list[dict], can_counter: bool) -> dict:
        return {"action": "trade_reject"}  # scripted brain rejects all trade offers

    def score_interaction(self, agent: Agent, world: World, other_name: str, sent: str, received: str | None) -> tuple[float, str, str]:
        return 0.0, "", ""

    def compress_memories(self, agent: Agent, memories: list, compression_type: str) -> list:
        return []

# ---------------------------------------------------------------------------
# Ollama brain (real LLM)
# ---------------------------------------------------------------------------

class OllamaBrain:
    """Drives the agent using a local Ollama model with per-agent conversation history."""

    # Keep last N exchanges in history to stay within the context window.
    # Each exchange = 1 user msg (~300 tokens) + 1 assistant msg (~50 tokens).
    # With num_ctx=2048 and system prompt ~200 tokens, 3 exchanges fits comfortably.
    MAX_HISTORY_EXCHANGES: int = 3

    def __init__(self, model: str = OLLAMA_MODEL) -> None:
        self.model = model
        self._history: list[dict] = []  # alternating user/assistant messages

    def decide(self, agent: Agent, world: World) -> dict:
        user_msg = build_tick_context(agent, world)
        # World state goes into system so the model receives it as first-person
        # perception, not as a request from a "user". History stays as user/assistant
        # pairs for past-context continuity. An empty user turn triggers the response.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n---\n\n" + user_msg},
            *self._history,
            {"role": "user", "content": ""},
        ]
        log.debug("[%s] LLM prompt:\n%s", agent.name, user_msg)
        response = _ollama_chat(
            model=self.model,
            messages=messages,
            format="json",
            options=OLLAMA_OPTIONS,
            think=True,
        )
        raw_content = response.message.content
        thinking = getattr(response.message, "thinking", None)
        if thinking:
            log.debug("[%s] LLM thinking:\n%s", agent.name, thinking)
        result = _parse_response(response)
        # Append this exchange to history and trim to window
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": raw_content})
        max_msgs = self.MAX_HISTORY_EXCHANGES * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]
        reasoning = result.pop("reasoning", None)
        if reasoning:
            log.debug("[%s] reasoning: %s", agent.name, reasoning)
        log.debug("[%s] LLM response: %s → %s", agent.name, raw_content, result)
        return result

    def respond_to_message(self, agent: Agent, world: World, sender_name: str, message: str) -> dict:
        """Called when another agent sends a message; returns the response action."""
        user_msg = build_tick_context(agent, world)
        # World state is system context; the incoming message is legitimately input
        # from another entity so it stays as the user turn.
        incoming = (
            f"Incoming message from {sender_name}: \"{message}\"\n"
            f"Reply with {{\"action\": \"interact\", \"target\": \"{sender_name}\", \"message\": \"...\"}} "
            f"or choose any other action to ignore."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n---\n\n" + user_msg},
            *self._history,
            {"role": "user", "content": incoming},
        ]
        log.debug("[%s] respond_to_message from %s: %s", agent.name, sender_name, message)
        response = _ollama_chat(
            model=self.model,
            messages=messages,
            format="json",
            options=OLLAMA_OPTIONS,
            think=True,
        )
        thinking = getattr(response.message, "thinking", None)
        if thinking:
            log.debug("[%s] respond_to_message thinking:\n%s", agent.name, thinking)
        result = _parse_response(response)
        log.debug("[%s] respond_to_message result: %s", agent.name, result)
        return result

    def respond_to_trade(self, agent: Agent, world: World, sender_name: str, offer_give: list[dict], offer_take: list[dict], can_counter: bool) -> dict:
        """Called when another agent proposes a trade. Returns accept, reject, or counter-offer."""
        user_msg = build_tick_context(agent, world)
        give_str = _format_trade_items(offer_give) or "nothing"
        take_str = _format_trade_items(offer_take) or "nothing"
        counter_line = (
            f'\nTo counter: {{"action": "trade_counter", "give": [...], "take": [...]}}'
            if can_counter else ""
        )
        incoming = (
            f"{sender_name} offers a trade: they give you {give_str}; they want {take_str}.\n"
            f'Accept: {{"action": "trade_accept"}}\n'
            f'Reject: {{"action": "trade_reject"}}'
            f"{counter_line}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n---\n\n" + user_msg},
            *self._history,
            {"role": "user", "content": incoming},
        ]
        log.debug("[%s] respond_to_trade from %s: give=%s take=%s", agent.name, sender_name, give_str, take_str)
        response = _ollama_chat(
            model=self.model,
            messages=messages,
            format="json",
            options=OLLAMA_OPTIONS,
            think=True,
        )
        thinking = getattr(response.message, "thinking", None)
        if thinking:
            log.debug("[%s] respond_to_trade thinking:\n%s", agent.name, thinking)
        result = _parse_trade_response(response, can_counter)
        log.debug("[%s] respond_to_trade result: %s", agent.name, result)
        return result

    def compress_memories(self, agent: Agent, memories: list[Memory], compression_type: str) -> list[Memory]:
        """Compress a list of memories into 1-3 factual summaries via LLM."""
        if not memories:
            return []
        tier_label = "day" if compression_type == "day" else "week"
        next_tier = "st" if compression_type == "day" else "lt"
        content_list = "\n".join(f"- [tick {m.tick}] {m.content}" for m in memories)
        prompt = (
            f"You are {agent.name}. Here are your memories from the past {tier_label}:\n\n"
            f"{content_list}\n\n"
            f"Write 1-3 short factual sentences summarizing what happened. "
            f"Mention key events, people you interacted with, and your physical state. "
            f"Be direct and factual — no storytelling, no embellishment. "
            f"Rate the overall importance of this period (1-10).\n"
            f"Reply as JSON: {{\"summary\": \"<sentences>\", \"importance\": <1-10>}}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = _ollama_chat(
                model=self.model,
                messages=messages,
                format="json",
                options=OLLAMA_OPTIONS,
                think=True,
            )
            data = json.loads(response.message.content)
            summary = str(data.get("summary", "")).strip()
            importance = float(data.get("importance", 5.0))
            importance = max(1.0, min(10.0, importance))
        except Exception as exc:
            log.warning("[%s] compress_memories (%s) failed: %s", agent.name, compression_type, exc)
            # Fallback: most important memory's content
            best = max(memories, key=lambda m: m.importance)
            summary = best.content
            importance = best.importance
        if not summary:
            return []
        latest_tick = max(m.tick for m in memories)
        log.info("[%s] memory compressed (%s→%s) imp=%.1f: %s", agent.name, compression_type, next_tier, importance, summary[:80])
        return [Memory(content=summary, tick=latest_tick, last_access=latest_tick, importance=importance, tier=next_tier)]

    def score_interaction(self, agent: Agent, world: World, other_name: str, sent: str, received: str | None) -> tuple[float, str, str]:
        """Score an interaction from this agent's perspective. Returns (score, note, memory)."""
        if received is not None:
            exchange = f"You said: \"{sent}\"\n{other_name} replied: \"{received}\""
        else:
            exchange = f"You said: \"{sent}\"\n{other_name} ignored your message."
        prompt = (
            f"You are {agent.name}. Rate your feelings toward {other_name} after this exchange:\n\n"
            f"{exchange}\n\n"
            f"Reply as JSON: {{\"score\": <float -1.0 to 1.0>, \"note\": \"<max 8 words>\", "
            f"\"memory\": \"<optional: your impression, what you learned, or leave empty>\"}}"
        )
        messages = [{"role": "user", "content": prompt}]
        response = _ollama_chat(
            model=self.model,
            messages=messages,
            format="json",
            options=OLLAMA_OPTIONS,
            think=True,
        )
        return _parse_score_response(response)

# ---------------------------------------------------------------------------
# Context builder (pure functions — testable without Ollama)
# ---------------------------------------------------------------------------

def build_tick_context(agent: Agent, world: World) -> str:
    tile = world.get_tile(agent.x, agent.y)
    if tile is not None:
        tile_desc = f"{tile.resource_type.value.capitalize()} ({tile.resource_type.value[0].upper()}{tile.amount:.0f})"
    else:
        tile_desc = "empty"

    carry = agent.carry_total
    inventory_parts = [
        f"{rtype.value}: {amount:.1f}"
        for rtype, amount in agent.inventory.items()
        if amount > 0
    ]
    inventory_str = ", ".join(inventory_parts) if inventory_parts else "empty"

    tips = get_tips(agent, world)
    tips_block = ""
    if tips:
        tips_lines = "\n".join(f"  {t}" for t in tips)
        tips_block = f"\nHow you feel:\n{tips_lines}\n"

    radius = 1
    surroundings = build_surroundings(agent, world, radius=radius)
    grid_size = 2 * radius + 1

    last_actions_block = ""
    if agent.last_actions:
        lines = "\n".join(f"  {r.detail}" for r in agent.last_actions)
        last_actions_block = f"\nLast actions:\n{lines}\n"

    nearby_agents = [
        a for a in world.agents
        if a.alive and a is not agent
        and abs(a.x - agent.x) <= 1 and abs(a.y - agent.y) <= 1
    ]
    nearby_block = ""
    if nearby_agents:
        nb_lines = "\n".join(
            f"  {a.name} at ({a.x},{a.y}) — satiation={a.satiation:.1f} warmth={a.warmth:.1f}"
            for a in nearby_agents
        )
        nearby_block = f"\nNearby agents (within 1 tile):\n{nb_lines}\n"

    # Build retrieval query from current context (needs nearby_agents, defined above)
    _query_parts = [f"({agent.x},{agent.y})", f"satiation {agent.satiation:.0f}", f"warmth {agent.warmth:.0f}"]
    if nearby_agents:
        _query_parts.extend(a.name for a in nearby_agents)
    _query = " ".join(_query_parts)
    retrieved = agent.memory.retrieve(_query, world.tick_count, k=5)
    memory_block = ""
    if retrieved:
        _tier_labels = {"vst": "today", "st": "this week", "lt": "long ago"}
        mem_lines = "\n".join(
            f"  [{_tier_labels.get(m.tier, m.tier)}] {m.content}"
            for m in retrieved
        )
        memory_block = f"\nMemories:\n{mem_lines}\n"

    relationships_block = ""
    if agent.relationships:
        rel_lines = "\n".join(
            f"  {name}: {r.score:+.2f} — {r.note}"
            for name, r in agent.relationships.items()
        )
        relationships_block = f"\nRelationships:\n{rel_lines}\n"

    return (
        f"Time: {world.time}\n"
        f"Position: ({agent.x},{agent.y}) — tile: {tile_desc}\n"
        f"Satiation: {agent.satiation:.1f}/100 | Warmth: {agent.warmth:.1f}/100 | Health: {agent.health:.1f}/100 | Carrying: {carry:.1f}/20 ({inventory_str})"
        f"{tips_block}"
        f"\nSurroundings ({grid_size}x{grid_size}):\n{surroundings}"
        f"{nearby_block}"
        f"{relationships_block}"
        f"{memory_block}"
        f"{last_actions_block}"
    )


def build_surroundings(agent: Agent, world: World, radius: int = 1) -> str:
    col_w = 5
    x0, y0 = agent.x - radius, agent.y - radius
    x1, y1 = agent.x + radius, agent.y + radius

    # Column header
    col_header = "     " + "".join(str(cx).rjust(col_w) for cx in range(x0, x1 + 1))
    rows = [col_header]

    for cy in range(y0, y1 + 1):
        row_label = f"{cy:>3}  "
        cells = []
        for cx in range(x0, x1 + 1):
            if cx == agent.x and cy == agent.y:
                tile = world.get_tile(cx, cy)
                if tile is not None:
                    char = tile.resource_type.value[0].upper()
                    cell = f"[{char}{tile.amount:.0f}]"
                else:
                    cell = "[ ]"
            elif cx < 0 or cx >= world.width or cy < 0 or cy >= world.height:
                cell = "~"
            elif (cx, cy) not in agent.revealed:
                cell = "?"
            elif any(a.alive and a.x == cx and a.y == cy for a in world.agents if a is not agent):
                cell = "@"
            else:
                tile = world.get_tile(cx, cy)
                if tile is None:
                    cell = "."
                else:
                    char = tile.resource_type.value[0].upper()
                    cell = f"{char}{tile.amount:.0f}"
            cells.append(cell.rjust(col_w))
        rows.append(row_label + "".join(cells))

    return "\n".join(rows)


def _parse_score_response(response) -> tuple[float, str, str]:
    try:
        content = response.message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        data = json.loads(content)
        score = float(data.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        note = str(data.get("note", ""))[:60]
        memory = str(data.get("memory", "")).strip()[:100]
        return score, note, memory
    except Exception:
        return 0.0, "", ""


_VALID_ACTIONS = {"move", "harvest", "eat", "burn", "rest", "interact", "trade"}

_VALID_TRADE_RESPONSES = {"trade_accept", "trade_reject", "trade_counter"}


def _format_trade_items(items: list[dict]) -> str:
    """Human-readable summary of trade items for LLM prompts."""
    parts = [f"{item['qty']:.1f} {item['resource']}" for item in items if item.get("qty", 0) > 0]
    return ", ".join(parts)


def _parse_trade_response(response, can_counter: bool) -> dict:
    """Parse a trade response (accept/reject/counter). Falls back to reject on any error."""
    try:
        content = response.message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        parsed = json.loads(content)
        action = parsed.get("action", "")
        if action not in _VALID_TRADE_RESPONSES:
            return {"action": "trade_reject"}
        if action == "trade_counter" and not can_counter:
            log.warning("_parse_trade_response: counter-offer not allowed at this stage — rejecting")
            return {"action": "trade_reject"}
        if action == "trade_counter":
            give = [i for i in parsed.get("give", []) if isinstance(i, dict) and float(i.get("qty", 0)) > 0]
            take = [i for i in parsed.get("take", []) if isinstance(i, dict) and float(i.get("qty", 0)) > 0]
            if not give and not take:
                return {"action": "trade_reject"}
            parsed["give"] = give
            parsed["take"] = take
        return parsed
    except Exception:
        return {"action": "trade_reject"}


def _parse_response(response) -> dict:
    try:
        content = response.message.content.strip()
        # Strip markdown code fences if the model wraps its output
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]  # drop ```json line
            content = content.rsplit("```", 1)[0]  # drop trailing ```
        parsed = json.loads(content)
        action = parsed.get("action", "")
        if action not in _VALID_ACTIONS:
            # Model emitted an unrecognised action (e.g. duplicate-key JSON like
            # {"action":"interact","target":"Ada","action":"hello"} where the
            # parser resolves to action="hello"). Fall back to rest.
            log.warning("_parse_response: unrecognised action %r — falling back to rest", action)
            return {"action": "rest"}
        # interact requires a non-empty message; guard here as a belt-and-braces
        # check (world.py Phase 1.5 also validates, but this keeps the contract clean).
        if action == "interact" and not str(parsed.get("message", "")).strip():
            log.warning("_parse_response: interact missing 'message' — falling back to rest")
            return {"action": "rest"}
        if action == "trade":
            if not str(parsed.get("target", "")).strip():
                log.warning("_parse_response: trade missing 'target' — falling back to rest")
                return {"action": "rest"}
            give = [i for i in parsed.get("give", []) if isinstance(i, dict) and float(i.get("qty", 0)) > 0]
            take = [i for i in parsed.get("take", []) if isinstance(i, dict) and float(i.get("qty", 0)) > 0]
            if not give and not take:
                log.warning("_parse_response: trade with both give and take empty — falling back to rest")
                return {"action": "rest"}
            parsed["give"] = give
            parsed["take"] = take
        return parsed
    except Exception:
        return {"action": "rest"}
