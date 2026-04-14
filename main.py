from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from godmode.agent import Agent
from godmode.brain import OllamaBrain
from godmode.display import render_world
from godmode.save import SAVE_PATH, load_state, save_state
from godmode.time import WorldTime
from godmode.world import World

TICKS = 720           # 30 in-world days
TICK_SLEEP = 0.35     # seconds between ticks (animation speed)
CLEAR = "\033[2J\033[H"
LOG_PATH = "godmode.log"

_NAMES = [
    "Ada", "Bo", "Cal", "Dee", "Eve",
    "Finn", "Gus", "Hal", "Ida", "Jax",
    "Kay", "Leo", "Mae", "Ned", "Ora",
    "Pip", "Que", "Rex", "Sue", "Ted",
]

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GodMode simulation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--new", action="store_true", help="Start a fresh world (clears save and log)")
    group.add_argument("--resume", action="store_true", help="Resume from the last saved state")
    return parser.parse_args()


def _setup_logging(clear: bool = False) -> None:
    if clear and os.path.exists(LOG_PATH):
        open(LOG_PATH, "w").close()
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.DEBUG,
        format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress HTTP transport noise — each LLM call generates ~10 lines at DEBUG
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _fib(n: int) -> int:
    """nth Fibonacci number (0-indexed): 0, 1, 1, 2, 3, 5, 8, 13, ..."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _alive_count(world: World) -> int:
    return sum(1 for a in world.agents if a.alive)


def _spawn(world: World, name_idx: int) -> Agent:
    name = _NAMES[name_idx % len(_NAMES)]
    x = world._rng.randint(1, world.width - 2)
    y = world._rng.randint(1, world.height - 2)
    agent = Agent(name=name, birth_tick=world.tick_count, x=x, y=y, brain=OllamaBrain())
    world.add_agent(agent)
    log.info("Spawned %s at tick=%d  pop=%d  pos=(%d,%d)", name, world.tick_count, _alive_count(world), x, y)
    return agent


def render_agent_status(agent: Agent) -> str:
    if not agent.alive:
        return f"  [{agent.name}] DEAD"
    inv_parts = [f"{rtype.value}: {amt:.1f}" for rtype, amt in agent.inventory.items() if amt > 0]
    inv_str = ", ".join(inv_parts) if inv_parts else "empty"
    last = agent.last_actions[-1].detail if agent.last_actions else "—"
    rel_str = ""
    if agent.relationships:
        rel_parts = [f"{name}:{r.score:+.1f}" for name, r in agent.relationships.items()]
        rel_str = f"\n    relations=[{', '.join(rel_parts)}]"
    return (
        f"  [{agent.name}] ({agent.x},{agent.y})  "
        f"satiation={agent.satiation:.1f}  warmth={agent.warmth:.1f}  health={agent.health:.1f}  "
        f"carry={agent.carry_total:.1f}/20  inv=[{inv_str}]\n"
        f"    → {last}"
        f"{rel_str}"
    )


def _render_frame(world: World, total_spawned: int, last_spawn_tick: int, next_interval_ticks: int) -> None:
    alive = _alive_count(world)
    ticks_left = max(0, last_spawn_tick + next_interval_ticks - world.tick_count)
    fib_days = next_interval_ticks // WorldTime.TICKS_PER_DAY

    header = (
        f"=== GodMode  pop={alive}/{total_spawned}  "
        f"next spawn in {ticks_left} ticks  (fib[{alive}]={fib_days}d) ==="
    )

    frame = CLEAR + header + "\n\n"
    frame += render_world(world) + "\n"
    for agent in world.agents:
        frame += render_agent_status(agent) + "\n"
    if world._last_interactions:
        frame += "\nInteractions:\n"
        for ix in world._last_interactions:
            if ix.get("type") == "trade":
                give_str = ", ".join(f"{i['qty']:.1f} {i['resource']}" for i in ix["give"]) or "nothing"
                take_str = ", ".join(f"{i['qty']:.1f} {i['resource']}" for i in ix["take"]) or "nothing"
                frame += (
                    f"  {ix['initiator']} ↔ {ix['target']}: trade "
                    f"(give {give_str} / want {take_str}) — {ix['outcome']}\n"
                    f"  [scores: {ix['initiator']} {ix['initiator_score']:+.2f}  "
                    f"{ix['target']} {ix['target_score']:+.2f}]\n"
                )
            elif ix["reply"]:
                frame += (
                    f"  {ix['initiator']} → {ix['target']}: \"{ix['message']}\"\n"
                    f"  {ix['target']} → {ix['initiator']}: \"{ix['reply']}\"\n"
                    f"  [scores: {ix['initiator']} {ix['initiator_score']:+.2f}  "
                    f"{ix['target']} {ix['target_score']:+.2f}]\n"
                )
            else:
                frame += (
                    f"  {ix['initiator']} → {ix['target']}: \"{ix['message']}\" (ignored)\n"
                    f"  [{ix['initiator']} score: {ix['initiator_score']:+.2f}]\n"
                )
    sys.stdout.write(frame)
    sys.stdout.flush()


def _resolve_mode(args: argparse.Namespace) -> str:
    """Return 'new' or 'resume'."""
    if args.new:
        return "new"
    if args.resume:
        return "resume" if os.path.exists(SAVE_PATH) else "new"
    # Auto-detect: prompt only when a save file is present.
    if os.path.exists(SAVE_PATH):
        sys.stdout.write("Save file found. [r]esume or [n]ew world? ")
        sys.stdout.flush()
        choice = input().strip().lower()
        return "resume" if choice.startswith("r") else "new"
    return "new"


def main() -> None:
    args = _parse_args()
    mode = _resolve_mode(args)

    _setup_logging(clear=(mode == "new"))

    if mode == "resume":
        world, total_spawned, last_spawn_tick = load_state(SAVE_PATH)
        alive = _alive_count(world)
        next_interval = _fib(alive) * WorldTime.TICKS_PER_DAY
        ticks_done = world.tick_count
    else:
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
        world = World(width=7, height=7, resource_density=0.35, seed=42)
        total_spawned = 0
        _spawn(world, total_spawned)
        total_spawned += 1
        last_spawn_tick = world.tick_count  # 0
        alive = _alive_count(world)
        next_interval = _fib(alive) * WorldTime.TICKS_PER_DAY  # fib[1]*24 = 24
        ticks_done = 0

    _render_frame(world, total_spawned, last_spawn_tick, next_interval)
    time.sleep(1.0)

    ticks_left_in_run = TICKS - ticks_done
    for _ in range(ticks_left_in_run):
        world.tick()

        # Dynamic interval: recomputes from current alive count each tick,
        # so deaths shrink the gap and trigger faster recovery spawns.
        alive = _alive_count(world)
        ticks_since = world.tick_count - last_spawn_tick
        next_interval = _fib(alive) * WorldTime.TICKS_PER_DAY

        if ticks_since >= next_interval:
            _spawn(world, total_spawned)
            total_spawned += 1
            last_spawn_tick = world.tick_count
            save_state(SAVE_PATH, world, total_spawned, last_spawn_tick)  # commit spawn atomically
            alive = _alive_count(world)
            next_interval = _fib(alive) * WorldTime.TICKS_PER_DAY

        save_state(SAVE_PATH, world, total_spawned, last_spawn_tick)
        _render_frame(world, total_spawned, last_spawn_tick, next_interval)

        if alive == 0:
            sys.stdout.write(f"\n\nAll agents perished at {world.time}.\n")
            sys.stdout.flush()
            break

        time.sleep(TICK_SLEEP)

    if _alive_count(world) > 0:
        sys.stdout.write(f"\n\n=== End  ({world.time}) ===\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
