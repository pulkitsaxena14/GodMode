from __future__ import annotations

from godmode.resources import ResourceType
from godmode.world import World

_COL_WIDTH = 5  # wide enough for "W15!" and padding

_TYPE_CHAR = {
    ResourceType.FOOD: "F",
    ResourceType.WOOD: "W",
}


def _cell_str(tile) -> str:
    if tile is None:
        return "."
    char = _TYPE_CHAR[tile.resource_type]
    amount = int(tile.amount)
    marker = "!" if tile.is_depleted else ""
    return f"{char}{amount}{marker}"


def render_world(world: World) -> str:
    """Return the full world grid as a string (no side effects)."""
    lines: list[str] = []

    header = f"{world.time}  tick={world.tick_count}  ({world.width}x{world.height})"
    lines.append(header)

    # Agent position lookup
    agent_positions: dict[tuple[int, int], str] = {}
    for agent in world.agents:
        if agent.alive:
            key = (agent.x, agent.y)
            if key not in agent_positions:
                agent_positions[key] = "@"

    # Column index header
    col_labels = "".join(str(x).rjust(_COL_WIDTH) for x in range(world.width))
    lines.append(f"{'':3}{col_labels}")

    for y in range(world.height):
        row_str = "".join(
            agent_positions.get(
                (x, y),
                "?" if not world.is_revealed(x, y) else _cell_str(world.grid[y][x])
            ).rjust(_COL_WIDTH)
            for x in range(world.width)
        )
        lines.append(f"{y:<3}{row_str}")

    return "\n".join(lines)


def print_world(world: World) -> None:
    print(render_world(world))


def print_tile_detail(world: World, x: int, y: int) -> None:
    tile = world.get_tile(x, y)
    if tile is None:
        print(f"({x},{y}): empty")
        return
    cfg = tile.config
    print(
        f"({x},{y}): {tile.resource_type.value}  "
        f"amount={tile.amount:.2f}/{cfg.max_value}  "
        f"depleted={tile.is_depleted}  "
        f"recovery_ticks_left={tile.depleted_ticks_remaining}  "
        f"growth_rate={cfg.depleted_growth_rate if tile.is_depleted else cfg.growth_rate}/tick"
    )
