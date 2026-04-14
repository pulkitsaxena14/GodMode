# GodMode

A local LLM-driven agent simulation. Autonomous agents survive in a procedurally generated world — foraging for food, gathering wood, managing warmth and hunger, forming relationships, and trading with each other. Each agent thinks and acts independently, powered by a locally running model via Ollama.

## What it does

- **Survival mechanics** — agents manage satiation and warmth. Neglect either long enough and health decays.
- **Fog of war** — the world starts dark. Agents reveal tiles as they explore.
- **Resources** — food tiles regrow over time; wood tiles regrow slower. Both deplete if over-harvested.
- **Memory** — agents record notable events. Memories compress from recent (VST) → daily (ST) → weekly (LT) summaries over time.
- **Social interaction** — agents can message each other when adjacent. Exchanges form relationship scores that persist and influence future behaviour.
- **Trade** — agents negotiate resource trades (food ↔ wood). Either side can counter-offer once; the other then accepts or rejects.
- **Dynamic population** — new agents spawn at Fibonacci-interval days. Deaths accelerate the next spawn.
- **Save / resume** — full state (world, agents, memories, brain history, relationships) persists to disk after every tick.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally with a thinking-capable model pulled:
  ```
  ollama pull gemma4:e4b-it-q8_0
  ```

## Setup

```bash
pip install uv        # if not already installed
uv sync
```

## Running

```bash
uv run python main.py           # auto-detects save file, prompts if found
uv run python main.py --new     # start fresh (clears save and log)
uv run python main.py --resume  # resume from last save (falls back to new if none)
```

Simulation runs for 720 ticks (30 in-world days). Progress is saved after every tick. Logs go to `godmode.log`.

## Project structure

```
main.py              — entry point, render loop, spawn logic
godmode/
  agent.py           — Agent dataclass, action execution, survival mechanics
  brain.py           — LLM interface (OllamaBrain), prompt construction, response parsing
  world.py           — World grid, tick loop, interaction and trade resolution
  resources.py       — ResourceType, ResourceConfig, ResourceTile
  memory.py          — MemoryStore, three-tier compression (VST/ST/LT)
  save.py            — JSON serialisation / deserialisation of full sim state
  display.py         — World grid renderer
  time.py            — In-world clock (ticks → hours/days/months/years)
tests/               — pytest suite (~227 tests, no Ollama required)
```

## Tests

```bash
uv run python -m pytest tests/ -v
uv run python -m pytest tests/ --cov=godmode --cov-report=term-missing
```
