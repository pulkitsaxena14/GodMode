from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECAY_FACTOR: float = 0.995   # recency decay per tick (1 tick ≈ 1 game-hour)
MIN_IMPORTANCE: float = 3.0   # observations below this threshold are dropped
TICKS_PER_DAY: int = 24
TICKS_PER_WEEK: int = 168     # 7 × 24


# ---------------------------------------------------------------------------
# Memory dataclass
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    content: str
    tick: int           # creation tick
    last_access: int    # updated on retrieval — drives recency score
    importance: float   # 1–10
    tier: str           # "vst" | "st" | "lt"


# ---------------------------------------------------------------------------
# Importance heuristic (no LLM call)
# ---------------------------------------------------------------------------

def score_importance(
    action: str,
    detail: str,
    satiation: float,
    warmth: float,
    health: float,
) -> float:
    """Rule-based importance score 1–10 for a completed action."""
    # Near-death overrides everything
    if health < 30:
        return 9.0
    # Active crisis: starving or freezing
    if satiation <= 0 or warmth <= 0:
        return 8.0
    # Recovery when in distress
    if action == "eat" and satiation < 25:
        return 7.0
    if action == "burn" and warmth < 25:
        return 7.0
    # Social interactions (conversation recorded separately with full content)
    if action == "interact":
        return 8.0
    # Successful resource gathering
    if action == "harvest":
        if "nothing" in detail or "too much" in detail:
            return 2.0
        return 4.0
    # Eating / burning at normal state
    if action in ("eat", "burn"):
        return 5.0
    # Routine
    if action == "move":
        return 2.0
    if action == "rest":
        return 1.0
    return 2.0


# ---------------------------------------------------------------------------
# MemoryStore — three-tier memory system
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Three tiers, each compressed by an LLM on a time schedule:

    VST (very short term): raw action observations recorded each tick.
        Compressed every night (tick % 24 == 0) → summaries move to ST.
    ST (short term): daily summaries accumulated across a week.
        Compressed every week (tick % 168 == 0) → summaries move to LT.
    LT (long term): weekly summaries kept indefinitely.
    """

    def __init__(self) -> None:
        self.vst: list[Memory] = []   # current day's raw observations
        self.st: list[Memory] = []    # current week's daily summaries
        self.lt: list[Memory] = []    # all weekly summaries (never cleared)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, content: str, tick: int, importance: float) -> None:
        """Add a VST observation.  Silently drops events below MIN_IMPORTANCE."""
        if importance < MIN_IMPORTANCE:
            return
        m = Memory(
            content=content,
            tick=tick,
            last_access=tick,
            importance=importance,
            tier="vst",
        )
        self.vst.append(m)
        log.debug("memory[VST] imp=%.1f  %s", importance, content[:80])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, current_tick: int, k: int = 5) -> list[Memory]:
        """
        Return top-k memories scored by:
          0.5 × recency  +  0.3 × importance  +  0.2 × keyword_relevance
        Updates last_access on returned memories.
        """
        all_memories: list[Memory] = self.vst + self.st + self.lt
        if not all_memories:
            return []

        query_words = set(query.lower().split())
        scored: list[tuple[float, Memory]] = []

        for m in all_memories:
            age = current_tick - m.last_access
            recency = DECAY_FACTOR ** max(age, 0)

            importance_norm = m.importance / 10.0

            # Fraction of query words present in the memory text (capped at 1)
            mem_words = set(m.content.lower().split())
            overlap = len(query_words & mem_words)
            relevance = min(1.0, overlap / max(len(query_words), 1) * 2)

            score = 0.5 * recency + 0.3 * importance_norm + 0.2 * relevance
            scored.append((score, m))

        scored.sort(key=lambda x: -x[0])
        result: list[Memory] = []
        for _, m in scored[:k]:
            m.last_access = current_tick   # refresh recency on access
            result.append(m)
        return result

    # ------------------------------------------------------------------
    # Compression scheduling
    # ------------------------------------------------------------------

    def needs_day_compression(self, tick: int) -> bool:
        """True when it's the end of a day and there are VST memories to compress."""
        return tick > 0 and tick % TICKS_PER_DAY == 0 and bool(self.vst)

    def needs_week_compression(self, tick: int) -> bool:
        """True when it's the end of a week and there are ST memories to compress."""
        return tick > 0 and tick % TICKS_PER_WEEK == 0 and bool(self.st)

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_vst_to_st(self, summaries: list[Memory]) -> None:
        """Move compressed day summaries into ST and clear VST."""
        self.st.extend(summaries)
        self.vst.clear()
        log.debug("memory VST→ST: %d summaries, ST now %d", len(summaries), len(self.st))

    def promote_st_to_lt(self, summaries: list[Memory]) -> None:
        """Move compressed week summaries into LT and clear ST."""
        self.lt.extend(summaries)
        self.st.clear()
        log.debug("memory ST→LT: %d summaries, LT now %d", len(summaries), len(self.lt))

    def __repr__(self) -> str:
        return f"MemoryStore(vst={len(self.vst)}, st={len(self.st)}, lt={len(self.lt)})"
