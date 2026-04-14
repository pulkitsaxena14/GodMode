"""Tests for the three-tier memory system (godmode/memory.py)."""
from __future__ import annotations

import pytest

from godmode.memory import (
    DECAY_FACTOR,
    MIN_IMPORTANCE,
    TICKS_PER_DAY,
    TICKS_PER_WEEK,
    Memory,
    MemoryStore,
    score_importance,
)


# ---------------------------------------------------------------------------
# Memory dataclass
# ---------------------------------------------------------------------------

def test_memory_fields():
    m = Memory(content="did something", tick=5, last_access=5, importance=6.0, tier="vst")
    assert m.content == "did something"
    assert m.tick == 5
    assert m.last_access == 5
    assert m.importance == 6.0
    assert m.tier == "vst"


# ---------------------------------------------------------------------------
# score_importance heuristic
# ---------------------------------------------------------------------------

def test_importance_rest():
    assert score_importance("rest", "rested", 80, 80, 80) == 1.0

def test_importance_move():
    assert score_importance("move", "moved to (3,3)", 80, 80, 80) == 2.0

def test_importance_harvest_success():
    assert score_importance("harvest", "harvested 3.0 food", 80, 80, 80) == 4.0

def test_importance_harvest_nothing():
    assert score_importance("harvest", "nothing to harvest here", 80, 80, 80) == 2.0

def test_importance_eat_normal():
    assert score_importance("eat", "ate 2.0 food", 60, 80, 80) == 5.0

def test_importance_eat_when_hungry():
    assert score_importance("eat", "ate 2.0 food", 20, 80, 80) == 7.0

def test_importance_burn_when_cold():
    assert score_importance("burn", "burned 2.0 wood", 80, 20, 80) == 7.0

def test_importance_interact():
    assert score_importance("interact", "spoke to Ada", 80, 80, 80) == 8.0

def test_importance_starvation_crisis():
    assert score_importance("move", "moved to (3,3)", 0, 80, 80) == 8.0

def test_importance_freezing_crisis():
    assert score_importance("rest", "rested", 80, 0, 80) == 8.0

def test_importance_near_death():
    assert score_importance("rest", "rested", 50, 50, 25) == 9.0

def test_importance_near_death_overrides_all():
    # Even rest scores 9.0 when health is critically low
    assert score_importance("rest", "rested", 0, 0, 10) == 9.0


# ---------------------------------------------------------------------------
# MemoryStore — record
# ---------------------------------------------------------------------------

def test_store_empty_initially():
    store = MemoryStore()
    assert store.vst == []
    assert store.st == []
    assert store.lt == []

def test_record_adds_to_vst():
    store = MemoryStore()
    store.record("ate some food", tick=5, importance=5.0)
    assert len(store.vst) == 1
    assert store.vst[0].content == "ate some food"
    assert store.vst[0].tier == "vst"

def test_record_drops_low_importance():
    store = MemoryStore()
    store.record("rested", tick=5, importance=1.0)
    assert store.vst == []

def test_record_at_threshold_is_kept():
    store = MemoryStore()
    store.record("exactly threshold", tick=5, importance=MIN_IMPORTANCE)
    assert len(store.vst) == 1

def test_record_preserves_tick():
    store = MemoryStore()
    store.record("something happened", tick=42, importance=5.0)
    assert store.vst[0].tick == 42
    assert store.vst[0].last_access == 42


# ---------------------------------------------------------------------------
# MemoryStore — retrieval
# ---------------------------------------------------------------------------

def test_retrieve_empty_store():
    store = MemoryStore()
    assert store.retrieve("anything", current_tick=10) == []

def test_retrieve_returns_up_to_k():
    store = MemoryStore()
    for i in range(10):
        store.record(f"event {i}", tick=i, importance=5.0)
    result = store.retrieve("event", current_tick=10, k=3)
    assert len(result) <= 3

def test_retrieve_updates_last_access():
    store = MemoryStore()
    store.record("some event", tick=0, importance=5.0)
    store.retrieve("some event", current_tick=50)
    assert store.vst[0].last_access == 50

def test_retrieve_prefers_recent():
    store = MemoryStore()
    store.record("old event", tick=0, importance=5.0)
    store.record("recent event", tick=99, importance=5.0)
    result = store.retrieve("event", current_tick=100, k=1)
    assert result[0].content == "recent event"

def test_retrieve_prefers_high_importance():
    store = MemoryStore()
    # Same tick so recency is equal — importance should decide
    store.record("low importance event", tick=10, importance=3.0)
    store.record("high importance event", tick=10, importance=9.0)
    result = store.retrieve("event", current_tick=10, k=1)
    assert result[0].content == "high importance event"

def test_retrieve_keyword_relevance_boosts_score():
    store = MemoryStore()
    store.record("Ada spoke to me today", tick=5, importance=4.0)
    store.record("ate some food near river", tick=5, importance=4.0)
    # Query mentions Ada — Ada memory should rank higher
    result = store.retrieve("Ada nearby", current_tick=5, k=1)
    assert "Ada" in result[0].content

def test_retrieve_searches_across_all_tiers():
    store = MemoryStore()
    store.vst.append(Memory("vst memory", tick=10, last_access=10, importance=5.0, tier="vst"))
    store.st.append(Memory("st memory", tick=5, last_access=5, importance=5.0, tier="st"))
    store.lt.append(Memory("lt memory", tick=0, last_access=0, importance=5.0, tier="lt"))
    result = store.retrieve("memory", current_tick=10, k=5)
    tiers = {m.tier for m in result}
    assert tiers == {"vst", "st", "lt"}


# ---------------------------------------------------------------------------
# Compression scheduling
# ---------------------------------------------------------------------------

def test_needs_day_compression_false_at_tick_0():
    store = MemoryStore()
    store.record("something", tick=0, importance=5.0)
    assert not store.needs_day_compression(0)

def test_needs_day_compression_true_at_end_of_day():
    store = MemoryStore()
    store.record("something", tick=1, importance=5.0)
    assert store.needs_day_compression(TICKS_PER_DAY)

def test_needs_day_compression_false_when_vst_empty():
    store = MemoryStore()
    assert not store.needs_day_compression(TICKS_PER_DAY)

def test_needs_week_compression_true_at_end_of_week():
    store = MemoryStore()
    store.st.append(Memory("a summary", tick=1, last_access=1, importance=5.0, tier="st"))
    assert store.needs_week_compression(TICKS_PER_WEEK)

def test_needs_week_compression_false_when_st_empty():
    store = MemoryStore()
    assert not store.needs_week_compression(TICKS_PER_WEEK)

def test_end_of_week_also_triggers_day_compression():
    store = MemoryStore()
    store.record("today's event", tick=TICKS_PER_WEEK - 1, importance=5.0)
    # At TICKS_PER_WEEK both conditions should be true
    assert store.needs_day_compression(TICKS_PER_WEEK)
    assert TICKS_PER_WEEK % TICKS_PER_DAY == 0  # verify mathematical relationship


# ---------------------------------------------------------------------------
# Tier promotion
# ---------------------------------------------------------------------------

def test_promote_vst_to_st():
    store = MemoryStore()
    store.record("daily event", tick=1, importance=5.0)
    summary = Memory("day summary", tick=24, last_access=24, importance=6.0, tier="st")
    store.promote_vst_to_st([summary])
    assert store.vst == []
    assert len(store.st) == 1
    assert store.st[0].content == "day summary"

def test_promote_st_to_lt():
    store = MemoryStore()
    store.st.append(Memory("week summary", tick=168, last_access=168, importance=7.0, tier="lt"))
    lt_summary = Memory("long term memory", tick=168, last_access=168, importance=7.0, tier="lt")
    store.promote_st_to_lt([lt_summary])
    assert store.st == []
    assert len(store.lt) == 1

def test_promote_with_empty_summaries_clears_tier():
    store = MemoryStore()
    store.record("something", tick=1, importance=5.0)
    store.promote_vst_to_st([])   # no summaries produced
    assert store.vst == []


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_shows_counts():
    store = MemoryStore()
    store.record("event", tick=1, importance=5.0)
    r = repr(store)
    assert "vst=1" in r
    assert "st=0" in r
    assert "lt=0" in r


# ---------------------------------------------------------------------------
# Integration with Agent
# ---------------------------------------------------------------------------

def test_agent_has_memory_store():
    from godmode.brain import ScriptedBrain
    from godmode.agent import Agent
    agent = Agent(name="Test", birth_tick=0, x=0, y=0, brain=ScriptedBrain([]))
    assert isinstance(agent.memory, MemoryStore)

def test_agent_memory_independent_per_instance():
    from godmode.brain import ScriptedBrain
    from godmode.agent import Agent
    a1 = Agent(name="A", birth_tick=0, x=0, y=0, brain=ScriptedBrain([]))
    a2 = Agent(name="B", birth_tick=0, x=0, y=0, brain=ScriptedBrain([]))
    a1.memory.record("only for a1", tick=1, importance=5.0)
    assert a2.memory.vst == []


# ---------------------------------------------------------------------------
# Save/load round-trip
# ---------------------------------------------------------------------------

def test_memory_save_load_roundtrip(tmp_path):
    from godmode.memory import Memory, MemoryStore
    from godmode.save import _serialize_memory_store, _deserialize_memory_store

    store = MemoryStore()
    store.record("vst observation", tick=3, importance=5.0)
    store.st.append(Memory("st summary", tick=24, last_access=24, importance=7.0, tier="st"))
    store.lt.append(Memory("lt summary", tick=168, last_access=168, importance=8.0, tier="lt"))

    data = _serialize_memory_store(store)
    restored = _deserialize_memory_store(data)

    assert len(restored.vst) == 1
    assert restored.vst[0].content == "vst observation"
    assert restored.vst[0].importance == 5.0
    assert len(restored.st) == 1
    assert restored.st[0].tier == "st"
    assert len(restored.lt) == 1
    assert restored.lt[0].content == "lt summary"
