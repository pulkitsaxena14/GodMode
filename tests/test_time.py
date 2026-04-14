from __future__ import annotations

import pytest
from godmode.time import WorldTime


def test_initial_time():
    t = WorldTime(tick=0)
    assert t.year == 1
    assert t.month == 1
    assert t.day == 1
    assert t.hour == 0


def test_hour_within_first_day():
    t = WorldTime(tick=23)
    assert t.hour == 23
    assert t.day == 1


def test_hour_wraps_at_day_boundary():
    t = WorldTime(tick=24)
    assert t.hour == 0
    assert t.day == 2


def test_day_wraps_at_month_boundary():
    # tick=720 = 24*30: first tick of month 2
    t = WorldTime(tick=720)
    assert t.day == 1
    assert t.month == 2


def test_month_wraps_at_year_boundary():
    # tick=8640 = 24*30*12: first tick of year 2
    t = WorldTime(tick=8640)
    assert t.month == 1
    assert t.year == 2


def test_mid_values():
    # tick=750: 720 (month 2 start) + 30 ticks = 1 full day + 6 hours into day 2
    t = WorldTime(tick=750)
    assert t.year == 1
    assert t.month == 2
    assert t.day == 2
    assert t.hour == 6


def test_str_format():
    t = WorldTime(tick=0)
    assert str(t) == "Y1 M01 D01 H00"


def test_str_format_mid_game():
    # tick=750: Y1 M02 D02 H06
    t = WorldTime(tick=750)
    assert str(t) == "Y1 M02 D02 H06"


def test_frozen():
    t = WorldTime(tick=5)
    with pytest.raises(Exception):
        t.tick = 10  # type: ignore[misc]
