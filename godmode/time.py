from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class WorldTime:
    tick: int

    HOURS_PER_DAY: ClassVar[int] = 24
    DAYS_PER_MONTH: ClassVar[int] = 30
    MONTHS_PER_YEAR: ClassVar[int] = 12
    TICKS_PER_DAY: ClassVar[int] = 24
    TICKS_PER_MONTH: ClassVar[int] = 720   # 24 * 30
    TICKS_PER_YEAR: ClassVar[int] = 8640  # 24 * 30 * 12

    @property
    def hour(self) -> int:
        return self.tick % self.HOURS_PER_DAY

    @property
    def day(self) -> int:
        return (self.tick // self.TICKS_PER_DAY) % self.DAYS_PER_MONTH + 1

    @property
    def month(self) -> int:
        return (self.tick // self.TICKS_PER_MONTH) % self.MONTHS_PER_YEAR + 1

    @property
    def year(self) -> int:
        return self.tick // self.TICKS_PER_YEAR + 1

    def __str__(self) -> str:
        return f"Y{self.year} M{self.month:02d} D{self.day:02d} H{self.hour:02d}"
