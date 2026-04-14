from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResourceType(Enum):
    FOOD = "food"
    WOOD = "wood"


@dataclass
class ResourceConfig:
    resource_type: ResourceType
    starting_value: float
    max_value: float
    growth_rate: float          # amount added per tick (healthy)
    depleted_growth_rate: float # amount added per tick (depleted)
    depletion_duration: int     # ticks the slow-recovery penalty lasts
    harvest_yield: float = 0.0  # max units an agent can take per action (0 = no cap)
    tradeable: bool = True      # can this resource be offered in a trade?


@dataclass
class ResourceTile:
    resource_type: ResourceType
    config: ResourceConfig
    amount: float
    depleted_ticks_remaining: int = field(default=0)

    @property
    def is_depleted(self) -> bool:
        return self.depleted_ticks_remaining > 0

    def tick(self) -> None:
        if self.depleted_ticks_remaining > 0:
            rate = self.config.depleted_growth_rate
            self.depleted_ticks_remaining -= 1
        else:
            rate = self.config.growth_rate
        self.amount = min(self.amount + rate, self.config.max_value)

    def harvest(self, requested: float) -> float:
        actual = min(requested, self.amount)
        self.amount -= actual
        if self.amount <= 0:
            self.amount = 0.0
            self.depleted_ticks_remaining = self.config.depletion_duration
        return actual


DEFAULT_CONFIGS: dict[ResourceType, ResourceConfig] = {
    ResourceType.FOOD: ResourceConfig(
        resource_type=ResourceType.FOOD,
        starting_value=5.0,
        max_value=10.0,
        growth_rate=0.5,
        depleted_growth_rate=0.25,
        depletion_duration=5,
        harvest_yield=5.0,
    ),
    ResourceType.WOOD: ResourceConfig(
        resource_type=ResourceType.WOOD,
        starting_value=8.0,
        max_value=15.0,
        growth_rate=0.3,
        depleted_growth_rate=0.15,
        depletion_duration=8,
        harvest_yield=3.0,
    ),
}
