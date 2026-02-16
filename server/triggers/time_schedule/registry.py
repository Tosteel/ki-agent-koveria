from __future__ import annotations

from typing import Any, Dict

from server.triggers.trigger_registry import TriggerRegistry

from .models import TimeScheduleConfig
from .time_schedule import TimeScheduleTrigger


def register(registry: TriggerRegistry) -> None:
    def factory(config: Dict[str, Any]) -> TimeScheduleTrigger:
        req = TimeScheduleConfig(**config)
        return TimeScheduleTrigger(interval_seconds=req.interval_seconds, fire_immediately=req.fire_immediately)

    registry.register("time_schedule", config_model=TimeScheduleConfig, factory=factory)

