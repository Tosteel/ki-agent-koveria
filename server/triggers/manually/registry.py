from __future__ import annotations

from typing import Any, Dict

from server.triggers.trigger_registry import TriggerRegistry

from .manually import ManuallyTrigger
from .models import ManuallyConfig

TRIGGER_TYPE = "manually"


def register(registry: TriggerRegistry) -> None:
    def factory(config: Dict[str, Any]) -> ManuallyTrigger:
        _ = ManuallyConfig(**config)
        return ManuallyTrigger()

    registry.register(TRIGGER_TYPE, config_model=ManuallyConfig, factory=factory)

