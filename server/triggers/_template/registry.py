from __future__ import annotations

from typing import Any, Dict

from server.triggers.trigger_registry import TriggerRegistry

from .models import TemplateTriggerConfig
from .trigger_template import TemplateTrigger

TRIGGER_TYPE = "template_trigger"


def register(registry: TriggerRegistry) -> None:
    def factory(config: Dict[str, Any]) -> TemplateTrigger:
        req = TemplateTriggerConfig(**config)
        return TemplateTrigger(interval_seconds=req.interval_seconds)

    registry.register(TRIGGER_TYPE, config_model=TemplateTriggerConfig, factory=factory)

