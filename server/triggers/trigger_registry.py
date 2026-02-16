from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Protocol, Type

from pydantic import BaseModel


class TriggerInstance(Protocol):
    def poll(self, now_utc: datetime) -> List[Dict[str, Any]]:
        ...


TriggerFactory = Callable[[Dict[str, Any]], TriggerInstance]


@dataclass
class TriggerDef:
    name: str
    config_model: Type[BaseModel]
    factory: TriggerFactory


class TriggerRegistry:
    def __init__(self) -> None:
        self._defs: Dict[str, TriggerDef] = {}

    def register(self, name: str, *, config_model: Type[BaseModel], factory: TriggerFactory) -> None:
        self._defs[name] = TriggerDef(name=name, config_model=config_model, factory=factory)

    def create_instance(self, trigger_type: str, config: Dict[str, Any]) -> TriggerInstance:
        d = self._defs.get(trigger_type)
        if d is None:
            raise ValueError(f"Unknown trigger type: {trigger_type}")
        validated = d.config_model(**(config or {}))
        return d.factory(validated.model_dump())

    def available_types(self) -> List[Dict[str, Any]]:
        return [{"name": d.name, "config_schema": d.config_model.model_json_schema()} for d in self._defs.values()]

