from __future__ import annotations

import importlib
from pathlib import Path

from .trigger_registry import TriggerRegistry


def register_all_triggers(registry: TriggerRegistry) -> TriggerRegistry:
    base_dir = Path(__file__).resolve().parent
    for child in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name == "__pycache__":
            continue
        if not (child / "registry.py").exists():
            continue
        module = importlib.import_module(f"server.triggers.{child.name}.registry")
        register_fn = getattr(module, "register", None)
        if callable(register_fn):
            register_fn(registry)
    return registry

