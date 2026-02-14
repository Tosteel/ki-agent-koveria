from __future__ import annotations

import importlib
from pathlib import Path

from server.agent.tool_registry import ToolRegistry


def register_all_tools(registry: ToolRegistry) -> ToolRegistry:
    base_dir = Path(__file__).resolve().parent
    for child in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name == "__pycache__":
            continue
        if not (child / "registry.py").exists():
            continue

        module = importlib.import_module(f"server.tools.{child.name}.registry")
        register_fn = getattr(module, "register", None)
        if callable(register_fn):
            register_fn(registry)

    return registry
