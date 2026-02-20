from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

from fastapi import APIRouter
from server.agent.tool_registry import ToolRegistry


def _is_discoverable_dir(path: Path, base_dir: Path) -> bool:
    rel = path.relative_to(base_dir)
    for part in rel.parts:
        if part.startswith("_") or part == "__pycache__":
            return False
    return True


def _module_name_for_file(base_dir: Path, file_path: Path) -> str:
    rel_dir = file_path.parent.relative_to(base_dir)
    if not rel_dir.parts:
        raise ValueError(f"Invalid tool module directory: {file_path.parent}")
    return "server.tools." + ".".join(rel_dir.parts) + f".{file_path.stem}"


def _iter_tool_module_files(base_dir: Path, filename: str) -> list[Path]:
    files: list[Path] = []
    for file_path in sorted(base_dir.rglob(filename), key=lambda p: str(p.relative_to(base_dir))):
        if not file_path.is_file():
            continue
        if not _is_discoverable_dir(file_path.parent, base_dir):
            continue
        files.append(file_path)
    return files


def register_all_tools(registry: ToolRegistry) -> ToolRegistry:
    base_dir = Path(__file__).resolve().parent
    for file_path in _iter_tool_module_files(base_dir, "registry.py"):
        module = importlib.import_module(_module_name_for_file(base_dir, file_path))
        register_fn = getattr(module, "register", None)
        if callable(register_fn):
            register_fn(registry)

    return registry


def create_all_tool_api_router(*, ensure_user_dirs: Callable) -> APIRouter:
    router = APIRouter()
    base_dir = Path(__file__).resolve().parent
    for file_path in _iter_tool_module_files(base_dir, "api.py"):
        module = importlib.import_module(_module_name_for_file(base_dir, file_path))
        create_router_fn = getattr(module, "create_router", None)
        if callable(create_router_fn):
            child_router = create_router_fn(ensure_user_dirs=ensure_user_dirs)
            router.include_router(child_router)

    return router
