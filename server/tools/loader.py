from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Iterable

from fastapi import APIRouter
from server.agent.tool_registry import ToolRegistry


def _is_discoverable_dir(path: Path, base_dir: Path) -> bool:
    rel = path.relative_to(base_dir)
    for part in rel.parts:
        if part.startswith("_") or part == "__pycache__":
            return False
    return True


def _module_name_for_file(base_dir: Path, module_prefix: str, file_path: Path) -> str:
    rel_dir = file_path.parent.relative_to(base_dir)
    if not rel_dir.parts:
        raise ValueError(f"Invalid tool module directory: {file_path.parent}")
    return module_prefix + "." + ".".join(rel_dir.parts) + f".{file_path.stem}"


def _module_roots() -> Iterable[tuple[Path, str]]:
    project_server_dir = Path(__file__).resolve().parents[1]
    return (
        (project_server_dir / "tools", "server.tools"),
        (project_server_dir / "workflows", "server.workflows"),
    )


def _iter_tool_module_files(filename: str) -> list[tuple[Path, str, Path]]:
    files: list[tuple[Path, str, Path]] = []
    for base_dir, module_prefix in _module_roots():
        if not base_dir.exists():
            continue
        for file_path in sorted(base_dir.rglob(filename), key=lambda p: str(p.relative_to(base_dir))):
            if not file_path.is_file():
                continue
            if not _is_discoverable_dir(file_path.parent, base_dir):
                continue
            files.append((base_dir, module_prefix, file_path))
    return files


def register_all_tools(registry: ToolRegistry) -> ToolRegistry:
    for base_dir, module_prefix, file_path in _iter_tool_module_files("registry.py"):
        module = importlib.import_module(_module_name_for_file(base_dir, module_prefix, file_path))
        register_fn = getattr(module, "register", None)
        if callable(register_fn):
            register_fn(registry)

    return registry


def create_all_tool_api_router(*, ensure_user_dirs: Callable) -> APIRouter:
    router = APIRouter()
    for base_dir, module_prefix, file_path in _iter_tool_module_files("api.py"):
        module = importlib.import_module(_module_name_for_file(base_dir, module_prefix, file_path))
        create_router_fn = getattr(module, "create_router", None)
        if callable(create_router_fn):
            child_router = create_router_fn(ensure_user_dirs=ensure_user_dirs)
            router.include_router(child_router)

    return router
