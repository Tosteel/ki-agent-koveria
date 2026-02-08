from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from fastapi import HTTPException

from ..core.models import (
    FileReadRequest, FileReadResponse,
    FileWriteRequest, FileWriteResponse,
    RagQueryRequest, RagQueryResponse,
    PdfExportRequest, PdfExportResponse,
)

@dataclass
class ToolContext:
    user_id: str
    settings: Any
    api_key: str  # <- neu

ToolHandler = Callable[[ToolContext, Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        self._tools[name] = handler

    def dispatch(self, name: str, ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {name}")
        return self._tools[name](ctx, args)
