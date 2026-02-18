from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .langsearch import search_langsearch
from .models import LangSearchRequest, LangSearchResponse


TOOL_NAME = "langsearch"


def register(registry: ToolRegistry) -> None:
    def tool_langsearch(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LangSearchRequest(**args)
        result = search_langsearch(
            query=req.query,
            count=req.count,
            summary=req.summary,
            freshness=req.freshness,
        )
        return LangSearchResponse(**result).model_dump()

    registry.register(TOOL_NAME, tool_langsearch, request_model=LangSearchRequest)

