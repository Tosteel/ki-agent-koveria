from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .google_search import search_google_custom
from .models import GoogleSearchRequest, GoogleSearchResponse


TOOL_NAME = "google_search"


def register(registry: ToolRegistry) -> None:
    def tool_google_search(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = GoogleSearchRequest(**args)
        result = search_google_custom(
            query=req.query,
            num=req.num,
            start=req.start,
            gl=req.gl,
            hl=req.hl,
            safe=req.safe,
            site_search=req.site_search,
        )
        return GoogleSearchResponse(**result).model_dump()

    registry.register(
        TOOL_NAME,
        tool_google_search,
        request_model=GoogleSearchRequest,
        response_model=GoogleSearchResponse,
    )
