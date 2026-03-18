from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .ebay import ebay_search
from .models import EbaySearchRequest, EbaySearchResponse


TOOL_NAME = "ebay_search"


def register(registry: ToolRegistry) -> None:
    def tool_search_ebay(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = EbaySearchRequest(**args)
        result = ebay_search(query=req.query, limit=req.limit, sort_order=req.sort_order)
        return EbaySearchResponse(**result).model_dump()

    registry.register(
        TOOL_NAME,
        tool_search_ebay,
        request_model=EbaySearchRequest,
        response_model=EbaySearchResponse,
    )
