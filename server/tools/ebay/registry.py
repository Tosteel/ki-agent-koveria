from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .ebay import search_ebay
from .models import EbaySearchRequest, EbaySearchResponse


TOOL_NAME = "search_ebay"


def register(registry: ToolRegistry) -> None:
    def tool_search_ebay(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = EbaySearchRequest(**args)
        result = search_ebay(query=req.query, limit=req.limit, sort_order=req.sort_order)
        return EbaySearchResponse(**result).model_dump()

    registry.register(TOOL_NAME, tool_search_ebay, request_model=EbaySearchRequest)
