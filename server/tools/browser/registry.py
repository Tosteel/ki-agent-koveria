from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .browser import browse_website, view_website
from .models import BrowseWebsiteRequest, ViewWebsiteRequest, WebsiteSearchResponse


TOOL_VIEW = "view_website"
TOOL_BROWSE = "browse_website"


def register(registry: ToolRegistry) -> None:
    def tool_view_website(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = ViewWebsiteRequest(**args)
        result = view_website(
            url=req.url,
            query=req.query,
            selector=req.selector,
            max_matches=req.max_matches,
            context_chars=req.context_chars,
            timeout_ms=req.timeout_ms,
        )
        return WebsiteSearchResponse(**result).model_dump()

    def tool_browse_website(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BrowseWebsiteRequest(**args)
        result = browse_website(
            url=req.url,
            query=req.query,
            selector=req.selector,
            max_matches=req.max_matches,
            context_chars=req.context_chars,
            timeout_ms=req.timeout_ms,
            max_pages=req.max_pages,
            click_selectors=req.click_selectors,
            follow_links_matching=req.follow_links_matching,
        )
        return WebsiteSearchResponse(**result).model_dump()

    registry.register(TOOL_VIEW, tool_view_website, request_model=ViewWebsiteRequest)
    registry.register(TOOL_BROWSE, tool_browse_website, request_model=BrowseWebsiteRequest)
