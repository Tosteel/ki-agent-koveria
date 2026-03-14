from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .browser import web_crawl_site, web_crawl_site_whitelist, web_fetch_page, web_search_page
from .models import (
    BrowseWebsiteRequest,
    BrowseWhitelistRequest,
    GetWebsiteRequest,
    GetWebsiteResponse,
    ViewWebsiteRequest,
    WebsiteSearchResponse,
)


TOOL_VIEW = "web_search_page"
TOOL_GET = "web_fetch_page"
TOOL_BROWSE = "web_crawl_site"
TOOL_BROWSE_WHITELIST = "web_crawl_site_whitelist"

def register(registry: ToolRegistry) -> None:
    def tool_web_search_page(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = ViewWebsiteRequest(**args)
        result = web_search_page(
            url=req.url,
            query=req.query,
            selector=req.selector,
            max_matches=req.max_matches,
            context_chars=req.context_chars,
            timeout_ms=req.timeout_ms,
            include_full_text=req.include_full_text,
            full_text_max_chars=req.full_text_max_chars,
        )
        return WebsiteSearchResponse(**result).model_dump()

    def tool_web_fetch_page(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = GetWebsiteRequest(**args)
        result = web_fetch_page(
            url=req.url,
            selector=req.selector,
            timeout_ms=req.timeout_ms,
            max_chars=req.max_chars,
            include_image_urls=req.include_image_urls,
        )
        return GetWebsiteResponse(**result).model_dump()

    def tool_web_crawl_site(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BrowseWebsiteRequest(**args)
        result = web_crawl_site(
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

    def tool_web_crawl_site_whitelist(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BrowseWhitelistRequest(**args)
        result = web_crawl_site_whitelist(
            url=req.url,
            query=req.query,
            selector=req.selector,
            max_matches=req.max_matches,
            context_chars=req.context_chars,
            timeout_ms=req.timeout_ms,
            max_pages=req.max_pages,
            click_selectors=req.click_selectors,
            follow_links_matching=req.follow_links_matching,
            allowed_domains=req.allowed_domains,
        )
        return WebsiteSearchResponse(**result).model_dump()

    registry.register(
        TOOL_VIEW,
        tool_web_search_page,
        request_model=ViewWebsiteRequest,
        response_model=WebsiteSearchResponse,
    )
    registry.register(
        TOOL_GET,
        tool_web_fetch_page,
        request_model=GetWebsiteRequest,
        response_model=GetWebsiteResponse,
    )
    registry.register(
        TOOL_BROWSE,
        tool_web_crawl_site,
        request_model=BrowseWebsiteRequest,
        response_model=WebsiteSearchResponse,
    )
    registry.register(
        TOOL_BROWSE_WHITELIST,
        tool_web_crawl_site_whitelist,
        request_model=BrowseWhitelistRequest,
        response_model=WebsiteSearchResponse,
    )
