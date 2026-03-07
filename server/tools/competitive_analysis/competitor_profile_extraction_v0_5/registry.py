from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_profile_extraction_v0_5 import extract_competitor_profiles_v0_5
from .models import CompetitorProfileExtractionV05Request, CompetitorProfileExtractionV05Response


TOOL_NAME = "competitor_profile_extraction_v0_5"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_profile_extraction_v0_5(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorProfileExtractionV05Request(**args)
        result = extract_competitor_profiles_v0_5(
            competitor_search_results=req.competitor_search_results,
            competitor_search_results_path=req.competitor_search_results_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_competitors=req.max_competitors,
            exclude_same_manufacturer=req.exclude_same_manufacturer,
            top_n_by_relevance=req.top_n_by_relevance,
            include_page_fetch=req.include_page_fetch,
            page_fetch_timeout_s=req.page_fetch_timeout_s,
            page_fetch_max_chars=req.page_fetch_max_chars,
            verbose_terminal=req.verbose_terminal,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorProfileExtractionV05Response(competitor_profile_results=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_profile_extraction_v0_5, request_model=CompetitorProfileExtractionV05Request)
