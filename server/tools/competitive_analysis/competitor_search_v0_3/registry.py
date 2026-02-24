from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_search_v0_3 import search_competitors_v0_3
from .models import CompetitorSearchRequest, CompetitorSearchResponse


TOOL_NAME = "competitor_search_v0_3"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_search_v0_3(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorSearchRequest(**args)
        result = search_competitors_v0_3(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            product_competitors=req.product_competitors,
            product_competitors_path=req.product_competitors_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            shortlist_size=req.shortlist_size,
            min_relevance_score=req.min_relevance_score,
            verbose_terminal=req.verbose_terminal,
            verbose_search_hits=req.verbose_search_hits,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorSearchResponse(competitor_search_results=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_search_v0_3, request_model=CompetitorSearchRequest)
