from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_search_v0_4 import search_competitors_v0_4
from .models import CompetitorSearchRequest, CompetitorSearchResponse


TOOL_NAME = "competitor_search_v0_4"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_search_v0_4(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorSearchRequest(**args)
        result = search_competitors_v0_4(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            shortlist_size=req.shortlist_size,
            max_candidates_to_check=req.max_candidates_to_check,
            min_relevance_score=req.min_relevance_score,
            search_timeout_ms=req.search_timeout_ms,
            verbose_terminal=req.verbose_terminal,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorSearchResponse(competitor_search_results=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_search_v0_4, request_model=CompetitorSearchRequest)
