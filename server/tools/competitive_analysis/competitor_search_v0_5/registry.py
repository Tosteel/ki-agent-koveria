from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_search_v0_5 import search_competitors_v0_5
from .models import CompetitorSearchV05Request, CompetitorSearchV05Response


TOOL_NAME = "competitor_search_v0_5"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_search_v0_5(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorSearchV05Request(**args)
        result = search_competitors_v0_5(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            max_candidates_to_check=req.max_candidates_to_check,
            verbose_terminal=req.verbose_terminal,
            verbose_search_hits=req.verbose_search_hits,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorSearchV05Response(competitor_search_results=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_search_v0_5, request_model=CompetitorSearchV05Request)
