from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_identification import identify_competitors
from .models import CompetitorIdentificationRequest, CompetitorIdentificationResponse


TOOL_NAME = "competitive_identify_competitors"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_identification(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorIdentificationRequest(**args)
        result = identify_competitors(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            shortlist_size=req.shortlist_size,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorIdentificationResponse(competitor_list=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_identification, request_model=CompetitorIdentificationRequest)
