from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_identification_v0_2 import identify_competitors_v0_2
from .models import CompetitorIdentificationRequest, CompetitorIdentificationResponse


TOOL_NAME = "competitor_identification_v0_2"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_identification_v0_2(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorIdentificationRequest(**args)
        result = identify_competitors_v0_2(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            shortlist_size=req.shortlist_size,
            min_relevance_score=req.min_relevance_score,
            min_similarity_score=req.min_similarity_score,
            url_priority_weight=req.url_priority_weight,
            datasheet_priority_weight=req.datasheet_priority_weight,
            exclude_below_threshold=req.exclude_below_threshold,
            exhaust_all_attempts=req.exhaust_all_attempts,
            verbose_terminal=req.verbose_terminal,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorIdentificationResponse(competitor_list=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_identification_v0_2, request_model=CompetitorIdentificationRequest)
