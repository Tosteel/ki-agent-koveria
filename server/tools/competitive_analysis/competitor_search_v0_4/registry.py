from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_search_v0_4 import search_competitors_v0_4
from .models import CompetitorSearchV04Request, CompetitorSearchV04Response


TOOL_NAME = "competitor_search_v0_4"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_search_v0_4(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorSearchV04Request(**args)
        result = search_competitors_v0_4(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            max_candidates_to_check=req.max_candidates_to_check,
            use_llm_feature_enrichment=req.use_llm_feature_enrichment,
            llm_min_relevance_for_enrichment=req.llm_min_relevance_for_enrichment,
            include_page_fetch=req.include_page_fetch,
            page_fetch_timeout_s=req.page_fetch_timeout_s,
            page_fetch_max_chars=req.page_fetch_max_chars,
            verbose_terminal=req.verbose_terminal,
            verbose_search_hits=req.verbose_search_hits,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorSearchV04Response(competitor_search_results=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_search_v0_4, request_model=CompetitorSearchV04Request)
