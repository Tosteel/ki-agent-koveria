from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_product_results_v0_6 import build_competitor_product_results_v0_6
from .models import CompetitorProductResultsV06Request, CompetitorProductResultsV06Response


TOOL_NAME = "competitor_product_results_v0_6"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_product_results_v0_6(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorProductResultsV06Request(**args)
        result = build_competitor_product_results_v0_6(
            competitor_search_results=req.competitor_search_results,
            competitor_search_results_path=req.competitor_search_results_path,
            provider=req.provider,
            top_n=req.top_n,
            verbose_terminal=req.verbose_terminal,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorProductResultsV06Response(competitor_product_results=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_product_results_v0_6, request_model=CompetitorProductResultsV06Request)

