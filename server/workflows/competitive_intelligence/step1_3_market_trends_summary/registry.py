from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .market_trends_summary import run_step_1_3_market_trends_summary
from .models import Step13MarketTrendsSummaryRequest, Step13MarketTrendsSummaryResponse


TOOL_NAME = "step1_3_market_trends_summary"


def register(registry: ToolRegistry) -> None:
    def tool_step1_3_market_trends_summary(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step13MarketTrendsSummaryRequest(**args)
        result = run_step_1_3_market_trends_summary(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step13MarketTrendsSummaryResponse(market_trends_summary=result).model_dump()

    registry.register(TOOL_NAME, tool_step1_3_market_trends_summary, request_model=Step13MarketTrendsSummaryRequest)
