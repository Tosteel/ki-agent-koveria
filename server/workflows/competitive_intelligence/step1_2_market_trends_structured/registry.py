from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .market_trends_structured import run_step_1_2_market_trends_structured
from .models import Step12MarketTrendsStructuredRequest, Step12MarketTrendsStructuredResponse


TOOL_NAME = "step1_2_market_trends_structured"


def register(registry: ToolRegistry) -> None:
    def tool_step1_2_market_trends_structured(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step12MarketTrendsStructuredRequest(**args)
        result = run_step_1_2_market_trends_structured(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step12MarketTrendsStructuredResponse(market_trends_structured=result).model_dump()

    registry.register(TOOL_NAME, tool_step1_2_market_trends_structured, request_model=Step12MarketTrendsStructuredRequest)
