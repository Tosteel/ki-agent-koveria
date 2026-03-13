from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .market_trends_raw import run_step_1_1_market_trends_raw
from .models import Step11MarketTrendsRawRequest, Step11MarketTrendsRawResponse


TOOL_NAME = "step1_1_market_trends_raw"


def register(registry: ToolRegistry) -> None:
    def tool_step1_1_market_trends_raw(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        _ = ctx
        req = Step11MarketTrendsRawRequest(**args)
        result = run_step_1_1_market_trends_raw(req=req)
        return Step11MarketTrendsRawResponse(market_trends_raw=result).model_dump()

    registry.register(TOOL_NAME, tool_step1_1_market_trends_raw, request_model=Step11MarketTrendsRawRequest)
