from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_trends import run_step_2_4_competitor_trends
from .models import Step24CompetitorTrendsRequest, Step24CompetitorTrendsResponse


TOOL_NAME = "step2_4_competitor_trends"


def register(registry: ToolRegistry) -> None:
    def tool_step2_4_competitor_trends(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step24CompetitorTrendsRequest(**args)
        result = run_step_2_4_competitor_trends(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step24CompetitorTrendsResponse(competitor_trends=result).model_dump()

    registry.register(TOOL_NAME, tool_step2_4_competitor_trends, request_model=Step24CompetitorTrendsRequest)
