from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import Step41InsightsRequest, Step41InsightsResponse
from .step4_1_insights import run_step_4_1_insights


TOOL_NAME = "step4_1_insights"


def register(registry: ToolRegistry) -> None:
    def tool_step4_1_insights(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step41InsightsRequest(**args)
        result = run_step_4_1_insights(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step41InsightsResponse(insights=result).model_dump()

    registry.register(TOOL_NAME, tool_step4_1_insights, request_model=Step41InsightsRequest)

