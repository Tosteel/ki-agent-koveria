from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import Step51RecommendationsRequest, Step51RecommendationsResponse
from .step5_1_recommendations import run_step_5_1_recommendations


TOOL_NAME = "step5_1_recommendations"


def register(registry: ToolRegistry) -> None:
    def tool_step5_1_recommendations(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step51RecommendationsRequest(**args)
        result = run_step_5_1_recommendations(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step51RecommendationsResponse(recommendations=result).model_dump()

    registry.register(TOOL_NAME, tool_step5_1_recommendations, request_model=Step51RecommendationsRequest)

