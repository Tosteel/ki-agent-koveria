from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .adaptive_analysis_plan_v0_2 import generate_adaptive_analysis_plan_v0_2
from .models import AdaptiveAnalysisPlanRequest, AdaptiveAnalysisPlanResponse


TOOL_NAME = "competitive_generate_analysis_plan_v0_2"


def register(registry: ToolRegistry) -> None:
    def tool_competitive_generate_analysis_plan_v0_2(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = AdaptiveAnalysisPlanRequest(**args)
        plan = generate_adaptive_analysis_plan_v0_2(
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return AdaptiveAnalysisPlanResponse(analysis_plan=plan).model_dump()

    registry.register(TOOL_NAME, tool_competitive_generate_analysis_plan_v0_2, request_model=AdaptiveAnalysisPlanRequest)
