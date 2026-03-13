from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_profile_structured import run_step_2_3_competitor_profile_structured
from .models import Step23CompetitorProfileStructuredRequest, Step23CompetitorProfileStructuredResponse


TOOL_NAME = "step2_3_competitor_profile_structured"


def register(registry: ToolRegistry) -> None:
    def tool_step2_3_competitor_profile_structured(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step23CompetitorProfileStructuredRequest(**args)
        result = run_step_2_3_competitor_profile_structured(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step23CompetitorProfileStructuredResponse(competitor_profile_structured=result).model_dump()

    registry.register(TOOL_NAME, tool_step2_3_competitor_profile_structured, request_model=Step23CompetitorProfileStructuredRequest)
