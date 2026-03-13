from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_profile_raw import run_step_2_2_competitor_profile_raw
from .models import Step22CompetitorProfileRawRequest, Step22CompetitorProfileRawResponse


TOOL_NAME = "step2_2_competitor_profile_raw"


def register(registry: ToolRegistry) -> None:
    def tool_step2_2_competitor_profile_raw(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step22CompetitorProfileRawRequest(**args)
        result = run_step_2_2_competitor_profile_raw(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step22CompetitorProfileRawResponse(competitor_profile_raw=result).model_dump()

    registry.register(TOOL_NAME, tool_step2_2_competitor_profile_raw, request_model=Step22CompetitorProfileRawRequest)
