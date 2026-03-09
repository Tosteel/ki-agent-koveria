from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import StartupMatchupStep3Request, StartupMatchupStep3Response
from .startup_matchup_step import run_step_3

TOOL_NAME = "startup_matchup_step_3_gap_analysis"


def register(registry: ToolRegistry) -> None:
    def tool_handler(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = StartupMatchupStep3Request(**args)
        result = run_step_3(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return StartupMatchupStep3Response(gap_analysis=result).model_dump()

    registry.register(TOOL_NAME, tool_handler, request_model=StartupMatchupStep3Request)
