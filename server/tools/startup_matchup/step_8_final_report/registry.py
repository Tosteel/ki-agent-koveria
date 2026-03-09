from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import StartupMatchupStep8Request, StartupMatchupStep8Response
from .startup_matchup_step import run_step_8

TOOL_NAME = "startup_matchup_step_8_final_report"


def register(registry: ToolRegistry) -> None:
    def tool_handler(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = StartupMatchupStep8Request(**args)
        result = run_step_8(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return StartupMatchupStep8Response(final_report=result).model_dump()

    registry.register(TOOL_NAME, tool_handler, request_model=StartupMatchupStep8Request)
