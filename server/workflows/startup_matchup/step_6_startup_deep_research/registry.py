from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import StartupMatchupStep6Request, StartupMatchupStep6Response
from .startup_matchup_step import run_step_6

TOOL_NAME = "startup_matchup_step_6_startup_deep_research"


def register(registry: ToolRegistry) -> None:
    def tool_handler(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = StartupMatchupStep6Request(**args)
        result = run_step_6(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return StartupMatchupStep6Response(startup_deep_profiles_raw=result).model_dump()

    registry.register(TOOL_NAME, tool_handler, request_model=StartupMatchupStep6Request)
