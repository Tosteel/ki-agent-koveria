from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import StartupMatchupStep5Request, StartupMatchupStep5Response
from .startup_matchup_step import run_step_5

TOOL_NAME = "startup_matchup_step_5_startup_ranking"


def register(registry: ToolRegistry) -> None:
    def tool_handler(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = StartupMatchupStep5Request(**args)
        result = run_step_5(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return StartupMatchupStep5Response(startup_ranked_list=result).model_dump()

    registry.register(TOOL_NAME, tool_handler, request_model=StartupMatchupStep5Request)
