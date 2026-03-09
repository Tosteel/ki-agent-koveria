from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import StartupMatchupStep4Request, StartupMatchupStep4Response
from .startup_matchup_step import run_step_4

TOOL_NAME = "startup_matchup_step_4_startup_search"


def register(registry: ToolRegistry) -> None:
    def tool_handler(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = StartupMatchupStep4Request(**args)
        result = run_step_4(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return StartupMatchupStep4Response(startup_candidates_raw=result).model_dump()

    registry.register(TOOL_NAME, tool_handler, request_model=StartupMatchupStep4Request)
