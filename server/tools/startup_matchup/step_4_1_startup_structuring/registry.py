from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import StartupMatchupStep41Request, StartupMatchupStep41Response
from .startup_matchup_step import run_step_41

TOOL_NAME = "startup_matchup_step_4_1_startup_structuring"


def register(registry: ToolRegistry) -> None:
    def tool_handler(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = StartupMatchupStep41Request(**args)
        result = run_step_41(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return StartupMatchupStep41Response(startup_structured_list=result).model_dump()

    registry.register(TOOL_NAME, tool_handler, request_model=StartupMatchupStep41Request)
