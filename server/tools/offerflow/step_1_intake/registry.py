from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import OfferflowStep1Request
from .offerflow_step import run_step_1

TOOL_NAME = "offerflow_step_1_intake"


def register(registry: ToolRegistry) -> None:
    def tool_handler(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = OfferflowStep1Request(**args)
        return run_step_1(s=ctx.settings, user_id=ctx.user_id, api_key=ctx.api_key, req=req)

    registry.register(TOOL_NAME, tool_handler, request_model=OfferflowStep1Request)
