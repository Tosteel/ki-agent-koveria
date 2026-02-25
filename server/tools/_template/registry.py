from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import TemplateToolRequest, TemplateToolResponse
from .tool_template import run_tool


# 1) Rename tool name here
TOOL_NAME = "template_tool"


def register(registry: ToolRegistry) -> None:
    # 2) Add auth/user-aware logic via ctx if needed
    def tool_handler(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = TemplateToolRequest(**args)
        result = run_tool(text=req.text, extra=req.extra)
        return TemplateToolResponse(**result).model_dump()

    # 3) Register tool with request/output models used by planner schema
    registry.register(
        TOOL_NAME,
        tool_handler,
        request_model=TemplateToolRequest,
        response_model=TemplateToolResponse,
    )
