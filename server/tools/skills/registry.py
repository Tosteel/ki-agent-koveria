from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import ListSkillsRequest, ListSkillsResponse
from .skills import skills_list


TOOL_NAME = "skills_list"


def register(registry: ToolRegistry) -> None:
    def tool_list_skills(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = ListSkillsRequest(**args)
        result = skills_list(include_descriptions=req.include_descriptions)
        return ListSkillsResponse(**result).model_dump()

    registry.register(
        TOOL_NAME,
        tool_list_skills,
        request_model=ListSkillsRequest,
        response_model=ListSkillsResponse,
    )
