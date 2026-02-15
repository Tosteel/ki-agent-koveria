from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import ListSkillsRequest, ListSkillsResponse
from .skills import list_skills


TOOL_NAME = "list_skills"


def register(registry: ToolRegistry) -> None:
    def tool_list_skills(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = ListSkillsRequest(**args)
        result = list_skills(include_descriptions=req.include_descriptions)
        return ListSkillsResponse(**result).model_dump()

    registry.register(TOOL_NAME, tool_list_skills, request_model=ListSkillsRequest)

