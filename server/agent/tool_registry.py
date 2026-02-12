from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

from fastapi import HTTPException
from pydantic import BaseModel

from .policies import PHASE1_ALLOWED_TOOLS


@dataclass
class ToolContext:
    user_id: str
    settings: Any
    api_key: str
    goal: str = ""


ToolHandler = Callable[[ToolContext, Dict[str, Any]], Dict[str, Any]]


@dataclass
class ToolDef:
    name: str
    handler: ToolHandler
    request_model: Type[BaseModel]


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}

    def register(self, name: str, handler: ToolHandler, *, request_model: Type[BaseModel]) -> None:
        self._tools[name] = ToolDef(name=name, handler=handler, request_model=request_model)

    def dispatch(self, name: str, ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {name}")
        return tool.handler(ctx, args)

    def planner_schema(self) -> dict:
        one_of = []

        for tool in self._tools.values():
            if tool.name not in PHASE1_ALLOWED_TOOLS:
                continue

            one_of.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tool": {"const": tool.name},
                        "args": tool.request_model.model_json_schema(),
                    },
                    "required": ["tool", "args"],
                }
            )

        return {
            "name": "tool_plan",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"oneOf": one_of},
                    }
                },
                "required": ["steps"],
            },
        }
