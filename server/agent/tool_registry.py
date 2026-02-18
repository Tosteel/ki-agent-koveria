from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

from fastapi import HTTPException
from pydantic import BaseModel

from .policies import is_phase1_tool_allowed


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

    def expected_input(self, name: str) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {}
        try:
            schema = tool.request_model.model_json_schema()
        except Exception:
            return {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        required_set = {str(x) for x in required}
        fields: Dict[str, Dict[str, Any]] = {}
        for key, val in props.items():
            if not isinstance(val, dict):
                continue
            v_type = val.get("type")
            if isinstance(v_type, list):
                type_name = "|".join(str(t) for t in v_type)
            else:
                type_name = str(v_type or "")
            fields[str(key)] = {
                "type": type_name,
                "required": str(key) in required_set,
            }
        return {
            "required": [str(x) for x in required],
            "fields": fields,
        }

    def planner_schema(self) -> dict:
        one_of = []

        for tool in self._tools.values():
            if not is_phase1_tool_allowed(tool.name):
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
