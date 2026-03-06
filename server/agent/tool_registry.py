from __future__ import annotations

import json
from dataclasses import field
from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from fastapi import HTTPException
from pydantic import BaseModel

from .policies import tools_allowed


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
    response_model: Optional[Type[BaseModel]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _load_tool_metadata_map() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    base_dir = Path(__file__).resolve().parents[1] / "tools"
    if not base_dir.exists():
        return out

    for path in base_dir.rglob("metadata.json"):
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        tool_name = str(path.parent.name or "").strip()
        if not tool_name:
            continue
        display_name = str(raw.get("name") or tool_name).strip() or tool_name
        out[tool_name] = {
            "name": display_name,
            "description": str(raw.get("description") or "").strip(),
            "input": str(raw.get("input") or "").strip(),
            "output": str(raw.get("output") or "").strip(),
        }
    return out


def _tool_schema_description(tool_name: str, meta: Dict[str, Any]) -> str:
    parts = [f"Tool: {tool_name}"]
    desc = str(meta.get("description") or "").strip()
    inp = str(meta.get("input") or "").strip()
    out = str(meta.get("output") or "").strip()
    if desc:
        parts.append(f"Description: {desc}")
    if inp:
        parts.append(f"Input: {inp}")
    if out:
        parts.append(f"Output: {out}")
    return "\n".join(parts)


def _output_schema_hint(response_model: Optional[Type[BaseModel]]) -> str:
    if response_model is None:
        return ""
    try:
        schema = response_model.model_json_schema()
    except Exception:
        return ""

    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    field_names = [str(k) for k in props.keys()]
    if not field_names:
        return ""
    fields_txt = ", ".join(field_names[:12])
    req_txt = ", ".join(str(x) for x in required[:12]) if required else "-"
    return f"Output fields: {fields_txt}\nRequired output fields: {req_txt}"


def _inline_local_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve local #/$defs/* references into inline schemas."""
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        out = deepcopy(schema)
        if isinstance(out, dict):
            out.pop("$defs", None)
        return out

    def _resolve(node: Any, seen: set[str]) -> Any:
        if isinstance(node, list):
            return [_resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node

        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            key = ref.split("/", 2)[-1]
            if key in seen:
                return {}
            target = defs.get(key)
            if not isinstance(target, dict):
                return {}
            resolved = _resolve(deepcopy(target), seen | {key})
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            if siblings:
                if isinstance(resolved, dict):
                    merged = deepcopy(resolved)
                    for k, v in siblings.items():
                        merged[k] = _resolve(v, seen)
                    return merged
                return {"allOf": [resolved, _resolve(siblings, seen)]}
            return resolved

        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k == "$defs":
                continue
            out[k] = _resolve(v, seen)
        return out

    return _resolve(deepcopy(schema), set())


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        request_model: Type[BaseModel],
        response_model: Optional[Type[BaseModel]] = None,
    ) -> None:
        meta = _load_tool_metadata_map().get(name, {})
        self._tools[name] = ToolDef(
            name=name,
            handler=handler,
            request_model=request_model,
            response_model=response_model,
            metadata=meta,
        )

    def get_tool(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def dispatch(self, name: str, ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.get_tool(name)
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
            if not tools_allowed(tool.name):
                continue
            args_schema = _inline_local_refs(tool.request_model.model_json_schema())
            if isinstance(args_schema, dict) and "additionalProperties" not in args_schema:
                args_schema["additionalProperties"] = False
            desc = _tool_schema_description(tool.name, tool.metadata)
            out_hint = _output_schema_hint(tool.response_model)
            if out_hint:
                desc = f"{desc}\n{out_hint}"

            one_of.append(
                {
                    "type": "object",
                    "description": desc,
                    "additionalProperties": False,
                    "properties": {
                        "tool": {"const": tool.name},
                        "args": args_schema,
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
