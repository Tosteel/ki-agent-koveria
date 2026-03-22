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
from .tool_policies import build_tool_policy


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

    def _pick_meta(raw_meta: Dict[str, Any], default_name: str) -> Dict[str, Any]:
        display_name = str(raw_meta.get("name") or default_name).strip() or default_name
        picked: Dict[str, Any] = {
            "name": display_name,
            "description": str(raw_meta.get("description") or "").strip(),
            "input": str(raw_meta.get("input") or "").strip(),
            "output": str(raw_meta.get("output") or "").strip(),
            "version": str(raw_meta.get("version") or "").strip(),
            "owner": str(raw_meta.get("owner") or "").strip(),
        }
        for key in (
            "capabilities",
            "side_effect_level",
            "requires",
            "result_contract",
            "retry_policy",
            "fallback",
            "quality_signals",
            "allows_goal_injection",
        ):
            if key in raw_meta:
                picked[key] = raw_meta.get(key)
        return picked

    project_server_dir = Path(__file__).resolve().parents[1]
    for base_dir in (project_server_dir / "tools", project_server_dir / "workflows"):
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("metadata.json"):
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            tool_name_by_folder = str(path.parent.name or "").strip()
            if not tool_name_by_folder:
                continue

            # Preferred format for multi-tool modules: {"tools": {"tool_name": {...}}}
            raw_tools = raw.get("tools")
            if isinstance(raw_tools, dict):
                base_meta = _pick_meta(raw, tool_name_by_folder)
                base_meta.pop("name", None)
                for tool_name_raw, tool_meta_raw in raw_tools.items():
                    tool_name = str(tool_name_raw).strip()
                    if not tool_name or not isinstance(tool_meta_raw, dict):
                        continue
                    tool_meta = dict(base_meta)
                    tool_meta.update(_pick_meta(tool_meta_raw, tool_name))
                    out[tool_name] = tool_meta
                continue

            out[tool_name_by_folder] = _pick_meta(raw, tool_name_by_folder)
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
        meta = dict(_load_tool_metadata_map().get(name, {}))
        meta["policy"] = build_tool_policy(name, meta)
        self._tools[name] = ToolDef(
            name=name,
            handler=handler,
            request_model=request_model,
            response_model=response_model,
            metadata=meta,
        )

    def get_tool(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def tool_metadata(self, name: str) -> Dict[str, Any]:
        tool = self.get_tool(name)
        if tool is None:
            return {}
        return dict(tool.metadata or {})

    def tool_policy(self, name: str) -> Dict[str, Any]:
        tool = self.get_tool(name)
        if tool is None:
            return build_tool_policy(name, {})
        policy = tool.metadata.get("policy") if isinstance(tool.metadata, dict) else {}
        if isinstance(policy, dict):
            return dict(policy)
        return build_tool_policy(name, tool.metadata if isinstance(tool.metadata, dict) else {})

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

    def planner_schema(self, goal: str = "") -> dict:
        one_of = []

        for tool in self._tools.values():
            if not tools_allowed(tool.name, goal=goal):
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
