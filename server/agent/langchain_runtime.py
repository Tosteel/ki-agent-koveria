from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import HTTPException

from .tool_registry import ToolContext, ToolRegistry

try:
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import StructuredTool

    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False
    RunnableLambda = None  # type: ignore[assignment]
    StructuredTool = None  # type: ignore[assignment]


def langchain_enabled() -> bool:
    flag = os.getenv("KOVERIA_USE_LANGCHAIN", "1").strip().lower()
    return HAS_LANGCHAIN and flag not in {"0", "false", "no", "off"}


def planner_runtime_mode() -> str:
    return "langchain" if langchain_enabled() else "legacy"


def tool_dispatch_runtime_mode() -> str:
    return "langchain" if langchain_enabled() else "legacy"


def run_planner_chain(*, llm: Any, goal: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not langchain_enabled():
        return llm.plan_steps(goal=goal, tool_schema=tool_schema)

    planner_chain = RunnableLambda(
        lambda state: llm.plan_steps(
            goal=str(state.get("goal") or ""),
            tool_schema=state.get("tool_schema") or {},
        )
    )
    out = planner_chain.invoke({"goal": goal, "tool_schema": tool_schema})
    if not isinstance(out, dict):
        return {"steps": []}
    steps = out.get("steps")
    if not isinstance(steps, list):
        return {"steps": []}
    return {"steps": steps}


def dispatch_tool_chain(*, registry: ToolRegistry, tool_name: str, ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if not langchain_enabled():
        return registry.dispatch(tool_name, ctx, args)

    tool = registry.get_tool(tool_name)
    if tool is None:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

    validated = tool.request_model(**args)
    payload = validated.model_dump()

    lc_tool = StructuredTool.from_function(
        name=tool.name,
        description=f"Koveria tool: {tool.name}",
        args_schema=tool.request_model,
        func=lambda **kwargs: tool.handler(ctx, kwargs),
    )
    dispatch_chain = RunnableLambda(lambda state: lc_tool.invoke(state))
    out = dispatch_chain.invoke(payload)
    if isinstance(out, dict):
        return out
    return {"value": out}
