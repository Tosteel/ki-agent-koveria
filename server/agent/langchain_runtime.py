from __future__ import annotations

import os
from typing import Any, Dict, TypedDict

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

try:
    from langgraph.graph import StateGraph

    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False
    StateGraph = None  # type: ignore[assignment]


def _requested_runtime() -> str:
    raw = os.getenv("KOVERIA_RUNTIME")
    if raw is not None and str(raw).strip():
        mode = str(raw).strip().lower()
    else:
        # Backward compatibility with existing deployments.
        legacy_flag = os.getenv("KOVERIA_USE_LANGCHAIN")
        if legacy_flag is not None:
            mode = "langchain" if str(legacy_flag).strip().lower() not in {"0", "false", "no", "off"} else "legacy"
        else:
            mode = "langgraph"

    if mode not in {"legacy", "langchain", "langgraph"}:
        return "langgraph"
    return mode


def _effective_runtime() -> str:
    requested = _requested_runtime()
    if requested == "legacy":
        return "legacy"
    if requested == "langgraph":
        if HAS_LANGGRAPH:
            return "langgraph"
        if HAS_LANGCHAIN:
            return "langchain"
        return "legacy"
    if requested == "langchain":
        if HAS_LANGCHAIN:
            return "langchain"
        return "legacy"
    return "legacy"


def planner_runtime_mode() -> str:
    return _effective_runtime()


def tool_dispatch_runtime_mode() -> str:
    return _effective_runtime()


def run_planner_chain(*, llm: Any, goal: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    mode = planner_runtime_mode()
    if mode == "legacy":
        return llm.plan_steps(goal=goal, tool_schema=tool_schema)

    if mode == "langgraph":
        out = _run_planner_langgraph(llm=llm, goal=goal, tool_schema=tool_schema)
    else:
        out = _run_planner_langchain(llm=llm, goal=goal, tool_schema=tool_schema)

    if not isinstance(out, dict):
        return {"steps": []}
    steps = out.get("steps")
    if not isinstance(steps, list):
        return {"steps": []}
    return {"steps": steps}


def _run_planner_langchain(*, llm: Any, goal: str, tool_schema: Dict[str, Any]) -> Any:
    planner_chain = RunnableLambda(
        lambda state: llm.plan_steps(
            goal=str(state.get("goal") or ""),
            tool_schema=state.get("tool_schema") or {},
        )
    )
    return planner_chain.invoke({"goal": goal, "tool_schema": tool_schema})


class _PlannerState(TypedDict, total=False):
    goal: str
    tool_schema: Dict[str, Any]
    plan: Dict[str, Any]


def _run_planner_langgraph(*, llm: Any, goal: str, tool_schema: Dict[str, Any]) -> Any:
    if not HAS_LANGGRAPH:
        return llm.plan_steps(goal=goal, tool_schema=tool_schema)

    def _plan_node(state: _PlannerState) -> Dict[str, Any]:
        return {
            "plan": llm.plan_steps(
                goal=str(state.get("goal") or ""),
                tool_schema=state.get("tool_schema") or {},
            )
        }

    graph_builder = StateGraph(_PlannerState)
    graph_builder.add_node("plan", _plan_node)
    graph_builder.set_entry_point("plan")
    graph_builder.set_finish_point("plan")
    graph = graph_builder.compile()
    out = graph.invoke({"goal": goal, "tool_schema": tool_schema})
    if isinstance(out, dict):
        return out.get("plan")
    return out


def dispatch_tool_chain(*, registry: ToolRegistry, tool_name: str, ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    mode = tool_dispatch_runtime_mode()
    if mode == "legacy":
        return registry.dispatch(tool_name, ctx, args)

    if mode == "langgraph":
        return _dispatch_tool_langgraph(registry=registry, tool_name=tool_name, ctx=ctx, args=args)
    return _dispatch_tool_langchain(registry=registry, tool_name=tool_name, ctx=ctx, args=args)


def _resolve_tool_call(*, registry: ToolRegistry, tool_name: str, args: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
    tool = registry.get_tool(tool_name)
    if tool is None:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

    validated = tool.request_model(**args)
    payload = validated.model_dump()
    return tool, payload


def _dispatch_tool_langchain(*, registry: ToolRegistry, tool_name: str, ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    tool, payload = _resolve_tool_call(registry=registry, tool_name=tool_name, args=args)
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


class _DispatchState(TypedDict, total=False):
    payload: Dict[str, Any]
    result: Any


def _dispatch_tool_langgraph(*, registry: ToolRegistry, tool_name: str, ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    tool, payload = _resolve_tool_call(registry=registry, tool_name=tool_name, args=args)

    if not HAS_LANGGRAPH:
        out = tool.handler(ctx, payload)
        return out if isinstance(out, dict) else {"value": out}

    def _dispatch_node(state: _DispatchState) -> Dict[str, Any]:
        out = tool.handler(ctx, state.get("payload") or {})
        return {"result": out}

    graph_builder = StateGraph(_DispatchState)
    graph_builder.add_node("dispatch", _dispatch_node)
    graph_builder.set_entry_point("dispatch")
    graph_builder.set_finish_point("dispatch")
    graph = graph_builder.compile()

    out_state = graph.invoke({"payload": payload})
    out = out_state.get("result") if isinstance(out_state, dict) else out_state
    if isinstance(out, dict):
        return out
    return {"value": out}
