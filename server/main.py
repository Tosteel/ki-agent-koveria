# uvicorn server.main:app --host 0.0.0.0 --port 8012 --reload
from __future__ import annotations

import json
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pathlib import Path

from .core.logging import setup_logging
from .core.settings import Settings, get_settings
from .core.models import (
    AgentRunRequest, AgentRunResponse,
    AgentAskRequest, AgentAskResponse,
    AgentPlanRequest,
    AgentPlanResponse,
    AgentClarifyRequest, AgentClarifyResponse,
    AgentFinalizeRequest, AgentFinalizeResponse,
)
from .deps import get_current_user, settings as dep_settings
from .auth import get_token_for_user

from .tools.loader import create_all_tool_api_router, register_all_tools
from .triggers import TriggerRegistry, TriggerRuntime, register_all_triggers
from .triggers.store import load_user_triggers, save_user_triggers
from .api.agent_routes import create_agent_router
from .api.trigger_routes import create_trigger_router
from .api.user_routes import router as user_router
from .services import memory_service
from .services import agent_service

from .agent.tool_registry import ToolRegistry, ToolContext
from .agent.orchestrator import Orchestrator
from pydantic import BaseModel, Field

app = FastAPI(title="ki-agent-koveria", version="0.1.0")


class TriggerCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    trigger_type: str = Field(..., min_length=1)
    task_id: int = Field(..., ge=1)
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class TriggerUpdateRequest(BaseModel):
    name: Optional[str] = None
    trigger_type: Optional[str] = None
    task_id: Optional[int] = Field(default=None, ge=1)
    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class TaskMemorySyncRequest(BaseModel):
    tasks: List[Dict[str, Any]] = Field(default_factory=list)


class AgentMemorySyncRequest(BaseModel):
    agents: List[Dict[str, Any]] = Field(default_factory=list)


def _now_iso() -> str:
    return memory_service.now_iso()


def _user_tasks_memory_path(s: Settings, user_id: str) -> Path:
    return s.user_dir(user_id) / "tasks_memory.json"


def _user_agents_memory_path(s: Settings, user_id: str) -> Path:
    return s.user_dir(user_id) / "agents_config.json"


def _normalize_tasks_payload(raw: Any) -> List[Dict[str, Any]]:
    return memory_service.normalize_tasks_payload(raw)


def _load_tasks_memory_for_user(s: Settings, user_id: str) -> Dict[str, Any]:
    return memory_service.load_tasks_memory_for_user(s, user_id)


def _save_tasks_memory_for_user(s: Settings, user_id: str, tasks: List[Dict[str, Any]]) -> None:
    memory_service.save_tasks_memory_for_user(s, user_id, tasks)


def _normalize_agents_payload(raw: Any) -> List[Dict[str, Any]]:
    return memory_service.normalize_agents_payload(raw)


def _load_agents_memory_for_user(s: Settings, user_id: str) -> Dict[str, Any]:
    return memory_service.load_agents_memory_for_user(s, user_id)


def _save_agents_memory_for_user(s: Settings, user_id: str, agents: List[Dict[str, Any]]) -> None:
    memory_service.save_agents_memory_for_user(s, user_id, agents)


def _first_nonempty_str(*values: Any) -> str:
    return agent_service.first_nonempty_str(*values)


def _extract_hit_source(hit: Dict[str, Any]) -> str:
    return agent_service.extract_hit_source(hit)


def _extract_hit_link(hit: Dict[str, Any]) -> str:
    return agent_service.extract_hit_link(hit)


def _extract_hit_text(hit: Dict[str, Any]) -> str:
    return agent_service.extract_hit_text(hit)


def _rag_result_to_text(query: str, rag_result: Dict[str, Any]) -> str:
    return agent_service.rag_result_to_text(query, rag_result)


def _search_result_to_text(user_prompt: str, result: Dict[str, Any]) -> str:
    return agent_service.search_result_to_text(user_prompt, result)


def _wants_summary(goal: str) -> bool:
    return agent_service.wants_summary(goal)


def _rewrite_summarize_to_compose(steps: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return agent_service.rewrite_summarize_to_compose(steps)


def _inject_llm_summary_before_pdf(steps: list[Dict[str, Any]], goal: str) -> list[Dict[str, Any]]:
    return agent_service.inject_llm_summary_before_pdf(steps, goal)


def _compact_tool_outputs(tool_outputs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return agent_service.compact_tool_outputs(tool_outputs)


def _sanitize_execution_steps(steps: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return agent_service.sanitize_execution_steps(steps)


def _outputs_for_final_answer(tool_outputs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return agent_service.outputs_for_final_answer(tool_outputs)


def _extract_execution_answer(tool_outputs: list[Dict[str, Any]]) -> str:
    return agent_service.extract_execution_answer(tool_outputs)


def _run_clarification_gate(llm: Any, goal: str) -> Dict[str, Any]:
    return agent_service.run_clarification_gate(llm, goal)


def _run_planner_guard(llm: Any, provider: str, goal: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return agent_service.run_planner_guard(llm, provider, goal, steps)


def _clarification_response(req_goal: str, gate: Dict[str, Any]) -> AgentAskResponse:
    questions = [str(q).strip() for q in (gate.get("questions") or []) if str(q).strip()]
    if not questions:
        questions = ["Welche Informationen fehlen genau, damit ich starten kann?"]
    answer = "Bevor ich starte, brauche ich noch:\n" + "\n".join(f"- {q}" for q in questions)
    return AgentAskResponse(
        ok=True,
        goal=req_goal,
        steps=[],
        tool_outputs=[],
        answer=answer,
        requires_user_input=True,
        missing_fields=list(gate.get("missing_fields") or []),
        questions=questions,
    )


def _history_to_context(history: Any, max_items: int = 12) -> str:
    return agent_service.history_to_context(history, max_items=max_items)


def _goal_with_context(llm: Any, provider: str, goal: str, history: Any) -> str:
    return agent_service.build_goal_with_context(llm, provider, goal, history)

from .services.llm_openai import LlmOpenai
from server.services.llm_ionos import IonosLLM

from .agent.planner import Planner

from dotenv import load_dotenv
load_dotenv()

@app.on_event("startup")
def _startup() -> None:
    setup_logging()
    s = get_settings()
    trigger_registry = _build_trigger_registry()
    runtime = TriggerRuntime(settings=s, registry=trigger_registry, step_executor=_execute_steps_for_trigger, poll_seconds=1.0)
    runtime.start()
    app.state.trigger_runtime = runtime


@app.on_event("shutdown")
def _shutdown() -> None:
    runtime = getattr(app.state, "trigger_runtime", None)
    if runtime is not None:
        runtime.stop()

def _ensure_user_dirs(s: Settings, user_id: str) -> None:
    memory_service.ensure_user_dirs(s, user_id)

app.include_router(user_router)
app.include_router(
    create_all_tool_api_router(
        ensure_user_dirs=_ensure_user_dirs,
    )
)


@app.get("/tasks/memory")
def get_tasks_memory(
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    memory = _load_tasks_memory_for_user(s, user_id)
    return {"user_id": user_id, **memory}


@app.post("/tasks/memory/sync")
def sync_tasks_memory(
    req: TaskMemorySyncRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    tasks = _normalize_tasks_payload(req.tasks)
    _save_tasks_memory_for_user(s, user_id, tasks)
    return {"ok": True, "user_id": user_id, "count": len(tasks)}


@app.get("/agents/memory")
def get_agents_memory(
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    memory = _load_agents_memory_for_user(s, user_id)
    return {"user_id": user_id, **memory}


@app.post("/agents/memory/sync")
def sync_agents_memory(
    req: AgentMemorySyncRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    agents = _normalize_agents_payload(req.agents)
    _save_agents_memory_for_user(s, user_id, agents)
    return {"ok": True, "user_id": user_id, "count": len(agents)}


def _get_trigger_runtime() -> TriggerRuntime:
    runtime = getattr(app.state, "trigger_runtime", None)
    if runtime is None:
        raise RuntimeError("trigger runtime not initialized")
    return runtime


# ----------------------------- Phase 1: Agent (Tool Registry + Orchestrator) -----------------------------
def _build_registry(user_id: str, s: Settings) -> ToolRegistry:
    return agent_service.build_registry(settings=s, user_id=user_id)


def _build_trigger_registry() -> TriggerRegistry:
    registry = TriggerRegistry()
    return register_all_triggers(registry)


def _execute_steps_for_trigger(user_id: str, steps: List[Dict[str, Any]], goal: str) -> List[Dict[str, Any]]:
    s = get_settings()
    _ensure_user_dirs(s, user_id)
    registry = _build_registry(user_id, s)
    orch = Orchestrator(registry)
    token = get_token_for_user(user_id)
    ctx = ToolContext(user_id=user_id, settings=s, api_key=token, goal=goal)
    return orch.run_steps(ctx, steps)


def _provider_key(provider: str) -> str:
    return agent_service.provider_key(provider)


def _llm_for_provider(provider: str) -> Any:
    return agent_service.llm_for_provider(provider)


def _run_steps_internal(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    goal: str,
    steps: List[Dict[str, Any]],
    log_label: str = "PLANNED STEPS",
) -> tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]], str]:
    return agent_service.run_steps_internal(
        user_id=user_id,
        settings=settings,
        api_key=api_key,
        goal=goal,
        steps=steps,
        log_label=log_label,
    )


def _append_agent_tools_hint(goal: str, s: Settings, user_id: str) -> str:
    return agent_service.append_agent_tools_hint(goal, s, user_id)


def _finalize_internal(*, provider: str, goal: str, tool_outputs_full: List[Dict[str, Any]]) -> str:
    return agent_service.finalize_internal(provider=provider, goal=goal, tool_outputs_full=tool_outputs_full)


app.include_router(
    create_trigger_router(
        ensure_user_dirs=_ensure_user_dirs,
        build_trigger_registry=_build_trigger_registry,
        load_user_triggers=load_user_triggers,
        save_user_triggers=save_user_triggers,
        now_iso=_now_iso,
        get_trigger_runtime=_get_trigger_runtime,
    )
)



app.include_router(
    create_agent_router(
        ensure_user_dirs=_ensure_user_dirs,
        build_registry=_build_registry,
        append_agent_tools_hint=_append_agent_tools_hint,
        llm_for_provider=_llm_for_provider,
        run_clarification_gate=_run_clarification_gate,
        run_planner_guard=_run_planner_guard,
        clarification_response=_clarification_response,
        goal_with_context=_goal_with_context,
        inject_llm_summary_before_pdf=_inject_llm_summary_before_pdf,
        run_steps_internal=_run_steps_internal,
        finalize_internal=_finalize_internal,
        sanitize_execution_steps=_sanitize_execution_steps,
    )
)
