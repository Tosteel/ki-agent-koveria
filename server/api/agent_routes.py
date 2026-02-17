from __future__ import annotations

from typing import Any, Callable, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from server.core.models import (
    AgentRunRequest,
    AgentRunResponse,
    AgentAskRequest,
    AgentAskResponse,
    AgentPlanRequest,
    AgentPlanResponse,
    AgentClarifyRequest,
    AgentClarifyResponse,
    AgentFinalizeRequest,
    AgentFinalizeResponse,
)
from server.deps import get_current_user, settings as dep_settings
from server.core.settings import Settings
from server.agent.planner import Planner

security = HTTPBearer(auto_error=False)


def create_agent_router(
    *,
    ensure_user_dirs: Callable[[Settings, str], None],
    build_registry: Callable[[], Any],
    llm_for_provider: Callable[[str], Any],
    run_clarification_gate: Callable[[Any, str], Dict[str, Any]],
    clarification_response: Callable[[str, Dict[str, Any]], AgentAskResponse],
    goal_with_context: Callable[[str, Any], str],
    inject_llm_summary_before_pdf: Callable[[List[Dict[str, Any]], str], List[Dict[str, Any]]],
    run_steps_internal: Callable[..., Any],
    finalize_internal: Callable[..., str],
    sanitize_execution_steps: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> APIRouter:
    router = APIRouter()

    @router.post('/agent/run', response_model=AgentRunResponse)
    def agent_run(
        req: AgentRunRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> AgentRunResponse:
        ensure_user_dirs(s, user_id)

        if not req.steps:
            raise HTTPException(status_code=422, detail='steps must not be empty for /agent/run')

        steps = sanitize_execution_steps([step.model_dump() for step in req.steps])
        log_label = (req.log_label or '').strip() or 'PLANNED STEPS'
        ok, _tool_outputs_full, tool_outputs_compact, answer = run_steps_internal(
            user_id=user_id,
            settings=s,
            api_key=credentials.credentials,
            goal='',
            steps=steps,
            log_label=log_label,
        )
        print('\n===== FINAL ANSWER =====')
        print(answer)
        print('========================\n')

        return AgentRunResponse(ok=ok, outputs=tool_outputs_compact)

    @router.post('/agent/plan', response_model=AgentPlanResponse)
    def agent_plan(
        req: AgentPlanRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> AgentPlanResponse:
        ensure_user_dirs(s, user_id)

        registry = build_registry()
        llm = llm_for_provider('ionos')
        effective_goal = req.goal
        additional_props = req.additional_props if isinstance(req.additional_props, dict) else {}
        raw_steps = additional_props.get('planned_steps')
        existing_steps = [str(step).strip() for step in (raw_steps or []) if str(step).strip()]
        if existing_steps:
            effective_goal = (
                f"{req.goal}\n\n"
                'Bestehende PLANNED STEPS (als Grundlage verwenden):\n'
                + '\n'.join(existing_steps)
            )
        planner = Planner(llm, registry)
        steps = planner.create_steps(goal=effective_goal)
        steps = inject_llm_summary_before_pdf(steps, effective_goal)

        print('\n===== PLANNED STEPS =====')
        for i, step in enumerate(steps, 1):
            print(f"{i}. tool={step.get('tool')} args={step.get('args')}")
        print('=========================\n')

        return AgentPlanResponse(
            ok=True,
            goal=req.goal,
            normalized_goal=effective_goal,
            status='ready',
            steps=steps,
            requires_user_input=False,
            missing_fields=[],
            questions=[],
        )

    @router.post('/agent/clarify', response_model=AgentClarifyResponse)
    def agent_clarify(
        req: AgentClarifyRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> AgentClarifyResponse:
        ensure_user_dirs(s, user_id)
        goal_ctx = goal_with_context(req.goal, req.history)
        llm = llm_for_provider(req.provider)
        gate = run_clarification_gate(llm, goal_ctx)

        provider = str(req.provider or '').strip().lower()
        if provider not in {'openai', 'ionos'}:
            provider = 'ionos'

        print('\n===== CLARIFICATION =====')
        print(f'provider={provider}')
        print(f"status={gate.get('status')}")
        print(f"normalized_goal={gate.get('normalized_goal')}")
        print(f"missing_fields={gate.get('missing_fields')}")
        print(f"questions={gate.get('questions')}")
        print('=========================\n')

        if gate['status'] == 'needs_info':
            clar = clarification_response(req.goal, gate)
            return AgentClarifyResponse(
                ok=True,
                goal=req.goal,
                status='needs_info',
                normalized_goal=str(gate.get('normalized_goal') or req.goal),
                requires_user_input=True,
                missing_fields=list(gate.get('missing_fields') or []),
                questions=list(gate.get('questions') or []),
                answer=clar.answer,
            )

        return AgentClarifyResponse(
            ok=True,
            goal=req.goal,
            status='ready',
            normalized_goal=str(gate.get('normalized_goal') or req.goal),
            requires_user_input=False,
            missing_fields=[],
            questions=[],
            answer='',
        )

    @router.post('/agent/finalize', response_model=AgentFinalizeResponse)
    def agent_finalize(
        req: AgentFinalizeRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> AgentFinalizeResponse:
        ensure_user_dirs(s, user_id)
        answer = finalize_internal(provider=req.provider, goal=req.goal, tool_outputs_full=req.tool_outputs)
        print('\n===== FINAL ANSWER =====')
        print(answer)
        print('========================\n')
        return AgentFinalizeResponse(ok=True, goal=req.goal, answer=answer)

    @router.post('/agent/askOpenAI', response_model=AgentAskResponse)
    def agent_askOpenAI(
        req: AgentAskRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> AgentAskResponse:
        ensure_user_dirs(s, user_id)
        goal_ctx = goal_with_context(req.goal, req.history)
        llm = llm_for_provider('openai')
        gate = run_clarification_gate(llm, goal_ctx)
        if gate['status'] == 'needs_info':
            return clarification_response(req.goal, gate)

        effective_goal = str(gate.get('normalized_goal') or req.goal)
        planner = Planner(llm, build_registry())
        steps = planner.create_steps(goal=effective_goal)
        steps = inject_llm_summary_before_pdf(steps, effective_goal)

        ok, tool_outputs_full, tool_outputs_compact, fallback_answer = run_steps_internal(
            user_id=user_id,
            settings=s,
            api_key=credentials.credentials,
            goal=effective_goal,
            steps=steps,
            log_label='PLANNED STEPS',
        )
        answer = finalize_internal(provider='openai', goal=effective_goal, tool_outputs_full=tool_outputs_full)
        if not str(answer).strip():
            answer = fallback_answer

        print('\n===== FINAL ANSWER =====')
        print(answer)
        print('========================\n')

        return AgentAskResponse(ok=ok, goal=req.goal, steps=steps, tool_outputs=tool_outputs_compact, answer=answer)

    @router.post('/agent/askIonos', response_model=AgentAskResponse)
    def agent_askIonos(
        req: AgentAskRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> AgentAskResponse:
        ensure_user_dirs(s, user_id)
        goal_ctx = goal_with_context(req.goal, req.history)
        llm = llm_for_provider('ionos')
        gate = run_clarification_gate(llm, goal_ctx)

        print('\n===== CLARIFICATION =====')
        print(f"status={gate.get('status')}")
        print(f"normalized_goal={gate.get('normalized_goal')}")
        print(f"missing_fields={gate.get('missing_fields')}")
        print(f"questions={gate.get('questions')}")
        print('=========================\n')

        if gate['status'] == 'needs_info':
            return clarification_response(req.goal, gate)

        effective_goal = str(gate.get('normalized_goal') or req.goal)
        planner = Planner(llm, build_registry())
        steps = planner.create_steps(goal=effective_goal)
        steps = inject_llm_summary_before_pdf(steps, effective_goal)

        ok, tool_outputs_full, tool_outputs_compact, fallback_answer = run_steps_internal(
            user_id=user_id,
            settings=s,
            api_key=credentials.credentials,
            goal=effective_goal,
            steps=steps,
            log_label='PLANNED STEPS',
        )
        answer = finalize_internal(provider='ionos', goal=effective_goal, tool_outputs_full=tool_outputs_full)
        if not str(answer).strip():
            answer = fallback_answer

        print('\n===== FINAL ANSWER =====')
        print(answer)
        print('========================\n')

        return AgentAskResponse(ok=ok, goal=req.goal, steps=steps, tool_outputs=tool_outputs_compact, answer=answer)

    return router
