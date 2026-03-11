from __future__ import annotations

from typing import Any, Callable, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from server.agent.langchain_runtime import planner_runtime_mode, tool_dispatch_runtime_mode
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
from server.agent.tool_registry import ToolRegistry

security = HTTPBearer(auto_error=False)


def create_agent_router(
    *,
    ensure_user_dirs: Callable[[Settings, str], None],
    build_registry: Callable[[str, Settings], Any],
    append_agent_tools_hint: Callable[[str, Settings, str], str],
    llm_for_provider: Callable[[str], Any],
    run_clarification_gate: Callable[[Any, str], Dict[str, Any]],
    run_planner_guard: Callable[[Any, str, str, List[Dict[str, Any]], ToolRegistry | None], Dict[str, Any]],
    clarification_response: Callable[[str, Dict[str, Any]], AgentAskResponse],
    goal_with_context: Callable[[Any, str, str, Any], str],
    inject_llm_summary_before_pdf: Callable[[List[Dict[str, Any]], str], List[Dict[str, Any]]],
    run_steps_internal: Callable[..., Any],
    finalize_internal: Callable[..., str],
    sanitize_execution_steps: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> APIRouter:
    router = APIRouter()

    def _build_effective_goal(base_goal: str, additional_props: Dict[str, Any], s: Settings, user_id: str) -> str:
        goal = str(base_goal or "").strip()
        props = additional_props if isinstance(additional_props, dict) else {}
        raw_steps = props.get('planned_steps')
        existing_steps = [str(step).strip() for step in (raw_steps or []) if str(step).strip()]
        if existing_steps:
            goal = (
                f"{goal}\n\n"
                'Bestehende PLANNED STEPS (als Grundlage verwenden):\n'
                + '\n'.join(existing_steps)
            )
        guard_missing = [str(x).strip() for x in (props.get('missing') or []) if str(x).strip()]
        guard_reasons = [str(x).strip() for x in (props.get('reasons') or []) if str(x).strip()]
        if guard_missing or guard_reasons:
            lines: List[str] = []
            if guard_missing:
                lines.append("Guard missing:")
                lines.extend(f"- {m}" for m in guard_missing)
            if guard_reasons:
                lines.append("Guard reasons:")
                lines.extend(f"- {r}" for r in guard_reasons)
            goal = f"{goal}\n\nZusatzhinweise fuer Replan:\n" + "\n".join(lines)
        return append_agent_tools_hint(goal, s, user_id)

    def _log_runtime_mode() -> None:
        print("\n===== RUNTIME =====")
        print(f"planner_runtime={planner_runtime_mode()}")
        print(f"tool_dispatch_runtime={tool_dispatch_runtime_mode()}")
        print("===================\n")

    def _apply_planner_guard(
        llm: Any,
        provider: str,
        planner: Planner,
        goal: str,
        steps: List[Dict[str, Any]],
        s: Settings,
        user_id: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        gate = run_planner_guard(llm, provider, goal, steps, planner.registry)
        print('\n===== PLANNER GUARD =====')
        print(f"status={gate.get('status')}")
        print(f"missing={gate.get('missing')}")
        print(f"reasons={gate.get('reasons')}")
        print('=========================\n')
        if gate.get("status") != "replan":
            return steps, gate

        replan_additional_props = {
            "missing": list(gate.get("missing") or []),
            "reasons": list(gate.get("reasons") or []),
        }
        guarded_goal = _build_effective_goal(goal, replan_additional_props, s, user_id)
        replanned = planner.create_steps(goal=guarded_goal)
        replanned = inject_llm_summary_before_pdf(replanned, guarded_goal)
        gate2 = run_planner_guard(llm, provider, goal, replanned, planner.registry)
        print('\n===== PLANNER GUARD (REPLAN) =====')
        print(f"status={gate2.get('status')}")
        print(f"missing={gate2.get('missing')}")
        print(f"reasons={gate2.get('reasons')}")
        print('==================================\n')
        return replanned, gate2

    def _extract_execution_replan_reason(tool_outputs: List[Dict[str, Any]]) -> str:
        for out in reversed(tool_outputs):
            if not isinstance(out, dict):
                continue
            status = str(out.get("status") or "").strip().lower()
            tool = str(out.get("tool") or "").strip()
            if status == "replan_required" or tool == "__replan__":
                return str(out.get("error") or "execution_replan_required").strip() or "execution_replan_required"
        return ""

    @router.post('/agent/run', response_model=AgentRunResponse)
    def agent_run(
        req: AgentRunRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> AgentRunResponse:
        ensure_user_dirs(s, user_id)
        _log_runtime_mode()

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
        _log_runtime_mode()

        registry = build_registry(user_id, s)
        llm = llm_for_provider(req.provider)
        additional_props = req.additional_props if isinstance(req.additional_props, dict) else {}
        effective_goal = _build_effective_goal(req.goal, additional_props, s, user_id)
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
        provider = str(req.provider or "ionos").strip().lower()
        if provider not in {"ionos", "openai"}:
            provider = "ionos"
        llm = llm_for_provider(provider)
        goal_ctx = goal_with_context(llm, provider, req.goal, req.history)
        gate = run_clarification_gate(llm, goal_ctx)

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

    @router.post('/agent/ask', response_model=AgentAskResponse)
    def agent_ask(
        req: AgentAskRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> AgentAskResponse:
        ensure_user_dirs(s, user_id)
        _log_runtime_mode()
        provider = str(req.provider or "ionos").strip().lower()
        if provider not in {"ionos", "openai"}:
            provider = "ionos"
        llm = llm_for_provider(provider)
        goal_ctx = goal_with_context(llm, provider, req.goal, req.history)
        gate = run_clarification_gate(llm, goal_ctx)

        print('\n===== CLARIFICATION =====')
        print(f"provider={provider}")
        print(f"status={gate.get('status')}")
        print(f"normalized_goal={gate.get('normalized_goal')}")
        print(f"missing_fields={gate.get('missing_fields')}")
        print(f"questions={gate.get('questions')}")
        print('=========================\n')

        if gate['status'] == 'needs_info':
            return clarification_response(req.goal, gate)

        effective_goal = str(gate.get('normalized_goal') or req.goal)
        effective_goal = append_agent_tools_hint(effective_goal, s, user_id)
        planner = Planner(llm, build_registry(user_id, s))
        steps = planner.create_steps(goal=effective_goal)
        steps = inject_llm_summary_before_pdf(steps, effective_goal)
        steps, plan_gate = _apply_planner_guard(llm, provider, planner, effective_goal, steps, s, user_id)
        if plan_gate.get("status") != "ready":
            missing = [str(x) for x in (plan_gate.get("missing") or [])]
            questions = [f"Bitte ergänze die Planung: {r}" for r in (plan_gate.get("reasons") or [])]
            if not questions:
                questions = ["Bitte präzisiere das Ziel, damit ein vollständiger Plan erstellt werden kann."]
            return AgentAskResponse(
                ok=False,
                goal=req.goal,
                steps=steps,
                tool_outputs=[],
                answer="Planung unvollständig. Ich brauche eine kurze Präzisierung, bevor ich starte.",
                requires_user_input=True,
                missing_fields=missing,
                questions=questions,
            )

        ok, tool_outputs_full, tool_outputs_compact, fallback_answer = run_steps_internal(
            user_id=user_id,
            settings=s,
            api_key=credentials.credentials,
            goal=effective_goal,
            steps=steps,
            log_label='PLANNED STEPS',
        )
        exec_replan_reason = _extract_execution_replan_reason(tool_outputs_full)
        if exec_replan_reason:
            replan_additional_props = {
                "missing": ["execution_replan"],
                "reasons": [exec_replan_reason],
            }
            repl_goal = _build_effective_goal(effective_goal, replan_additional_props, s, user_id)
            replanner = Planner(llm, build_registry(user_id, s))
            replanned_steps = replanner.create_steps(goal=repl_goal)
            replanned_steps = inject_llm_summary_before_pdf(replanned_steps, repl_goal)
            replanned_steps, repl_gate = _apply_planner_guard(
                llm, provider, replanner, effective_goal, replanned_steps, s, user_id
            )
            if repl_gate.get("status") == "ready":
                ok, tool_outputs_full, tool_outputs_compact, fallback_answer = run_steps_internal(
                    user_id=user_id,
                    settings=s,
                    api_key=credentials.credentials,
                    goal=repl_goal,
                    steps=replanned_steps,
                    log_label='REPLANNED STEPS',
                )
                steps = replanned_steps
                effective_goal = repl_goal
            else:
                missing = [str(x) for x in (repl_gate.get("missing") or [])]
                questions = [f"Bitte ergänze die Planung: {r}" for r in (repl_gate.get("reasons") or [])]
                if not questions:
                    questions = ["Bitte präzisiere das Ziel, damit ein vollständiger Replan erstellt werden kann."]
                return AgentAskResponse(
                    ok=False,
                    goal=req.goal,
                    steps=replanned_steps,
                    tool_outputs=tool_outputs_compact,
                    answer="Die Ausführung konnte nicht robust abgeschlossen werden. Ich brauche eine kurze Präzisierung für den Replan.",
                    requires_user_input=True,
                    missing_fields=missing,
                    questions=questions,
                )

        answer = finalize_internal(provider=provider, goal=effective_goal, tool_outputs_full=tool_outputs_full)
        if not str(answer).strip():
            answer = fallback_answer

        print('\n===== FINAL ANSWER =====')
        print(answer)
        print('========================\n')

        return AgentAskResponse(ok=ok, goal=req.goal, steps=steps, tool_outputs=tool_outputs_compact, answer=answer)

    return router
