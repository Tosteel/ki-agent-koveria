from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, TypedDict

from pydantic import ValidationError

from ..agent.langchain_runtime import dispatch_tool_chain, tool_dispatch_runtime_mode
from ..agent.models import AgentStep
from ..agent.policies import tools_allowed
from ..agent.tool_policies import classify_error_kind, fallback_candidates_for_capabilities
from ..agent.tool_registry import ToolContext, ToolRegistry

try:
    from langgraph.graph import END, StateGraph

    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False
    END = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


class _WorkflowState(TypedDict, total=False):
    steps: List[Dict[str, Any]]
    index: int
    payload: Dict[str, Any]
    outputs: List[Dict[str, Any]]
    retry_counts: Dict[str, int]
    fallback_history: List[str]
    pending_fallback_step: Dict[str, Any]
    pending_fallback_key: str
    decision: str
    decision_reason: str
    replan_required: bool
    replan_reason: str


class Orchestrator:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run_steps(self, ctx: ToolContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if tool_dispatch_runtime_mode() == "langgraph" and HAS_LANGGRAPH:
            return self._run_steps_langgraph(ctx, steps)
        return self._run_steps_legacy(ctx, steps)

    def _run_steps_legacy(self, ctx: ToolContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        payload: Dict[str, Any] = {}

        for i, raw_step in enumerate(steps, start=1):
            payload, entry = self._execute_once(
                step_no=i,
                raw_step=raw_step,
                ctx=ctx,
                payload=payload,
                outputs=outputs,
                fallback_active=False,
            )
            outputs.append(entry)
            self._log_step_output(entry)

        return outputs

    def _run_steps_langgraph(self, ctx: ToolContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not HAS_LANGGRAPH:
            return self._run_steps_legacy(ctx, steps)

        def _execute_step(state: _WorkflowState) -> Dict[str, Any]:
            idx = int(state.get("index") or 0)
            raw_steps = list(state.get("steps") or [])
            if idx < 0 or idx >= len(raw_steps):
                return {"decision": "end", "decision_reason": "steps_exhausted"}

            payload = dict(state.get("payload") or {})
            outputs = list(state.get("outputs") or [])
            payload, entry = self._execute_once(
                step_no=idx + 1,
                raw_step=raw_steps[idx],
                ctx=ctx,
                payload=payload,
                outputs=outputs,
                fallback_active=bool(state.get("fallback_history")),
            )
            outputs.append(entry)
            self._log_step_output(entry)
            return {"payload": payload, "outputs": outputs}

        def _evaluate_result(state: _WorkflowState) -> Dict[str, Any]:
            idx = int(state.get("index") or 0)
            steps_now = list(state.get("steps") or [])
            outputs = list(state.get("outputs") or [])
            retry_counts = dict(state.get("retry_counts") or {})
            fallback_history = list(state.get("fallback_history") or [])
            if idx < 0 or idx >= len(steps_now):
                return {"decision": "end", "decision_reason": "steps_exhausted"}
            if not outputs:
                return {"decision": "end", "decision_reason": "no_outputs"}

            entry = outputs[-1] if isinstance(outputs[-1], dict) else {}
            tool = str(entry.get("tool") or "").strip()
            status = str(entry.get("status") or ("success" if entry.get("ok") else "permanent_error")).strip()
            envelope = entry.get("envelope") if isinstance(entry.get("envelope"), dict) else {}
            policy = self.registry.tool_policy(tool)
            step_args = envelope.get("step_args") if isinstance(envelope.get("step_args"), dict) else {}
            error_kind = str(envelope.get("error_kind") or "")
            error_text = str(entry.get("error") or "").lower()

            if status == "success":
                return {"decision": "next", "decision_reason": "success"}

            retry_key = f"{idx}:{tool}"
            retry_policy = policy.get("retry_policy") if isinstance(policy.get("retry_policy"), dict) else {}
            max_retries = int(retry_policy.get("max_retries") or 0)
            retry_on = [str(x).strip().lower() for x in (retry_policy.get("retry_on") or []) if str(x).strip()]
            current_retries = int(retry_counts.get(retry_key) or 0)

            if status == "transient_error" and current_retries < max_retries:
                err_token = error_kind.lower().strip()
                retry_match = any(token in error_text for token in retry_on)
                if not retry_on or err_token in retry_on or "transient_error" in retry_on or retry_match:
                    return {
                        "decision": "retry",
                        "decision_reason": f"retry:{tool}:{current_retries + 1}/{max_retries}",
                    }

            fallback_step, fallback_key = self._select_fallback_step(
                tool=tool,
                step_args=step_args,
                goal=getattr(ctx, "goal", "") or "",
                status=status,
                policy=policy,
                fallback_history=fallback_history,
            )
            if fallback_step is not None:
                return {
                    "decision": "fallback",
                    "decision_reason": f"fallback:{tool}->{fallback_step.get('tool')}",
                    "pending_fallback_step": fallback_step,
                    "pending_fallback_key": fallback_key,
                }

            caps = [str(c).strip() for c in (policy.get("capabilities") or []) if str(c).strip()]
            needs_replan = status in {"empty", "low_quality", "transient_error", "permanent_error"} and (
                any(c in {"knowledge_search", "web_search"} for c in caps)
                or str(policy.get("side_effect_level") or "") == "high"
                or tool in {"mail_send", "mail_answer"}
            )
            if needs_replan:
                reason = str(entry.get("error") or f"{tool}:{status}")
                return {
                    "decision": "replan",
                    "decision_reason": f"replan_required:{tool}:{status}:{reason}",
                }

            # Keep backward compatibility for non-critical tools: continue with next step.
            return {"decision": "next", "decision_reason": f"continue_on_{status}"}

        def _advance_index(state: _WorkflowState) -> Dict[str, Any]:
            idx = int(state.get("index") or 0)
            return {"index": idx + 1}

        def _retry_step(state: _WorkflowState) -> Dict[str, Any]:
            idx = int(state.get("index") or 0)
            steps_now = list(state.get("steps") or [])
            retry_counts = dict(state.get("retry_counts") or {})
            outputs = list(state.get("outputs") or [])
            if idx < 0 or idx >= len(steps_now):
                return {}
            step_raw = steps_now[idx] if isinstance(steps_now[idx], dict) else {}
            tool = str(step_raw.get("tool") or "").strip()
            policy = self.registry.tool_policy(tool)
            retry_policy = policy.get("retry_policy") if isinstance(policy.get("retry_policy"), dict) else {}
            retry_key = f"{idx}:{tool}"
            retry_counts[retry_key] = int(retry_counts.get(retry_key) or 0) + 1

            backoff_ms = int(retry_policy.get("backoff_ms") or 0)
            if backoff_ms > 0:
                time.sleep(min(backoff_ms / 1000.0, 2.0))
            if outputs and isinstance(outputs[-1], dict):
                outputs[-1]["handled"] = True
            return {"retry_counts": retry_counts, "outputs": outputs}

        def _insert_fallback(state: _WorkflowState) -> Dict[str, Any]:
            idx = int(state.get("index") or 0)
            steps_now = list(state.get("steps") or [])
            outputs = list(state.get("outputs") or [])
            fallback_step = state.get("pending_fallback_step") if isinstance(state.get("pending_fallback_step"), dict) else None
            fallback_key = str(state.get("pending_fallback_key") or "").strip()
            if fallback_step is None or idx < 0 or idx >= len(steps_now):
                return {}

            updated_steps = list(steps_now)
            insert_at = idx + 1
            updated_steps.insert(insert_at, fallback_step)
            fallback_history = list(state.get("fallback_history") or [])
            if fallback_key:
                fallback_history.append(fallback_key)
            if outputs and isinstance(outputs[-1], dict):
                outputs[-1]["handled"] = True
            return {
                "steps": updated_steps,
                "index": insert_at,
                "fallback_history": fallback_history,
                "pending_fallback_step": {},
                "pending_fallback_key": "",
                "outputs": outputs,
            }

        def _mark_replan(state: _WorkflowState) -> Dict[str, Any]:
            outputs = list(state.get("outputs") or [])
            idx = int(state.get("index") or 0)
            reason = str(state.get("decision_reason") or "replan_required").strip() or "replan_required"
            payload = dict(state.get("payload") or {})
            entry = {
                "step": idx + 1,
                "tool": "__replan__",
                "ok": False,
                "status": "replan_required",
                "error": reason,
                "payload": payload,
                "envelope": {"status": "replan_required", "reason": reason},
            }
            outputs.append(entry)
            self._log_step_output(entry)
            return {"outputs": outputs, "replan_required": True, "replan_reason": reason}

        def _after_adjustment_route(state: _WorkflowState) -> str:
            if bool(state.get("replan_required")):
                return "end"
            idx = int(state.get("index") or 0)
            raw_steps = state.get("steps") or []
            return "continue" if idx < len(raw_steps) else "end"

        def _decision_route(state: _WorkflowState) -> str:
            decision = str(state.get("decision") or "").strip().lower()
            if decision in {"next", "retry", "fallback", "replan", "end"}:
                return decision
            return "next"

        graph_builder = StateGraph(_WorkflowState)
        graph_builder.add_node("execute_step", _execute_step)
        graph_builder.add_node("evaluate_result", _evaluate_result)
        graph_builder.add_node("advance_index", _advance_index)
        graph_builder.add_node("retry_step", _retry_step)
        graph_builder.add_node("insert_fallback", _insert_fallback)
        graph_builder.add_node("mark_replan", _mark_replan)

        graph_builder.set_entry_point("execute_step")
        graph_builder.add_edge("execute_step", "evaluate_result")
        graph_builder.add_conditional_edges(
            "evaluate_result",
            _decision_route,
            {
                "next": "advance_index",
                "retry": "retry_step",
                "fallback": "insert_fallback",
                "replan": "mark_replan",
                "end": END,
            },
        )
        graph_builder.add_conditional_edges(
            "advance_index",
            _after_adjustment_route,
            {
                "continue": "execute_step",
                "end": END,
            },
        )
        graph_builder.add_conditional_edges(
            "retry_step",
            _after_adjustment_route,
            {
                "continue": "execute_step",
                "end": END,
            },
        )
        graph_builder.add_conditional_edges(
            "insert_fallback",
            _after_adjustment_route,
            {
                "continue": "execute_step",
                "end": END,
            },
        )
        graph_builder.add_edge("mark_replan", END)

        graph = graph_builder.compile()
        out = graph.invoke(
            {
                "steps": steps,
                "index": 0,
                "payload": {},
                "outputs": [],
                "retry_counts": {},
                "fallback_history": [],
                "pending_fallback_step": {},
                "pending_fallback_key": "",
                "decision": "",
                "decision_reason": "",
                "replan_required": False,
                "replan_reason": "",
            }
        )

        if isinstance(out, dict) and isinstance(out.get("outputs"), list):
            return out.get("outputs") or []
        return []

    def _execute_once(
        self,
        *,
        step_no: int,
        raw_step: Dict[str, Any],
        ctx: ToolContext,
        payload: Dict[str, Any],
        outputs: List[Dict[str, Any]],
        fallback_active: bool = False,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            step = AgentStep.model_validate(raw_step).model_dump()
        except ValidationError as e:
            entry = {
                "step": step_no,
                "tool": str((raw_step or {}).get("tool") if isinstance(raw_step, dict) else ""),
                "ok": False,
                "status": "permanent_error",
                "error": f"invalid_step_schema: {e.errors()}",
                "payload": payload,
                "envelope": {"status": "permanent_error", "error_kind": "invalid_step_schema", "step_args": {}},
            }
            return payload, entry

        tool = (step.get("tool") or "").strip()
        policy = self.registry.tool_policy(tool)
        args = self._merge_with_payload(step.get("args") or {}, payload)
        goal = (getattr(ctx, "goal", "") or "").strip()
        if goal and "goal" not in args and bool(policy.get("allows_goal_injection", True)):
            args["goal"] = goal
        args = self._resolve_placeholders(args, outputs, payload)
        args = self._promote_legacy_result_refs_after_fallback(
            args,
            outputs=outputs,
            payload=payload,
            fallback_active=fallback_active,
        )
        expected = self.registry.expected_input(tool)
        self._log_step_input(step_no, tool, args, expected)

        if not tools_allowed(tool, goal=goal):
            entry = {
                "step": step_no,
                "tool": tool,
                "ok": False,
                "status": "permanent_error",
                "error": "tool_not_allowed",
                "payload": payload,
                "envelope": {
                    "status": "permanent_error",
                    "error_kind": "tool_not_allowed",
                    "capabilities": list(policy.get("capabilities") or []),
                    "step_args": args,
                },
            }
            return payload, entry

        gate_reason = self._side_effect_gate_reason(tool=tool, args=args, outputs=outputs, policy=policy)
        if gate_reason:
            entry = {
                "step": step_no,
                "tool": tool,
                "ok": False,
                "status": "permanent_error",
                "error": f"side_effect_gate_blocked: {gate_reason}",
                "payload": payload,
                "envelope": {
                    "status": "permanent_error",
                    "error_kind": "side_effect_gate_blocked",
                    "capabilities": list(policy.get("capabilities") or []),
                    "quality_ok": False,
                    "step_args": args,
                },
            }
            return payload, entry

        try:
            res = dispatch_tool_chain(registry=self.registry, tool_name=tool, ctx=ctx, args=args)
            new_payload = self._as_payload(res, step_no, tool)
            envelope = self._build_success_envelope(tool=tool, args=args, result=res, policy=policy)
            status = str(envelope.get("status") or "success")
            entry = {
                "step": step_no,
                "tool": tool,
                "ok": status == "success",
                "status": status,
                "result": res,
                "payload": new_payload,
                "envelope": envelope,
            }
            if status != "success":
                entry["error"] = str(envelope.get("reason") or status)
            return new_payload, entry
        except Exception as e:
            err = str(e)
            status = classify_error_kind(err)
            if status not in {"transient_error", "permanent_error"}:
                status = "permanent_error"
            entry = {
                "step": step_no,
                "tool": tool,
                "ok": False,
                "status": status,
                "error": err,
                "payload": payload,
                "envelope": {
                    "status": status,
                    "error_kind": classify_error_kind(err),
                    "capabilities": list(policy.get("capabilities") or []),
                    "step_args": args,
                },
            }
            return payload, entry

    def _build_success_envelope(
        self,
        *,
        tool: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = result if isinstance(result, dict) else {"value": result}
        capabilities = [str(c).strip() for c in (policy.get("capabilities") or []) if str(c).strip()]
        contract = policy.get("result_contract") if isinstance(policy.get("result_contract"), dict) else {}
        quality = policy.get("quality_signals") if isinstance(policy.get("quality_signals"), dict) else {}

        hit_count = self._count_items(payload)
        has_sources = self._has_sources(payload)
        text = self._extract_primary_text(payload)
        text_len = len(text)

        has_signal = self._has_any_success_signal(payload, list(contract.get("success_if_any") or []))
        if not has_signal:
            has_signal = bool(hit_count > 0 or text_len > 0)
        if payload.get("sent") is True:
            has_signal = True

        status = "success"
        reason = ""
        if not has_signal:
            status = "empty"
            reason = "empty_result"

        min_hits = int(quality.get("min_hits") or 0)
        require_sources = bool(quality.get("require_sources"))
        min_text_length = int(quality.get("min_text_length") or 0)
        is_search_capability = any(cap in {"knowledge_search", "web_search"} for cap in capabilities)
        quality_ok = True
        if min_hits > 0 and hit_count < min_hits:
            quality_ok = False
        if require_sources and not has_sources:
            quality_ok = False
        if min_text_length > 0 and ((not is_search_capability) or hit_count <= 0) and text_len < min_text_length:
            quality_ok = False
        if payload.get("sent") is True:
            quality_ok = True

        if status == "success" and not quality_ok:
            status = "low_quality"
            reason = "quality_threshold_not_met"

        return {
            "status": status,
            "reason": reason,
            "quality_ok": quality_ok,
            "capabilities": capabilities,
            "metrics": {
                "items": hit_count,
                "has_sources": has_sources,
                "text_length": text_len,
            },
            "step_args": args,
        }

    def _select_fallback_step(
        self,
        *,
        tool: str,
        step_args: Dict[str, Any],
        goal: str,
        status: str,
        policy: Dict[str, Any],
        fallback_history: List[str],
    ) -> tuple[Dict[str, Any] | None, str]:
        fallback = policy.get("fallback") if isinstance(policy.get("fallback"), dict) else {}
        current_capabilities = [str(c).strip() for c in (policy.get("capabilities") or []) if str(c).strip()]
        candidates: List[str] = []
        for name in (fallback.get("fallback_candidates") or []):
            candidate = str(name).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        cap_key = "on_empty_capabilities" if status in {"empty", "low_quality"} else "on_transient_error_capabilities"
        capability_targets = [str(x).strip() for x in (fallback.get(cap_key) or []) if str(x).strip()]
        if not capability_targets:
            # Auto-fallback based on the tool's own capabilities if no explicit fallback policy exists.
            capability_targets = list(current_capabilities)

        if tool == "web_search_page" and "web_crawl_site" not in candidates:
            candidates.append("web_crawl_site")
        elif tool == "web_crawl_site" and "web_search_page" not in candidates:
            candidates.append("web_search_page")

        for candidate in fallback_candidates_for_capabilities(capability_targets):
            if candidate not in candidates:
                candidates.append(candidate)
        for reg_tool_name in self.registry.tool_names():
            if reg_tool_name in candidates:
                continue
            reg_caps = [str(c).strip() for c in (self.registry.tool_policy(reg_tool_name).get("capabilities") or []) if str(c).strip()]
            if any(cap in capability_targets for cap in reg_caps):
                candidates.append(reg_tool_name)

        for candidate in candidates:
            if candidate == tool or self.registry.get_tool(candidate) is None:
                continue
            key = f"{tool}->{candidate}"
            if key in fallback_history:
                continue
            fallback_args = self._fallback_args(candidate=candidate, source_args=step_args, goal=goal)
            if fallback_args is None:
                continue
            return {"tool": candidate, "args": fallback_args}, key

        return None, ""

    @staticmethod
    def _fallback_args(candidate: str, source_args: Dict[str, Any], goal: str) -> Dict[str, Any] | None:
        prompt = (
            str(source_args.get("query") or "").strip()
            or str(source_args.get("user_prompt") or "").strip()
            or str(source_args.get("text") or "").strip()
            or str(goal or "").strip()
        )
        if candidate in {"websearch_table", "search_multitable"}:
            return {"user_prompt": prompt} if prompt else None
        if candidate == "langsearch":
            return {"query": prompt} if prompt else None
        if candidate == "rag_knowledgebase":
            return {"query": prompt} if prompt else None
        if candidate in {"web_search_page", "web_crawl_site", "web_crawl_site_whitelist"}:
            url = str(source_args.get("url") or "").strip()
            if not url:
                return None
            args: Dict[str, Any] = {"url": url}
            if prompt:
                args["query"] = prompt
            return args
        return {}

    def _side_effect_gate_reason(
        self,
        *,
        tool: str,
        args: Dict[str, Any],
        outputs: List[Dict[str, Any]],
        policy: Dict[str, Any],
    ) -> str:
        if str(policy.get("side_effect_level") or "none") != "high":
            return ""

        if tool in {"mail_send", "mail_answer", "file_write"}:
            unresolved = self._has_unresolved_placeholder(args)
            if unresolved:
                return "unresolved_placeholders"

        if tool == "mail_send":
            body = str(args.get("body") or args.get("text") or "").strip()
            min_text_len = int((policy.get("quality_signals") or {}).get("min_text_length") or 40)
            if len(body) < min_text_len:
                return "body_too_short"

        return ""

    @classmethod
    def _has_unresolved_placeholder(cls, value: Any) -> bool:
        if isinstance(value, dict):
            return any(cls._has_unresolved_placeholder(v) for v in value.values())
        if isinstance(value, list):
            return any(cls._has_unresolved_placeholder(v) for v in value)
        if not isinstance(value, str):
            return False
        txt = value.strip()
        patterns = (r"\{\{[^{}]+\}\}", r"\$\{[^{}]+\}", r"\{steps\[[0-9]+\][^}]*\}", r"\{last[^}]*\}")
        return any(re.search(p, txt) for p in patterns)

    @staticmethod
    def _has_any_success_signal(payload: Dict[str, Any], keys: List[str]) -> bool:
        for key in keys:
            val = payload.get(str(key))
            if isinstance(val, str) and val.strip():
                return True
            if isinstance(val, list) and len(val) > 0:
                return True
            if isinstance(val, dict) and len(val) > 0:
                return True
            if isinstance(val, bool) and val is True:
                return True
            if isinstance(val, (int, float)) and val > 0:
                return True
        return False

    @staticmethod
    def _count_items(payload: Dict[str, Any]) -> int:
        for key in ("hits", "rows", "results", "items", "data", "matches"):
            val = payload.get(key)
            if isinstance(val, list):
                return len(val)
        return 0

    @staticmethod
    def _has_sources(payload: Dict[str, Any]) -> bool:
        for key in ("hits", "rows", "results", "items", "data", "matches"):
            val = payload.get(key)
            if not isinstance(val, list):
                continue
            for item in val:
                if not isinstance(item, dict):
                    continue
                for source_key in ("source", "url", "link", "source_url", "file", "document", "href"):
                    source = item.get(source_key)
                    if isinstance(source, str) and source.strip():
                        return True
        return False

    @staticmethod
    def _extract_primary_text(payload: Dict[str, Any]) -> str:
        for key in ("composed_text", "summary", "text", "answer", "content", "markdown"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    @staticmethod
    def _merge_with_payload(args: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if isinstance(payload, dict):
            merged.update(payload)
        merged.update(args)
        return merged

    @staticmethod
    def _as_payload(result: Any, step: int, tool: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"_step": step, "_tool": tool}
        if isinstance(result, dict):
            payload.update(result)
        else:
            payload["value"] = result
        return payload

    @classmethod
    def _resolve_placeholders(cls, value: Any, outputs: List[Dict[str, Any]], payload: Dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {k: cls._resolve_placeholders(v, outputs, payload) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._resolve_placeholders(v, outputs, payload) for v in value]
        if isinstance(value, str):
            return cls._resolve_string(value, outputs, payload)
        return value

    @classmethod
    def _promote_legacy_result_refs_after_fallback(
        cls,
        value: Any,
        *,
        outputs: List[Dict[str, Any]],
        payload: Dict[str, Any],
        fallback_active: bool,
    ) -> Any:
        if not fallback_active:
            return value

        if isinstance(value, dict):
            return {
                k: cls._promote_legacy_result_refs_after_fallback(
                    v,
                    outputs=outputs,
                    payload=payload,
                    fallback_active=fallback_active,
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                cls._promote_legacy_result_refs_after_fallback(
                    v,
                    outputs=outputs,
                    payload=payload,
                    fallback_active=fallback_active,
                )
                for v in value
            ]
        if not isinstance(value, str):
            return value

        # Only promote unresolved legacy references like {steps[n].result}
        # after fallback insertions changed runtime step order.
        expr = cls._extract_full_steps_expr(value)
        if not expr or ".result" not in expr.lower():
            return value
        resolved = cls._resolve_expr(expr, outputs, payload)
        if resolved is not None:
            return value

        fallback_text = cls._fallback_text_value(outputs, payload)
        return fallback_text if fallback_text else value

    @staticmethod
    def _extract_full_steps_expr(text: str) -> str | None:
        src = str(text or "").strip()
        patterns = (
            r"\{\s*((?:steps\[\d+\]|steps\.\d+)(?:\.[A-Za-z0-9_]+|\[\d+\])*)\s*\}",
            r"\$\{\s*((?:steps\[\d+\]|steps\.\d+)(?:\.[A-Za-z0-9_]+|\[\d+\])*)\s*\}",
            r"\{\{\s*((?:steps\[\d+\]|steps\.\d+)(?:\.[A-Za-z0-9_]+|\[\d+\])*)\s*\}\}",
        )
        for pattern in patterns:
            m = re.fullmatch(pattern, src)
            if m:
                return str(m.group(1) or "").strip() or None
        return None

    @classmethod
    def _fallback_text_value(cls, outputs: List[Dict[str, Any]], payload: Dict[str, Any]) -> str:
        for expr in ("last.text", "last.summary", "last.composed_text"):
            val = cls._resolve_expr(expr, outputs, payload)
            if isinstance(val, str) and val.strip():
                return val.strip()
        val = payload.get("text") if isinstance(payload, dict) else None
        if isinstance(val, str) and val.strip():
            return val.strip()
        return ""

    @classmethod
    def _resolve_string(cls, text: str, outputs: List[Dict[str, Any]], payload: Dict[str, Any]) -> Any:
        full_dollar_legacy = re.fullmatch(r"\$\{\s*((?:steps\[\d+\]|last)(?:\.[A-Za-z0-9_]+|\[\d+\])*)\s*\}", text)
        if full_dollar_legacy:
            resolved = cls._resolve_expr(full_dollar_legacy.group(1), outputs, payload)
            return resolved if resolved is not None else text

        full_double = re.fullmatch(r"\{\{\s*([^{}]+?)\s*\}\}", text)
        if full_double:
            resolved = cls._resolve_expr(full_double.group(1), outputs, payload)
            return resolved if resolved is not None else text

        full_legacy = re.fullmatch(r"\{\s*((?:steps\[\d+\]|last)(?:\.[A-Za-z0-9_]+|\[\d+\])*)\s*\}", text)
        if full_legacy:
            resolved = cls._resolve_expr(full_legacy.group(1), outputs, payload)
            return resolved if resolved is not None else text

        def repl_double(match: re.Match[str]) -> str:
            resolved = cls._resolve_expr(match.group(1), outputs, payload)
            return str(resolved) if resolved is not None else match.group(0)

        def repl_legacy(match: re.Match[str]) -> str:
            resolved = cls._resolve_expr(match.group(1), outputs, payload)
            return str(resolved) if resolved is not None else match.group(0)

        out = re.sub(r"\$\{((?:steps\[\d+\]|last)(?:\.[A-Za-z0-9_]+|\[\d+\])*)\}", repl_legacy, text)
        out = re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", repl_double, out)
        out = re.sub(r"\{((?:steps\[\d+\]|last)(?:\.[A-Za-z0-9_]+|\[\d+\])*)\}", repl_legacy, out)
        return out

    @classmethod
    def _resolve_expr(cls, expr: str, outputs: List[Dict[str, Any]], payload: Dict[str, Any]) -> Any:
        expr = (expr or "").strip()
        if not expr:
            return None

        if expr == "payload":
            return payload

        if expr.startswith("last"):
            base = outputs[-1] if outputs else payload
            tokens = cls._path_tokens(expr[len("last"):])
            return cls._walk_path(base, tokens)

        if expr.startswith("steps["):
            end = expr.find("]")
            if end <= 6:
                return None
            idx_raw = expr[6:end]
            if not idx_raw.isdigit():
                return None
            idx = int(idx_raw)
            if idx < 0 or idx >= len(outputs):
                return None
            base = outputs[idx]
            tokens = cls._path_tokens(expr[end + 1 :])
            return cls._walk_path(base, tokens)

        if expr.startswith("steps."):
            rem = expr[len("steps."):]
            parts = rem.split(".")
            if not parts or not parts[0].isdigit():
                return None
            step_no = int(parts[0])  # 1-based
            idx = step_no - 1
            if idx < 0 or idx >= len(outputs):
                return None
            base = outputs[idx]
            tail = "." + ".".join(parts[1:]) if len(parts) > 1 else ""
            tokens = cls._path_tokens(tail)
            return cls._walk_path(base, tokens)

        return None

    @staticmethod
    def _path_tokens(path: str) -> List[Any]:
        tokens: List[Any] = []
        p = (path or "").strip()
        if not p:
            return tokens

        i = 0
        while i < len(p):
            if p[i] == ".":
                i += 1
                start = i
                while i < len(p) and p[i] not in ".[":
                    i += 1
                token = p[start:i]
                if token:
                    tokens.append(int(token) if token.isdigit() else token)
                continue
            if p[i] == "[":
                j = p.find("]", i)
                if j == -1:
                    break
                idx_raw = p[i + 1 : j].strip()
                if idx_raw.isdigit():
                    tokens.append(int(idx_raw))
                i = j + 1
                continue
            i += 1
        return tokens

    @classmethod
    def _walk_path(cls, current: Any, tokens: List[Any]) -> Any:
        cur = current
        for token in tokens:
            if isinstance(token, int):
                if isinstance(cur, list) and 0 <= token < len(cur):
                    cur = cur[token]
                    continue
                return None

            if isinstance(cur, dict):
                if token in cur:
                    cur = cur[token]
                    continue
                result = cur.get("result")
                if isinstance(result, dict) and token in result:
                    cur = result[token]
                    continue
                step_payload = cur.get("payload")
                if isinstance(step_payload, dict) and token in step_payload:
                    cur = step_payload[token]
                    continue
            return None
        return cur

    @staticmethod
    def _log_step_output(entry: Dict[str, Any]) -> None:
        step = entry.get("step")
        tool = entry.get("tool")
        ok = bool(entry.get("ok"))
        status = str(entry.get("status") or ("success" if ok else "error"))
        print(f"----- STEP OUTPUT {step} -----")
        print(f"tool={tool} ok={ok} status={status}")
        if not ok and entry.get("error"):
            print(f"error={entry.get('error')}")
        payload = entry.get("payload")
        if isinstance(payload, dict):
            try:
                payload_str = json.dumps(payload, ensure_ascii=False)
            except Exception:
                payload_str = str(payload)
            if len(payload_str) > 4000:
                payload_str = payload_str[:4000] + "...(truncated)"
            print(f"payload={payload_str}")
        else:
            print(f"payload={payload}")
        print("---------------------------")

    @staticmethod
    def _log_step_input(step: int, tool: str, args: Dict[str, Any], expected: Dict[str, Any]) -> None:
        print(f"----- STEP INPUT {step} -----")
        print(f"tool={tool}")
        try:
            args_str = json.dumps(args, ensure_ascii=False)
        except Exception:
            args_str = str(args)
        if len(args_str) > 4000:
            args_str = args_str[:4000] + "...(truncated)"
        print(f"args={args_str}")
        try:
            expected_str = json.dumps(expected, ensure_ascii=False)
        except Exception:
            expected_str = str(expected)
        if len(expected_str) > 4000:
            expected_str = expected_str[:4000] + "...(truncated)"
        print(f"expected={expected_str}")
        print("--------------------------")
