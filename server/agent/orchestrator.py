from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict

from pydantic import ValidationError

from ..agent.models import AgentStep
from ..agent.langchain_runtime import dispatch_tool_chain, tool_dispatch_runtime_mode
from ..agent.policies import tools_allowed
from ..agent.tool_registry import ToolRegistry, ToolContext

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
            payload, entry = self._execute_single_step(
                step_no=i,
                raw_step=raw_step,
                ctx=ctx,
                payload=payload,
                outputs=outputs,
            )
            outputs.append(entry)
            self._log_step_output(entry)

        return outputs

    def _run_steps_langgraph(self, ctx: ToolContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not HAS_LANGGRAPH:
            return self._run_steps_legacy(ctx, steps)

        def _execute_step(state: _WorkflowState) -> Dict[str, Any]:
            idx = int(state.get("index") or 0)
            raw_steps = state.get("steps") or []
            if idx < 0 or idx >= len(raw_steps):
                return {}

            payload = dict(state.get("payload") or {})
            outputs = list(state.get("outputs") or [])
            payload, entry = self._execute_single_step(
                step_no=idx + 1,
                raw_step=raw_steps[idx],
                ctx=ctx,
                payload=payload,
                outputs=outputs,
            )
            outputs.append(entry)
            self._log_step_output(entry)
            return {"index": idx + 1, "payload": payload, "outputs": outputs}

        def _route(state: _WorkflowState) -> str:
            idx = int(state.get("index") or 0)
            raw_steps = state.get("steps") or []
            return "continue" if idx < len(raw_steps) else "end"

        graph_builder = StateGraph(_WorkflowState)
        graph_builder.add_node("execute_step", _execute_step)
        graph_builder.set_entry_point("execute_step")
        graph_builder.add_conditional_edges(
            "execute_step",
            _route,
            {
                "continue": "execute_step",
                "end": END,
            },
        )
        graph = graph_builder.compile()
        out = graph.invoke({"steps": steps, "index": 0, "payload": {}, "outputs": []})

        if isinstance(out, dict) and isinstance(out.get("outputs"), list):
            return out.get("outputs") or []
        return []

    def _execute_single_step(
        self,
        *,
        step_no: int,
        raw_step: Dict[str, Any],
        ctx: ToolContext,
        payload: Dict[str, Any],
        outputs: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            step = AgentStep.model_validate(raw_step).model_dump()
        except ValidationError as e:
            entry = {
                "step": step_no,
                "tool": str((raw_step or {}).get("tool") if isinstance(raw_step, dict) else ""),
                "ok": False,
                "error": f"invalid_step_schema: {e.errors()}",
                "payload": payload,
            }
            return payload, entry

        tool = (step.get("tool") or "").strip()
        args = self._merge_with_payload(step.get("args") or {}, payload)
        goal = (getattr(ctx, "goal", "") or "").strip()
        if goal and "goal" not in args:
            args["goal"] = goal
        args = self._resolve_placeholders(args, outputs, payload)
        expected = self.registry.expected_input(tool)
        self._log_step_input(step_no, tool, args, expected)

        if not tools_allowed(tool):
            entry = {"step": step_no, "tool": tool, "ok": False, "error": "tool_not_allowed", "payload": payload}
            return payload, entry

        try:
            res = dispatch_tool_chain(registry=self.registry, tool_name=tool, ctx=ctx, args=args)
            new_payload = self._as_payload(res, step_no, tool)
            entry = {"step": step_no, "tool": tool, "ok": True, "result": res, "payload": new_payload}
            return new_payload, entry
        except Exception as e:
            entry = {"step": step_no, "tool": tool, "ok": False, "error": str(e), "payload": payload}
            return payload, entry

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
        print(f"----- STEP OUTPUT {step} -----")
        print(f"tool={tool} ok={ok}")
        if not ok:
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
