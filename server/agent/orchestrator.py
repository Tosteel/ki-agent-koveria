from __future__ import annotations

import re
from typing import Any, Dict, List

from ..agent.policies import is_phase1_tool_allowed
from ..agent.tool_registry import ToolRegistry, ToolContext


class Orchestrator:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run_steps(self, ctx: ToolContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        payload: Dict[str, Any] = {}

        for i, step in enumerate(steps, start=1):
            tool = (step.get("tool") or "").strip()
            args = self._merge_with_payload(step.get("args") or {}, payload)
            goal = (getattr(ctx, "goal", "") or "").strip()
            if goal and "goal" not in args:
                args["goal"] = goal
            args = self._resolve_placeholders(args, outputs, payload)

            if not is_phase1_tool_allowed(tool):
                outputs.append({"step": i, "tool": tool, "ok": False, "error": "tool_not_allowed", "payload": payload})
                continue

            try:
                res = self.registry.dispatch(tool, ctx, args)
                payload = self._as_payload(res, i, tool)
                outputs.append({"step": i, "tool": tool, "ok": True, "result": res, "payload": payload})
            except Exception as e:
                outputs.append({"step": i, "tool": tool, "ok": False, "error": str(e), "payload": payload})

        return outputs

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
