from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..agent.policies import PHASE1_ALLOWED_TOOLS
from ..agent.tool_registry import ToolRegistry, ToolContext


class Orchestrator:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run_steps(self, ctx: ToolContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for i, step in enumerate(steps, start=1):
            tool = (step.get("tool") or "").strip()
            args = self._resolve_args(step.get("args") or {}, outputs)

            if tool not in PHASE1_ALLOWED_TOOLS:
                outputs.append({"step": i, "tool": tool, "ok": False, "error": "tool_not_allowed"})
                continue

            try:
                res = self.registry.dispatch(tool, ctx, args)
                outputs.append({"step": i, "tool": tool, "ok": True, "result": res})
            except Exception as e:
                outputs.append({"step": i, "tool": tool, "ok": False, "error": str(e)})
        return outputs

    def _resolve_args(self, value: Any, outputs: List[Dict[str, Any]]) -> Any:
        if isinstance(value, dict):
            return {k: self._resolve_args(v, outputs) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_args(v, outputs) for v in value]
        if isinstance(value, str):
            return self._resolve_string(value, outputs)
        return value

    def _resolve_string(self, value: str, outputs: List[Dict[str, Any]]) -> Any:
        matches = list(re.finditer(r"\{\{\s*([^{}]+?)\s*\}\}", value))
        if not matches:
            return value

        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            resolved = self._lookup(matches[0].group(1), outputs)
            return resolved if resolved is not None else value

        rendered = value
        for m in matches:
            expr = m.group(1)
            resolved = self._lookup(expr, outputs)
            rendered = rendered.replace(m.group(0), "" if resolved is None else self._stringify(resolved))
        return rendered

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _lookup(self, expr: str, outputs: List[Dict[str, Any]]) -> Any:
        path = expr.strip()
        if not path:
            return None

        if path == "last":
            return outputs[-1] if outputs else None

        if path.startswith("last."):
            base: Any = outputs[-1] if outputs else None
            return self._walk(base, path[5:])

        if path.startswith("steps."):
            rest = path[6:]
            if rest and rest[0].isdigit():
                idx_token, _, tail = rest.partition(".")
                try:
                    step_number = int(idx_token)
                except ValueError:
                    return None
                if step_number < 1 or step_number > len(outputs):
                    return None
                return self._walk(outputs[step_number - 1], tail)

        if path.startswith("steps["):
            # 0-based index via steps[0]
            idx_end = path.find("]")
            if idx_end <= 6:
                return None
            try:
                idx = int(path[6:idx_end])
            except ValueError:
                return None
            if idx < 0 or idx >= len(outputs):
                return None
            tail = path[idx_end + 1 :].lstrip(".")
            return self._walk(outputs[idx], tail)

        return None

    def _walk(self, base: Any, tail: str) -> Any:
        if base is None:
            return None
        if not tail:
            return base

        current = base
        for token in tail.split("."):
            if isinstance(current, dict):
                if token not in current:
                    return None
                current = current[token]
                continue

            if isinstance(current, list) and token.isdigit():
                idx = int(token)
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
                continue

            return None

        return current
