from __future__ import annotations

from typing import Any, Dict, List

from ..agent.policies import PHASE1_ALLOWED_TOOLS
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

            if tool not in PHASE1_ALLOWED_TOOLS:
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
