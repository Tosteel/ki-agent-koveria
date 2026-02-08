from __future__ import annotations

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
            args = step.get("args") or {}

            if tool not in PHASE1_ALLOWED_TOOLS:
                outputs.append({"step": i, "tool": tool, "ok": False, "error": "tool_not_allowed"})
                continue

            try:
                res = self.registry.dispatch(tool, ctx, args)
                outputs.append({"step": i, "tool": tool, "ok": True, "result": res})
            except Exception as e:
                outputs.append({"step": i, "tool": tool, "ok": False, "error": str(e)})
        return outputs
