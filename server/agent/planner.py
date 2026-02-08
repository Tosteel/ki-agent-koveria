from __future__ import annotations
from typing import Any, Dict, List

from .policies import PHASE1_ALLOWED_TOOLS
from ..services.llm_openai import LlmRuntime

class Planner:
    def __init__(self, llm: LlmRuntime):
        self.llm = llm

    def _plan_schema(self) -> Dict[str, Any]:
        # Minimaler Schema-Contract: steps = [{tool, args}]
        # Tool muss in allowed tools sein.
        return {
            "name": "tool_plan",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["steps"],
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["tool", "args"],
                            "properties": {
                                "tool": {"type": "string", "enum": sorted(list(PHASE1_ALLOWED_TOOLS))},
                                "args": {"type": "object"},
                            },
                        },
                    }
                },
            },
        }

    def create_steps(self, goal: str) -> List[Dict[str, Any]]:
        # Fallback (ohne OPENAI_API_KEY): sehr simple Heuristik
        if not self.llm.enabled():
            return [
                {"tool": "query_rag", "args": {"query": goal, "top_k": 5}},
            ]

        plan = self.llm.plan_steps(goal=goal, tool_schema=self._plan_schema())
        steps = plan.get("steps") or []
        return steps
