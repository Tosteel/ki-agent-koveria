from typing import Any, Dict, List

from ..agent.tool_registry import ToolRegistry
from ..services.llm_openai import LlmRuntime

class Planner:
    def __init__(self, llm: LlmRuntime, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry

    def create_steps(self, goal: str) -> List[Dict[str, Any]]:
        if not self.llm.enabled():
            return []

        schema = self.registry.planner_schema()

        plan = self.llm.plan_steps(
            goal=goal,
            tool_schema=schema,
        )
        return plan.get("steps", [])