from typing import Any, Dict, List

from pydantic import ValidationError

from ..agent.models import AgentToolPlan
from ..agent.tool_registry import ToolRegistry
from ..agent.langchain_runtime import run_planner_chain
from ..services.llm_openai import LlmOpenai

class Planner:
    def __init__(self, llm: LlmOpenai, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry

    def create_steps(self, goal: str) -> List[Dict[str, Any]]:
        if not self.llm.enabled():
            return []

        schema = self.registry.planner_schema(goal=goal)
        plan = run_planner_chain(llm=self.llm, goal=goal, tool_schema=schema)
        try:
            parsed = AgentToolPlan.model_validate(plan)
        except ValidationError:
            return []
        return [step.model_dump() for step in parsed.steps]
