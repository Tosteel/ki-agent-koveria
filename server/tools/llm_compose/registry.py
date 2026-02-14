from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .llm_compose import llm_compose_text
from .models import LlmComposeRequest, LlmComposeResponse


def register(registry: ToolRegistry) -> None:
    def tool_llm_compose(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LlmComposeRequest(**args)
        result = llm_compose_text(
            text=req.text,
            goal=req.goal,
            instruction=req.instruction,
            max_chars=req.max_chars,
        )
        return LlmComposeResponse(**result).model_dump()

    registry.register("llm_compose", tool_llm_compose, request_model=LlmComposeRequest)
