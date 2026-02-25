from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .llm_summary import llm_summarize_text
from .models import LlmSummaryRequest, LlmSummaryResponse


def register(registry: ToolRegistry) -> None:
    def tool_llm_summarize(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LlmSummaryRequest(**args)
        result = llm_summarize_text(
            text=req.text,
            goal=req.goal,
            instruction=req.instruction,
            max_chars=req.max_chars,
        )
        return LlmSummaryResponse(**result).model_dump()

    registry.register(
        "llm_summarize",
        tool_llm_summarize,
        request_model=LlmSummaryRequest,
        response_model=LlmSummaryResponse,
    )
