from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .llm_smalltalk import llm_smalltalk
from .models import LlmSmalltalkRequest, LlmSmalltalkResponse


def register(registry: ToolRegistry) -> None:
    def tool_llm_smalltalk(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LlmSmalltalkRequest(**args)
        result = llm_smalltalk(message=req.message, tone=req.tone, max_chars=req.max_chars)
        return LlmSmalltalkResponse(**result).model_dump()

    registry.register("llm_smalltalk", tool_llm_smalltalk, request_model=LlmSmalltalkRequest)

