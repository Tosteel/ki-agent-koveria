from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .llm_text import llm_text_chat, llm_text_compose, llm_text_summarize
from .models import (
    LlmTextChatRequest,
    LlmTextChatResponse,
    LlmTextComposeRequest,
    LlmTextComposeResponse,
    LlmTextSummarizeRequest,
    LlmTextSummarizeResponse,
)


def register(registry: ToolRegistry) -> None:
    def tool_llm_text_compose(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LlmTextComposeRequest(**args)
        result = llm_text_compose(
            text=req.text,
            goal=req.goal,
            instruction=req.instruction,
            max_chars=req.max_chars,
        )
        return LlmTextComposeResponse(**result).model_dump()

    def tool_llm_text_summarize(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LlmTextSummarizeRequest(**args)
        result = llm_text_summarize(
            text=req.text,
            goal=req.goal,
            instruction=req.instruction,
            max_chars=req.max_chars,
        )
        return LlmTextSummarizeResponse(**result).model_dump()

    def tool_llm_text_chat(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LlmTextChatRequest(**args)
        result = llm_text_chat(
            message=req.message,
            tone=req.tone,
            max_chars=req.max_chars,
        )
        return LlmTextChatResponse(**result).model_dump()

    registry.register(
        "llm_text_compose",
        tool_llm_text_compose,
        request_model=LlmTextComposeRequest,
        response_model=LlmTextComposeResponse,
    )
    registry.register(
        "llm_text_summarize",
        tool_llm_text_summarize,
        request_model=LlmTextSummarizeRequest,
        response_model=LlmTextSummarizeResponse,
    )
    registry.register(
        "llm_text_chat",
        tool_llm_text_chat,
        request_model=LlmTextChatRequest,
        response_model=LlmTextChatResponse,
    )

