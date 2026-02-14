from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LlmComposeRequest(BaseModel):
    text: str
    instruction: str = ""
    goal: str = ""
    max_chars: int = Field(3000, ge=200, le=20000)


class LlmComposeResponse(BaseModel):
    composed_text: str
    text: str
    fallback_used: bool = False
    model: str = ""
    usage: Optional[Dict[str, Any]] = None


__all__ = ["LlmComposeRequest", "LlmComposeResponse"]
