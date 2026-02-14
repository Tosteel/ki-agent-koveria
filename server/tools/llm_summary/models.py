from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LlmSummaryRequest(BaseModel):
    text: str
    instruction: str = ""
    goal: str = ""
    max_chars: int = Field(1200, ge=200, le=10000)


class LlmSummaryResponse(BaseModel):
    summary: str
    text: str
    fallback_used: bool = False
    model: str = ""
    usage: Optional[Dict[str, Any]] = None


__all__ = ["LlmSummaryRequest", "LlmSummaryResponse"]
