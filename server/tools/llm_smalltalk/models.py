from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LlmSmalltalkRequest(BaseModel):
    message: str = Field(..., min_length=1)
    tone: str = "freundlich"
    max_chars: int = Field(280, ge=60, le=1200)


class LlmSmalltalkResponse(BaseModel):
    reply: str
    text: str
    fallback_used: bool = False
    model: str = ""
    usage: Optional[Dict[str, Any]] = None

