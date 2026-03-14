from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LlmTextComposeRequest(BaseModel):
    text: str
    instruction: str = ""
    goal: str = ""
    max_chars: int = Field(3000, ge=200, le=20000)


class LlmTextComposeResponse(BaseModel):
    text: str
    composed_text: str
    fallback_used: bool = False
    model: str = ""
    usage: Optional[Dict[str, Any]] = None


class LlmTextSummarizeRequest(BaseModel):
    text: str
    instruction: str = ""
    goal: str = ""
    max_chars: int = Field(1200, ge=200, le=10000)


class LlmTextSummarizeResponse(BaseModel):
    summary: str
    text: str
    fallback_used: bool = False
    model: str = ""
    usage: Optional[Dict[str, Any]] = None


class LlmTextChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    tone: str = "freundlich"
    max_chars: int = Field(280, ge=60, le=1200)


class LlmTextChatResponse(BaseModel):
    reply: str
    text: str
    fallback_used: bool = False
    model: str = ""
    usage: Optional[Dict[str, Any]] = None

