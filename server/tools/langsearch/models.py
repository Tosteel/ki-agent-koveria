from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class LangSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    count: int = Field(5, ge=1, le=20)
    summary: bool = True
    freshness: str | None = None


class LangSearchItem(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    score: float | None = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class LangSearchResponse(BaseModel):
    query: str
    count: int
    items: List[LangSearchItem] = Field(default_factory=list)
    summary_text: str = ""
    text: str = ""

