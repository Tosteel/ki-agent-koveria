from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=50)
    classification: Optional[str] = None


class RagHit(BaseModel):
    source: str
    score: float
    text: str


class RagQueryResponse(BaseModel):
    query: str
    hits: List[RagHit]


__all__ = [
    "RagQueryRequest",
    "RagHit",
    "RagQueryResponse",
]
