from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class GoogleSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    num: int = Field(5, ge=1, le=10)
    start: int = Field(1, ge=1, le=91)
    gl: str | None = None
    hl: str | None = None
    safe: str | None = None
    site_search: str | None = None


class GoogleSearchItem(BaseModel):
    title: str = ""
    link: str = ""
    snippet: str = ""
    display_link: str = ""


class GoogleSearchResponse(BaseModel):
    query: str
    count: int
    total_results: int | None = None
    items: List[GoogleSearchItem] = Field(default_factory=list)
    text: str = ""

