from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class WebsiteMatch(BaseModel):
    tag: str
    text: str
    snippet: str
    href: str = ""


class ViewWebsiteRequest(BaseModel):
    url: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    selector: str = "body"
    max_matches: int = Field(8, ge=1, le=30)
    context_chars: int = Field(180, ge=60, le=600)
    timeout_ms: int = Field(15000, ge=2000, le=120000)


class BrowseWebsiteRequest(BaseModel):
    url: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    selector: str = "body"
    max_matches: int = Field(8, ge=1, le=50)
    context_chars: int = Field(180, ge=60, le=600)
    timeout_ms: int = Field(15000, ge=2000, le=120000)
    max_pages: int = Field(3, ge=1, le=10)
    click_selectors: List[str] = Field(default_factory=list)
    follow_links_matching: str = ""


class WebsiteSearchResponse(BaseModel):
    url: str
    final_url: str
    title: str
    query: str
    count: int
    matches: List[WebsiteMatch] = Field(default_factory=list)
    visited_urls: List[str] = Field(default_factory=list)
    text: str
