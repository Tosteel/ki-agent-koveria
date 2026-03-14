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
    include_full_text: bool = True
    full_text_max_chars: int = Field(300000, ge=1000, le=2000000)


class GetWebsiteRequest(BaseModel):
    url: str = Field(..., min_length=1)
    selector: str = "body"
    timeout_ms: int = Field(15000, ge=2000, le=120000)
    max_chars: int = Field(300000, ge=1000, le=2000000)
    include_image_urls: bool = True


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


class BrowseWhitelistRequest(BaseModel):
    url: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    selector: str = "body"
    max_matches: int = Field(8, ge=1, le=50)
    context_chars: int = Field(180, ge=60, le=600)
    timeout_ms: int = Field(15000, ge=2000, le=120000)
    max_pages: int = Field(3, ge=1, le=10)
    click_selectors: List[str] = Field(default_factory=list)
    follow_links_matching: str = ""
    allowed_domains: List[str] = Field(
        default_factory=list,
        description="Optionale Whitelist erlaubter Domains (z. B. ['tagesschau.de', 'zdf.de']).",
    )


class WebsiteSearchResponse(BaseModel):
    url: str
    final_url: str
    title: str
    query: str
    count: int
    matches: List[WebsiteMatch] = Field(default_factory=list)
    visited_urls: List[str] = Field(default_factory=list)
    text: str


class GetWebsiteResponse(BaseModel):
    url: str
    final_url: str
    title: str
    selector: str
    content_type: str = ""
    status_code: int
    text: str
    html: str = ""
