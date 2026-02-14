from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EbaySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=50)
    sort_order: str = "BestMatch"


class EbayItem(BaseModel):
    item_id: str = ""
    title: str = ""
    price: str = ""
    currency: str = ""
    location: str = ""
    url: str = ""


class EbaySearchResponse(BaseModel):
    query: str
    count: int
    items: List[EbayItem] = Field(default_factory=list)
