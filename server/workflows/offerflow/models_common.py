from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class OfferflowMetadata(BaseModel):
    trade: str = ""
    region: str = ""
    project_type: str = ""
    scope_tags: List[str] = Field(default_factory=list)
    size: str = ""
    outcome: str = ""


class OfferflowBaseResponse(BaseModel):
    ok: bool = True
    offer_id: str
    step: int
    message: str = ""
