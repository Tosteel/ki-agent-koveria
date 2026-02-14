from __future__ import annotations

from pydantic import BaseModel, Field


class TemplateToolRequest(BaseModel):
    text: str = Field(..., min_length=1)
    extra: str = ""


class TemplateToolResponse(BaseModel):
    text: str
    ok: bool = True
