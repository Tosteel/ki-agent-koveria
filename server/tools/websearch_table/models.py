from __future__ import annotations

from pydantic import BaseModel


class SearchGenerateJsonRequest(BaseModel):
    user_prompt: str


class SearchGenerateJsonResponse(BaseModel):
    text: str = ""

    model_config = {"extra": "allow"}


__all__ = ["SearchGenerateJsonRequest", "SearchGenerateJsonResponse"]
