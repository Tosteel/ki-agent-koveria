from __future__ import annotations

from pydantic import BaseModel


class SearchGenerateJsonRequest(BaseModel):
    user_prompt: str


__all__ = ["SearchGenerateJsonRequest"]
