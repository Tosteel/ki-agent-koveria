from __future__ import annotations

from pydantic import BaseModel


class VideoAnalyzeSyncRequest(BaseModel):
    prompt: str


__all__ = ["VideoAnalyzeSyncRequest"]
