from __future__ import annotations

from pydantic import BaseModel


class VideoAnalyzeSyncRequest(BaseModel):
    prompt: str


class VideoAnalyzeSyncResponse(BaseModel):
    text: str = ""

    model_config = {"extra": "allow"}


__all__ = ["VideoAnalyzeSyncRequest", "VideoAnalyzeSyncResponse"]
