from __future__ import annotations

from pydantic import BaseModel, Field


class TemplateTriggerConfig(BaseModel):
    interval_seconds: int = Field(60, ge=10, le=86400)

