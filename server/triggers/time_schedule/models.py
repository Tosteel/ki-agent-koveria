from __future__ import annotations

from pydantic import BaseModel, Field


class TimeScheduleConfig(BaseModel):
    interval_seconds: int = Field(300, ge=10, le=86400)
    fire_immediately: bool = False

