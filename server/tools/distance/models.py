from __future__ import annotations

from pydantic import BaseModel, Field


class DistanceCheckRequest(BaseModel):
    origin: str = Field(..., min_length=2)
    destination: str = Field(..., min_length=2)
    max_distance_km: float = Field(default=0.0, ge=0.0)


class DistanceCheckResponse(BaseModel):
    origin: str = ""
    destination: str = ""
    distance_km: float = 0.0
    estimated_duration_minutes: int = 0
    within_limit: bool = True
    text: str = ""
