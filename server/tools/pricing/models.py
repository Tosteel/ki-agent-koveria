from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class PricingComputeQuoteRequest(BaseModel):
    facts: Dict[str, Any] = Field(default_factory=dict)
    pricing_rules: Dict[str, Any] = Field(default_factory=dict)
    booking_rules: Dict[str, Any] = Field(default_factory=dict)
    distance_km: float = Field(default=0.0, ge=0.0)


class PricingComputeQuoteResponse(BaseModel):
    total_eur: float = 0.0
    currency: str = "EUR"
    breakdown: Dict[str, float] = Field(default_factory=dict)
    overnight_included: bool = False
    text: str = ""
