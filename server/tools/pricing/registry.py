from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import PricingComputeQuoteRequest, PricingComputeQuoteResponse
from .pricing import pricing_compute_quote


def register(registry: ToolRegistry) -> None:
    def tool_pricing_compute_quote(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PricingComputeQuoteRequest(**args)
        result = pricing_compute_quote(
            facts=req.facts,
            pricing_rules=req.pricing_rules,
            booking_rules=req.booking_rules,
            distance_km=req.distance_km,
        )
        return PricingComputeQuoteResponse(**result).model_dump()

    registry.register(
        "pricing_compute_quote",
        tool_pricing_compute_quote,
        request_model=PricingComputeQuoteRequest,
        response_model=PricingComputeQuoteResponse,
    )
