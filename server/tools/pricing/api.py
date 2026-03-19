from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import PricingComputeQuoteRequest, PricingComputeQuoteResponse
from .pricing import pricing_compute_quote


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/tools/pricing/compute-quote", response_model=PricingComputeQuoteResponse)
    def compute_route(
        req: PricingComputeQuoteRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> PricingComputeQuoteResponse:
        ensure_user_dirs(s, user_id)
        return PricingComputeQuoteResponse(
            **pricing_compute_quote(
                facts=req.facts,
                pricing_rules=req.pricing_rules,
                booking_rules=req.booking_rules,
                distance_km=req.distance_km,
            )
        )

    return router
