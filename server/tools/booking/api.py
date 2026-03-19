from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .booking import booking_decision_engine, booking_extract_facts, booking_validate_completeness
from .models import (
    BookingDecisionEngineRequest,
    BookingDecisionEngineResponse,
    BookingExtractFactsRequest,
    BookingExtractFactsResponse,
    BookingValidateCompletenessRequest,
    BookingValidateCompletenessResponse,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/tools/booking/extract-facts", response_model=BookingExtractFactsResponse)
    def extract_facts_route(
        req: BookingExtractFactsRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingExtractFactsResponse:
        ensure_user_dirs(s, user_id)
        return BookingExtractFactsResponse(**booking_extract_facts(text=req.text, timezone_name=req.timezone))

    @router.post("/tools/booking/validate-completeness", response_model=BookingValidateCompletenessResponse)
    def validate_route(
        req: BookingValidateCompletenessRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingValidateCompletenessResponse:
        ensure_user_dirs(s, user_id)
        return BookingValidateCompletenessResponse(
            **booking_validate_completeness(facts=req.facts, required_fields=req.required_fields)
        )

    @router.post("/tools/booking/decision-engine", response_model=BookingDecisionEngineResponse)
    def decision_route(
        req: BookingDecisionEngineRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingDecisionEngineResponse:
        ensure_user_dirs(s, user_id)
        return BookingDecisionEngineResponse(
            **booking_decision_engine(
                facts=req.facts,
                profile_rules=req.profile_rules,
                completeness=req.completeness,
                distance=req.distance,
                quote=req.quote,
            )
        )

    return router
