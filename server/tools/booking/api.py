from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .booking import (
    booking_decision_engine,
    booking_extract_facts,
    booking_instruction_check,
    booking_reply_score,
    booking_validate_completeness,
)
from .models import (
    BookingDecisionEngineRequest,
    BookingDecisionEngineResponse,
    BookingExtractFactsRequest,
    BookingExtractFactsResponse,
    BookingInstructionCheckRequest,
    BookingInstructionCheckResponse,
    BookingReplyScoreRequest,
    BookingReplyScoreResponse,
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

    @router.post("/tools/booking/reply-score", response_model=BookingReplyScoreResponse)
    def reply_score_route(
        req: BookingReplyScoreRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingReplyScoreResponse:
        ensure_user_dirs(s, user_id)
        return BookingReplyScoreResponse(
            **booking_reply_score(
                user_message=req.user_message,
                draft_reply=req.draft_reply,
                booking_decision=req.booking_decision,
                facts=req.facts,
                required_fields=req.required_fields,
                missing_fields=req.missing_fields,
                knowledge_evidence=req.knowledge_evidence,
                require_actionable=req.require_actionable,
            )
        )

    @router.post("/tools/booking/instruction-check", response_model=BookingInstructionCheckResponse)
    def instruction_check_route(
        req: BookingInstructionCheckRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> BookingInstructionCheckResponse:
        ensure_user_dirs(s, user_id)
        return BookingInstructionCheckResponse(
            **booking_instruction_check(
                instructions=req.instructions,
                user_message=req.user_message,
                draft_reply=req.draft_reply,
                booking_decision=req.booking_decision,
                facts=req.facts,
            )
        )

    return router
