from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

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


def register(registry: ToolRegistry) -> None:
    def tool_booking_extract_facts(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingExtractFactsRequest(**args)
        result = booking_extract_facts(text=req.text, timezone_name=req.timezone, required_fields=req.required_fields)
        return BookingExtractFactsResponse(**result).model_dump()

    def tool_booking_validate_completeness(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingValidateCompletenessRequest(**args)
        result = booking_validate_completeness(facts=req.facts, required_fields=req.required_fields)
        return BookingValidateCompletenessResponse(**result).model_dump()

    def tool_booking_decision_engine(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingDecisionEngineRequest(**args)
        result = booking_decision_engine(
            facts=req.facts,
            profile_rules=req.profile_rules,
            completeness=req.completeness,
            distance=req.distance,
            quote=req.quote,
            require_price_confirmation=req.require_price_confirmation,
        )
        return BookingDecisionEngineResponse(**result).model_dump()

    def tool_booking_reply_score(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingReplyScoreRequest(**args)
        result = booking_reply_score(
            user_message=req.user_message,
            draft_reply=req.draft_reply,
            booking_decision=req.booking_decision,
            facts=req.facts,
            required_fields=req.required_fields,
            missing_fields=req.missing_fields,
            knowledge_evidence=req.knowledge_evidence,
            require_actionable=req.require_actionable,
        )
        return BookingReplyScoreResponse(**result).model_dump()

    def tool_booking_instruction_check(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingInstructionCheckRequest(**args)
        result = booking_instruction_check(
            instructions=req.instructions,
            user_message=req.user_message,
            draft_reply=req.draft_reply,
            booking_decision=req.booking_decision,
            facts=req.facts,
        )
        return BookingInstructionCheckResponse(**result).model_dump()

    registry.register(
        "booking_extract_facts",
        tool_booking_extract_facts,
        request_model=BookingExtractFactsRequest,
        response_model=BookingExtractFactsResponse,
    )
    registry.register(
        "booking_validate_completeness",
        tool_booking_validate_completeness,
        request_model=BookingValidateCompletenessRequest,
        response_model=BookingValidateCompletenessResponse,
    )
    # Alias with legacy typo requested by user.
    registry.register(
        "booking_booking_validate_completness",
        tool_booking_validate_completeness,
        request_model=BookingValidateCompletenessRequest,
        response_model=BookingValidateCompletenessResponse,
    )
    registry.register(
        "booking_decision_engine",
        tool_booking_decision_engine,
        request_model=BookingDecisionEngineRequest,
        response_model=BookingDecisionEngineResponse,
    )
    registry.register(
        "booking_reply_score",
        tool_booking_reply_score,
        request_model=BookingReplyScoreRequest,
        response_model=BookingReplyScoreResponse,
    )
    registry.register(
        "booking_instruction_check",
        tool_booking_instruction_check,
        request_model=BookingInstructionCheckRequest,
        response_model=BookingInstructionCheckResponse,
    )
    # Alias with legacy typo requested by user.
    registry.register(
        "booking_descision_enginge",
        tool_booking_decision_engine,
        request_model=BookingDecisionEngineRequest,
        response_model=BookingDecisionEngineResponse,
    )
