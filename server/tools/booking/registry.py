from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .booking import booking_decision_engine, booking_extract_facts, booking_validate_completeness
from .models import (
    BookingDecisionEngineRequest,
    BookingDecisionEngineResponse,
    BookingExtractFactsRequest,
    BookingExtractFactsResponse,
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
    # Alias with legacy typo requested by user.
    registry.register(
        "booking_descision_enginge",
        tool_booking_decision_engine,
        request_model=BookingDecisionEngineRequest,
        response_model=BookingDecisionEngineResponse,
    )
