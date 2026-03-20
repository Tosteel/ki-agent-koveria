from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .calendar import (
    calendar_check_availability,
    calendar_create_event,
    calendar_hold_event,
    calendar_propose_slots,
    calendar_update_event,
)
from .models import (
    CalendarCheckAvailabilityRequest,
    CalendarCheckAvailabilityResponse,
    CalendarCreateEventRequest,
    CalendarCreateEventResponse,
    CalendarHoldEventRequest,
    CalendarHoldEventResponse,
    CalendarProposeSlotsRequest,
    CalendarProposeSlotsResponse,
    CalendarUpdateEventRequest,
    CalendarUpdateEventResponse,
)


def register(registry: ToolRegistry) -> None:
    def tool_calendar_check_availability(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CalendarCheckAvailabilityRequest(**args)
        result = calendar_check_availability(
            start_iso=req.start_iso,
            end_iso=req.end_iso,
            calendar_id=req.calendar_id,
            timezone_name=req.timezone,
        )
        return CalendarCheckAvailabilityResponse(**result).model_dump()

    def tool_calendar_create_event(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CalendarCreateEventRequest(**args)
        result = calendar_create_event(
            summary=req.summary,
            start_iso=req.start_iso,
            end_iso=req.end_iso,
            description=req.description,
            location=req.location,
            attendees=req.attendees,
            calendar_id=req.calendar_id,
            timezone_name=req.timezone,
            send_updates=req.send_updates,
        )
        return CalendarCreateEventResponse(**result).model_dump()

    def tool_calendar_propose_slots(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CalendarProposeSlotsRequest(**args)
        result = calendar_propose_slots(
            start_iso=req.start_iso,
            end_iso=req.end_iso,
            duration_minutes=req.duration_minutes,
            max_slots=req.max_slots,
            step_minutes=req.step_minutes,
            calendar_id=req.calendar_id,
            timezone_name=req.timezone,
        )
        return CalendarProposeSlotsResponse(**result).model_dump()

    def tool_calendar_update_event(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CalendarUpdateEventRequest(**args)
        result = calendar_update_event(
            event_id=req.event_id,
            calendar_id=req.calendar_id,
            summary=req.summary,
            description=req.description,
            location=req.location,
            send_updates=req.send_updates,
        )
        return CalendarUpdateEventResponse(**result).model_dump()

    def tool_calendar_hold_event(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CalendarHoldEventRequest(**args)
        result = calendar_hold_event(
            summary=req.summary,
            start_iso=req.start_iso,
            end_iso=req.end_iso,
            description=req.description,
            location=req.location,
            attendees=req.attendees,
            calendar_id=req.calendar_id,
            timezone_name=req.timezone,
            hold_minutes=req.hold_minutes,
            send_updates=req.send_updates,
        )
        return CalendarHoldEventResponse(**result).model_dump()

    registry.register(
        "calendar_check_availability",
        tool_calendar_check_availability,
        request_model=CalendarCheckAvailabilityRequest,
        response_model=CalendarCheckAvailabilityResponse,
    )
    registry.register(
        "calendar_create_event",
        tool_calendar_create_event,
        request_model=CalendarCreateEventRequest,
        response_model=CalendarCreateEventResponse,
    )
    registry.register(
        "calendar_propose_slots",
        tool_calendar_propose_slots,
        request_model=CalendarProposeSlotsRequest,
        response_model=CalendarProposeSlotsResponse,
    )
    registry.register(
        "calendar_update_event",
        tool_calendar_update_event,
        request_model=CalendarUpdateEventRequest,
        response_model=CalendarUpdateEventResponse,
    )
    registry.register(
        "calendar_hold_event",
        tool_calendar_hold_event,
        request_model=CalendarHoldEventRequest,
        response_model=CalendarHoldEventResponse,
    )
    # Alias with requested spelling.
    registry.register(
        "calender_hold_event",
        tool_calendar_hold_event,
        request_model=CalendarHoldEventRequest,
        response_model=CalendarHoldEventResponse,
    )
