from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

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


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/tools/google/calendar/check-availability", response_model=CalendarCheckAvailabilityResponse)
    def check_availability_route(
        req: CalendarCheckAvailabilityRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CalendarCheckAvailabilityResponse:
        ensure_user_dirs(s, user_id)
        result = calendar_check_availability(
            start_iso=req.start_iso,
            end_iso=req.end_iso,
            calendar_id=req.calendar_id,
            timezone_name=req.timezone,
        )
        return CalendarCheckAvailabilityResponse(**result)

    @router.post("/tools/google/calendar/create-event", response_model=CalendarCreateEventResponse)
    def create_event_route(
        req: CalendarCreateEventRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CalendarCreateEventResponse:
        ensure_user_dirs(s, user_id)
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
        return CalendarCreateEventResponse(**result)

    @router.post("/tools/google/calendar/hold-event", response_model=CalendarHoldEventResponse)
    def hold_event_route(
        req: CalendarHoldEventRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CalendarHoldEventResponse:
        ensure_user_dirs(s, user_id)
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
        return CalendarHoldEventResponse(**result)

    @router.post("/tools/google/calendar/update-event", response_model=CalendarUpdateEventResponse)
    def update_event_route(
        req: CalendarUpdateEventRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CalendarUpdateEventResponse:
        ensure_user_dirs(s, user_id)
        result = calendar_update_event(
            event_id=req.event_id,
            calendar_id=req.calendar_id,
            summary=req.summary,
            description=req.description,
            location=req.location,
            send_updates=req.send_updates,
        )
        return CalendarUpdateEventResponse(**result)

    @router.post("/tools/google/calendar/propose-slots", response_model=CalendarProposeSlotsResponse)
    def propose_slots_route(
        req: CalendarProposeSlotsRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CalendarProposeSlotsResponse:
        ensure_user_dirs(s, user_id)
        result = calendar_propose_slots(
            start_iso=req.start_iso,
            end_iso=req.end_iso,
            duration_minutes=req.duration_minutes,
            max_slots=req.max_slots,
            step_minutes=req.step_minutes,
            calendar_id=req.calendar_id,
            timezone_name=req.timezone,
        )
        return CalendarProposeSlotsResponse(**result)

    return router
