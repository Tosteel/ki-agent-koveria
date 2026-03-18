from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CalendarCheckAvailabilityRequest(BaseModel):
    start_iso: str = Field(..., min_length=10, description="Startzeitpunkt als ISO-8601.")
    end_iso: str = Field(..., min_length=10, description="Endzeitpunkt als ISO-8601.")
    calendar_id: str = Field(default="primary", description="Google Calendar ID.")
    timezone: str = Field(default="Europe/Berlin", description="Zeitzone für die Anfrage.")


class CalendarBusyInterval(BaseModel):
    start: str = ""
    end: str = ""


class CalendarCheckAvailabilityResponse(BaseModel):
    calendar_id: str = "primary"
    start_iso: str = ""
    end_iso: str = ""
    timezone: str = "Europe/Berlin"
    is_available: bool = False
    busy: List[CalendarBusyInterval] = Field(default_factory=list)
    text: str = ""


class CalendarCreateEventRequest(BaseModel):
    summary: str = Field(..., min_length=1)
    start_iso: str = Field(..., min_length=10)
    end_iso: str = Field(..., min_length=10)
    description: str = ""
    location: str = ""
    attendees: List[str] = Field(default_factory=list)
    calendar_id: str = Field(default="primary")
    timezone: str = Field(default="Europe/Berlin")
    send_updates: str = Field(
        default="none",
        description="none|all|externalOnly, steuert Google Event-Benachrichtigungen.",
    )


class CalendarCreateEventResponse(BaseModel):
    created: bool = False
    event_id: str = ""
    html_link: str = ""
    status: str = ""
    start_iso: str = ""
    end_iso: str = ""
    text: str = ""


class CalendarProposeSlotsRequest(BaseModel):
    start_iso: str = Field(..., min_length=10, description="Suchfenster Start als ISO-8601.")
    end_iso: str = Field(..., min_length=10, description="Suchfenster Ende als ISO-8601.")
    duration_minutes: int = Field(default=30, ge=15, le=240)
    max_slots: int = Field(default=5, ge=1, le=20)
    step_minutes: int = Field(default=30, ge=5, le=120)
    calendar_id: str = Field(default="primary")
    timezone: str = Field(default="Europe/Berlin")


class CalendarSlot(BaseModel):
    start: str = ""
    end: str = ""


class CalendarProposeSlotsResponse(BaseModel):
    calendar_id: str = "primary"
    timezone: str = "Europe/Berlin"
    duration_minutes: int = 30
    slots: List[CalendarSlot] = Field(default_factory=list)
    text: str = ""

