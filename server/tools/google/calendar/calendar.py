from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import requests
from fastapi import HTTPException

_TOKEN_CACHE: Dict[str, str] = {"access_token": ""}


def _parse_iso(value: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise HTTPException(status_code=422, detail="ISO timestamp is required")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid ISO timestamp: {raw}") from exc


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _google_cfg() -> Dict[str, str]:
    token = _resolve_access_token()
    if not token:
        raise HTTPException(
            status_code=500,
            detail=(
                "Google OAuth token is missing. "
                "Set GOOGLE_ACCESS_TOKEN or configure refresh via "
                "GOOGLE_OAUTH_CLIENT_ID/GOOGLE_OAUTH_CLIENT_SECRET/GOOGLE_OAUTH_REFRESH_TOKEN."
            ),
        )
    return {
        "base_url": "https://www.googleapis.com/calendar/v3",
        "token": token,
    }


def _refresh_access_token() -> str:
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    refresh_token = os.getenv("GOOGLE_OAUTH_REFRESH_TOKEN", "").strip()
    if not client_id or not client_secret or not refresh_token:
        return ""

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    try:
        resp = requests.post("https://oauth2.googleapis.com/token", data=payload, timeout=20)
    except requests.RequestException:
        return ""
    if resp.status_code >= 400:
        return ""
    try:
        data = resp.json()
    except Exception:
        return ""
    token = str(data.get("access_token") or "").strip()
    if token:
        _TOKEN_CACHE["access_token"] = token
    return token


def _resolve_access_token() -> str:
    cached = str(_TOKEN_CACHE.get("access_token") or "").strip()
    if cached:
        return cached
    direct = os.getenv("GOOGLE_ACCESS_TOKEN", "").strip()
    if direct:
        _TOKEN_CACHE["access_token"] = direct
        return direct
    return _refresh_access_token()


def _google_request(
    *,
    method: str,
    path: str,
    params: Dict[str, Any] | None = None,
    json_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = _google_cfg()
    url = f"{cfg['base_url']}{path}"

    def _do_request(token: str) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        return requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params or None,
            json=json_payload or None,
            timeout=30,
        )

    try:
        resp = _do_request(cfg["token"])
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Google Calendar request failed: {exc}") from exc

    if resp.status_code == 401:
        refreshed = _refresh_access_token()
        if refreshed:
            try:
                resp = _do_request(refreshed)
            except requests.RequestException as exc:
                raise HTTPException(status_code=502, detail=f"Google Calendar request failed: {exc}") from exc

    if resp.status_code >= 400:
        detail = ""
        try:
            detail = str(resp.json())
        except Exception:
            detail = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"Google Calendar API error: {detail}")
    try:
        data = resp.json()
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def calendar_check_availability(
    *,
    start_iso: str,
    end_iso: str,
    calendar_id: str = "primary",
    timezone_name: str = "Europe/Berlin",
) -> Dict[str, Any]:
    start = _parse_iso(start_iso)
    end = _parse_iso(end_iso)
    if end <= start:
        raise HTTPException(status_code=422, detail="end_iso must be after start_iso")

    payload = {
        "timeMin": _to_iso(start),
        "timeMax": _to_iso(end),
        "timeZone": timezone_name,
        "items": [{"id": calendar_id}],
    }
    data = _google_request(method="POST", path="/freeBusy", json_payload=payload)
    calendars = data.get("calendars") if isinstance(data.get("calendars"), dict) else {}
    info = calendars.get(calendar_id) if isinstance(calendars.get(calendar_id), dict) else {}
    busy_raw = info.get("busy") if isinstance(info.get("busy"), list) else []
    busy: List[Dict[str, str]] = []
    for item in busy_raw:
        if not isinstance(item, dict):
            continue
        busy.append({"start": str(item.get("start") or ""), "end": str(item.get("end") or "")})

    is_available = len(busy) == 0
    text = (
        f"Kalender '{calendar_id}' ist im Zeitraum frei."
        if is_available
        else f"Kalender '{calendar_id}' ist im Zeitraum belegt ({len(busy)} Blocker)."
    )
    return {
        "calendar_id": calendar_id,
        "start_iso": _to_iso(start),
        "end_iso": _to_iso(end),
        "timezone": timezone_name,
        "is_available": is_available,
        "busy": busy,
        "text": text,
    }


def calendar_create_event(
    *,
    summary: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    location: str = "",
    attendees: List[str] | None = None,
    calendar_id: str = "primary",
    timezone_name: str = "Europe/Berlin",
    send_updates: str = "none",
) -> Dict[str, Any]:
    start = _parse_iso(start_iso)
    end = _parse_iso(end_iso)
    if end <= start:
        raise HTTPException(status_code=422, detail="end_iso must be after start_iso")
    send_updates_clean = str(send_updates or "none").strip()
    if send_updates_clean not in {"none", "all", "externalOnly"}:
        send_updates_clean = "none"

    payload: Dict[str, Any] = {
        "summary": str(summary or "").strip(),
        "description": str(description or "").strip(),
        "location": str(location or "").strip(),
        "start": {"dateTime": _to_iso(start), "timeZone": timezone_name},
        "end": {"dateTime": _to_iso(end), "timeZone": timezone_name},
    }
    att = []
    for mail in (attendees or []):
        em = str(mail or "").strip()
        if em:
            att.append({"email": em})
    if att:
        payload["attendees"] = att

    params = {"sendUpdates": send_updates_clean}
    data = _google_request(
        method="POST",
        path=f"/calendars/{calendar_id}/events",
        params=params,
        json_payload=payload,
    )
    event_id = str(data.get("id") or "")
    html_link = str(data.get("htmlLink") or "")
    status = str(data.get("status") or "")
    return {
        "created": bool(event_id),
        "event_id": event_id,
        "html_link": html_link,
        "status": status,
        "start_iso": _to_iso(start),
        "end_iso": _to_iso(end),
        "text": f"Termin {'erstellt' if event_id else 'nicht erstellt'}: {event_id or '-'}",
    }


def _slots_from_busy(
    *,
    window_start: datetime,
    window_end: datetime,
    busy: List[Dict[str, str]],
    duration_minutes: int,
    max_slots: int,
    step_minutes: int,
) -> List[Dict[str, str]]:
    busy_ranges: List[tuple[datetime, datetime]] = []
    for item in busy:
        try:
            s = _parse_iso(str(item.get("start") or ""))
            e = _parse_iso(str(item.get("end") or ""))
            if e > s:
                busy_ranges.append((s, e))
        except Exception:
            continue
    busy_ranges.sort(key=lambda x: x[0])

    slots: List[Dict[str, str]] = []
    cur = window_start
    delta_duration = timedelta(minutes=int(duration_minutes))
    delta_step = timedelta(minutes=int(step_minutes))
    while cur + delta_duration <= window_end and len(slots) < max_slots:
        candidate_end = cur + delta_duration
        overlap = False
        for b_start, b_end in busy_ranges:
            if cur < b_end and candidate_end > b_start:
                overlap = True
                break
        if not overlap:
            slots.append({"start": _to_iso(cur), "end": _to_iso(candidate_end)})
        cur = cur + delta_step
    return slots


def calendar_propose_slots(
    *,
    start_iso: str,
    end_iso: str,
    duration_minutes: int = 30,
    max_slots: int = 5,
    step_minutes: int = 30,
    calendar_id: str = "primary",
    timezone_name: str = "Europe/Berlin",
) -> Dict[str, Any]:
    window_start = _parse_iso(start_iso)
    window_end = _parse_iso(end_iso)
    if window_end <= window_start:
        raise HTTPException(status_code=422, detail="end_iso must be after start_iso")

    check = calendar_check_availability(
        start_iso=_to_iso(window_start),
        end_iso=_to_iso(window_end),
        calendar_id=calendar_id,
        timezone_name=timezone_name,
    )
    busy = check.get("busy") if isinstance(check.get("busy"), list) else []
    slots = _slots_from_busy(
        window_start=window_start,
        window_end=window_end,
        busy=busy,  # type: ignore[arg-type]
        duration_minutes=duration_minutes,
        max_slots=max_slots,
        step_minutes=step_minutes,
    )
    return {
        "calendar_id": calendar_id,
        "timezone": timezone_name,
        "duration_minutes": int(duration_minutes),
        "slots": slots,
        "text": f"Vorschläge gefunden: {len(slots)}",
    }
