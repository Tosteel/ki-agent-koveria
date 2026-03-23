from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from fastapi import HTTPException

from server.agent.langchain_runtime import dispatch_tool_chain
from server.agent.tool_registry import ToolContext, ToolRegistry
from server.core.settings import Settings
from server.services.agent_service import build_registry
from server.services.llm_ionos import IonosLLM

from .models import (
    ActionType,
    BookingAssistantV3OperatorChatRequest,
    BookingAssistantV3OperatorChatResponse,
    BookingAssistantV3PendingApplyRequest,
    BookingAssistantV3PendingNextResponse,
    BookingAssistantV3ReviewActionResponse,
    BookingAssistantV3ReviewItem,
    BookingAssistantV3ReviewsResponse,
    BookingAssistantV3RunItem,
    BookingAssistantV3RunRequest,
    BookingAssistantV3RunResponse,
    BookingAssistantV3SimpleActionRequest,
    BookingAssistantV3StatusResponse,
)
from .store import (
    acquire_run_lock,
    add_review,
    append_activity,
    append_case_history,
    append_run_history,
    find_review,
    get_thread_case,
    has_processed,
    list_reviews,
    load_state,
    mark_processed,
    release_run_lock,
    save_state,
    upsert_thread_case,
)

_TRACE_ENABLED: ContextVar[bool] = ContextVar("booking_assistant_v3_trace_enabled", default=False)
_TRACE_STEP: ContextVar[int] = ContextVar("booking_assistant_v3_trace_step", default=0)


# ----------------------- basics -----------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_since_alias(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    now_utc = datetime.now(timezone.utc)
    if low in {"now", "jetzt"}:
        return now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if low in {"today", "heute"}:
        return now_utc.replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    if low in {"yesterday", "gestern"}:
        d = (now_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return d.isoformat().replace("+00:00", "Z")
    return raw


def _trace_log(title: str, lines: List[str] | None = None) -> None:
    if not _TRACE_ENABLED.get():
        return
    print("")
    print(f"===== {title} =====")
    for line in (lines or []):
        print(line)
    print("====================")
    print("")


def _tool_call(*, registry: ToolRegistry, ctx: ToolContext, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if registry.get_tool(tool) is None:
        raise HTTPException(status_code=400, detail=f"Required tool missing: {tool}")
    if _TRACE_ENABLED.get():
        step = _TRACE_STEP.get() + 1
        _TRACE_STEP.set(step)
        _trace_log(
            f"STEP INPUT {step}",
            [f"tool={tool}", f"args={json.dumps(args, ensure_ascii=False)[:2600]}"],
        )
    out = dispatch_tool_chain(registry=registry, tool_name=tool, ctx=ctx, args=args)
    payload = out if isinstance(out, dict) else {"value": out}
    if _TRACE_ENABLED.get():
        _trace_log(
            f"STEP OUTPUT {_TRACE_STEP.get()}",
            [f"tool={tool}", f"payload={json.dumps(payload, ensure_ascii=False)[:3600]}"],
        )
    return payload


def _safe_tool_call(*, registry: ToolRegistry, ctx: ToolContext, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if registry.get_tool(tool) is None:
        return {"_error": f"missing_tool:{tool}"}
    try:
        return _tool_call(registry=registry, ctx=ctx, tool=tool, args=args)
    except Exception as exc:
        return {"_error": str(exc)}


def _float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _dedupe(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        v = str(raw or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# ----------------------- profile -----------------------

def _normalize_required_fields(required_fields: List[str] | None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in (required_fields or []):
        key = str(raw or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    if out:
        return out
    return ["event_date", "start_time", "duration_hours", "location", "occasion", "client_name", "price_confirmed"]


def _rules_dict(rules: Dict[str, Any], key: str, alias: str = "") -> Dict[str, Any]:
    section = rules.get(key)
    if isinstance(section, dict):
        return section
    if alias:
        alt = rules.get(alias)
        if isinstance(alt, dict):
            return alt
    return {}


def _is_placeholder_text(value: str) -> bool:
    low = str(value or "").strip().lower()
    if not low:
        return True
    if low in {"string", "null", "none", "n/a"}:
        return True
    if low.startswith("additionalprop"):
        return True
    return False


def _sanitize_instructions_add(values: List[str] | None) -> List[str]:
    out: List[str] = []
    for raw in (values or []):
        v = str(raw or "").strip()
        if not v or _is_placeholder_text(v):
            continue
        out.append(v)
    return _dedupe(out)


def _sanitize_rules_patch_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k or "").strip()
            if not key or _is_placeholder_text(key):
                continue
            nested = _sanitize_rules_patch_value(v)
            if nested is None:
                continue
            if isinstance(nested, dict) and not nested:
                continue
            if isinstance(nested, list) and not nested:
                continue
            cleaned[key] = nested
        return cleaned
    if isinstance(value, list):
        cleaned_list: List[Any] = []
        for item in value:
            nested = _sanitize_rules_patch_value(item)
            if nested is None:
                continue
            if isinstance(nested, str) and _is_placeholder_text(nested):
                continue
            if isinstance(nested, dict) and not nested:
                continue
            if isinstance(nested, list) and not nested:
                continue
            cleaned_list.append(nested)
        return cleaned_list
    if isinstance(value, str):
        v = value.strip()
        if not v or _is_placeholder_text(v):
            return None
        return v
    return value


def _sanitize_rules_patch(patch: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        return {}
    cleaned = _sanitize_rules_patch_value(patch)
    if isinstance(cleaned, dict):
        return cleaned
    return {}


def _default_profile(name: str, codename: str = "") -> Dict[str, Any]:
    return {
        "assistent_profile_name": name,
        "codename": codename or "Booking Assistant",
        "instructions": [
            "Du bist ein Event-Booking-Assistent.",
            "Bleibe freundlich, präzise und prozessklar.",
            "Bestätige Termine nie final ohne Human-in-the-loop-Freigabe.",
        ],
        "rules": {
            "offering": {
                "summary": "Event-DJ für private und geschäftliche Veranstaltungen.",
            },
            "required_fields": [
                "event_date",
                "start_time",
                "duration_hours",
                "location",
                "occasion",
                "client_name",
                "price_confirmed",
            ],
            "rejection": {
                "weekend_only": False,
                "max_duration_hours": 0,
                "max_distance_km": 0,
                "blocked_weekdays": [],
                "latest_start_hour": 0,
            },
            "human_review": {
                "always_manual": False,
                "duration_over_hours": 0,
                "start_after_hour": 0,
                "distance_over_km": 0,
            },
            "calendar": {
                "calendar_id": "primary",
                "auto_decline_if_busy": True,
                "timezone": "Europe/Berlin",
            },
            "pricing": {
                "hourly_rate_eur": 120,
                "travel_per_km_eur": 0.7,
                "overnight_flat_eur": 120,
                "setup_flat_eur": 80,
                "teardown_flat_eur": 60,
                "travel_round_trip": True,
            },
            "booking": {
                "base_address": "Pforzheim, Deutschland",
                "max_distance_km": 200,
            },
            "mail": {
                "never_auto_send": False,
                "block_auto_reply_topics": [],
            },
        },
    }


def _ensure_profile(*, registry: ToolRegistry, ctx: ToolContext, req: BookingAssistantV3RunRequest) -> Dict[str, Any]:
    profile_name = str(req.assistant_profile_name or "booking_default").strip() or "booking_default"
    get_out = _safe_tool_call(registry=registry, ctx=ctx, tool="assistent_profile_get", args={"assistent_profile_name": profile_name})
    if get_out.get("_error"):
        if not req.profile_bootstrap:
            raise HTTPException(status_code=404, detail=f"Assistant profile not found: {profile_name}")
        base = _default_profile(profile_name, codename=req.assistant_codename)
        _tool_call(
            registry=registry,
            ctx=ctx,
            tool="assistent_profile_create",
            args={
                "assistent_profile_name": profile_name,
                "codename": str(base.get("codename") or ""),
                "instructions": list(base.get("instructions") or []),
                "rules": dict(base.get("rules") or {}),
            },
        )
        get_out = _tool_call(registry=registry, ctx=ctx, tool="assistent_profile_get", args={"assistent_profile_name": profile_name})

    clean_instructions_add = _sanitize_instructions_add(req.profile_instructions_add or [])
    clean_rules_patch = _sanitize_rules_patch(req.profile_rules_patch or {})

    if clean_instructions_add or clean_rules_patch:
        _tool_call(
            registry=registry,
            ctx=ctx,
            tool="assistent_profile_update",
            args={
                "assistent_profile_name": profile_name,
                "instructions_add": clean_instructions_add,
                "rules_patch": clean_rules_patch,
            },
        )
        get_out = _tool_call(registry=registry, ctx=ctx, tool="assistent_profile_get", args={"assistent_profile_name": profile_name})

    profile = get_out.get("profile") if isinstance(get_out.get("profile"), dict) else {}
    if not profile:
        raise HTTPException(status_code=500, detail="Assistant profile could not be loaded")
    return profile


# ----------------------- extraction / checks -----------------------

def _is_missing_required_field(name: str, value: Any) -> bool:
    key = str(name or "").strip()
    if key == "price_confirmed":
        return bool(value) is not True
    if key == "duration_hours":
        try:
            return float(value) <= 0.0
        except Exception:
            return True
    if value is None:
        return True
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return not value.strip()
    return False


def _serialize_required_value(name: str, value: Any) -> Any:
    if _is_missing_required_field(name, value):
        if str(name).strip() == "price_confirmed":
            return False
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def _build_required_status(*, required_fields: List[str], facts: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    missing: List[str] = []
    present: List[str] = []
    for key in required_fields:
        v = facts.get(key)
        values[key] = _serialize_required_value(key, v)
        if _is_missing_required_field(key, v):
            missing.append(key)
        else:
            present.append(key)
    return {
        "required_field_names": list(required_fields),
        "required_fields": values,
        "missing_required_fields": missing,
        "present_required_fields": present,
        "complete": len(missing) == 0,
        "updated_at": _now_iso(),
    }


def _merge_facts(base: Dict[str, Any], incoming: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    out = dict(base or {})
    for key in required_fields:
        value = incoming.get(key)
        if _is_missing_required_field(key, value):
            continue
        out[key] = value
    return out


def _normalize_fact_value(key: str, value: Any) -> Any:
    k = str(key or "").strip()
    if _is_missing_required_field(k, value):
        return ""
    if k == "duration_hours":
        try:
            return round(float(value), 2)
        except Exception:
            return ""
    if isinstance(value, str):
        return value.strip()
    return value


def _offer_signature_fields(required_fields: List[str]) -> List[str]:
    # price_confirmed is process-state and may change after the offer was sent;
    # it must not invalidate the offer signature.
    out: List[str] = []
    seen: set[str] = set()
    for raw in (required_fields or []):
        key = str(raw or "").strip()
        if not key or key == "price_confirmed" or key in seen:
            continue
        seen.add(key)
        out.append(key)
    if out:
        return out
    return ["event_date", "start_time", "duration_hours", "location", "occasion", "client_name"]


def _offer_signature_from_facts(facts: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    keys = _offer_signature_fields(required_fields)
    return {k: _normalize_fact_value(k, facts.get(k)) for k in keys}


def _offer_signature_matches(offer_state: Dict[str, Any], facts: Dict[str, Any], required_fields: List[str]) -> bool:
    if not isinstance(offer_state, dict):
        return False
    sig = offer_state.get("signature") if isinstance(offer_state.get("signature"), dict) else {}
    if not sig:
        return False
    current = _offer_signature_from_facts(facts, required_fields)
    for k, v in current.items():
        if _normalize_fact_value(k, sig.get(k)) != v:
            return False
    return True


def _weekday_de(date_iso: str) -> str:
    raw = str(date_iso or "").strip()
    if not raw:
        return ""
    try:
        d = datetime.fromisoformat(raw)
    except Exception:
        try:
            d = datetime.strptime(raw, "%Y-%m-%d")
        except Exception:
            return ""
    names = ["montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag"]
    return names[d.weekday()]


def _is_weekend(date_iso: str) -> bool | None:
    raw = str(date_iso or "").strip()
    if not raw:
        return None
    try:
        d = datetime.fromisoformat(raw)
    except Exception:
        try:
            d = datetime.strptime(raw, "%Y-%m-%d")
        except Exception:
            return None
    return d.weekday() >= 5


def _evaluate_rejection_rules(*, facts: Dict[str, Any], distance_km: float, rules: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []

    weekend_only = bool(rules.get("weekend_only", False))
    event_date = str(facts.get("event_date") or "").strip()
    if weekend_only and event_date:
        wk = _is_weekend(event_date)
        if wk is False:
            reasons.append("Buchungen sind nur am Wochenende möglich.")

    max_duration = _float_or(rules.get("max_duration_hours"), 0.0)
    duration = _float_or(facts.get("duration_hours"), 0.0)
    if max_duration > 0 and duration > max_duration:
        reasons.append(f"Anfrage überschreitet Maximaldauer ({duration:.1f}h > {max_duration:.1f}h).")

    max_distance = _float_or(rules.get("max_distance_km"), 0.0)
    if max_distance > 0 and distance_km > max_distance:
        reasons.append(f"Distanz überschreitet Limit ({distance_km:.1f}km > {max_distance:.1f}km).")

    blocked_weekdays = [str(x).strip().lower() for x in (rules.get("blocked_weekdays") or []) if str(x).strip()]
    if blocked_weekdays and event_date:
        w = _weekday_de(event_date)
        if w and w in blocked_weekdays:
            reasons.append(f"Anfragen am {w.capitalize()} werden nicht angenommen.")

    latest_start = int(_float_or(rules.get("latest_start_hour"), 0.0))
    start_time = str(facts.get("start_time") or "").strip()
    if latest_start > 0 and start_time:
        try:
            hh = int(start_time.split(":", 1)[0])
            if hh > latest_start:
                reasons.append(f"Startzeit liegt nach {latest_start:02d}:00 Uhr.")
        except Exception:
            pass

    return reasons


def _evaluate_human_review_rules(*, facts: Dict[str, Any], distance_km: float, rules: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []

    if bool(rules.get("always_manual", False)):
        reasons.append("Profilregel verlangt manuelle Prüfung.")

    duration_threshold = _float_or(rules.get("duration_over_hours"), 0.0)
    duration = _float_or(facts.get("duration_hours"), 0.0)
    if duration_threshold > 0 and duration > duration_threshold:
        reasons.append(f"Dauer über Human-Review-Schwelle ({duration:.1f}h > {duration_threshold:.1f}h).")

    start_after = int(_float_or(rules.get("start_after_hour"), 0.0))
    start_time = str(facts.get("start_time") or "").strip()
    if start_after > 0 and start_time:
        try:
            hh = int(start_time.split(":", 1)[0])
            if hh >= start_after:
                reasons.append(f"Startzeit erfordert manuelle Prüfung (>= {start_after:02d}:00).")
        except Exception:
            pass

    distance_threshold = _float_or(rules.get("distance_over_km"), 0.0)
    if distance_threshold > 0 and distance_km > distance_threshold:
        reasons.append(f"Distanz erfordert manuelle Prüfung ({distance_km:.1f}km > {distance_threshold:.1f}km).")

    return reasons


def _combine_start_end_iso(facts: Dict[str, Any]) -> Tuple[str, str]:
    event_date = str(facts.get("event_date") or "").strip()
    start_time = str(facts.get("start_time") or "").strip()
    duration = _float_or(facts.get("duration_hours"), 0.0)
    if not event_date or not start_time or duration <= 0:
        return "", ""
    try:
        hh, mm = [int(x) for x in start_time.split(":", 1)]
    except Exception:
        return "", ""
    try:
        start = datetime.fromisoformat(event_date).replace(hour=hh, minute=mm, second=0, microsecond=0)
    except Exception:
        return "", ""
    end = start + timedelta(hours=duration)
    return start.isoformat(), end.isoformat()


def _calendar_check(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    calendar_id: str,
    facts: Dict[str, Any],
) -> Dict[str, Any]:
    start_iso, end_iso = _combine_start_end_iso(facts)
    if not start_iso or not end_iso:
        return {
            "has_term": False,
            "checked": False,
            "is_available": None,
            "calendar_id": calendar_id,
            "start_iso": "",
            "end_iso": "",
            "text": "Kein vollständiger Termin in der Anfrage enthalten.",
        }
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="calendar_check_availability",
        args={"start_iso": start_iso, "end_iso": end_iso, "calendar_id": calendar_id},
    )
    if out.get("_error"):
        return {
            "has_term": True,
            "checked": False,
            "is_available": None,
            "calendar_id": calendar_id,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "text": str(out.get("_error") or "calendar_check_failed"),
        }
    return {
        "has_term": True,
        "checked": True,
        "is_available": bool(out.get("is_available")),
        "calendar_id": str(out.get("calendar_id") or calendar_id),
        "start_iso": str(out.get("start_iso") or start_iso),
        "end_iso": str(out.get("end_iso") or end_iso),
        "busy": out.get("busy") if isinstance(out.get("busy"), list) else [],
        "text": str(out.get("text") or "").strip(),
    }


def _case_calendar_snapshot(case: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(case, dict):
        return {}
    cal = case.get("calendar")
    if isinstance(cal, dict):
        return cal
    return {}


def _is_busy_due_to_own_hold(*, case: Dict[str, Any] | None, cal: Dict[str, Any]) -> bool:
    if not isinstance(cal, dict):
        return False
    if not bool(cal.get("has_term")) or not bool(cal.get("checked")):
        return False
    if cal.get("is_available") is not False:
        return False
    snap = _case_calendar_snapshot(case)
    hold_event_id = str(snap.get("hold_event_id") or "").strip()
    if not hold_event_id:
        return False
    hold_start = str(snap.get("start_iso") or "").strip()
    hold_end = str(snap.get("end_iso") or "").strip()
    start_iso = str(cal.get("start_iso") or "").strip()
    end_iso = str(cal.get("end_iso") or "").strip()
    if not hold_start or not hold_end or not start_iso or not end_iso:
        return False
    return hold_start == start_iso and hold_end == end_iso


def _apply_own_hold_override(*, case: Dict[str, Any] | None, cal: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_busy_due_to_own_hold(case=case, cal=cal):
        return cal
    out = dict(cal)
    out["is_available"] = True
    base = str(out.get("text") or "").strip()
    out["text"] = (base + " Eigener Kalender-Blocker für dieses Thread-Zeitfenster erkannt.").strip()
    return out


def _ensure_calendar_blocker(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    case: Dict[str, Any] | None,
    cal: Dict[str, Any],
    facts: Dict[str, Any],
    calendar_id: str,
    from_email: str,
    hold_minutes: int,
) -> Dict[str, Any]:
    if not bool(cal.get("has_term")) or not bool(cal.get("checked")) or cal.get("is_available") is not True:
        return {"created": False, "reason": "no_free_slot"}

    start_iso = str(cal.get("start_iso") or "").strip()
    end_iso = str(cal.get("end_iso") or "").strip()
    if not start_iso or not end_iso:
        return {"created": False, "reason": "missing_slot"}

    snap = _case_calendar_snapshot(case)
    existing_hold_id = str(snap.get("hold_event_id") or "").strip()
    existing_start = str(snap.get("start_iso") or "").strip()
    existing_end = str(snap.get("end_iso") or "").strip()
    if existing_hold_id and existing_start == start_iso and existing_end == end_iso:
        return {
            "created": False,
            "reason": "existing_hold",
            "event_id": existing_hold_id,
            "html_link": str(snap.get("hold_html_link") or "").strip(),
            "status": str(snap.get("hold_status") or "").strip(),
            "text": "Kalender-Blocker bereits vorhanden.",
        }

    summary = f"Booking-Anfrage: {str(facts.get('occasion') or 'Event')}".strip()
    description = "Vorläufiger Kalender-Blocker aus booking_assistant_v3."
    attendees = [str(from_email).strip()] if str(from_email).strip() else []

    args = {
        "summary": summary,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "description": description,
        "location": str(facts.get("location") or "").strip(),
        "attendees": attendees,
        "calendar_id": calendar_id,
        "hold_minutes": max(15, min(int(hold_minutes or 90), 1440)),
        "send_updates": "none",
    }
    out = _safe_tool_call(registry=registry, ctx=ctx, tool="calendar_hold_event", args=args)
    if out.get("_error"):
        return {"created": False, "reason": str(out.get("_error") or "hold_failed")}

    event_id = str(out.get("event_id") or "").strip()
    return {
        "created": bool(out.get("created") or event_id),
        "reason": "hold_created" if event_id else "hold_unknown",
        "event_id": event_id,
        "html_link": str(out.get("html_link") or "").strip(),
        "status": str(out.get("status") or "").strip(),
        "text": str(out.get("text") or "").strip(),
        "hold_expires_at": str(out.get("hold_expires_at") or "").strip(),
    }


# ----------------------- intent + retrieval -----------------------

def _classify_intent(*, registry: ToolRegistry, ctx: ToolContext, mail_payload: Dict[str, Any]) -> Dict[str, Any]:
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="mail_classify",
        args={
            "text": str(mail_payload.get("text") or "").strip(),
            "subject": str(mail_payload.get("subject") or "").strip(),
            "body_text": str(mail_payload.get("body_text") or "").strip(),
            "from_email": str(mail_payload.get("from_email") or "").strip(),
        },
    )
    if out.get("_error"):
        body = str(mail_payload.get("body_text") or "").lower()
        if any(x in body for x in ("termin", "buch", "verfügbar", "angebot", "hochzeit", "geburtstag", "firmen")):
            return {"intent": "termin", "confidence": 0.45, "reason": "heuristic_fallback"}
        if "newsletter" in body:
            return {"intent": "newsletter", "confidence": 0.45, "reason": "heuristic_fallback"}
        return {"intent": "info", "confidence": 0.0, "reason": "classify_failed"}

    intent = str(out.get("intent") or "info").strip().lower()
    if intent not in {"info", "termin", "newsletter", "angebot", "eskalation", "beschwerde"}:
        intent = "info"
    return {
        "intent": intent,
        "confidence": _float_or(out.get("confidence"), 0.0),
        "reason": str(out.get("reason") or "").strip(),
    }


def _read_mail_context(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    mailbox: str,
    mail_id: str,
    include_thread: bool,
) -> Dict[str, Any]:
    base = _tool_call(registry=registry, ctx=ctx, tool="gmail_read_mail", args={"mail_id": mail_id, "mailbox": mailbox, "max_chars": 20000})
    parts: List[str] = [str(base.get("text") or "").strip()]
    thread_facts_text = ""
    if include_thread:
        thread = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="gmail_read_mail_thread",
            args={"mail_id": mail_id, "mailbox": mailbox, "max_messages": 20, "max_chars": 10000},
        )
        t = str(thread.get("text") or "").strip()
        if t:
            parts.append("THREAD:\n" + t)
        msgs = thread.get("messages") if isinstance(thread.get("messages"), list) else []
        fact_parts: List[str] = []
        for item in reversed(msgs):
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject") or "").strip()
            body = str(item.get("body_text") or "").strip()
            if subject:
                fact_parts.append(f"Subject: {subject}")
            if body:
                fact_parts.append(body)
        thread_facts_text = "\n\n".join(x for x in fact_parts if x).strip()

    merged = "\n\n".join(x for x in parts if x).strip()
    out = dict(base)
    out["text"] = merged
    out["thread_facts_text"] = thread_facts_text
    return out


def _extract_sources(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if not isinstance(payload, dict):
        return out
    for key in ("final_url", "url", "source", "link", "href"):
        v = str(payload.get(key) or "").strip()
        if v:
            out.append(v)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    for item in items:
        if isinstance(item, dict):
            v = str(item.get("url") or item.get("href") or "").strip()
            if v:
                out.append(v)
    return _dedupe(out)


def _retrieve_info_context(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    query: str,
    rag_top_k: int,
    web_sources: List[str],
    web_whitelist_domains: List[str],
    max_context_chars: int,
) -> Dict[str, Any]:
    parts: List[str] = []
    sources: List[str] = []

    rag = _safe_tool_call(registry=registry, ctx=ctx, tool="rag_knowledgebase", args={"query": query, "top_k": rag_top_k})
    if not rag.get("_error"):
        t = str(rag.get("text") or "").strip()
        if t:
            parts.append("RAG:\n" + t)
        sources.extend(_extract_sources(rag))

    for raw_url in web_sources:
        url = str(raw_url or "").strip()
        if not url:
            continue
        out = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="web_crawl_site_whitelist",
            args={
                "url": url,
                "query": query,
                "allowed_domains": web_whitelist_domains,
                "max_pages": 10,
                "max_matches": 8,
            },
        )
        if out.get("_error"):
            out = _safe_tool_call(registry=registry, ctx=ctx, tool="web_crawl_site", args={"url": url, "query": query, "max_pages": 10, "max_matches": 8})
        if out.get("_error"):
            out = _safe_tool_call(registry=registry, ctx=ctx, tool="web_fetch_page", args={"url": url, "query": query})
        if out.get("_error"):
            continue
        t = str(out.get("text") or "").strip()
        if t:
            parts.append(f"WEB({url}):\n{t}")
        sources.extend(_extract_sources(out))

    merged = "\n\n".join(parts).strip()
    if len(merged) > max_context_chars:
        merged = merged[:max_context_chars].rstrip() + "…"

    return {"context_text": merged, "sources": _dedupe(sources)}


def _compose_info_reply(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    mail_payload: Dict[str, Any],
    profile: Dict[str, Any],
    context_text: str,
    sources: List[str],
) -> str:
    profile_txt = json.dumps(profile, ensure_ascii=False, indent=2)
    src_txt = "\n".join(f"- {s}" for s in sources[:20]) or "- keine"
    text = (
        "Kundenanfrage:\n"
        f"{str(mail_payload.get('text') or '').strip()}\n\n"
        "Assistentenprofil:\n"
        f"{profile_txt}\n\n"
        "Kontext:\n"
        f"{context_text or 'Kein zusätzlicher Kontext gefunden.'}\n\n"
        "Quellen:\n"
        f"{src_txt}"
    )
    instruction = (
        "Schreibe eine präzise, kurze Antwort auf Deutsch. "
        "Wenn Infos fehlen, stelle genau eine klare Rückfrage."
    )
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="llm_text_compose",
        args={"text": text, "instruction": instruction, "max_chars": 2200},
    )
    if out.get("_error"):
        return "Vielen Dank für Ihre Nachricht. Ich prüfe Ihr Anliegen und melde mich zeitnah mit einer konkreten Rückmeldung."
    return str(out.get("text") or "").strip()


# ----------------------- message composition -----------------------

def _field_label(field: str) -> str:
    mapping = {
        "event_date": "Datum",
        "start_time": "Startzeit",
        "duration_hours": "Dauer",
        "location": "Ort",
        "occasion": "Anlass",
        "client_name": "Name",
        "price_confirmed": "Angebotsbestätigung",
    }
    return mapping.get(str(field), str(field))


def _fmt_fact_value(name: str, value: Any) -> str:
    key = str(name)
    if key == "duration_hours":
        v = _float_or(value, 0.0)
        if v > 0:
            return f"{v:.1f} Stunden"
    if key == "price_confirmed":
        return "bestätigt" if bool(value) else "offen"
    if key == "event_date":
        raw = str(value or "").strip()
        try:
            dt = datetime.fromisoformat(raw)
            return dt.strftime("%d.%m.%Y")
        except Exception:
            return raw
    return str(value or "").strip()


def _known_facts_lines(facts: Dict[str, Any], required_fields: List[str]) -> List[str]:
    lines: List[str] = []
    for field in required_fields:
        val = facts.get(field)
        if _is_missing_required_field(field, val):
            continue
        lines.append(f"- {_field_label(field)}: {_fmt_fact_value(field, val)}")
    return lines


def _compose_refusal_mail(reasons: List[str]) -> str:
    r = "\n".join(f"- {x}" for x in reasons if str(x).strip())
    return (
        "Vielen Dank für Ihre Anfrage.\n\n"
        "Leider kann ich die Buchung unter den angefragten Rahmenbedingungen nicht zusagen.\n\n"
        f"Gründe:\n{r}\n\n"
        "Wenn Sie möchten, können Sie eine angepasste Anfrage senden."
    ).strip()


def _compose_busy_decline_mail() -> str:
    return (
        "Vielen Dank für Ihre Anfrage.\n\n"
        "Der angefragte Termin ist in diesem Zeitfenster leider nicht verfügbar.\n\n"
        "Bitte nennen Sie einen alternativen Termin oder eine alternative Startzeit."
    ).strip()


def _compose_human_review_hold_mail() -> str:
    return (
        "Vielen Dank für Ihre Anfrage.\n\n"
        "Ich habe Ihr Anliegen zur persönlichen Prüfung weitergegeben und melde mich zeitnah mit einer Rückmeldung."
    ).strip()


def _compose_offer_mail(*, quote_text: str, facts: Dict[str, Any], required_fields: List[str]) -> str:
    known = _known_facts_lines(facts, required_fields)
    known_block = "\n".join(known) if known else "- keine"
    offer_block = str(quote_text or "").strip() or "Preisübersicht folgt im nächsten Schritt."
    return (
        "Vielen Dank, die Anfrage kann weiterbearbeitet werden.\n\n"
        "Bereits erfasst:\n"
        f"{known_block}\n\n"
        "Hier ist das Angebot:\n"
        f"{offer_block}\n\n"
        "Bitte bestätigen Sie das Angebot kurz schriftlich. Erst danach kann ich den Termin verbindlich weiterführen."
    ).strip()


def _compose_missing_fields_mail(
    *,
    missing_fields: List[str],
    facts: Dict[str, Any],
    required_fields: List[str],
    precheck_text: str,
    optional_offer_text: str,
) -> str:
    missing_lines = "\n".join(f"- {_field_label(x)}" for x in missing_fields)
    known = _known_facts_lines(facts, required_fields)
    known_block = "\n".join(known) if known else "- keine"

    parts = [
        "Vielen Dank für Ihre Anfrage.",
        "",
        "Für eine verbindliche Bearbeitung fehlen noch diese Angaben:",
        missing_lines or "- weitere Details",
        "",
        "Bereits erfasst:",
        known_block,
    ]
    if precheck_text.strip():
        parts.extend(["", precheck_text.strip()])
    if optional_offer_text.strip():
        parts.extend(["", "Vorläufiges Angebot:", optional_offer_text.strip()])
    parts.extend(["", "Bitte senden Sie die fehlenden Angaben, dann setze ich den Prozess direkt fort."])
    return "\n".join(parts).strip()


def _compose_provisional_confirmation_mail() -> str:
    return (
        "Vielen Dank für die Angebotsbestätigung.\n\n"
        "Ich habe den Termin unter Vorbehalt vorgemerkt. "
        "Die finale Bestätigung erfolgt nach interner Freigabe."
    ).strip()


def _compose_final_confirmation_mail(*, event_link: str, quote_text: str) -> str:
    link_line = f"\nKalender-Link: {event_link}" if event_link else ""
    quote_block = f"\n\nBestätigtes Angebot:\n{quote_text}" if quote_text.strip() else ""
    return (
        "Vielen Dank für Ihre Geduld.\n\n"
        "Der Termin ist jetzt final bestätigt."
        f"{quote_block}{link_line}\n\n"
        "Bei Rückfragen melden Sie sich jederzeit."
    ).strip()


def _compose_final_rejection_mail(reason: str = "") -> str:
    reason_block = f"\n\nGrund: {reason.strip()}" if reason.strip() else ""
    return (
        "Vielen Dank für Ihre Anfrage.\n\n"
        "Nach finaler Prüfung kann ich die Buchung leider nicht bestätigen."
        f"{reason_block}\n\n"
        "Wenn Sie möchten, prüfen wir gern eine alternative Anfrage."
    ).strip()


# ----------------------- review template generation -----------------------

def _summarize_for_template(*, registry: ToolRegistry, ctx: ToolContext, action: str, body: str) -> str:
    text = str(body or "").strip()
    if not text:
        return ""
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="llm_text_compose",
        args={
            "text": text,
            "instruction": (
                f"Fasse die geplante Kundenantwort fuer die Aktion '{action}' in 1-2 Saetzen zusammen. "
                "Keine Platzhalter, keine Meta-Texte."
            ),
            "max_chars": 280,
        },
    )
    if out.get("_error"):
        short = " ".join(text.split())
        return short[:260].rstrip() + ("…" if len(short) > 260 else "")
    return str(out.get("text") or "").strip() or text[:260]


def _build_rule_action_templates(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    approve_body: str,
    offer_body: str,
    reject_body: str,
) -> Dict[str, str]:
    approve_summary = _summarize_for_template(registry=registry, ctx=ctx, action="approve", body=approve_body)
    offer_summary = _summarize_for_template(registry=registry, ctx=ctx, action="offer", body=offer_body)
    reject_summary = _summarize_for_template(registry=registry, ctx=ctx, action="reject", body=reject_body)
    return {
        "approve": approve_summary or "Prozess wird mit konkreter Kundenantwort fortgesetzt.",
        "offer": offer_summary or "Es wird ein angepasstes, regelkonformes Angebot gesendet.",
        "reject": reject_summary or "Es wird eine höfliche Absage mit Begründung gesendet.",
    }


# ----------------------- scoring/policy/instruction -----------------------

def _score_reply(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    user_message: str,
    draft: str,
    sources: List[str],
    booking_decision: str,
    facts: Dict[str, Any],
    required_fields: List[str],
    missing_fields: List[str],
) -> Dict[str, Any]:
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="booking_reply_score",
        args={
            "user_message": user_message,
            "draft_reply": draft,
            "booking_decision": booking_decision,
            "facts": facts,
            "required_fields": required_fields,
            "missing_fields": missing_fields,
            "knowledge_evidence": sources[:20],
            "require_actionable": True,
        },
    )
    if out.get("_error"):
        out = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="customer_support_reply_score",
            args={
                "user_message": user_message,
                "draft_reply": draft,
                "knowledge_evidence": sources[:20],
                "require_actionable": True,
            },
        )
    if out and not out.get("_error"):
        total = max(0.0, min(1.0, _float_or(out.get("total_score"), 0.0)))
        return {
            "score_total": total,
            "verdict": str(out.get("verdict") or "needs_review").strip().lower(),
            "reason": "; ".join(str(x) for x in (out.get("reasons") or [])[:3] if str(x).strip()),
            "raw": out,
        }
    fallback = 0.72 if len(draft.strip()) >= 80 else 0.58
    return {"score_total": fallback, "verdict": "send" if fallback >= 0.8 else "needs_review", "reason": "score_fallback", "raw": {}}


def _policy_check(*, registry: ToolRegistry, ctx: ToolContext, text: str, strict_mode: bool) -> Dict[str, Any]:
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="customer_support_policy_check",
        args={"text": text, "policy_profile": "default", "strict_mode": strict_mode},
    )
    if out and not out.get("_error"):
        return out
    return {"allowed": True, "risk_level": "unknown", "violations": [], "warnings": ["policy_check_unavailable"]}


def _instruction_check(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    instructions: List[str],
    user_message: str,
    draft_reply: str,
    booking_decision: str,
    facts: Dict[str, Any],
) -> Dict[str, Any]:
    if not instructions or not draft_reply.strip():
        return {"allowed": True, "risk_level": "low", "reason": "no_instructions"}
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="booking_instruction_check",
        args={
            "instructions": instructions,
            "user_message": user_message,
            "draft_reply": draft_reply,
            "booking_decision": booking_decision,
            "facts": facts,
        },
    )
    if out.get("_error"):
        return {"allowed": True, "risk_level": "low", "reason": "instruction_check_unavailable"}
    return {
        "allowed": bool(out.get("allowed")),
        "risk_level": str(out.get("risk_level") or "low").strip().lower(),
        "reason": str(out.get("reason") or "").strip(),
        "violations": [str(x).strip() for x in (out.get("violations") or []) if str(x).strip()],
    }


# ----------------------- offer acceptance -----------------------

def _llm_detect_offer_acceptance(*, latest_body_text: str, thread_text: str) -> Dict[str, Any]:
    client = IonosLLM()
    if not client.enabled():
        return {"accepted": False, "confidence": 0.0, "reason": "llm_unavailable", "model": ""}

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_v3_offer_acceptance",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "accepted": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["accepted", "confidence", "reason"],
            },
            "strict": True,
        },
    }
    prompt = (
        "Prüfe, ob die LETZTE Kundenantwort ein zuvor gesendetes Angebot eindeutig annimmt.\n"
        "- true nur bei klarer Zustimmung.\n"
        "- false bei Unklarheit, Rückfragen oder Themenwechsel.\n\n"
        f"Letzte Kundenantwort:\n{latest_body_text.strip()}\n\n"
        f"Thread-Kontext:\n{thread_text[:5000]}"
    )
    try:
        completion = client.chat_completions(
            messages=[
                {"role": "system", "content": "Du bist ein präziser Klassifikator. Antworte strikt als JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            max_tokens=140,
            temperature=0.0,
            top_p=0.1,
        )
        raw = IonosLLM.extract_text(completion)
        parsed = json.loads(raw) if raw else {}
        return {
            "accepted": bool(parsed.get("accepted")),
            "confidence": max(0.0, min(1.0, _float_or(parsed.get("confidence"), 0.0))),
            "reason": str(parsed.get("reason") or "").strip(),
            "model": client.cfg.model,
        }
    except Exception as exc:
        return {"accepted": False, "confidence": 0.0, "reason": f"llm_error:{exc}", "model": client.cfg.model}


# ----------------------- review objects -----------------------

def _new_review(
    *,
    kind: str,
    mail_id: str,
    thread_id: str,
    mailbox: str,
    from_email: str,
    subject: str,
    mail_text: str,
    draft_body: str,
    booking_decision: str,
    score: Dict[str, Any],
    reason: str,
    action_templates: Dict[str, str],
    required_fields_status: Dict[str, Any],
) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "id": uuid4().hex,
        "kind": str(kind),
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "mail_id": mail_id,
        "thread_id": thread_id,
        "mailbox": mailbox,
        "from_email": from_email,
        "subject": subject,
        "mail_text": mail_text,
        "draft_body": draft_body,
        "booking_decision": booking_decision,
        "score_total": _float_or(score.get("score_total"), 0.0),
        "score": dict(score or {}),
        "reason": reason,
        "action_templates": dict(action_templates or {}),
        "required_fields_status": dict(required_fields_status or {}),
        "selected_action": "",
        "sent": False,
        "send_result": {},
    }


def _activity_record(
    *,
    mail_id: str,
    thread_id: str,
    decision: str,
    booking_decision: str = "",
    review_id: str = "",
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "mail_id": str(mail_id or "").strip(),
        "thread_id": str(thread_id or "").strip(),
        "decision": str(decision or "").strip(),
        "booking_decision": str(booking_decision or "").strip(),
        "review_id": str(review_id or "").strip(),
        "reason": str(reason or "").strip(),
    }


def _review_summary_item(review: Dict[str, Any]) -> Dict[str, Any]:
    templates = review.get("action_templates") if isinstance(review.get("action_templates"), dict) else {}
    return {
        "id": str(review.get("id") or "").strip(),
        "kind": str(review.get("kind") or "").strip(),
        "status": str(review.get("status") or "").strip(),
        "mail_id": str(review.get("mail_id") or "").strip(),
        "thread_id": str(review.get("thread_id") or "").strip(),
        "from_email": str(review.get("from_email") or "").strip(),
        "subject": str(review.get("subject") or "").strip(),
        "reason": str(review.get("reason") or "").strip(),
        "created_at": str(review.get("created_at") or ""),
        "score_total": _float_or(review.get("score_total"), 0.0),
        "required_fields_status": review.get("required_fields_status") if isinstance(review.get("required_fields_status"), dict) else {},
        "options": {k: str(v or "").strip() for k, v in templates.items() if str(v or "").strip()},
    }


# ----------------------- run core -----------------------

def _should_auto_send(
    *,
    never_auto_send: bool,
    booking_decision: str,
    policy: Dict[str, Any],
    instruction: Dict[str, Any],
    score: Dict[str, Any],
    threshold: float,
) -> bool:
    if never_auto_send:
        return False
    if not bool(policy.get("allowed")):
        return False
    if str(policy.get("risk_level") or "").strip().lower() in {"high", "critical"}:
        return False

    # Auto-declines (e.g. slot unavailable / profile rejection) should be sent
    # without being blocked by instruction-score friction.
    if str(booking_decision or "").strip().lower() == "auto_decline":
        return True

    if not bool(instruction.get("allowed", True)):
        return False

    if booking_decision in {"need_clarification", "auto_decline", "provisional_confirmation", "info_reply"}:
        if booking_decision == "info_reply":
            return _float_or(score.get("score_total"), 0.0) >= threshold and str(score.get("verdict") or "") == "send"
        return True

    if booking_decision in {"human_review", "final_confirmation_pending"}:
        return False

    return _float_or(score.get("score_total"), 0.0) >= threshold and str(score.get("verdict") or "") == "send"


def _send_answer(*, registry: ToolRegistry, ctx: ToolContext, mail_id: str, mailbox: str, body: str, subject: str = "") -> Dict[str, Any]:
    args: Dict[str, Any] = {"mail_id": mail_id, "mailbox": mailbox, "body": body}
    if subject.strip():
        args["subject"] = subject.strip()
    return _tool_call(registry=registry, ctx=ctx, tool="gmail_answer_mail", args=args)


def _contains_blocked_topic(text: str, blocked_topics: List[str]) -> str:
    t = str(text or "").lower()
    for topic in blocked_topics:
        key = str(topic or "").strip().lower()
        if key and key in t:
            return key
    return ""


def run_once(*, user_id: str, settings: Settings, api_key: str, req: BookingAssistantV3RunRequest) -> BookingAssistantV3RunResponse:
    trace_token = _TRACE_ENABLED.set(bool(req.trace_steps))
    step_token = _TRACE_STEP.set(0)

    run_id = f"run_{uuid4().hex[:12]}"
    started_at = _now_iso()

    _trace_log(
        "BOOKING ASSISTANT V3 RUN",
        [
            f"user_id={user_id}",
            f"mailbox={req.mailbox}",
            f"limit={req.limit}",
            f"assistant_profile_name={req.assistant_profile_name}",
        ],
    )

    registry = build_registry(settings=settings, user_id=user_id)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_v3_run_once")

    state = load_state(settings, user_id)
    lock = acquire_run_lock(state, run_id=run_id, ttl_seconds=7200)
    if not bool(lock.get("acquired")):
        finished_at = _now_iso()
        append_run_history(
            state,
            {
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "lock_blocked": True,
                "lock_reason": str(lock.get("reason") or "run_already_active"),
                "processed_count": 0,
                "sent_count": 0,
                "review_count": 0,
                "skipped_count": 0,
            },
        )
        save_state(settings, user_id, state)
        _TRACE_ENABLED.reset(trace_token)
        _TRACE_STEP.reset(step_token)
        return BookingAssistantV3RunResponse(
            ok=True,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            lock_blocked=True,
            lock_reason=str(lock.get("reason") or "run_already_active"),
        )

    items: List[BookingAssistantV3RunItem] = []
    sent_count = 0
    review_count = 0
    skipped_count = 0
    processed_count = 0

    try:
        profile = _ensure_profile(registry=registry, ctx=ctx, req=req)
        instructions = [str(x).strip() for x in (profile.get("instructions") or []) if str(x).strip()]
        rules = profile.get("rules") if isinstance(profile.get("rules"), dict) else {}
        required_fields = _normalize_required_fields([str(x).strip() for x in (rules.get("required_fields") or []) if str(x).strip()])
        detail_required_fields = [x for x in required_fields if x != "price_confirmed"]
        rejection_rules = rules.get("rejection") if isinstance(rules.get("rejection"), dict) else {}
        human_review_rules = rules.get("human_review") if isinstance(rules.get("human_review"), dict) else {}
        calendar_rules = _rules_dict(rules, "calendar")
        pricing_rules = rules.get("pricing") if isinstance(rules.get("pricing"), dict) else {}
        booking_rules = rules.get("booking") if isinstance(rules.get("booking"), dict) else {}
        mail_rules = rules.get("mail") if isinstance(rules.get("mail"), dict) else {}

        never_auto_send = bool(mail_rules.get("never_auto_send"))
        blocked_topics = [str(x).strip() for x in (mail_rules.get("block_auto_reply_topics") or []) if str(x).strip()]
        calendar_id = str(calendar_rules.get("calendar_id") or "primary").strip() or "primary"
        auto_decline_if_busy = bool(calendar_rules.get("auto_decline_if_busy", True))

        fetch_limit = max(1, min(50, int(req.limit) * 5))
        inbox = _tool_call(registry=registry, ctx=ctx, tool="gmail_fetch_unanswered_mails", args={"mailbox": req.mailbox, "limit": fetch_limit})
        emails = inbox.get("emails") if isinstance(inbox.get("emails"), list) else []
        _trace_log("INBOX RESULT", [f"emails_found={len(emails)}", f"scan_limit={fetch_limit}", f"process_limit={req.limit}"])

        eligible_seen = 0
        for raw in emails:
            if not isinstance(raw, dict):
                continue
            mail_id = str(raw.get("uid") or "").strip()
            subject = str(raw.get("subject") or "").strip()
            from_email = str(raw.get("from_email") or "").strip()
            if not mail_id:
                continue

            _trace_log("MAIL START", [f"mail_id={mail_id}", f"from={from_email}", f"subject={subject}"])

            if has_processed(state, mail_id):
                skipped_count += 1
                items.append(BookingAssistantV3RunItem(mail_id=mail_id, subject=subject, from_email=from_email, decision="skipped", reason="already_processed"))
                append_activity(state, _activity_record(mail_id=mail_id, thread_id="", decision="skipped", reason="already_processed"))
                continue

            eligible_seen += 1
            if eligible_seen > int(req.limit):
                break

            try:
                mail_payload = _read_mail_context(
                    registry=registry,
                    ctx=ctx,
                    mailbox=req.mailbox,
                    mail_id=mail_id,
                    include_thread=req.include_thread,
                )
            except Exception as exc:
                items.append(BookingAssistantV3RunItem(mail_id=mail_id, subject=subject, from_email=from_email, decision="failed", reason=f"read_mail_failed:{exc}"))
                append_activity(state, _activity_record(mail_id=mail_id, thread_id="", decision="failed", reason=f"read_mail_failed:{exc}"))
                continue

            thread_id = str(mail_payload.get("thread_id") or "").strip() or mail_id
            intent_info = _classify_intent(registry=registry, ctx=ctx, mail_payload=mail_payload)
            intent = str(intent_info.get("intent") or "info")
            _trace_log(
                "INTENT",
                [f"mail_id={mail_id}", f"intent={intent}", f"confidence={_float_or(intent_info.get('confidence'), 0.0):.2f}", f"reason={intent_info.get('reason')}"],
            )

            if intent == "newsletter":
                skipped_count += 1
                processed_count += 1
                mark_processed(state, mail_id)
                items.append(BookingAssistantV3RunItem(mail_id=mail_id, thread_id=thread_id, subject=subject, from_email=from_email, intent=intent, decision="skipped", reason="intent=newsletter"))
                append_activity(state, _activity_record(mail_id=mail_id, thread_id=thread_id, decision="skipped", reason="intent=newsletter"))
                continue

            draft = ""
            booking_decision = ""
            reasons: List[str] = []
            sources: List[str] = []
            score: Dict[str, Any] = {"score_total": 0.0, "verdict": "needs_review", "reason": ""}
            required_status: Dict[str, Any] = {}
            facts_for_instruction: Dict[str, Any] = {}

            if intent == "info":
                blocked = _contains_blocked_topic(str(mail_payload.get("text") or ""), blocked_topics)
                if blocked:
                    draft = _compose_human_review_hold_mail()
                    booking_decision = "human_review"
                    reasons = [f"mail_topic_blocked:{blocked}"]
                else:
                    query = "\n".join(
                        x for x in [str(mail_payload.get("subject") or "").strip(), str(mail_payload.get("body_text") or "").strip()] if x
                    ).strip()
                    context = _retrieve_info_context(
                        registry=registry,
                        ctx=ctx,
                        query=query,
                        rag_top_k=req.rag_top_k,
                        web_sources=req.web_sources,
                        web_whitelist_domains=req.web_whitelist_domains,
                        max_context_chars=req.max_context_chars,
                    )
                    sources = [str(x).strip() for x in (context.get("sources") or []) if str(x).strip()]
                    draft = _compose_info_reply(
                        registry=registry,
                        ctx=ctx,
                        mail_payload=mail_payload,
                        profile=profile,
                        context_text=str(context.get("context_text") or ""),
                        sources=sources,
                    )
                    booking_decision = "info_reply"

                score = _score_reply(
                    registry=registry,
                    ctx=ctx,
                    user_message=str(mail_payload.get("text") or ""),
                    draft=draft,
                    sources=sources,
                    booking_decision=booking_decision,
                    facts={},
                    required_fields=[],
                    missing_fields=[],
                )
            else:
                # booking path
                case = get_thread_case(state, thread_id)
                case_required = case.get("required_fields") if isinstance(case, dict) and isinstance(case.get("required_fields"), dict) else {}
                case_facts = case.get("facts") if isinstance(case, dict) and isinstance(case.get("facts"), dict) else {}

                latest_text = "\n".join(
                    x for x in [str(mail_payload.get("subject") or "").strip(), str(mail_payload.get("body_text") or "").strip()] if x
                ).strip() or str(mail_payload.get("text") or "")

                ext_latest = _safe_tool_call(registry=registry, ctx=ctx, tool="booking_extract_facts", args={"text": latest_text, "required_fields": required_fields})
                ext_thread = _safe_tool_call(
                    registry=registry,
                    ctx=ctx,
                    tool="booking_extract_facts",
                    args={"text": str(mail_payload.get("thread_facts_text") or str(mail_payload.get("text") or "")), "required_fields": required_fields},
                )
                facts_latest = ext_latest.get("facts") if isinstance(ext_latest.get("facts"), dict) else {}
                facts_thread = ext_thread.get("facts") if isinstance(ext_thread.get("facts"), dict) else {}

                # Merge priority: case fallback < thread context < latest customer message.
                facts = _merge_facts({}, case_facts, required_fields)
                facts = _merge_facts(facts, case_required, required_fields)
                facts = _merge_facts(facts, facts_thread, required_fields)
                facts = _merge_facts(facts, facts_latest, required_fields)
                facts_for_instruction = dict(facts)
                case_offer = case.get("offer") if isinstance(case, dict) and isinstance(case.get("offer"), dict) else {}
                offer_stale = bool(case_offer) and not _offer_signature_matches(case_offer, facts, required_fields)

                required_status = _build_required_status(required_fields=required_fields, facts=facts)
                missing_detail_fields = [x for x in detail_required_fields if x in required_status.get("missing_required_fields", [])]

                # Distance/quote if possible
                distance_km = 0.0
                quote_text = ""
                quote_payload: Dict[str, Any] = {}
                if str(facts.get("location") or "").strip():
                    distance = _safe_tool_call(
                        registry=registry,
                        ctx=ctx,
                        tool="distance_check",
                        args={
                            "origin": str(booking_rules.get("base_address") or "Pforzheim, Deutschland"),
                            "destination": str(facts.get("location") or "").strip(),
                            "max_distance_km": _float_or(booking_rules.get("max_distance_km"), 0.0),
                        },
                    )
                    distance_km = _float_or(distance.get("distance_km"), 0.0)
                if _float_or(facts.get("duration_hours"), 0.0) > 0:
                    quote_payload = _safe_tool_call(
                        registry=registry,
                        ctx=ctx,
                        tool="pricing_compute_quote",
                        args={
                            "facts": facts,
                            "pricing_rules": pricing_rules,
                            "booking_rules": booking_rules,
                            "distance_km": distance_km,
                        },
                    )
                    quote_text = str(quote_payload.get("text") or "").strip()

                # Step 3: rules check on found fields
                reject_reasons = _evaluate_rejection_rules(facts=facts, distance_km=distance_km, rules=rejection_rules)
                human_reasons = _evaluate_human_review_rules(facts=facts, distance_km=distance_km, rules=human_review_rules)

                # Step 4: calendar check
                cal = _calendar_check(registry=registry, ctx=ctx, calendar_id=calendar_id, facts=facts)
                cal = _apply_own_hold_override(case=case, cal=cal)
                blocker: Dict[str, Any] = {"created": False, "reason": "not_applicable"}
                if not reject_reasons and cal.get("is_available") is True:
                    hold_minutes = int(_float_or(calendar_rules.get("hold_minutes"), 90.0))
                    blocker = _ensure_calendar_blocker(
                        registry=registry,
                        ctx=ctx,
                        case=case,
                        cal=cal,
                        facts=facts,
                        calendar_id=calendar_id,
                        from_email=from_email,
                        hold_minutes=hold_minutes,
                    )
                if str(blocker.get("event_id") or "").strip():
                    cal["hold_event_id"] = str(blocker.get("event_id") or "").strip()
                    cal["hold_html_link"] = str(blocker.get("html_link") or "").strip()
                    cal["hold_status"] = str(blocker.get("status") or "").strip()
                    cal["hold_expires_at"] = str(blocker.get("hold_expires_at") or "").strip()
                calendar_text = str(cal.get("text") or "").strip()
                blocker_text = str(blocker.get("text") or "").strip()
                if blocker_text:
                    calendar_text = (calendar_text + "\n" + blocker_text).strip()
                elif str(blocker.get("event_id") or "").strip():
                    calendar_text = (calendar_text + "\nKalender-Blocker wurde gesetzt.").strip()

                # persist case snapshot
                case_calendar = _case_calendar_snapshot(case)
                case_patch: Dict[str, Any] = {
                    "assistant_profile_name": str(req.assistant_profile_name or "booking_default").strip() or "booking_default",
                    "last_mail_id": mail_id,
                    "from_email": from_email,
                    "subject": subject,
                    "required_field_names": required_fields,
                    "required_fields": dict(required_status.get("required_fields") or {}),
                    "missing_required_fields": list(required_status.get("missing_required_fields") or []),
                    "facts": {k: facts.get(k) for k in required_fields if k in facts},
                    "calendar": {
                        "has_term": bool(cal.get("has_term")),
                        "checked": bool(cal.get("checked")),
                        "is_available": cal.get("is_available"),
                        "calendar_id": str(cal.get("calendar_id") or calendar_id),
                        "start_iso": str(cal.get("start_iso") or ""),
                        "end_iso": str(cal.get("end_iso") or ""),
                        "hold_event_id": str(cal.get("hold_event_id") or case_calendar.get("hold_event_id") or ""),
                        "hold_html_link": str(cal.get("hold_html_link") or case_calendar.get("hold_html_link") or ""),
                        "hold_status": str(cal.get("hold_status") or case_calendar.get("hold_status") or ""),
                        "hold_expires_at": str(cal.get("hold_expires_at") or case_calendar.get("hold_expires_at") or ""),
                        "text": calendar_text,
                    },
                }
                if offer_stale:
                    case_patch["offer"] = {}
                upsert_thread_case(state, thread_id=thread_id, patch=case_patch)

                if reject_reasons:
                    booking_decision = "auto_decline"
                    reasons = list(reject_reasons)
                    draft = _compose_refusal_mail(reasons)
                elif bool(cal.get("has_term")) and bool(cal.get("checked")) and cal.get("is_available") is False and auto_decline_if_busy:
                    booking_decision = "auto_decline"
                    reasons = ["Angefragtes Zeitfenster ist im Kalender belegt."]
                    draft = _compose_busy_decline_mail()
                elif human_reasons:
                    # step 3 -> human review
                    booking_decision = "human_review"
                    reasons = list(human_reasons)
                    draft = _compose_human_review_hold_mail()
                    approve_body = _resume_rule_review_after_approve(
                        state=state,
                        registry=registry,
                        ctx=ctx,
                        review={
                            "thread_id": thread_id,
                            "from_email": from_email,
                        },
                        profile_rules=rules,
                    )
                    offer_body = _compose_offer_action_body(
                        state=state,
                        registry=registry,
                        ctx=ctx,
                        review={"thread_id": thread_id, "from_email": from_email},
                        profile_rules=rules,
                    )
                    reject_body = _compose_reject_action_body(
                        review={"reason": "; ".join(reasons)},
                        explicit_reason="; ".join(reasons),
                    )
                    action_templates = _build_rule_action_templates(
                        registry=registry,
                        ctx=ctx,
                        approve_body=approve_body,
                        offer_body=offer_body,
                        reject_body=reject_body,
                    )

                    review = _new_review(
                        kind="rule_review",
                        mail_id=mail_id,
                        thread_id=thread_id,
                        mailbox=req.mailbox,
                        from_email=from_email,
                        subject=subject,
                        mail_text=str(mail_payload.get("text") or ""),
                        draft_body=draft,
                        booking_decision=booking_decision,
                        score={"score_total": 0.85, "verdict": "needs_review", "reason": "human_review_required", "raw": {"bypassed": True}},
                        reason="; ".join(reasons),
                        action_templates=action_templates,
                        required_fields_status=required_status,
                    )
                    add_review(state, review)
                    mark_processed(state, mail_id)
                    processed_count += 1
                    review_count += 1
                    append_activity(
                        state,
                        _activity_record(
                            mail_id=mail_id,
                            thread_id=thread_id,
                            decision="needs_human",
                            booking_decision=booking_decision,
                            review_id=str(review.get("id") or ""),
                            reason="; ".join(reasons),
                        ),
                    )
                    append_case_history(
                        state,
                        thread_id=thread_id,
                        history_item={"mail_id": mail_id, "booking_decision": booking_decision, "reason": "; ".join(reasons)},
                    )
                    items.append(
                        BookingAssistantV3RunItem(
                            mail_id=mail_id,
                            thread_id=thread_id,
                            subject=subject,
                            from_email=from_email,
                            intent=intent,
                            decision="needs_human",
                            booking_decision=booking_decision,
                            review_id=str(review.get("id") or ""),
                            score_total=0.85,
                            reason="; ".join(reasons),
                        )
                    )
                    continue
                elif missing_detail_fields:
                    # Step 5: ask for missing fields, optional offer if duration known and calendar free
                    optional_offer = ""
                    if _float_or(facts.get("duration_hours"), 0.0) > 0 and cal.get("is_available") is True and quote_text.strip():
                        optional_offer = quote_text
                        upsert_thread_case(
                            state,
                            thread_id=thread_id,
                            patch={
                                "offer": {
                                    "status": "sent",
                                    "quote_text": quote_text,
                                    "sent_at": _now_iso(),
                                    "signature": _offer_signature_from_facts(facts, required_fields),
                                }
                            },
                        )
                    booking_decision = "need_clarification"
                    reasons = ["missing_required_fields"]
                    draft = _compose_missing_fields_mail(
                        missing_fields=missing_detail_fields,
                        facts=facts,
                        required_fields=required_fields,
                        precheck_text=calendar_text,
                        optional_offer_text=optional_offer,
                    )
                else:
                    # Step 6: full fields path
                    case_now = get_thread_case(state, thread_id) or {}
                    offer_state = case_now.get("offer") if isinstance(case_now.get("offer"), dict) else {}
                    offer_status = str(offer_state.get("status") or "").strip().lower()
                    offer_present = (
                        offer_status in {"sent", "accepted"}
                        and bool(str(offer_state.get("quote_text") or "").strip())
                        and _offer_signature_matches(offer_state, facts, required_fields)
                    )

                    if not offer_present:
                        booking_decision = "need_clarification"
                        reasons = ["offer_not_sent_yet"]
                        draft = _compose_offer_mail(quote_text=quote_text, facts=facts, required_fields=required_fields)
                        upsert_thread_case(
                            state,
                            thread_id=thread_id,
                            patch={
                                "offer": {
                                    "status": "sent",
                                    "quote_text": quote_text,
                                    "sent_at": _now_iso(),
                                    "signature": _offer_signature_from_facts(facts, required_fields),
                                },
                                "status": "waiting_customer",
                            },
                        )
                    else:
                        accept = _llm_detect_offer_acceptance(
                            latest_body_text=str(mail_payload.get("body_text") or ""),
                            thread_text=str(mail_payload.get("text") or ""),
                        )
                        accepted = bool(accept.get("accepted")) and _float_or(accept.get("confidence"), 0.0) >= 0.7
                        if not accepted:
                            booking_decision = "need_clarification"
                            reasons = ["offer_confirmation_missing"]
                            draft = _compose_offer_mail(
                                quote_text=str(offer_state.get("quote_text") or quote_text),
                                facts=facts,
                                required_fields=required_fields,
                            )
                        else:
                            booking_decision = "provisional_confirmation"
                            reasons = ["offer_accepted_waiting_final_confirmation"]
                            facts["price_confirmed"] = True
                            required_status = _build_required_status(required_fields=required_fields, facts=facts)
                            draft = _compose_provisional_confirmation_mail()

                            # create final confirmation review (step 6)
                            review = _new_review(
                                kind="final_confirmation",
                                mail_id=mail_id,
                                thread_id=thread_id,
                                mailbox=req.mailbox,
                                from_email=from_email,
                                subject=subject,
                                mail_text=str(mail_payload.get("text") or ""),
                                draft_body=draft,
                                booking_decision="final_confirmation_pending",
                                score={"score_total": 0.9, "verdict": "needs_review", "reason": "final_confirmation_required", "raw": {"bypassed": True}},
                                reason="Finale manuelle Bestätigung erforderlich.",
                                action_templates={
                                    "final_confirmation": "Termin final bestätigen und verbindlich zusagen.",
                                    "final_rejection": "Anfrage final ablehnen.",
                                },
                                required_fields_status=required_status,
                            )
                            add_review(state, review)
                            review_count += 1
                            append_activity(
                                state,
                                _activity_record(
                                    mail_id=mail_id,
                                    thread_id=thread_id,
                                    decision="needs_human",
                                    booking_decision="final_confirmation_pending",
                                    review_id=str(review.get("id") or ""),
                                    reason="final_confirmation_required",
                                ),
                            )
                            upsert_thread_case(
                                state,
                                thread_id=thread_id,
                                patch={
                                    "status": "waiting_final_confirmation",
                                    "offer": {
                                        "status": "accepted",
                                        "quote_text": str(offer_state.get("quote_text") or quote_text),
                                        "accepted_at": _now_iso(),
                                        "accept_reason": str(accept.get("reason") or ""),
                                        "signature": _offer_signature_from_facts(facts, required_fields),
                                    },
                                },
                            )

                # Common send/review gate for non-human-review branch
                score = _score_reply(
                    registry=registry,
                    ctx=ctx,
                    user_message=str(mail_payload.get("text") or ""),
                    draft=draft,
                    sources=sources,
                    booking_decision=booking_decision,
                    facts=facts,
                    required_fields=required_fields,
                    missing_fields=list(required_status.get("missing_required_fields") or []),
                )

            instruction = _instruction_check(
                registry=registry,
                ctx=ctx,
                instructions=instructions,
                user_message=str(mail_payload.get("text") or ""),
                draft_reply=draft,
                booking_decision=booking_decision,
                facts=facts_for_instruction,
            )
            policy = _policy_check(registry=registry, ctx=ctx, text=draft, strict_mode=req.strict_policy)

            can_auto_send = _should_auto_send(
                never_auto_send=never_auto_send,
                booking_decision=booking_decision,
                policy=policy,
                instruction=instruction,
                score=score,
                threshold=req.auto_send_threshold,
            )

            if can_auto_send:
                send_out = _send_answer(
                    registry=registry,
                    ctx=ctx,
                    mail_id=mail_id,
                    mailbox=req.mailbox,
                    body=draft,
                )
                sent_count += 1
                processed_count += 1
                mark_processed(state, mail_id)
                append_activity(
                    state,
                    _activity_record(
                        mail_id=mail_id,
                        thread_id=thread_id,
                        decision="auto_sent",
                        booking_decision=booking_decision or "info_reply",
                        reason=str(score.get("reason") or "auto_sent"),
                    ),
                )
                append_case_history(
                    state,
                    thread_id=thread_id,
                    history_item={
                        "mail_id": mail_id,
                        "booking_decision": booking_decision,
                        "reason": str(score.get("reason") or "auto_sent"),
                    },
                )
                items.append(
                    BookingAssistantV3RunItem(
                        mail_id=mail_id,
                        thread_id=thread_id,
                        subject=subject,
                        from_email=from_email,
                        intent=intent,
                        decision="auto_sent",
                        booking_decision=booking_decision,
                        sent=bool(send_out.get("sent")),
                        score_total=_float_or(score.get("score_total"), 0.0),
                        reason=str(score.get("reason") or "auto_sent"),
                    )
                )
                continue

            # fallback to review
            reason_parts: List[str] = []
            if never_auto_send:
                reason_parts.append("never_auto_send")
            if not bool(policy.get("allowed")):
                reason_parts.append("policy_blocked")
            if not bool(instruction.get("allowed", True)):
                reason_parts.append("instruction_blocked")
            if booking_decision:
                reason_parts.append(f"booking_decision={booking_decision}")
            base_reason = str(score.get("reason") or "needs_human_review")
            reason_text = " | ".join([base_reason] + reason_parts)

            review = _new_review(
                kind="rule_review",
                mail_id=mail_id,
                thread_id=thread_id,
                mailbox=req.mailbox,
                from_email=from_email,
                subject=subject,
                mail_text=str(mail_payload.get("text") or ""),
                draft_body=draft,
                booking_decision=booking_decision,
                score=score,
                reason=reason_text,
                action_templates=_build_rule_action_templates(
                    registry=registry,
                    ctx=ctx,
                    approve_body=_resume_rule_review_after_approve(
                        state=state,
                        registry=registry,
                        ctx=ctx,
                        review={"thread_id": thread_id, "from_email": from_email},
                        profile_rules=rules,
                    ),
                    offer_body=_compose_offer_action_body(
                        state=state,
                        registry=registry,
                        ctx=ctx,
                        review={"thread_id": thread_id, "from_email": from_email},
                        profile_rules=rules,
                    ),
                    reject_body=_compose_reject_action_body(
                        review={"reason": reason_text},
                        explicit_reason=reason_text,
                    ),
                ),
                required_fields_status=required_status,
            )
            add_review(state, review)
            review_count += 1
            processed_count += 1
            mark_processed(state, mail_id)
            append_activity(
                state,
                _activity_record(
                    mail_id=mail_id,
                    thread_id=thread_id,
                    decision="needs_human",
                    booking_decision=booking_decision,
                    review_id=str(review.get("id") or ""),
                    reason=reason_text,
                ),
            )
            append_case_history(
                state,
                thread_id=thread_id,
                history_item={"mail_id": mail_id, "booking_decision": booking_decision, "reason": reason_text},
            )
            items.append(
                BookingAssistantV3RunItem(
                    mail_id=mail_id,
                    thread_id=thread_id,
                    subject=subject,
                    from_email=from_email,
                    intent=intent,
                    decision="needs_human",
                    booking_decision=booking_decision,
                    review_id=str(review.get("id") or ""),
                    sent=False,
                    score_total=_float_or(score.get("score_total"), 0.0),
                    reason=reason_text,
                )
            )

        finished_at = _now_iso()
        release_run_lock(state, run_id=run_id)
        append_run_history(
            state,
            {
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "lock_blocked": False,
                "lock_reason": "",
                "processed_count": processed_count,
                "sent_count": sent_count,
                "review_count": review_count,
                "skipped_count": skipped_count,
            },
        )
        save_state(settings, user_id, state)
        return BookingAssistantV3RunResponse(
            ok=True,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            lock_blocked=False,
            lock_reason="",
            processed_count=processed_count,
            sent_count=sent_count,
            review_count=review_count,
            skipped_count=skipped_count,
            items=items,
        )
    finally:
        _TRACE_ENABLED.reset(trace_token)
        _TRACE_STEP.reset(step_token)


# ----------------------- operator / hitl -----------------------

def _apply_action_send(
    *,
    state: Dict[str, Any],
    registry: ToolRegistry,
    ctx: ToolContext,
    review: Dict[str, Any],
    body: str,
    subject: str,
    action: str,
) -> Dict[str, Any]:
    send_result = _send_answer(
        registry=registry,
        ctx=ctx,
        mail_id=str(review.get("mail_id") or ""),
        mailbox=str(review.get("mailbox") or "INBOX"),
        body=body,
        subject=subject,
    )
    review["sent"] = bool(send_result.get("sent"))
    review["send_result"] = send_result
    review["draft_body"] = body
    review["selected_action"] = action
    review["updated_at"] = _now_iso()
    return send_result


def _action_template(review: Dict[str, Any], action: str) -> str:
    templates = review.get("action_templates") if isinstance(review.get("action_templates"), dict) else {}
    return str(templates.get(action) or templates.get("default") or review.get("draft_body") or "").strip()


def _resume_rule_review_after_approve(
    *,
    state: Dict[str, Any],
    registry: ToolRegistry,
    ctx: ToolContext,
    review: Dict[str, Any],
    profile_rules: Dict[str, Any],
) -> str:
    thread_id = str(review.get("thread_id") or "").strip()
    case = get_thread_case(state, thread_id) if thread_id else None
    if not isinstance(case, dict):
        return "Vielen Dank. Ich setze den Prozess fort und melde mich mit dem nächsten Schritt."

    required_fields = _normalize_required_fields([str(x).strip() for x in (case.get("required_field_names") or []) if str(x).strip()])
    facts = case.get("facts") if isinstance(case.get("facts"), dict) else {}
    required_status = _build_required_status(required_fields=required_fields, facts=facts)
    missing_detail = [x for x in required_status.get("missing_required_fields", []) if x != "price_confirmed"]

    calendar_rules = _rules_dict(profile_rules, "calendar")
    pricing_rules = profile_rules.get("pricing") if isinstance(profile_rules.get("pricing"), dict) else {}
    booking_rules = profile_rules.get("booking") if isinstance(profile_rules.get("booking"), dict) else {}

    cal = _calendar_check(
        registry=registry,
        ctx=ctx,
        calendar_id=str(calendar_rules.get("calendar_id") or "primary"),
        facts=facts,
    )
    cal = _apply_own_hold_override(case=case, cal=cal)

    distance_km = 0.0
    if str(facts.get("location") or "").strip():
        dist = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="distance_check",
            args={
                "origin": str(booking_rules.get("base_address") or "Pforzheim, Deutschland"),
                "destination": str(facts.get("location") or "").strip(),
                "max_distance_km": _float_or(booking_rules.get("max_distance_km"), 0.0),
            },
        )
        distance_km = _float_or(dist.get("distance_km"), 0.0)

    quote_text = ""
    if _float_or(facts.get("duration_hours"), 0.0) > 0:
        quote = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="pricing_compute_quote",
            args={"facts": facts, "pricing_rules": pricing_rules, "booking_rules": booking_rules, "distance_km": distance_km},
        )
        quote_text = str(quote.get("text") or "").strip()

    blocker = _ensure_calendar_blocker(
        registry=registry,
        ctx=ctx,
        case=case,
        cal=cal,
        facts=facts,
        calendar_id=str(calendar_rules.get("calendar_id") or "primary"),
        from_email=str(review.get("from_email") or ""),
        hold_minutes=int(_float_or(calendar_rules.get("hold_minutes"), 90.0)),
    )
    if str(blocker.get("event_id") or "").strip():
        case_calendar = _case_calendar_snapshot(case)
        upsert_thread_case(
            state,
            thread_id=thread_id,
            patch={
                "calendar": {
                    "has_term": bool(cal.get("has_term")),
                    "checked": bool(cal.get("checked")),
                    "is_available": cal.get("is_available"),
                    "calendar_id": str(cal.get("calendar_id") or calendar_rules.get("calendar_id") or "primary"),
                    "start_iso": str(cal.get("start_iso") or ""),
                    "end_iso": str(cal.get("end_iso") or ""),
                    "hold_event_id": str(blocker.get("event_id") or case_calendar.get("hold_event_id") or ""),
                    "hold_html_link": str(blocker.get("html_link") or case_calendar.get("hold_html_link") or ""),
                    "hold_status": str(blocker.get("status") or case_calendar.get("hold_status") or ""),
                    "hold_expires_at": str(blocker.get("hold_expires_at") or case_calendar.get("hold_expires_at") or ""),
                    "text": str(cal.get("text") or ""),
                }
            },
        )

    if missing_detail:
        return _compose_missing_fields_mail(
            missing_fields=missing_detail,
            facts=facts,
            required_fields=required_fields,
            precheck_text=str(cal.get("text") or ""),
            optional_offer_text=quote_text if _float_or(facts.get("duration_hours"), 0.0) > 0 and cal.get("is_available") is True else "",
        )

    offer = case.get("offer") if isinstance(case.get("offer"), dict) else {}
    offer_status = str(offer.get("status") or "").strip().lower()
    if offer_status not in {"sent", "accepted"}:
        upsert_thread_case(
            state,
            thread_id=thread_id,
            patch={"offer": {"status": "sent", "quote_text": quote_text, "sent_at": _now_iso()}, "status": "waiting_customer"},
        )
        return _compose_offer_mail(quote_text=quote_text, facts=facts, required_fields=required_fields)

    if bool(facts.get("price_confirmed")):
        return _compose_provisional_confirmation_mail()

    return _compose_offer_mail(
        quote_text=str(offer.get("quote_text") or quote_text),
        facts=facts,
        required_fields=required_fields,
    )


def _compose_offer_action_body(
    *,
    state: Dict[str, Any],
    registry: ToolRegistry,
    ctx: ToolContext,
    review: Dict[str, Any],
    profile_rules: Dict[str, Any],
) -> str:
    thread_id = str(review.get("thread_id") or "").strip()
    case = get_thread_case(state, thread_id) if thread_id else None
    if not isinstance(case, dict):
        return "Vielen Dank für Ihre Anfrage. Ich kann Ihnen gern ein angepasstes, regelkonformes Angebot senden."

    required_fields = _normalize_required_fields([str(x).strip() for x in (case.get("required_field_names") or []) if str(x).strip()])
    facts = case.get("facts") if isinstance(case.get("facts"), dict) else {}
    required_status = _build_required_status(required_fields=required_fields, facts=facts)
    missing_detail = [x for x in required_status.get("missing_required_fields", []) if x != "price_confirmed"]

    calendar_rules = _rules_dict(profile_rules, "calendar")
    pricing_rules = profile_rules.get("pricing") if isinstance(profile_rules.get("pricing"), dict) else {}
    booking_rules = profile_rules.get("booking") if isinstance(profile_rules.get("booking"), dict) else {}

    cal = _calendar_check(
        registry=registry,
        ctx=ctx,
        calendar_id=str(calendar_rules.get("calendar_id") or "primary"),
        facts=facts,
    )
    cal = _apply_own_hold_override(case=case, cal=cal)

    distance_km = 0.0
    if str(facts.get("location") or "").strip():
        dist = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="distance_check",
            args={
                "origin": str(booking_rules.get("base_address") or "Pforzheim, Deutschland"),
                "destination": str(facts.get("location") or "").strip(),
                "max_distance_km": _float_or(booking_rules.get("max_distance_km"), 0.0),
            },
        )
        distance_km = _float_or(dist.get("distance_km"), 0.0)

    quote_text = ""
    if _float_or(facts.get("duration_hours"), 0.0) > 0:
        quote = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="pricing_compute_quote",
            args={"facts": facts, "pricing_rules": pricing_rules, "booking_rules": booking_rules, "distance_km": distance_km},
        )
        quote_text = str(quote.get("text") or "").strip()

    if missing_detail:
        return _compose_missing_fields_mail(
            missing_fields=missing_detail,
            facts=facts,
            required_fields=required_fields,
            precheck_text=str(cal.get("text") or ""),
            optional_offer_text=quote_text if _float_or(facts.get("duration_hours"), 0.0) > 0 and cal.get("is_available") is True else "",
        )

    if quote_text.strip():
        return _compose_offer_mail(quote_text=quote_text, facts=facts, required_fields=required_fields)

    return "Vielen Dank für Ihre Anfrage. Ich kann Ihnen gern ein angepasstes Angebot senden, sobald die fehlenden Details vorliegen."


def _compose_reject_action_body(*, review: Dict[str, Any], explicit_reason: str = "") -> str:
    reason = str(explicit_reason or "").strip()
    if not reason:
        raw = str(review.get("reason") or "").strip()
        reason = raw.split("|", 1)[0].strip() if raw else ""
    return _compose_final_rejection_mail(reason=reason)


def _polish_action_body_with_llm(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    action: str,
    body: str,
) -> str:
    base = str(body or "").strip()
    if not base:
        return base
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="llm_text_compose",
        args={
            "text": base,
            "instruction": (
                f"Formuliere die Kundenantwort fuer Aktion '{action}' klar, freundlich und verbindlich auf Deutsch. "
                "Inhaltliche Bedeutung unverändert lassen."
            ),
            "max_chars": 2600,
        },
    )
    if out.get("_error"):
        return base
    txt = str(out.get("text") or "").strip()
    return txt or base


def _final_confirmation_action(
    *,
    state: Dict[str, Any],
    registry: ToolRegistry,
    ctx: ToolContext,
    review: Dict[str, Any],
    profile_rules: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    thread_id = str(review.get("thread_id") or "").strip()
    case = get_thread_case(state, thread_id) if thread_id else None
    if not isinstance(case, dict):
        raise HTTPException(status_code=422, detail="No thread case found for final confirmation")

    facts = case.get("facts") if isinstance(case.get("facts"), dict) else {}
    calendar_rules = _rules_dict(profile_rules, "calendar")
    calendar_id = str(calendar_rules.get("calendar_id") or "primary")
    cal = _calendar_check(registry=registry, ctx=ctx, calendar_id=calendar_id, facts=facts)
    cal = _apply_own_hold_override(case=case, cal=cal)
    if not bool(cal.get("has_term")):
        raise HTTPException(status_code=422, detail="Final confirmation requires a complete term (date/time/duration)")
    if not bool(cal.get("checked")):
        raise HTTPException(status_code=422, detail=f"Calendar check failed: {cal.get('text')}")
    if cal.get("is_available") is not True:
        raise HTTPException(status_code=409, detail="Final confirmation not possible: slot is busy")

    case_calendar = _case_calendar_snapshot(case)
    hold_event_id = str(case_calendar.get("hold_event_id") or "").strip()
    event_id = ""
    html_link = ""
    created: Dict[str, Any] = {}

    if hold_event_id:
        updated = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="calendar_update_event",
            args={
                "event_id": hold_event_id,
                "calendar_id": calendar_id,
                "summary": f"Booking: {str(facts.get('occasion') or 'Event')}",
                "description": "Final bestätigte Buchung aus Booking Assistant v3",
                "location": str(facts.get("location") or ""),
                "send_updates": "none",
            },
        )
        if not updated.get("_error"):
            event_id = str(updated.get("event_id") or hold_event_id).strip()
            html_link = str(updated.get("html_link") or "").strip()
            created = dict(updated)

    if not event_id:
        created = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="calendar_create_event",
            args={
                "summary": f"Booking: {str(facts.get('occasion') or 'Event')}",
                "start_iso": str(cal.get("start_iso") or ""),
                "end_iso": str(cal.get("end_iso") or ""),
                "location": str(facts.get("location") or ""),
                "description": "Final bestätigte Buchung aus Booking Assistant v3",
                "attendees": [str(review.get("from_email") or "")] if str(review.get("from_email") or "").strip() else [],
                "calendar_id": calendar_id,
            },
        )
        if created.get("_error"):
            raise HTTPException(status_code=422, detail=f"Calendar create failed: {created.get('_error')}")
        event_id = str(created.get("event_id") or "").strip()
        html_link = str(created.get("html_link") or "").strip()

    offer = case.get("offer") if isinstance(case.get("offer"), dict) else {}
    quote_text = str(offer.get("quote_text") or "").strip()

    upsert_thread_case(
        state,
        thread_id=thread_id,
        patch={
            "status": "closed_confirmed",
            "calendar": {
                "has_term": True,
                "checked": True,
                "is_available": True,
                "calendar_id": calendar_id,
                "start_iso": str(cal.get("start_iso") or ""),
                "end_iso": str(cal.get("end_iso") or ""),
                "event_id": event_id,
                "html_link": html_link,
                "hold_event_id": hold_event_id,
                "updated_at": _now_iso(),
            },
            "offer": {
                "status": "accepted",
                "quote_text": quote_text,
                "confirmed_at": _now_iso(),
            },
        },
    )
    body = _compose_final_confirmation_mail(event_link=html_link, quote_text=quote_text)
    return body, created


def apply_pending_action(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV3PendingApplyRequest,
) -> BookingAssistantV3ReviewActionResponse:
    state = load_state(settings, user_id)
    review = find_review(state, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if str(review.get("status") or "").strip().lower() != "pending":
        raise HTTPException(status_code=409, detail="Review is not pending")

    action: ActionType = req.action
    kind = str(review.get("kind") or "").strip().lower()

    registry = build_registry(settings=settings, user_id=user_id)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_v3_pending_apply")

    # load profile to reuse rules for resume/final actions
    profile_name = "booking_default"
    case = get_thread_case(state, str(review.get("thread_id") or ""))
    if isinstance(case, dict):
        profile_name = str(case.get("assistant_profile_name") or "booking_default")
    get_profile = _safe_tool_call(registry=registry, ctx=ctx, tool="assistent_profile_get", args={"assistent_profile_name": profile_name})
    profile = get_profile.get("profile") if isinstance(get_profile.get("profile"), dict) else {}
    profile_rules = profile.get("rules") if isinstance(profile.get("rules"), dict) else {}

    if kind == "rule_review" and action not in {"approve", "offer", "reject"}:
        raise HTTPException(status_code=422, detail="rule_review supports actions: approve, offer, reject")
    if kind == "final_confirmation" and action not in {"final_confirmation", "final_rejection"}:
        raise HTTPException(status_code=422, detail="final_confirmation review supports: final_confirmation, final_rejection")

    body = ""
    send_result: Dict[str, Any] = {}

    if action == "approve":
        body = str(req.edited_body or "").strip() or _resume_rule_review_after_approve(
            state=state,
            registry=registry,
            ctx=ctx,
            review=review,
            profile_rules=profile_rules,
        )
        body = _polish_action_body_with_llm(registry=registry, ctx=ctx, action="approve", body=body)
        if bool(req.send_to_customer):
            send_result = _apply_action_send(state=state, registry=registry, ctx=ctx, review=review, body=body, subject=req.subject, action="approve")
        review["status"] = "approved"
    elif action == "offer":
        body = str(req.edited_body or "").strip() or _compose_offer_action_body(
            state=state,
            registry=registry,
            ctx=ctx,
            review=review,
            profile_rules=profile_rules,
        )
        body = _polish_action_body_with_llm(registry=registry, ctx=ctx, action="offer", body=body)
        if bool(req.send_to_customer):
            send_result = _apply_action_send(state=state, registry=registry, ctx=ctx, review=review, body=body, subject=req.subject, action="offer")
        review["status"] = "offered"
        thread_id = str(review.get("thread_id") or "").strip()
        if thread_id:
            upsert_thread_case(state, thread_id=thread_id, patch={"status": "waiting_customer"})
    elif action == "reject":
        body = str(req.edited_body or "").strip() or _compose_reject_action_body(
            review=review,
            explicit_reason=req.reason,
        )
        body = _polish_action_body_with_llm(registry=registry, ctx=ctx, action="reject", body=body)
        if bool(req.send_to_customer):
            send_result = _apply_action_send(state=state, registry=registry, ctx=ctx, review=review, body=body, subject=req.subject, action="reject")
        review["status"] = "rejected"
        thread_id = str(review.get("thread_id") or "").strip()
        if thread_id:
            upsert_thread_case(state, thread_id=thread_id, patch={"status": "closed_rejected"})
    elif action == "final_confirmation":
        body = str(req.edited_body or "").strip()
        if not body:
            body, _ = _final_confirmation_action(state=state, registry=registry, ctx=ctx, review=review, profile_rules=profile_rules)
        if bool(req.send_to_customer):
            send_result = _apply_action_send(state=state, registry=registry, ctx=ctx, review=review, body=body, subject=req.subject, action="final_confirmation")
        review["status"] = "final_confirmed"
    elif action == "final_rejection":
        body = str(req.edited_body or "").strip() or _compose_final_rejection_mail(reason=req.reason)
        if bool(req.send_to_customer):
            send_result = _apply_action_send(state=state, registry=registry, ctx=ctx, review=review, body=body, subject=req.subject, action="final_rejection")
        review["status"] = "final_rejected"
        thread_id = str(review.get("thread_id") or "").strip()
        if thread_id:
            upsert_thread_case(state, thread_id=thread_id, patch={"status": "closed_rejected"})
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported action: {action}")

    review["updated_at"] = _now_iso()
    review["selected_action"] = str(action)
    if req.reason.strip():
        review["reason"] = req.reason.strip()
    append_activity(
        state,
        _activity_record(
            mail_id=str(review.get("mail_id") or ""),
            thread_id=str(review.get("thread_id") or ""),
            decision=f"operator_{action}",
            booking_decision=str(review.get("booking_decision") or ""),
            review_id=str(review.get("id") or review_id),
            reason=str(review.get("reason") or ""),
        ),
    )
    save_state(settings, user_id, state)

    return BookingAssistantV3ReviewActionResponse(
        ok=True,
        review_id=str(review.get("id") or review_id),
        status=str(review.get("status") or ""),
        sent=bool(review.get("sent")),
        reason=str(review.get("reason") or ""),
    )


def get_reviews(*, user_id: str, settings: Settings, status: str = "pending", kind: str = "") -> BookingAssistantV3ReviewsResponse:
    state = load_state(settings, user_id)
    s = str(status or "").strip().lower()
    if s in {"", "all", "*"}:
        s = ""
    k = str(kind or "").strip().lower()
    if k in {"", "all", "*"}:
        k = ""
    reviews = [BookingAssistantV3ReviewItem(**r) for r in list_reviews(state, status=s, kind=k)]
    reviews.sort(key=lambda x: x.created_at, reverse=True)
    return BookingAssistantV3ReviewsResponse(ok=True, reviews=reviews)


def get_pending_next(*, user_id: str, settings: Settings, queue: str = "any") -> BookingAssistantV3PendingNextResponse:
    state = load_state(settings, user_id)
    q = str(queue or "any").strip().lower()
    pending = list_reviews(state, status="pending")
    pending.sort(key=lambda x: str(x.get("created_at") or ""))

    if q in {"rule_review", "final_confirmation"}:
        pending = [x for x in pending if str(x.get("kind") or "").strip().lower() == q]

    if not pending:
        return BookingAssistantV3PendingNextResponse(
            ok=True,
            has_pending=False,
            review={},
            options={},
            text="Keine offenen Pending-Fälle vorhanden.",
        )

    review = pending[0]
    summary = _review_summary_item(review)
    options = dict(summary.get("options") or {})
    summary["options"] = options
    return BookingAssistantV3PendingNextResponse(
        ok=True,
        has_pending=True,
        review=summary,
        options=options,
        text=f"Nächster Pending-Fall: {summary.get('id')} ({summary.get('kind')}).",
    )


def get_status(*, user_id: str, settings: Settings, since: str = "") -> BookingAssistantV3StatusResponse:
    state = load_state(settings, user_id)
    since_input = str(since or "").strip()
    since_effective = since_input or str(state.get("last_status_at") or "").strip()
    if not since_effective:
        since_effective = _normalize_since_alias("today")
    else:
        since_effective = _normalize_since_alias(since_effective)
    since_dt = _parse_iso_utc(since_effective)
    if since_effective and since_dt is None:
        raise HTTPException(status_code=422, detail="Invalid 'since' timestamp. Use ISO-8601.")

    activity = state.get("activity_log") if isinstance(state.get("activity_log"), list) else []
    filtered: List[Dict[str, Any]] = []
    for item in activity:
        if not isinstance(item, dict):
            continue
        ts = _parse_iso_utc(str(item.get("timestamp") or "").strip())
        if since_dt and (ts is None or ts < since_dt):
            continue
        filtered.append(item)

    incoming_ids = {str(x.get("mail_id") or "").strip() for x in filtered if str(x.get("mail_id") or "").strip()}
    incoming_count = len(incoming_ids)

    pending_rule = list_reviews(state, status="pending", kind="rule_review")
    pending_final = list_reviews(state, status="pending", kind="final_confirmation")

    rejected_count = 0
    auto_sent_count = 0
    blocker_counter: Dict[str, int] = {}

    for item in filtered:
        decision = str(item.get("decision") or "").strip().lower()
        booking_decision = str(item.get("booking_decision") or "").strip().lower()
        reason = str(item.get("reason") or "").strip()
        if decision == "auto_sent":
            auto_sent_count += 1
            if booking_decision == "auto_decline":
                rejected_count += 1
        if decision in {"operator_reject", "operator_final_rejection"}:
            rejected_count += 1
        if decision == "needs_human" and reason:
            blocker_counter[reason] = blocker_counter.get(reason, 0) + 1

    top_blockers = [
        {"reason": k, "count": v}
        for k, v in sorted(blocker_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]

    recommendations: List[str] = []
    if pending_final:
        recommendations.append("Es gibt final zu bestätigende Termine. Bitte final_confirmation/final_rejection durchführen.")
    if pending_rule:
        recommendations.append("Es gibt offene Regel-Reviews. Bitte approve/offer/reject entscheiden.")
    if not recommendations:
        recommendations.append("Aktuell keine offenen manuellen Entscheidungen.")

    text = (
        f"Heute gingen {incoming_count} Buchungsanfragen ein. "
        f"{len(pending_final)} warten auf finale Bestätigung, "
        f"{rejected_count} wurden abgelehnt und "
        f"{len(pending_rule)} erfordern deine Aufmerksamkeit."
    )

    state["last_status_at"] = _now_iso()
    save_state(settings, user_id, state)

    return BookingAssistantV3StatusResponse(
        ok=True,
        since=since_effective,
        generated_at=_now_iso(),
        incoming_count=incoming_count,
        pending_rule_review_count=len(pending_rule),
        pending_final_confirmation_count=len(pending_final),
        rejected_count=rejected_count,
        auto_sent_count=auto_sent_count,
        top_blockers=top_blockers,
        recommendations=recommendations,
        text=text,
    )


def operator_chat(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    req: BookingAssistantV3OperatorChatRequest,
) -> BookingAssistantV3OperatorChatResponse:
    low = str(req.message or "").strip().lower()

    if any(x in low for x in ("status", "zusammenfassung", "report", "meeting")):
        status = get_status(user_id=user_id, settings=settings, since="")
        return BookingAssistantV3OperatorChatResponse(
            ok=True,
            intent="status",
            action_taken="status",
            data=status.model_dump(),
            text=status.text,
        )

    if "final" in low and any(x in low for x in ("offen", "pending", "bestätigung", "bestaetigung")):
        nxt = get_pending_next(user_id=user_id, settings=settings, queue="final_confirmation")
        return BookingAssistantV3OperatorChatResponse(
            ok=True,
            intent="pending_final",
            action_taken="pending_next_final",
            data=nxt.model_dump(),
            text=nxt.text,
        )

    if any(x in low for x in ("offen", "pending", "aufmerksamkeit", "review")):
        nxt = get_pending_next(user_id=user_id, settings=settings, queue="rule_review")
        return BookingAssistantV3OperatorChatResponse(
            ok=True,
            intent="pending_rule",
            action_taken="pending_next_rule",
            data=nxt.model_dump(),
            text=nxt.text,
        )

    return BookingAssistantV3OperatorChatResponse(
        ok=True,
        intent="fallback",
        action_taken="fallback",
        data={},
        text="Nutze z. B. 'status', 'zeige offene Reviews' oder 'zeige finale Bestätigungen'.",
    )


# convenience wrappers for direct endpoints/tools

def approve_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV3SimpleActionRequest,
) -> BookingAssistantV3ReviewActionResponse:
    return apply_pending_action(
        user_id=user_id,
        settings=settings,
        api_key=api_key,
        review_id=review_id,
        req=BookingAssistantV3PendingApplyRequest(
            provider=req.provider,
            action="approve",
            edited_body=req.edited_body,
            subject=req.subject,
            reason=req.reason,
            send_to_customer=req.send_to_customer,
        ),
    )


def offer_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV3SimpleActionRequest,
) -> BookingAssistantV3ReviewActionResponse:
    return apply_pending_action(
        user_id=user_id,
        settings=settings,
        api_key=api_key,
        review_id=review_id,
        req=BookingAssistantV3PendingApplyRequest(
            provider=req.provider,
            action="offer",
            edited_body=req.edited_body,
            subject=req.subject,
            reason=req.reason,
            send_to_customer=req.send_to_customer,
        ),
    )


def reject_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV3SimpleActionRequest,
) -> BookingAssistantV3ReviewActionResponse:
    return apply_pending_action(
        user_id=user_id,
        settings=settings,
        api_key=api_key,
        review_id=review_id,
        req=BookingAssistantV3PendingApplyRequest(
            provider=req.provider,
            action="reject",
            edited_body=req.edited_body,
            subject=req.subject,
            reason=req.reason,
            send_to_customer=req.send_to_customer,
        ),
    )


def final_confirmation_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV3SimpleActionRequest,
) -> BookingAssistantV3ReviewActionResponse:
    return apply_pending_action(
        user_id=user_id,
        settings=settings,
        api_key=api_key,
        review_id=review_id,
        req=BookingAssistantV3PendingApplyRequest(
            provider=req.provider,
            action="final_confirmation",
            edited_body=req.edited_body,
            subject=req.subject,
            reason=req.reason,
            send_to_customer=req.send_to_customer,
        ),
    )


def final_rejection_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV3SimpleActionRequest,
) -> BookingAssistantV3ReviewActionResponse:
    return apply_pending_action(
        user_id=user_id,
        settings=settings,
        api_key=api_key,
        review_id=review_id,
        req=BookingAssistantV3PendingApplyRequest(
            provider=req.provider,
            action="final_rejection",
            edited_body=req.edited_body,
            subject=req.subject,
            reason=req.reason,
            send_to_customer=req.send_to_customer,
        ),
    )
