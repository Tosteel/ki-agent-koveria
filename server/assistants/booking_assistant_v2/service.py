from __future__ import annotations

import hashlib
import json
import re
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
    BookingAssistantV2ApproveRequest,
    BookingAssistantV2CounterofferRequest,
    BookingAssistantV2OperatorChatRequest,
    BookingAssistantV2OperatorChatResponse,
    BookingAssistantV2PendingApplyRequest,
    BookingAssistantV2PendingNextResponse,
    BookingAssistantV2RejectRequest,
    BookingAssistantV2ReviewActionResponse,
    BookingAssistantV2ReviewItem,
    BookingAssistantV2ReviewsResponse,
    BookingAssistantV2RunItem,
    BookingAssistantV2RunRequest,
    BookingAssistantV2RunResponse,
    BookingAssistantV2StatusMeetingResponse,
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

_TRACE_ENABLED: ContextVar[bool] = ContextVar("booking_assistant_v2_trace_enabled", default=False)
_TRACE_STEP: ContextVar[int] = ContextVar("booking_assistant_v2_trace_step", default=0)


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
        day = (now_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return day.isoformat().replace("+00:00", "Z")
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
        raise HTTPException(status_code=400, detail=f"Required tool is missing: {tool}")
    if _TRACE_ENABLED.get():
        step = _TRACE_STEP.get() + 1
        _TRACE_STEP.set(step)
        _trace_log(
            f"STEP INPUT {step}",
            [
                f"tool={tool}",
                f"args={json.dumps(args, ensure_ascii=False)[:2400]}",
            ],
        )
    out = dispatch_tool_chain(registry=registry, tool_name=tool, ctx=ctx, args=args)
    payload = out if isinstance(out, dict) else {"value": out}
    if _TRACE_ENABLED.get():
        _trace_log(
            f"STEP OUTPUT {_TRACE_STEP.get()}",
            [
                f"tool={tool}",
                f"payload={json.dumps(payload, ensure_ascii=False)[:3500]}",
            ],
        )
    return payload


def _safe_tool_call(*, registry: ToolRegistry, ctx: ToolContext, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if registry.get_tool(tool) is None:
        return {"_error": f"missing_tool:{tool}"}
    try:
        return _tool_call(registry=registry, ctx=ctx, tool=tool, args=args)
    except Exception as exc:
        return {"_error": str(exc)}


def _normalize_required_fields(required_fields: List[str] | None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in (required_fields or []):
        name = str(raw or "").strip()
        if not name or name in seen:
            continue
        out.append(name)
        seen.add(name)
    if out:
        return out
    return ["event_date", "start_time", "duration_hours", "location", "occasion", "client_name", "price_confirmed"]


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


def _serialize_required_field_value(name: str, value: Any) -> Any:
    if _is_missing_required_field(name, value):
        return False if str(name).strip() == "price_confirmed" else ""
    if isinstance(value, str):
        return value.strip()
    return value


def _build_required_fields_status(*, required_fields: List[str] | None, facts: Dict[str, Any]) -> Dict[str, Any]:
    required = _normalize_required_fields(required_fields)
    values: Dict[str, Any] = {}
    missing: List[str] = []
    present: List[str] = []
    for field in required:
        raw = facts.get(field)
        values[field] = _serialize_required_field_value(field, raw)
        if _is_missing_required_field(field, raw):
            missing.append(field)
        else:
            present.append(field)
    return {
        "required_fields": values,
        "required_field_names": required,
        "present_required_fields": present,
        "missing_required_fields": missing,
        "complete": len(missing) == 0,
        "updated_at": _now_iso(),
    }


def _extract_sources(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if not isinstance(payload, dict):
        return out
    for key in ("final_url", "url", "source", "link", "href", "document", "file"):
        val = str(payload.get(key) or "").strip()
        if val:
            out.append(val)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    for item in items:
        if isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            if url:
                out.append(url)
    matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
    for m in matches:
        if isinstance(m, dict):
            href = str(m.get("href") or "").strip()
            if href:
                out.append(href)
    seen: set[str] = set()
    deduped: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped


def _default_profile(name: str, codename: str = "") -> Dict[str, Any]:
    return {
        "assistent_profile_name": name,
        "codename": codename or "Event-DJ Assistant",
        "instructions": [
            "Du bist ein Event-DJ Booking-Assistent.",
            "Bestätige Termine erst, wenn alle Pflichtangaben vollständig sind.",
            "Kommuniziere Preise transparent und verlange eine Angebotsbestätigung.",
            "Bei Regelverletzungen oder Unsicherheit immer Human Review.",
        ],
        "rules": {
            "offering": {
                "summary": "Event-DJ mit Fokus auf Musikgestaltung für Feiern und Events.",
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
            "booking": {
                "weekend_only": True,
                "max_duration_hours": 8,
                "max_distance_km": 200,
                "overnight_distance_km": 60,
                "overnight_after_hour": 22,
                "base_address": "Pforzheim, Deutschland",
            },
            "pricing": {
                "hourly_rate_eur": 120,
                "travel_per_km_eur": 0.7,
                "overnight_flat_eur": 120,
                "setup_flat_eur": 80,
                "teardown_flat_eur": 60,
                "travel_round_trip": True,
            },
            "mail": {
                "never_auto_send": False,
                "block_auto_reply_topics": [],
            },
        },
    }


def _ensure_profile(*, registry: ToolRegistry, ctx: ToolContext, req: BookingAssistantV2RunRequest) -> Dict[str, Any]:
    profile_name = str(req.assistant_profile_name or "booking_default").strip() or "booking_default"
    get_out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="assistent_profile_get",
        args={"assistent_profile_name": profile_name},
    )
    if get_out.get("_error"):
        if req.profile_bootstrap:
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
            get_out = _tool_call(
                registry=registry,
                ctx=ctx,
                tool="assistent_profile_get",
                args={"assistent_profile_name": profile_name},
            )
        else:
            raise HTTPException(status_code=404, detail=f"Assistant profile not found: {profile_name}")

    if req.profile_instructions_add or req.profile_rules_patch:
        _tool_call(
            registry=registry,
            ctx=ctx,
            tool="assistent_profile_update",
            args={
                "assistent_profile_name": profile_name,
                "instructions_add": list(req.profile_instructions_add or []),
                "rules_patch": dict(req.profile_rules_patch or {}),
            },
        )
        get_out = _tool_call(
            registry=registry,
            ctx=ctx,
            tool="assistent_profile_get",
            args={"assistent_profile_name": profile_name},
        )

    profile = get_out.get("profile") if isinstance(get_out.get("profile"), dict) else {}
    if not profile:
        raise HTTPException(status_code=500, detail="Assistant profile could not be loaded")
    return profile


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
            return {"intent": "termin", "confidence": 0.4, "reason": "heuristic_fallback"}
        return {"intent": "info", "confidence": 0.0, "reason": str(out.get("_error") or "classify_failed")}
    intent = str(out.get("intent") or "info").strip().lower()
    if intent not in {"info", "beschwerde", "angebot", "termin", "eskalation", "newsletter"}:
        intent = "info"
    return {
        "intent": intent,
        "confidence": float(out.get("confidence") or 0.0),
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
    base = _tool_call(
        registry=registry,
        ctx=ctx,
        tool="gmail_read_mail",
        args={"mail_id": mail_id, "mailbox": mailbox, "max_chars": 20000},
    )
    parts: List[str] = [str(base.get("text") or "").strip()]
    thread_text = ""
    thread_facts_text = ""
    if include_thread:
        thread = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="gmail_read_mail_thread",
            args={"mail_id": mail_id, "mailbox": mailbox, "max_messages": 20, "max_chars": 12000},
        )
        t = str(thread.get("text") or "").strip()
        thread_text = t
        if t:
            parts.append("THREAD:\n" + t)
        msgs = thread.get("messages") if isinstance(thread.get("messages"), list) else []
        fact_parts: List[str] = []
        for item in reversed(msgs):
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject") or "").strip()
            body = str(item.get("body_text") or "").strip()
            if subj:
                fact_parts.append(f"Subject: {subj}")
            if body:
                fact_parts.append(body)
        thread_facts_text = "\n\n".join(x for x in fact_parts if x).strip()
    merged = "\n\n".join(p for p in parts if p).strip()
    out = dict(base)
    out["text"] = merged or str(base.get("text") or "").strip()
    out["thread_text"] = thread_text
    out["thread_facts_text"] = thread_facts_text
    return out


def _retrieve_web_context(
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

    rag = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="rag_knowledgebase",
        args={"query": query, "top_k": rag_top_k},
    )
    if not rag.get("_error"):
        txt = str(rag.get("text") or "").strip()
        if txt:
            parts.append("RAG:\n" + txt)
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
            out = _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="web_crawl_site",
                args={"url": url, "query": query, "max_pages": 10, "max_matches": 8},
            )
        if out.get("_error"):
            out = _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="web_fetch_page",
                args={"url": url, "query": query},
            )
        if out.get("_error"):
            continue
        txt = str(out.get("text") or "").strip()
        if txt:
            parts.append(f"WEB({url}):\n{txt}")
        sources.extend(_extract_sources(out))

    merged = "\n\n".join(parts).strip()
    if len(merged) > max_context_chars:
        merged = merged[:max_context_chars].rstrip() + "…"

    seen: set[str] = set()
    deduped: List[str] = []
    for s in sources:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)

    return {"context_text": merged, "sources": deduped}


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
    source_lines = "\n".join(f"- {s}" for s in sources[:20]) or "- keine"
    text = (
        "Eingehende Mail:\n"
        f"{str(mail_payload.get('text') or '').strip()}\n\n"
        "Assistentenprofil:\n"
        f"{profile_txt}\n\n"
        "Recherchierter Kontext:\n"
        f"{context_text or 'Kein zusätzlicher Kontext gefunden.'}\n\n"
        "Quellen:\n"
        f"{source_lines}"
    )
    instruction = (
        "Erstelle eine hilfreiche, präzise Antwort auf Deutsch. "
        "Nutze Informationen aus Assistentenprofil und Kontext. "
        "Wenn wichtige Infos fehlen, stelle eine kurze Rückfrage."
    )
    out = _tool_call(
        registry=registry,
        ctx=ctx,
        tool="llm_text_compose",
        args={"text": text, "instruction": instruction, "max_chars": 2400},
    )
    return str(out.get("text") or "").strip()


def _llm_classify_offer_confirmation(*, latest_body_text: str, thread_text: str) -> Dict[str, Any]:
    client = IonosLLM()
    if not client.enabled():
        return {
            "offer_present_in_thread": False,
            "offer_accepted_by_latest_reply": False,
            "confidence": 0.0,
            "reason": "llm_unavailable",
            "model": "",
        }

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_offer_confirmation_gate_v2",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "offer_present_in_thread": {"type": "boolean"},
                    "offer_accepted_by_latest_reply": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": [
                    "offer_present_in_thread",
                    "offer_accepted_by_latest_reply",
                    "confidence",
                    "reason",
                ],
            },
            "strict": True,
        },
    }

    prompt = (
        "Prüfe in einem Booking-Mailverlauf zwei Punkte:\n"
        "1) Wurde im Thread bereits ein konkretes Angebot kommuniziert?\n"
        "2) Nimmt die LETZTE Kundenantwort dieses Angebot verbindlich an?\n\n"
        "Regeln:\n"
        "- Nur bei klarer Annahme accepted=true.\n"
        "- Bei Unklarheit, Rückfrage, neuem Gegenvorschlag oder Themenwechsel -> false.\n\n"
        f"Letzte Kundenantwort:\n{latest_body_text.strip()}\n\n"
        f"Thread-Kontext:\n{thread_text[:5000]}"
    )
    try:
        completion = client.chat_completions(
            messages=[
                {"role": "system", "content": "Du bist ein präziser Klassifikator. Antworte strikt im JSON-Schema."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            max_tokens=180,
            temperature=0.0,
            top_p=0.1,
        )
        raw = IonosLLM.extract_text(completion)
        parsed = json.loads(raw) if raw else {}
        return {
            "offer_present_in_thread": bool(parsed.get("offer_present_in_thread")),
            "offer_accepted_by_latest_reply": bool(parsed.get("offer_accepted_by_latest_reply")),
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence") or 0.0))),
            "reason": str(parsed.get("reason") or "").strip(),
            "model": client.cfg.model,
        }
    except Exception as exc:
        return {
            "offer_present_in_thread": False,
            "offer_accepted_by_latest_reply": False,
            "confidence": 0.0,
            "reason": f"llm_error:{exc}",
            "model": client.cfg.model if client.enabled() else "",
        }


def _score_reply(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    user_message: str,
    draft: str,
    sources: List[str],
    booking_decision: str,
    facts: Dict[str, Any] | None = None,
    required_fields: List[str] | None = None,
    missing_fields: List[str] | None = None,
) -> Dict[str, Any]:
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="booking_reply_score",
        args={
            "user_message": user_message,
            "draft_reply": draft,
            "booking_decision": booking_decision,
            "facts": dict(facts or {}),
            "required_fields": [str(x).strip() for x in (required_fields or []) if str(x).strip()],
            "missing_fields": [str(x).strip() for x in (missing_fields or []) if str(x).strip()],
            "knowledge_evidence": sources[:20],
            "require_actionable": True,
        },
    )
    if out and out.get("_error"):
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
        total = max(0.0, min(1.0, float(out.get("total_score") or 0.0)))
        return {
            "score_total": total,
            "verdict": str(out.get("verdict") or "needs_review").strip().lower(),
            "reason": "; ".join(str(x) for x in (out.get("reasons") or [])[:3] if str(x).strip()),
            "raw": out,
        }
    text_len = len((draft or "").strip())
    fallback = 0.5 + (0.2 if text_len > 150 else 0.0) + (0.15 if sources else 0.0)
    fallback = max(0.0, min(1.0, fallback))
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
    if not draft_reply.strip() or not instructions:
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


def _activity_record(
    *,
    mail_id: str,
    thread_id: str,
    decision: str,
    booking_decision: str = "",
    event_id: str = "",
    review_id: str = "",
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "mail_id": str(mail_id or "").strip(),
        "thread_id": str(thread_id or "").strip(),
        "decision": str(decision or "").strip(),
        "booking_decision": str(booking_decision or "").strip(),
        "event_id": str(event_id or "").strip(),
        "review_id": str(review_id or "").strip(),
        "reason": str(reason or "").strip(),
    }


def _compose_decision_mail(*, decision: str, reasons: List[str], quote_text: str = "", event_link: str = "") -> str:
    rs = "\n".join(f"- {r}" for r in reasons if r)
    if decision == "auto_decline":
        return (
            "Vielen Dank für Ihre Anfrage. Leider kann ich den Termin auf Basis der aktuellen Rahmenbedingungen nicht zusagen.\n\n"
            f"Gründe:\n{rs}\n\n"
            "Wenn Sie möchten, können wir eine alternative Anfrage mit angepassten Rahmenbedingungen prüfen."
        ).strip()
    if decision == "auto_accept":
        link_line = f"\nKalender-Link: {event_link}" if event_link else ""
        quote_block = f"\n\nPreisübersicht:\n{quote_text}" if quote_text.strip() else ""
        return (
            "Vielen Dank für Ihre Rückmeldung. Der Termin wurde verbindlich im Kalender reserviert."
            f"{quote_block}{link_line}\n\n"
            "Vielen Dank für die Bestätigung. Bei Rückfragen melden Sie sich jederzeit."
        ).strip()
    return (
        "Vielen Dank für Ihre Anfrage. Für die verbindliche Bearbeitung benötige ich noch folgende Rückmeldung:\n\n"
        f"{rs}"
    ).strip()


def _compose_offer_request(*, quote_text: str, reasons: List[str] | None = None) -> str:
    rs = [str(x).strip() for x in (reasons or []) if str(x).strip()]
    reason_block = ""
    if rs:
        reason_block = "Hinweise:\n" + "\n".join(f"- {r}" for r in rs) + "\n\n"
    offer_block = str(quote_text or "").strip()
    offer_text = "Hier ist das konkrete Angebot:\n" + offer_block if offer_block else "Ein konkretes Angebot ist in Vorbereitung."
    return (
        "Vielen Dank, alle wichtigen Veranstaltungsdaten liegen vor.\n\n"
        f"{reason_block}"
        f"{offer_text}\n\n"
        "Bitte bestätigen Sie das Angebot kurz schriftlich. Erst danach reserviere ich den Termin verbindlich im Kalender."
    ).strip()


def _combine_start_end_iso(facts: Dict[str, Any]) -> Tuple[str, str]:
    date_iso = str(facts.get("event_date") or "").strip()
    start_time = str(facts.get("start_time") or "").strip() or "20:00"
    duration = float(facts.get("duration_hours") or 0.0)
    if not date_iso:
        return "", ""
    try:
        hh, mm = [int(x) for x in start_time.split(":", 1)]
    except Exception:
        hh, mm = 20, 0
    start_dt = datetime.fromisoformat(date_iso).replace(hour=hh, minute=mm, second=0, microsecond=0)
    if duration <= 0:
        duration = 4.0
    end_dt = start_dt + timedelta(hours=duration)
    return start_dt.isoformat(), end_dt.isoformat()


def _is_weekend_date(date_iso: str) -> bool | None:
    raw = str(date_iso or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d")
        except Exception:
            return None
    return dt.weekday() >= 5


def _offer_signature(*, facts: Dict[str, Any], quote_text: str) -> str:
    payload = {
        "event_date": str(facts.get("event_date") or "").strip(),
        "start_time": str(facts.get("start_time") or "").strip(),
        "duration_hours": float(facts.get("duration_hours") or 0.0),
        "location": str(facts.get("location") or "").strip(),
        "occasion": str(facts.get("occasion") or "").strip(),
        "quote_text": str(quote_text or "").strip(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _merge_facts(base: Dict[str, Any], candidate: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    out = dict(base or {})
    for key in required_fields:
        c_val = candidate.get(key)
        if _is_missing_required_field(key, c_val):
            continue
        out[key] = c_val
    return out


def _build_review(
    *,
    mail_id: str,
    thread_id: str,
    mailbox: str,
    from_email: str,
    subject: str,
    mail_text: str,
    draft_body: str,
    booking_decision: str,
    score: Dict[str, Any],
    sources: List[str],
    reason: str,
    action_templates: Dict[str, str],
    required_fields_status: Dict[str, Any],
) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "id": uuid4().hex,
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
        "score_total": float(score.get("score_total") or 0.0),
        "score": dict(score),
        "sources": list(sources),
        "ticket_id": "",
        "reason": reason,
        "action_templates": dict(action_templates or {}),
        "required_fields_status": dict(required_fields_status or {}),
        "selected_action": "",
        "sent": False,
        "send_result": {},
    }


def _build_action_templates(
    *,
    draft_body: str,
    facts: Dict[str, Any],
    quote_text: str,
    booking_decision: str,
) -> Dict[str, str]:
    client_name = str(facts.get("client_name") or "").strip()
    event_date = str(facts.get("event_date") or "").strip()
    start_time = str(facts.get("start_time") or "").strip()
    location = str(facts.get("location") or "").strip()
    duration = float(facts.get("duration_hours") or 0.0)

    greet = "Guten Tag"
    if client_name:
        greet = f"Guten Tag {client_name}"

    parts = [x for x in (event_date, start_time, location) if x]
    event_line = ", ".join(parts)

    approve_lines = [
        f"{greet},",
        "",
        "vielen Dank für Ihre Anfrage.",
        "Ich bestätige den Termin verbindlich.",
    ]
    if event_line:
        approve_lines.append(f"Termin: {event_line}")
    if duration > 0:
        approve_lines.append(f"Einsatzdauer: {duration:.1f} Stunden")
    if quote_text.strip():
        approve_lines.extend(["", "Vereinbartes Angebot:", quote_text.strip()])
    approve_lines.append("")
    approve_lines.append("Ich freue mich auf die Veranstaltung.")

    reject = (
        "Vielen Dank für Ihre Anfrage.\n\n"
        "Nach Prüfung kann ich den Auftrag unter den aktuell angefragten Rahmenbedingungen leider nicht bestätigen.\n\n"
        "Wenn Sie möchten, sende ich Ihnen gern einen Alternativvorschlag."
    )

    counter = (
        "Vielen Dank für Ihre Anfrage.\n\n"
        "Ich kann Ihnen einen angepassten Vorschlag anbieten.\n"
        "Bitte teilen Sie mir mit, welche Alternative für Sie passt."
    )

    if booking_decision == "auto_decline":
        approve_lines = [
            f"{greet},",
            "",
            "vielen Dank für Ihre Anfrage.",
            "ich bestätige den Auftrag ausnahmsweise trotz abweichender Rahmenbedingungen.",
        ]

    return {
        "approve": "\n".join(approve_lines).strip(),
        "reject": reject.strip(),
        "counteroffer": counter.strip(),
        "default": draft_body.strip(),
    }


def _contains_blocked_topic(*, text: str, blocked_topics: List[str]) -> str:
    t = str(text or "").lower()
    for item in blocked_topics:
        key = str(item or "").strip().lower()
        if key and key in t:
            return key
    return ""


def _quote_text_from_quote_payload(quote: Dict[str, Any]) -> str:
    return str(quote.get("text") or "").strip()


def _float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _facts_from_required_status(required_status: Dict[str, Any]) -> Dict[str, Any]:
    values = required_status.get("required_fields") if isinstance(required_status.get("required_fields"), dict) else {}
    out: Dict[str, Any] = {}
    for k, v in values.items():
        if _is_missing_required_field(str(k), v):
            continue
        out[str(k)] = v
    return out


def _build_resume_mail_for_approve(
    *,
    state: Dict[str, Any],
    registry: ToolRegistry,
    ctx: ToolContext,
    review: Dict[str, Any],
) -> str:
    thread_id = str(review.get("thread_id") or "").strip()
    case = get_thread_case(state, thread_id) if thread_id else None

    required_status = review.get("required_fields_status") if isinstance(review.get("required_fields_status"), dict) else {}
    if not required_status and isinstance(case, dict):
        req_names = case.get("required_field_names") if isinstance(case.get("required_field_names"), list) else []
        case_req = case.get("required_fields") if isinstance(case.get("required_fields"), dict) else {}
        case_facts = case.get("facts") if isinstance(case.get("facts"), dict) else {}
        merged_case_facts = dict(case_facts)
        for key, value in case_req.items():
            if _is_missing_required_field(str(key), value):
                continue
            merged_case_facts[str(key)] = value
        required_status = _build_required_fields_status(required_fields=[str(x).strip() for x in req_names], facts=merged_case_facts)

    required_names = _normalize_required_fields(
        [str(x).strip() for x in (required_status.get("required_field_names") or []) if str(x).strip()]
    )
    if not required_names:
        required_names = _normalize_required_fields(None)

    facts = _facts_from_required_status(required_status)
    if isinstance(case, dict):
        case_facts = case.get("facts") if isinstance(case.get("facts"), dict) else {}
        for key in required_names:
            if key in facts:
                continue
            c_val = case_facts.get(key)
            if _is_missing_required_field(key, c_val):
                continue
            facts[key] = c_val

    missing = [str(x).strip() for x in (required_status.get("missing_required_fields") or []) if str(x).strip()]
    missing_detail = [x for x in missing if x != "price_confirmed"]

    if missing_detail:
        clar = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="mail_compose_clarification",
            args={"missing_fields": missing_detail, "known_facts": facts},
        )
        body = str(clar.get("body") or "").strip()
        if body:
            return body
        return (
            "Vielen Dank für Ihre Rückmeldung.\n\n"
            "Für die verbindliche Bearbeitung benötige ich noch einige fehlende Angaben."
        ).strip()

    offer = case.get("offer") if isinstance(case, dict) and isinstance(case.get("offer"), dict) else {}
    offer_status = str(offer.get("status") or "").strip().lower()
    quote_text = str(offer.get("quote_text") or "").strip()
    if not quote_text:
        # Last resort: try to find quote text in previous review draft.
        draft = str(review.get("draft_body") or "").strip()
        if "Preisübersicht" in draft:
            quote_text = draft[draft.find("Preisübersicht"):].strip()

    price_confirmed = bool(facts.get("price_confirmed"))
    if offer_status not in {"sent", "accepted"}:
        if quote_text:
            body = _compose_offer_request(
                quote_text=quote_text,
                reasons=["Angebot wurde noch nicht bestätigt."],
            )
            if thread_id:
                upsert_thread_case(
                    state,
                    thread_id=thread_id,
                    patch={
                        "offer": {
                            "signature": str(offer.get("signature") or ""),
                            "quote_text": quote_text,
                            "status": "sent",
                            "sent_at": _now_iso(),
                        }
                    },
                )
            return body
        return (
            "Vielen Dank für Ihre Rückmeldung.\n\n"
            "Ich setze den Prozess fort und sende Ihnen im nächsten Schritt das konkrete Angebot."
        ).strip()

    if not price_confirmed:
        return _compose_offer_request(
            quote_text=quote_text,
            reasons=["Angebotsbestätigung ausstehend."],
        )

    # Erst wenn Angebot bestätigt ist, versuchen wir die finale Terminreservierung.
    start_iso, end_iso = _combine_start_end_iso(facts)
    if not start_iso or not end_iso:
        return (
            "Vielen Dank für Ihre Rückmeldung.\n\n"
            "Für die finale Reservierung benötige ich Datum, Startzeit und Dauer noch einmal eindeutig."
        ).strip()

    avail = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="calendar_check_availability",
        args={"start_iso": start_iso, "end_iso": end_iso, "calendar_id": "primary"},
    )
    if bool(avail.get("_error")):
        return (
            "Vielen Dank für Ihre Rückmeldung.\n\n"
            "Ich setze die finale Kalenderprüfung fort und melde mich mit der verbindlichen Bestätigung."
        ).strip()
    if not bool(avail.get("is_available")):
        return (
            "Vielen Dank für Ihre Rückmeldung. Das gewünschte Zeitfenster ist aktuell belegt.\n\n"
            "Bitte nennen Sie einen alternativen Termin oder eine alternative Startzeit."
        ).strip()

    created = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="calendar_create_event",
        args={
            "summary": f"DJ Booking: {str(facts.get('occasion') or 'Event')}",
            "start_iso": start_iso,
            "end_iso": end_iso,
            "location": str(facts.get("location") or ""),
            "description": "Verbindlich bestätigte Buchung aus Booking Assistant V2 (Operator approve)",
            "attendees": [str(review.get("from_email") or "")] if str(review.get("from_email") or "").strip() else [],
            "calendar_id": "primary",
        },
    )
    if created.get("_error"):
        return (
            "Vielen Dank für Ihre Rückmeldung.\n\n"
            "Ich habe die Freigabe erhalten und setze die finale Terminreservierung fort."
        ).strip()

    event_id = str(created.get("event_id") or "").strip()
    html_link = str(created.get("html_link") or "").strip()
    if thread_id and event_id:
        upsert_thread_case(
            state,
            thread_id=thread_id,
            patch={
                "status": "booked",
                "event": {
                    "event_id": event_id,
                    "html_link": html_link,
                    "updated_at": _now_iso(),
                },
                "offer": {
                    "signature": str(offer.get("signature") or ""),
                    "quote_text": quote_text,
                    "status": "accepted",
                    "confirmed_at": _now_iso(),
                },
            },
        )
    return _compose_decision_mail(
        decision="auto_accept",
        reasons=["Operator-Freigabe umgesetzt."],
        quote_text=quote_text,
        event_link=html_link,
    )


def run_once(*, user_id: str, settings: Settings, api_key: str, req: BookingAssistantV2RunRequest) -> BookingAssistantV2RunResponse:
    trace_token = _TRACE_ENABLED.set(bool(req.trace_steps))
    step_token = _TRACE_STEP.set(0)

    run_id = f"run_{uuid4().hex[:12]}"
    started_at = _now_iso()
    _trace_log(
        "BOOKING ASSISTANT V2 RUN",
        [
            f"user_id={user_id}",
            f"mailbox={req.mailbox}",
            f"limit={req.limit}",
            f"assistant_profile_name={req.assistant_profile_name}",
        ],
    )

    registry = build_registry(settings=settings, user_id=user_id)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_v2_run_once")

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
        return BookingAssistantV2RunResponse(
            ok=True,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            lock_blocked=True,
            lock_reason=str(lock.get("reason") or "run_already_active"),
        )

    run_items: List[BookingAssistantV2RunItem] = []
    sent_count = 0
    review_count = 0
    skipped_count = 0
    processed_count = 0

    try:
        profile = _ensure_profile(registry=registry, ctx=ctx, req=req)
        profile_instructions = (
            [str(x).strip() for x in (profile.get("instructions") or []) if str(x).strip()]
            if isinstance(profile.get("instructions"), list)
            else []
        )
        rules = profile.get("rules") if isinstance(profile.get("rules"), dict) else {}
        booking_rules = rules.get("booking") if isinstance(rules.get("booking"), dict) else {}
        pricing_rules = rules.get("pricing") if isinstance(rules.get("pricing"), dict) else {}
        mail_rules = rules.get("mail") if isinstance(rules.get("mail"), dict) else {}
        never_auto_send = bool(mail_rules.get("never_auto_send"))
        blocked_topics = [str(x).strip() for x in (mail_rules.get("block_auto_reply_topics") or []) if str(x).strip()]

        required_fields = _normalize_required_fields(
            [str(x).strip() for x in (rules.get("required_fields") or []) if str(x).strip()]
        )
        detail_required_fields = [x for x in required_fields if x != "price_confirmed"]

        fetch_limit = max(1, min(50, int(req.limit) * 5))
        inbox = _tool_call(
            registry=registry,
            ctx=ctx,
            tool="gmail_fetch_unanswered_mails",
            args={"mailbox": req.mailbox, "limit": fetch_limit},
        )
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
                run_items.append(
                    BookingAssistantV2RunItem(
                        mail_id=mail_id,
                        subject=subject,
                        from_email=from_email,
                        decision="skipped",
                        reason="already_processed",
                    )
                )
                append_activity(
                    state,
                    _activity_record(
                        mail_id=mail_id,
                        thread_id="",
                        decision="skipped",
                        reason="already_processed",
                    ),
                )
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
                run_items.append(
                    BookingAssistantV2RunItem(
                        mail_id=mail_id,
                        subject=subject,
                        from_email=from_email,
                        decision="failed",
                        reason=f"read_mail_failed:{exc}",
                    )
                )
                append_activity(
                    state,
                    _activity_record(mail_id=mail_id, thread_id="", decision="failed", reason=f"read_mail_failed:{exc}"),
                )
                continue

            thread_id = str(mail_payload.get("thread_id") or "").strip()
            intent_info = _classify_intent(registry=registry, ctx=ctx, mail_payload=mail_payload)
            intent = str(intent_info.get("intent") or "info")
            _trace_log(
                "INTENT",
                [
                    f"mail_id={mail_id}",
                    f"intent={intent}",
                    f"confidence={float(intent_info.get('confidence') or 0.0):.2f}",
                    f"reason={intent_info.get('reason')}",
                ],
            )

            if intent == "newsletter":
                skipped_count += 1
                processed_count += 1
                mark_processed(state, mail_id)
                run_items.append(
                    BookingAssistantV2RunItem(
                        mail_id=mail_id,
                        thread_id=thread_id,
                        subject=subject,
                        from_email=from_email,
                        decision="skipped",
                        reason="intent=newsletter",
                    )
                )
                append_activity(
                    state,
                    _activity_record(
                        mail_id=mail_id,
                        thread_id=thread_id,
                        decision="skipped",
                        reason="intent=newsletter",
                    ),
                )
                continue

            query = "\n".join(
                x
                for x in [
                    str(mail_payload.get("subject") or "").strip(),
                    str(mail_payload.get("body_text") or "").strip(),
                ]
                if x
            ).strip()

            draft = ""
            booking_decision = ""
            reasons: List[str] = []
            sources: List[str] = []
            score: Dict[str, Any] = {"score_total": 0.0, "verdict": "needs_review", "reason": ""}
            facts: Dict[str, Any] = {}
            required_fields_status: Dict[str, Any] = {}
            action_templates: Dict[str, str] = {}
            event_id = ""

            if intent == "info":
                blocked = _contains_blocked_topic(text=query, blocked_topics=blocked_topics)
                if blocked:
                    booking_decision = "human_review"
                    draft = (
                        "Vielen Dank für Ihre Anfrage. "
                        "Ich habe Ihr Anliegen intern zur persönlichen Prüfung weitergegeben "
                        "und melde mich zeitnah mit einer verbindlichen Rückmeldung."
                    )
                    reasons = [f"auto_reply_topic_blocked:{blocked}"]
                else:
                    context = _retrieve_web_context(
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
                )
            else:
                case = get_thread_case(state, thread_id) if thread_id else None
                case_required = {}
                if isinstance(case, dict):
                    case_required = case.get("required_fields") if isinstance(case.get("required_fields"), dict) else {}

                latest_text = "\n".join(
                    x
                    for x in [
                        str(mail_payload.get("subject") or "").strip(),
                        str(mail_payload.get("body_text") or "").strip(),
                    ]
                    if x
                ).strip()
                if not latest_text:
                    latest_text = str(mail_payload.get("text") or "")

                out_latest = _safe_tool_call(
                    registry=registry,
                    ctx=ctx,
                    tool="booking_extract_facts",
                    args={"text": latest_text, "required_fields": required_fields},
                )
                out_thread = _safe_tool_call(
                    registry=registry,
                    ctx=ctx,
                    tool="booking_extract_facts",
                    args={
                        "text": str(mail_payload.get("thread_facts_text") or str(mail_payload.get("text") or "")),
                        "required_fields": required_fields,
                    },
                )
                facts_latest = out_latest.get("facts") if isinstance(out_latest.get("facts"), dict) else {}
                facts_thread = out_thread.get("facts") if isinstance(out_thread.get("facts"), dict) else {}

                facts = _merge_facts({}, facts_thread, required_fields)
                facts = _merge_facts(facts, facts_latest, required_fields)
                facts = _merge_facts(facts, case_required, required_fields)

                required_fields_status = _build_required_fields_status(required_fields=required_fields, facts=facts)
                missing_detail_fields = [
                    x for x in detail_required_fields if x in required_fields_status.get("missing_required_fields", [])
                ]

                # Calendar precheck as soon as date is known.
                precheck_note = ""
                precheck_busy_hard = False
                if str(facts.get("event_date") or "").strip():
                    start_iso_pre, end_iso_pre = _combine_start_end_iso(facts)
                    if start_iso_pre and end_iso_pre:
                        precheck = _safe_tool_call(
                            registry=registry,
                            ctx=ctx,
                            tool="calendar_check_availability",
                            args={"start_iso": start_iso_pre, "end_iso": end_iso_pre, "calendar_id": "primary"},
                        )
                        if not precheck.get("_error"):
                            if bool(precheck.get("is_available")):
                                precheck_note = "Vorab-Check Kalender: Der Termin ist im aktuell geprüften Zeitfenster frei."
                            else:
                                precheck_note = "Vorab-Check Kalender: Das aktuell geprüfte Zeitfenster ist belegt."
                                start_known = bool(str(facts.get("start_time") or "").strip())
                                duration_known = _float_or(facts.get("duration_hours"), 0.0) > 0.0
                                precheck_busy_hard = bool(start_known and duration_known)

                quote_payload: Dict[str, Any] = {}
                quote_text = ""
                distance_payload: Dict[str, Any] = {}

                if not missing_detail_fields:
                    origin = str(booking_rules.get("base_address") or "Pforzheim, Deutschland")
                    destination = str(facts.get("location") or "").strip()
                    distance_payload = _safe_tool_call(
                        registry=registry,
                        ctx=ctx,
                        tool="distance_check",
                        args={
                            "origin": origin,
                            "destination": destination,
                            "max_distance_km": float(booking_rules.get("max_distance_km") or 0.0),
                        },
                    )
                    quote_payload = _safe_tool_call(
                        registry=registry,
                        ctx=ctx,
                        tool="pricing_compute_quote",
                        args={
                            "facts": facts,
                            "pricing_rules": pricing_rules,
                            "booking_rules": booking_rules,
                            "distance_km": float(distance_payload.get("distance_km") or 0.0),
                        },
                    )
                    quote_text = _quote_text_from_quote_payload(quote_payload)

                if precheck_busy_hard:
                    booking_decision = "need_clarification"
                    draft = (
                        "Vielen Dank für Ihre Anfrage. Das gewünschte Zeitfenster ist im Kalender bereits belegt.\n\n"
                        "Bitte nennen Sie einen alternativen Termin oder eine alternative Startzeit."
                    ).strip()
                    reasons = ["calendar_precheck_busy"]
                elif missing_detail_fields:
                    booking_decision = "need_clarification"
                    clar = _safe_tool_call(
                        registry=registry,
                        ctx=ctx,
                        tool="mail_compose_clarification",
                        args={"missing_fields": missing_detail_fields, "known_facts": facts},
                    )
                    draft = str(clar.get("body") or "").strip()
                    if precheck_note:
                        draft = f"{draft}\n\n{precheck_note}".strip()
                    reasons = ["missing_required_fields"]
                else:
                    # Deterministic early checks + engine check.
                    weekend_only = bool(booking_rules.get("weekend_only", True))
                    if weekend_only and _is_weekend_date(str(facts.get("event_date") or "")) is False:
                        booking_decision = "auto_decline"
                        reasons = ["Buchungen sind nur am Wochenende möglich."]
                        draft = _compose_decision_mail(decision=booking_decision, reasons=reasons)
                    else:
                        pre_dec = _safe_tool_call(
                            registry=registry,
                            ctx=ctx,
                            tool="booking_decision_engine",
                            args={
                                "facts": facts,
                                "profile_rules": booking_rules,
                                "completeness": {"complete": True, "missing_fields": []},
                                "distance": distance_payload,
                                "quote": quote_payload,
                                "require_price_confirmation": False,
                            },
                        )
                        pre_decision = str(pre_dec.get("decision") or "human_review").strip().lower()
                        reasons = [str(x).strip() for x in (pre_dec.get("reasons") or []) if str(x).strip()]

                        if pre_decision in {"auto_decline", "human_review"}:
                            booking_decision = pre_decision
                            if booking_decision == "auto_decline":
                                draft = _compose_decision_mail(decision=booking_decision, reasons=reasons, quote_text=quote_text)
                            else:
                                draft = (
                                    "Vielen Dank für Ihre Anfrage. "
                                    "Ich habe Ihr Anliegen intern zur persönlichen Prüfung weitergegeben "
                                    "und melde mich zeitnah mit einer verbindlichen Rückmeldung."
                                )
                        else:
                            offer_gate = _llm_classify_offer_confirmation(
                                latest_body_text=str(mail_payload.get("body_text") or ""),
                                thread_text=str(mail_payload.get("text") or ""),
                            )
                            gate_conf = float(offer_gate.get("confidence") or 0.0)
                            offer_present = bool(offer_gate.get("offer_present_in_thread")) and gate_conf >= 0.6
                            offer_accepted = bool(offer_gate.get("offer_accepted_by_latest_reply")) and gate_conf >= 0.7
                            sig = _offer_signature(facts=facts, quote_text=quote_text)
                            case_offer = case.get("offer") if isinstance(case, dict) and isinstance(case.get("offer"), dict) else {}
                            already_sent_same_offer = str(case_offer.get("signature") or "") == sig

                            if not offer_present and not already_sent_same_offer:
                                booking_decision = "need_clarification"
                                draft = _compose_offer_request(
                                    quote_text=quote_text,
                                    reasons=["Angebot wurde noch nicht bestätigt."],
                                )
                                reasons = ["offer_sent_waiting_confirmation"]
                            elif not offer_accepted:
                                booking_decision = "need_clarification"
                                draft = _compose_offer_request(
                                    quote_text=quote_text,
                                    reasons=["Angebotsbestätigung ausstehend."],
                                )
                                reasons = ["offer_confirmation_missing"]
                            else:
                                facts["price_confirmed"] = True
                                final_dec = _safe_tool_call(
                                    registry=registry,
                                    ctx=ctx,
                                    tool="booking_decision_engine",
                                    args={
                                        "facts": facts,
                                        "profile_rules": booking_rules,
                                        "completeness": {"complete": True, "missing_fields": []},
                                        "distance": distance_payload,
                                        "quote": quote_payload,
                                        "require_price_confirmation": True,
                                    },
                                )
                                booking_decision = str(final_dec.get("decision") or "human_review").strip().lower()
                                reasons = [str(x).strip() for x in (final_dec.get("reasons") or []) if str(x).strip()]

                                if booking_decision == "auto_accept":
                                    start_iso, end_iso = _combine_start_end_iso(facts)
                                    if start_iso and end_iso:
                                        avail = _safe_tool_call(
                                            registry=registry,
                                            ctx=ctx,
                                            tool="calendar_check_availability",
                                            args={"start_iso": start_iso, "end_iso": end_iso, "calendar_id": "primary"},
                                        )
                                        if bool(avail.get("is_available")):
                                            created = _safe_tool_call(
                                                registry=registry,
                                                ctx=ctx,
                                                tool="calendar_create_event",
                                                args={
                                                    "summary": f"DJ Booking: {str(facts.get('occasion') or 'Event')}",
                                                    "start_iso": start_iso,
                                                    "end_iso": end_iso,
                                                    "location": str(facts.get("location") or ""),
                                                    "description": "Verbindlich bestätigte Buchung aus Booking Assistant V2",
                                                    "attendees": [from_email] if from_email else [],
                                                    "calendar_id": "primary",
                                                },
                                            )
                                            event_id = str(created.get("event_id") or "").strip()
                                            link = str(created.get("html_link") or "").strip()
                                            draft = _compose_decision_mail(
                                                decision="auto_accept",
                                                reasons=reasons,
                                                quote_text=quote_text,
                                                event_link=link,
                                            )
                                        else:
                                            booking_decision = "need_clarification"
                                            draft = (
                                                "Vielen Dank für Ihre Rückmeldung. Das gewünschte Zeitfenster ist aktuell belegt.\n\n"
                                                "Bitte nennen Sie einen alternativen Termin oder eine alternative Startzeit."
                                            )
                                            reasons = ["calendar_busy"]
                                    else:
                                        booking_decision = "need_clarification"
                                        draft = (
                                            "Vielen Dank für Ihre Rückmeldung. Für die finale Reservierung fehlt eine valide Terminzeit.\n\n"
                                            "Bitte nennen Sie Datum und Startzeit erneut."
                                        )
                                        reasons = ["invalid_time_window"]
                                elif booking_decision == "auto_decline":
                                    draft = _compose_decision_mail(decision="auto_decline", reasons=reasons, quote_text=quote_text)
                                elif booking_decision == "human_review":
                                    draft = (
                                        "Vielen Dank für Ihre Anfrage. "
                                        "Ich habe Ihr Anliegen intern zur persönlichen Prüfung weitergegeben "
                                        "und melde mich zeitnah mit einer verbindlichen Rückmeldung."
                                    )
                                else:
                                    booking_decision = "need_clarification"
                                    draft = _compose_offer_request(
                                        quote_text=quote_text,
                                        reasons=["Angebotsbestätigung ausstehend."],
                                    )

                            # Update offer state if we sent/observed an offer.
                            if booking_decision == "need_clarification" and quote_text.strip():
                                upsert_thread_case(
                                    state,
                                    thread_id=thread_id or mail_id,
                                    patch={
                                        "offer": {
                                            "signature": sig,
                                            "quote_text": quote_text,
                                            "total_eur": float(quote_payload.get("total_eur") or 0.0),
                                            "currency": str(quote_payload.get("currency") or "EUR"),
                                            "status": "sent",
                                            "sent_at": _now_iso(),
                                        }
                                    },
                                )
                            if booking_decision == "auto_accept" and quote_text.strip():
                                upsert_thread_case(
                                    state,
                                    thread_id=thread_id or mail_id,
                                    patch={
                                        "offer": {
                                            "signature": sig,
                                            "quote_text": quote_text,
                                            "total_eur": float(quote_payload.get("total_eur") or 0.0),
                                            "currency": str(quote_payload.get("currency") or "EUR"),
                                            "status": "accepted",
                                            "confirmed_at": _now_iso(),
                                        },
                                    },
                                )

                required_fields_status = _build_required_fields_status(required_fields=required_fields, facts=facts)
                case_patch = {
                    "last_mail_id": mail_id,
                    "from_email": from_email,
                    "subject": subject,
                    "status": (
                        "booked"
                        if booking_decision == "auto_accept"
                        else "declined"
                        if booking_decision == "auto_decline"
                        else "human_review"
                        if booking_decision == "human_review"
                        else "gathering"
                    ),
                    "required_field_names": required_fields,
                    "required_fields": dict(required_fields_status.get("required_fields") or {}),
                    "missing_required_fields": list(required_fields_status.get("missing_required_fields") or []),
                    "facts": {k: facts.get(k) for k in required_fields if k in facts},
                }
                if event_id:
                    case_patch["event"] = {
                        "event_id": event_id,
                        "updated_at": _now_iso(),
                    }
                upsert_thread_case(state, thread_id=thread_id or mail_id, patch=case_patch)
                append_case_history(
                    state,
                    thread_id=thread_id or mail_id,
                    history_item={
                        "mail_id": mail_id,
                        "booking_decision": booking_decision,
                        "reason": "; ".join(reasons),
                    },
                )

                action_templates = _build_action_templates(
                    draft_body=draft,
                    facts=facts,
                    quote_text=quote_text,
                    booking_decision=booking_decision,
                )

                if booking_decision == "human_review":
                    score = {
                        "score_total": 0.85,
                        "verdict": "needs_review",
                        "reason": "human_review_required",
                        "raw": {"bypassed": True},
                    }
                else:
                    score = _score_reply(
                        registry=registry,
                        ctx=ctx,
                        user_message=str(mail_payload.get("text") or "").strip(),
                        draft=draft,
                        sources=sources,
                        booking_decision=booking_decision,
                        facts=facts,
                        required_fields=required_fields,
                        missing_fields=list(required_fields_status.get("missing_required_fields") or []),
                    )

            instruction = _instruction_check(
                registry=registry,
                ctx=ctx,
                instructions=profile_instructions,
                user_message=str(mail_payload.get("text") or "").strip(),
                draft_reply=draft,
                booking_decision=booking_decision,
                facts=facts,
            )
            instruction_allowed = bool(instruction.get("allowed"))

            policy = _policy_check(registry=registry, ctx=ctx, text=draft, strict_mode=req.strict_policy)
            policy_allowed = bool(policy.get("allowed"))
            policy_risk = str(policy.get("risk_level") or "").strip().lower()

            score_total = float(score.get("score_total") or 0.0)
            verdict = str(score.get("verdict") or "needs_review").strip().lower()

            if never_auto_send:
                can_auto_send = False
            elif booking_decision == "human_review":
                can_auto_send = False
            elif booking_decision in {"need_clarification", "auto_decline", "auto_accept"}:
                can_auto_send = bool(draft and policy_allowed and policy_risk not in {"high", "critical"} and instruction_allowed)
            else:
                can_auto_send = bool(
                    draft
                    and policy_allowed
                    and policy_risk not in {"high", "critical"}
                    and instruction_allowed
                    and score_total >= req.auto_send_threshold
                    and verdict == "send"
                )

            if can_auto_send:
                send_out = _tool_call(
                    registry=registry,
                    ctx=ctx,
                    tool="gmail_answer_mail",
                    args={"mail_id": mail_id, "mailbox": req.mailbox, "body": draft},
                )
                sent_count += 1
                processed_count += 1
                mark_processed(state, mail_id)
                run_items.append(
                    BookingAssistantV2RunItem(
                        mail_id=mail_id,
                        thread_id=thread_id,
                        subject=subject,
                        from_email=from_email,
                        decision="auto_sent",
                        booking_decision=booking_decision or "info_reply",
                        sent=bool(send_out.get("sent")),
                        score_total=score_total,
                        reason=str(score.get("reason") or "auto_sent"),
                        event_id=event_id,
                    )
                )
                append_activity(
                    state,
                    _activity_record(
                        mail_id=mail_id,
                        thread_id=thread_id,
                        decision="auto_sent",
                        booking_decision=booking_decision or "info_reply",
                        event_id=event_id,
                        reason=str(score.get("reason") or "auto_sent"),
                    ),
                )
                continue

            reason_parts: List[str] = []
            if never_auto_send:
                reason_parts.append("never_auto_send")
            if booking_decision:
                reason_parts.append(f"booking_decision={booking_decision}")
            if not instruction_allowed:
                reason_parts.append("instruction_blocked")
                viol = [str(x).strip() for x in (instruction.get("violations") or []) if str(x).strip()]
                if viol:
                    reason_parts.append("instruction_violations=" + "; ".join(viol))
            if not policy_allowed:
                reason_parts.append("policy_blocked")
            base_reason = str(score.get("reason") or "needs_human_review").strip()
            if base_reason:
                reason_parts.insert(0, base_reason)
            reason_text = " | ".join(x for x in reason_parts if x)

            review = _build_review(
                mail_id=mail_id,
                thread_id=thread_id,
                mailbox=req.mailbox,
                from_email=from_email,
                subject=subject,
                mail_text=str(mail_payload.get("text") or "").strip(),
                draft_body=draft,
                booking_decision=booking_decision,
                score=score,
                sources=sources,
                reason=reason_text,
                action_templates=action_templates
                or {
                    "approve": draft,
                    "reject": "Vielen Dank für Ihre Anfrage. Nach interner Prüfung kann ich diese Anfrage leider nicht zusagen.",
                    "counteroffer": "Vielen Dank für Ihre Anfrage. Gern prüfe ich eine angepasste Variante. Bitte teilen Sie mir mögliche Alternativen mit.",
                    "default": draft,
                },
                required_fields_status=required_fields_status,
            )
            add_review(state, review)
            mark_processed(state, mail_id)
            processed_count += 1
            review_count += 1
            review_id = str(review.get("id") or "")
            append_activity(
                state,
                _activity_record(
                    mail_id=mail_id,
                    thread_id=thread_id,
                    decision="needs_human",
                    booking_decision=booking_decision,
                    review_id=review_id,
                    event_id=event_id,
                    reason=reason_text,
                ),
            )
            run_items.append(
                BookingAssistantV2RunItem(
                    mail_id=mail_id,
                    thread_id=thread_id,
                    subject=subject,
                    from_email=from_email,
                    decision="needs_human",
                    booking_decision=booking_decision,
                    review_id=review_id,
                    event_id=event_id,
                    score_total=score_total,
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
        return BookingAssistantV2RunResponse(
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
            items=run_items,
        )
    finally:
        _TRACE_ENABLED.reset(trace_token)
        _TRACE_STEP.reset(step_token)


def get_reviews(*, user_id: str, settings: Settings, status: str = "pending") -> BookingAssistantV2ReviewsResponse:
    state = load_state(settings, user_id)
    filter_status = str(status or "").strip().lower()
    if filter_status in {"", "all", "*"}:
        filter_status = ""
    reviews = [BookingAssistantV2ReviewItem(**r) for r in list_reviews(state, status=filter_status)]
    reviews.sort(key=lambda x: x.created_at, reverse=True)
    return BookingAssistantV2ReviewsResponse(ok=True, reviews=reviews)


def _review_template_body(review: Dict[str, Any], template_key: str, fallback: str = "") -> str:
    templates = review.get("action_templates")
    if isinstance(templates, dict):
        value = str(templates.get(template_key) or "").strip()
        if value:
            return value
        default_value = str(templates.get("default") or "").strip()
        if default_value:
            return default_value
    return str(fallback or "").strip()


def approve_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV2ApproveRequest,
) -> BookingAssistantV2ReviewActionResponse:
    state = load_state(settings, user_id)
    review = find_review(state, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if str(review.get("status") or "").strip().lower() != "pending":
        raise HTTPException(status_code=409, detail="Review is not pending")

    manual_body = str(req.edited_body or "").strip()
    if manual_body:
        body = manual_body
    else:
        registry = build_registry(settings=settings, user_id=user_id)
        ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_v2_review_approve")
        body = _build_resume_mail_for_approve(
            state=state,
            registry=registry,
            ctx=ctx,
            review=review,
        )
    if not body:
        raise HTTPException(status_code=422, detail="No draft body available for approval")

    registry = build_registry(settings=settings, user_id=user_id)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_v2_review_approve")
    policy = _policy_check(registry=registry, ctx=ctx, text=body, strict_mode=True)
    if not bool(policy.get("allowed")):
        raise HTTPException(status_code=422, detail="manual_approve_blocked_by_policy")

    args: Dict[str, Any] = {
        "mail_id": str(review.get("mail_id") or ""),
        "mailbox": str(review.get("mailbox") or "INBOX"),
        "body": body,
    }
    subject = str(req.subject or "").strip()
    if subject:
        args["subject"] = subject

    send_result = _tool_call(registry=registry, ctx=ctx, tool="gmail_answer_mail", args=args)
    review["status"] = "approved"
    review["updated_at"] = _now_iso()
    review["sent"] = bool(send_result.get("sent"))
    review["send_result"] = send_result
    review["draft_body"] = body
    review["selected_action"] = "approve"
    append_activity(
        state,
        _activity_record(
            mail_id=str(review.get("mail_id") or ""),
            thread_id=str(review.get("thread_id") or ""),
            decision="operator_approve",
            booking_decision=str(review.get("booking_decision") or ""),
            review_id=str(review.get("id") or review_id),
            reason="review_approved_resume_flow",
        ),
    )
    save_state(settings, user_id, state)

    return BookingAssistantV2ReviewActionResponse(
        ok=True,
        review_id=str(review.get("id") or review_id),
        status=str(review.get("status") or "approved"),
        sent=bool(review.get("sent")),
    )


def reject_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV2RejectRequest,
) -> BookingAssistantV2ReviewActionResponse:
    state = load_state(settings, user_id)
    review = find_review(state, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if str(review.get("status") or "").strip().lower() != "pending":
        raise HTTPException(status_code=409, detail="Review is not pending")

    reason = str(req.reason or "").strip()
    body = str(req.edited_body or "").strip() or _review_template_body(
        review,
        "reject",
        fallback="Vielen Dank für Ihre Anfrage. Nach interner Prüfung kann ich den Auftrag leider nicht bestätigen.",
    )
    subject = str(req.subject or "").strip()
    send_to_customer = bool(req.send_to_customer)

    send_result: Dict[str, Any] = {}
    if send_to_customer:
        registry = build_registry(settings=settings, user_id=user_id)
        ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_v2_review_reject")
        policy = _policy_check(registry=registry, ctx=ctx, text=body, strict_mode=True)
        if not bool(policy.get("allowed")):
            raise HTTPException(status_code=422, detail="manual_reject_blocked_by_policy")
        args: Dict[str, Any] = {
            "mail_id": str(review.get("mail_id") or ""),
            "mailbox": str(review.get("mailbox") or "INBOX"),
            "body": body,
        }
        if subject:
            args["subject"] = subject
        send_result = _tool_call(registry=registry, ctx=ctx, tool="gmail_answer_mail", args=args)

    review["status"] = "rejected"
    review["updated_at"] = _now_iso()
    if reason:
        review["reason"] = reason
    review["sent"] = bool(send_result.get("sent"))
    review["send_result"] = send_result
    review["draft_body"] = body
    review["selected_action"] = "reject"
    append_activity(
        state,
        _activity_record(
            mail_id=str(review.get("mail_id") or ""),
            thread_id=str(review.get("thread_id") or ""),
            decision="operator_reject",
            booking_decision=str(review.get("booking_decision") or ""),
            review_id=str(review.get("id") or review_id),
            reason=reason or "review_rejected",
        ),
    )
    save_state(settings, user_id, state)

    return BookingAssistantV2ReviewActionResponse(
        ok=True,
        review_id=str(review.get("id") or review_id),
        status=str(review.get("status") or "rejected"),
        sent=bool(review.get("sent")),
        reason=str(review.get("reason") or ""),
    )


def counteroffer_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV2CounterofferRequest,
) -> BookingAssistantV2ReviewActionResponse:
    state = load_state(settings, user_id)
    review = find_review(state, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if str(review.get("status") or "").strip().lower() != "pending":
        raise HTTPException(status_code=409, detail="Review is not pending")

    body = str(req.edited_body or "").strip() or _review_template_body(
        review,
        "counteroffer",
        fallback=str(review.get("draft_body") or ""),
    )
    if not body:
        raise HTTPException(status_code=422, detail="No counteroffer body available")

    registry = build_registry(settings=settings, user_id=user_id)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_v2_review_counteroffer")

    policy = _policy_check(registry=registry, ctx=ctx, text=body, strict_mode=True)
    if not bool(policy.get("allowed")):
        raise HTTPException(status_code=422, detail="manual_counteroffer_blocked_by_policy")

    args: Dict[str, Any] = {
        "mail_id": str(review.get("mail_id") or ""),
        "mailbox": str(review.get("mailbox") or "INBOX"),
        "body": body,
    }
    subject = str(req.subject or "").strip()
    if subject:
        args["subject"] = subject

    send_result = _tool_call(registry=registry, ctx=ctx, tool="gmail_answer_mail", args=args)
    review["status"] = "counteroffered"
    review["updated_at"] = _now_iso()
    review["sent"] = bool(send_result.get("sent"))
    review["send_result"] = send_result
    review["draft_body"] = body
    review["selected_action"] = "counteroffer"
    append_activity(
        state,
        _activity_record(
            mail_id=str(review.get("mail_id") or ""),
            thread_id=str(review.get("thread_id") or ""),
            decision="operator_counteroffer",
            booking_decision=str(review.get("booking_decision") or ""),
            review_id=str(review.get("id") or review_id),
            reason="review_counteroffered",
        ),
    )
    save_state(settings, user_id, state)

    return BookingAssistantV2ReviewActionResponse(
        ok=True,
        review_id=str(review.get("id") or review_id),
        status=str(review.get("status") or "counteroffered"),
        sent=bool(review.get("sent")),
        reason=str(review.get("reason") or ""),
    )


def _review_summary_item(review: Dict[str, Any]) -> Dict[str, Any]:
    templates = review.get("action_templates") if isinstance(review.get("action_templates"), dict) else {}
    required = review.get("required_fields_status") if isinstance(review.get("required_fields_status"), dict) else {}
    return {
        "id": str(review.get("id") or "").strip(),
        "mail_id": str(review.get("mail_id") or "").strip(),
        "thread_id": str(review.get("thread_id") or "").strip(),
        "mailbox": str(review.get("mailbox") or "INBOX").strip(),
        "from_email": str(review.get("from_email") or "").strip(),
        "subject": str(review.get("subject") or "").strip(),
        "status": str(review.get("status") or "").strip(),
        "booking_decision": str(review.get("booking_decision") or "").strip(),
        "reason": str(review.get("reason") or "").strip(),
        "created_at": str(review.get("created_at") or "").strip(),
        "score_total": float(review.get("score_total") or 0.0),
        "options": {
            "approve": str(templates.get("approve") or "").strip(),
            "reject": str(templates.get("reject") or "").strip(),
            "counteroffer": str(templates.get("counteroffer") or "").strip(),
        },
        "required_fields_status": required,
    }


def _normalize_blocker_reason(reason: str) -> str:
    raw = str(reason or "").strip()
    if not raw:
        return "Unbekannter Blocker"
    low = raw.lower()
    if "maximaldauer" in low or "dauer" in low and ">" in low:
        return "Dauer ueber Maximum"
    if "wochenende" in low:
        return "Wochenendregel verletzt"
    if "angebot" in low and "best" in low:
        return "Angebot noch nicht bestaetigt"
    if "instruction_blocked" in low:
        return "Profil-Instruktionen blockieren Auto-Versand"
    if "policy_blocked" in low:
        return "Policy-Check blockiert Versand"
    if "kalender" in low and "belegt" in low:
        return "Kalenderfenster belegt"
    if "distance" in low or "distanz" in low:
        return "Distanzregel verletzt"
    return raw[:120]


def _build_status_recommendations(*, top_labels: List[str], pending_count: int) -> List[str]:
    recommendations: List[str] = []
    labels = {x for x in top_labels if x}
    if "Dauer ueber Maximum" in labels:
        recommendations.append("Bei langen Anfragen aktiv Counteroffer mit maximal erlaubter Dauer senden.")
    if "Wochenendregel verletzt" in labels:
        recommendations.append("Direkt naechste Wochenend-Termine als Alternativen vorschlagen.")
    if "Angebot noch nicht bestaetigt" in labels:
        recommendations.append("Angebot klar ausweisen und explizite Angebotsbestaetigung anfordern.")
    if "Kalenderfenster belegt" in labels:
        recommendations.append("Bei belegten Slots sofort 2-3 alternative Zeitfenster anbieten.")
    if pending_count > 0:
        recommendations.append("Offene Pendings mit pending_next nacheinander triagieren.")
    if not recommendations:
        recommendations.append("Aktueller Ablauf ist stabil; keine akuten Prozess-Blocker erkannt.")
    return recommendations[:5]


def get_status_meeting(
    *,
    user_id: str,
    settings: Settings,
    since: str = "",
) -> BookingAssistantV2StatusMeetingResponse:
    state = load_state(settings, user_id)
    since_input = str(since or "").strip()
    since_raw = _normalize_since_alias(since_input)
    since_dt = _parse_iso_utc(since_raw) if since_raw else None
    if since_raw and since_dt is None:
        raise HTTPException(status_code=422, detail="Invalid 'since' timestamp. Use ISO-8601.")

    activity = state.get("activity_log") if isinstance(state.get("activity_log"), list) else []
    filtered_activity: List[Dict[str, Any]] = []
    for item in activity:
        if not isinstance(item, dict):
            continue
        ts = _parse_iso_utc(str(item.get("timestamp") or "").strip())
        if since_dt and (ts is None or ts < since_dt):
            continue
        filtered_activity.append(dict(item))

    filtered_activity.sort(key=lambda x: str(x.get("timestamp") or ""))
    recent_activity = filtered_activity[-20:]

    confirmed_items: List[Dict[str, Any]] = []
    rejected_items: List[Dict[str, Any]] = []
    blocker_reasons: List[str] = []
    for item in filtered_activity:
        decision = str(item.get("decision") or "").strip().lower()
        booking_decision = str(item.get("booking_decision") or "").strip().lower()
        if decision == "operator_approve" or (decision == "auto_sent" and booking_decision == "auto_accept"):
            confirmed_items.append(
                {
                    "timestamp": str(item.get("timestamp") or "").strip(),
                    "mail_id": str(item.get("mail_id") or "").strip(),
                    "thread_id": str(item.get("thread_id") or "").strip(),
                    "event_id": str(item.get("event_id") or "").strip(),
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
        if decision == "operator_reject" or (decision == "auto_sent" and booking_decision == "auto_decline"):
            rejected_items.append(
                {
                    "timestamp": str(item.get("timestamp") or "").strip(),
                    "mail_id": str(item.get("mail_id") or "").strip(),
                    "thread_id": str(item.get("thread_id") or "").strip(),
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
        if decision in {"needs_human", "failed"}:
            blocker_reasons.append(str(item.get("reason") or "").strip())

    pending_reviews = list_reviews(state, status="pending")
    pending_items = [_review_summary_item(r) for r in pending_reviews if isinstance(r, dict)]
    for rv in pending_reviews:
        if isinstance(rv, dict):
            blocker_reasons.append(str(rv.get("reason") or "").strip())

    reason_counts: Dict[str, int] = {}
    for reason in blocker_reasons:
        label = _normalize_blocker_reason(reason)
        reason_counts[label] = int(reason_counts.get(label) or 0) + 1
    top_blockers = sorted(
        ({"reason": k, "count": v} for k, v in reason_counts.items()),
        key=lambda x: x["count"],
        reverse=True,
    )[:5]
    top_labels = [str(x.get("reason") or "") for x in top_blockers]
    recommendations = _build_status_recommendations(top_labels=top_labels, pending_count=len(pending_items))

    since_text = since_raw or ""
    generated_at = _now_iso()
    text = (
        f"Status seit {since_text or 'Beginn'}: "
        f"{len(confirmed_items)} bestaetigt, "
        f"{len(rejected_items)} abgelehnt, "
        f"{len(pending_items)} pending."
    )

    return BookingAssistantV2StatusMeetingResponse(
        ok=True,
        since=since_text,
        generated_at=generated_at,
        total_processed=len(filtered_activity),
        confirmed_count=len(confirmed_items),
        rejected_count=len(rejected_items),
        pending_count=len(pending_items),
        confirmed_items=confirmed_items[:20],
        rejected_items=rejected_items[:20],
        pending_items=pending_items[:20],
        top_blockers=top_blockers,
        recommendations=recommendations,
        recent_activity=recent_activity,
        text=text,
    )


def get_pending_next(*, user_id: str, settings: Settings) -> BookingAssistantV2PendingNextResponse:
    state = load_state(settings, user_id)
    pending = [r for r in list_reviews(state, status="pending") if isinstance(r, dict)]
    pending.sort(key=lambda x: str(x.get("created_at") or ""))
    if not pending:
        return BookingAssistantV2PendingNextResponse(
            ok=True,
            has_pending=False,
            review={},
            options={},
            text="Keine offenen Pending-Faelle vorhanden.",
        )

    review = pending[0]
    summary = _review_summary_item(review)
    options = dict(summary.get("options") or {})
    summary["options"] = options
    return BookingAssistantV2PendingNextResponse(
        ok=True,
        has_pending=True,
        review=summary,
        options=options,
        text=f"Naechster Pending-Fall: {summary.get('id')} ({summary.get('subject') or 'ohne Betreff'}).",
    )


def apply_pending_action(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: BookingAssistantV2PendingApplyRequest,
) -> BookingAssistantV2ReviewActionResponse:
    action = str(req.action or "").strip().lower()
    if action == "approve":
        return approve_review(
            user_id=user_id,
            settings=settings,
            api_key=api_key,
            review_id=review_id,
            req=BookingAssistantV2ApproveRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
            ),
        )
    if action == "reject":
        return reject_review(
            user_id=user_id,
            settings=settings,
            api_key=api_key,
            review_id=review_id,
            req=BookingAssistantV2RejectRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )
    if action == "counteroffer":
        return counteroffer_review(
            user_id=user_id,
            settings=settings,
            api_key=api_key,
            review_id=review_id,
            req=BookingAssistantV2CounterofferRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
            ),
        )
    raise HTTPException(status_code=422, detail=f"Unsupported action: {action}")


def operator_chat(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    req: BookingAssistantV2OperatorChatRequest,
) -> BookingAssistantV2OperatorChatResponse:
    low = str(req.message or "").strip().lower()

    if any(x in low for x in ("status", "zusammenfassung", "report", "meeting")):
        status = get_status_meeting(user_id=user_id, settings=settings, since="")
        return BookingAssistantV2OperatorChatResponse(
            ok=True,
            intent="status_meeting",
            action_taken="status_meeting",
            data=status.model_dump(),
            text=status.text,
        )

    if "pending" in low and any(x in low for x in ("next", "naechst", "nächst")):
        nxt = get_pending_next(user_id=user_id, settings=settings)
        return BookingAssistantV2OperatorChatResponse(
            ok=True,
            intent="triage_pending",
            action_taken="pending_next",
            data=nxt.model_dump(),
            text=nxt.text,
        )

    if "pending" in low or "offen" in low:
        reviews = get_reviews(user_id=user_id, settings=settings, status="pending")
        return BookingAssistantV2OperatorChatResponse(
            ok=True,
            intent="list_pending",
            action_taken="list_pending",
            data=reviews.model_dump(),
            text=f"Offene Pendings: {len(reviews.reviews)}",
        )

    return BookingAssistantV2OperatorChatResponse(
        ok=True,
        intent="fallback",
        action_taken="fallback",
        data={},
        text="Bitte nennen Sie eine konkrete Aktion, z. B. 'status', 'pending next' oder 'pending'.",
    )
