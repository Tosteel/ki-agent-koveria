from __future__ import annotations

import json
import re
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import HTTPException

from server.agent.langchain_runtime import dispatch_tool_chain
from server.agent.tool_registry import ToolContext, ToolRegistry
from server.core.settings import Settings
from server.services.agent_service import build_registry
from server.services.llm_ionos import IonosLLM

from .models import (
    BookingAssistantApproveRequest,
    BookingAssistantCounterofferRequest,
    BookingAssistantOperatorChatRequest,
    BookingAssistantOperatorChatResponse,
    BookingAssistantPendingApplyRequest,
    BookingAssistantPendingNextResponse,
    BookingAssistantRejectRequest,
    BookingAssistantReviewActionResponse,
    BookingAssistantReviewItem,
    BookingAssistantReviewsResponse,
    BookingAssistantRunItem,
    BookingAssistantRunRequest,
    BookingAssistantRunResponse,
    BookingAssistantStatusMeetingResponse,
)
from .store import (
    acquire_run_lock,
    add_review,
    append_activity,
    append_run_history,
    find_review,
    has_processed,
    list_reviews,
    load_state,
    mark_processed,
    release_run_lock,
    save_state,
)

_TRACE_ENABLED: ContextVar[bool] = ContextVar("booking_assistant_trace_enabled", default=False)
_TRACE_STEP: ContextVar[int] = ContextVar("booking_assistant_trace_step", default=0)


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
                f"args={json.dumps(args, ensure_ascii=False)[:2500]}",
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


def _dedupe_str_list(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        v = str(raw or "").strip()
        if not v or v in seen:
            continue
        out.append(v)
        seen.add(v)
    return out


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
    return _dedupe_str_list(out)


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


def _ensure_profile(*, registry: ToolRegistry, ctx: ToolContext, req: BookingAssistantRunRequest) -> Dict[str, Any]:
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
    thread_facts_text = ""
    if include_thread:
        thread = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="gmail_read_mail_thread",
            args={"mail_id": mail_id, "mailbox": mailbox, "max_messages": 20, "max_chars": 8000},
        )
        t = str(thread.get("text") or "").strip()
        if t:
            parts.append("THREAD:\n" + t)
        msgs = thread.get("messages") if isinstance(thread.get("messages"), list) else []
        fact_parts: List[str] = []
        # Newest-first helps downstream extraction prefer the latest user-provided facts.
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

    if len("\n\n".join(parts)) < 150:
        fallback = _safe_tool_call(registry=registry, ctx=ctx, tool="langsearch", args={"query": query, "count": 5})
        if not fallback.get("_error"):
            txt = str(fallback.get("text") or "").strip()
            if txt:
                parts.append("LANGSEARCH:\n" + txt)
            sources.extend(_extract_sources(fallback))

    merged = "\n\n".join(parts).strip()
    if len(merged) > max_context_chars:
        merged = merged[:max_context_chars].rstrip() + "…"
    return {"context_text": merged, "sources": _dedupe_str_list(sources)}


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


def _build_review(
    *,
    mail_id: str,
    mailbox: str,
    from_email: str,
    subject: str,
    mail_text: str,
    draft_body: str,
    booking_decision: str,
    score: Dict[str, Any],
    sources: List[str],
    reason: str,
    action_templates: Dict[str, str] | None = None,
    ticket_id: str = "",
) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "id": uuid4().hex,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "mail_id": mail_id,
        "mailbox": mailbox,
        "from_email": from_email,
        "subject": subject,
        "mail_text": mail_text,
        "draft_body": draft_body,
        "booking_decision": booking_decision,
        "score_total": float(score.get("score_total") or 0.0),
        "score": dict(score),
        "sources": list(sources),
        "ticket_id": ticket_id,
        "reason": reason,
        "action_templates": dict(action_templates or {}),
        "sent": False,
        "send_result": {},
    }


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
        args={"text": text, "instruction": instruction, "max_chars": 2500},
    )
    return str(out.get("text") or "").strip()


def _weekday_name_de(date_iso: str) -> str:
    raw = str(date_iso or "").strip()
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d")
        except Exception:
            return ""
    names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    return names[dt.weekday()]


def _next_weekend_dates(date_iso: str, *, count: int = 3) -> List[str]:
    base: datetime | None = None
    raw = str(date_iso or "").strip()
    if raw:
        try:
            base = datetime.fromisoformat(raw)
        except Exception:
            try:
                base = datetime.strptime(raw, "%Y-%m-%d")
            except Exception:
                base = None
    if base is None:
        base = datetime.now()

    out: List[str] = []
    cursor = base + timedelta(days=1)
    max_days = 90
    checked = 0
    while len(out) < max(1, count) and checked < max_days:
        if cursor.weekday() >= 5:
            out.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=1)
        checked += 1
    return out


def _llm_classify_counteroffer_acceptance(*, latest_body_text: str, thread_text: str) -> Dict[str, Any]:
    client = IonosLLM()
    if not client.enabled():
        return {
            "accepted": False,
            "confidence": 0.0,
            "reason": "llm_unavailable",
            "model": "",
        }

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_counteroffer_acceptance",
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
        "Entscheide, ob die LETZTE Kundenantwort einen zuvor gemachten Gegen-Vorschlag "
        "(Counteroffer) eindeutig annimmt.\n"
        "- true nur bei klarer Zustimmung zur vorgeschlagenen Anpassung.\n"
        "- false bei Unklarheit, Rückfragen, neuem Gegenvorschlag oder Themenwechsel.\n\n"
        f"Letzte Kundenantwort:\n{latest_body_text.strip()}\n\n"
        f"Thread-Kontext:\n{thread_text[:4000]}"
    )
    try:
        completion = client.chat_completions(
            messages=[
                {"role": "system", "content": "Du bist ein präziser Klassifikator. Antworte strikt im JSON-Schema."},
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
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence") or 0.0))),
            "reason": str(parsed.get("reason") or "").strip(),
            "model": client.cfg.model,
        }
    except Exception as exc:
        return {
            "accepted": False,
            "confidence": 0.0,
            "reason": f"llm_error:{exc}",
            "model": client.cfg.model if client.enabled() else "",
        }


def _extract_counteroffer_max_duration(thread_text: str) -> float | None:
    raw = str(thread_text or "")
    if not raw:
        return None
    # Example: "- Einsatzdauer: max. 8.0 Stunden statt 12.0 Stunden"
    m = re.search(r"max\.?\s*(\d+(?:[.,]\d+)?)\s*stunden\s*statt", raw, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except Exception:
        return None


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
            "name": "booking_offer_confirmation_gate",
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
        "Prüfe im Booking-Mailverlauf zwei Punkte:\n"
        "1) Wurde zuvor im Thread ein konkretes Angebot kommuniziert?\n"
        "2) Nimmt die LETZTE Kundenantwort dieses Angebot verbindlich an?\n\n"
        "Wichtig:\n"
        "- Beurteile die Annahme nur anhand der letzten Kundenantwort, nutze den Rest nur als Kontext.\n"
        "- Bei Unklarheit oder nur teilweiser Zustimmung -> offer_accepted_by_latest_reply=false.\n\n"
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


def _apply_counteroffer_acceptance(
    *,
    facts: Dict[str, Any],
    latest_body_text: str,
    thread_text: str,
) -> tuple[Dict[str, Any], bool, str]:
    out = dict(facts or {})
    cls = _llm_classify_counteroffer_acceptance(latest_body_text=latest_body_text, thread_text=thread_text)
    accepted = bool(cls.get("accepted"))
    confidence = float(cls.get("confidence") or 0.0)
    if not accepted or confidence < 0.7:
        return out, False, ""
    max_duration = _extract_counteroffer_max_duration(thread_text)
    if max_duration is None or max_duration <= 0:
        return out, False, ""

    current_duration = 0.0
    try:
        current_duration = float(out.get("duration_hours") or 0.0)
    except Exception:
        current_duration = 0.0

    changed = False
    if current_duration <= 0 or current_duration > max_duration:
        out["duration_hours"] = max_duration
        changed = True
    out["counteroffer_accepted"] = True
    note = (
        f"counteroffer_acceptance_applied:max_duration={max_duration:.1f};"
        f"confidence={confidence:.2f};reason={str(cls.get('reason') or '').strip()}"
    )
    return out, changed, note


def _compose_decision_mail(
    *,
    decision: str,
    reasons: List[str],
    quote_text: str,
    hold_link: str = "",
    event_date: str = "",
    repeated_same_rule: bool = False,
) -> str:
    rs = "\n".join(f"- {r}" for r in reasons if r)
    if decision == "auto_decline":
        weekend_rule = any("wochenende" in str(r).lower() for r in reasons)
        if weekend_rule:
            weekday = _weekday_name_de(event_date)
            suggestions = _next_weekend_dates(event_date, count=3)
            intro = (
                "Vielen Dank für Ihre Rückmeldung. Wie bereits erwähnt kann ich den Termin unter diesen Bedingungen nicht zusagen."
                if repeated_same_rule
                else "Vielen Dank für Ihre Anfrage. Leider kann ich den Termin auf Basis der aktuellen Rahmenbedingungen nicht zusagen."
            )
            lines = [intro, "", "Gründe:", rs]
            if event_date:
                weekday_info = f"{event_date} ist ein {weekday}." if weekday else f"{event_date} liegt nicht am Wochenende."
                lines.extend(["", f"Hinweis: {weekday_info}"])
            if suggestions:
                lines.extend(["", "Mögliche Wochenend-Alternativen:"])
                lines.extend(f"- {d}" for d in suggestions)
            lines.extend(["", "Wenn Sie möchten, wählen Sie einen der Vorschläge oder nennen Sie einen anderen Wochenendtermin."])
            return "\n".join(lines).strip()

        return (
            "Vielen Dank für Ihre Anfrage. Leider kann ich den Termin auf Basis der aktuellen Rahmenbedingungen nicht zusagen.\n\n"
            f"Gründe:\n{rs}\n\n"
            "Wenn Sie möchten, können wir eine alternative Anfrage mit angepassten Rahmenbedingungen prüfen."
        ).strip()
    if decision == "auto_accept":
        link_line = f"\nKalender-Link: {hold_link}" if hold_link else ""
        return (
            "Vielen Dank für Ihre Anfrage. Der Termin wurde im Kalender reserviert.\n\n"
            f"Preisübersicht:\n{quote_text}{link_line}\n\n"
            "Vielen Dank für die Bestätigung. Bei Rückfragen melden Sie sich jederzeit."
        ).strip()
    return (
        "Vielen Dank für Ihre Anfrage. Für die verbindliche Bearbeitung benötige ich noch folgende Rückmeldung:\n\n"
        f"{rs}"
    ).strip()


def _compose_quote_confirmation_mail(*, quote_text: str, reasons: List[str] | None = None) -> str:
    rs = [str(x).strip() for x in (reasons or []) if str(x).strip()]
    reason_block = ""
    if rs:
        reason_block = "Hinweise:\n" + "\n".join(f"- {r}" for r in rs) + "\n\n"
    offer_block = str(quote_text or "").strip()
    if offer_block:
        offer_text = "Hier ist das konkrete Angebot:\n" + offer_block + "\n\n"
        confirm_text = (
            "Bitte bestätigen Sie das Angebot kurz schriftlich. "
            "Erst danach reserviere ich den Termin verbindlich im Kalender."
        )
    else:
        offer_text = "Das konkrete Angebot wird gerade vorbereitet.\n\n"
        confirm_text = (
            "Sobald das Angebot vorliegt, erhalten Sie es zur Bestätigung."
        )
    return (
        "Vielen Dank, alle wichtigen Veranstaltungsdaten liegen vor.\n\n"
        f"{reason_block}"
        f"{offer_text}"
        f"{confirm_text}"
    ).strip()


def _build_review_action_templates(
    *,
    booking_decision: str,
    draft_body: str,
    facts: Dict[str, Any],
    booking_rules: Dict[str, Any],
    quote_text: str,
) -> Dict[str, str]:
    event_date = str(facts.get("event_date") or "").strip()
    start_time = str(facts.get("start_time") or "").strip()
    location = str(facts.get("location") or "").strip()
    occasion = str(facts.get("occasion") or "Event").strip()
    client_name = str(facts.get("client_name") or "").strip()
    price_confirmed = bool(facts.get("price_confirmed"))
    max_duration = float(booking_rules.get("max_duration_hours") or 8.0)
    facts_duration = float(facts.get("duration_hours") or 0.0)
    detail_parts: List[str] = []
    if event_date:
        detail_parts.append(event_date)
    if start_time:
        detail_parts.append(start_time)
    if location:
        detail_parts.append(location)
    base_event_line = ", ".join(detail_parts).strip(", ")

    salutation = "Guten Tag"
    if client_name:
        salutation = f"Guten Tag {client_name}"

    approve_lines = [f"{salutation},", "", "vielen Dank für Ihre Anfrage."]
    if base_event_line:
        approve_lines.append(f"Ich kann Ihnen den Termin ({base_event_line}) grundsätzlich zusagen.")
    else:
        approve_lines.append("Ich kann Ihnen den angefragten Termin grundsätzlich zusagen.")
    if facts_duration > 0:
        approve_lines.append(f"Der Einsatz ist mit {facts_duration:.1f} Stunden eingeplant.")
    if quote_text.strip():
        approve_lines.extend(["", "Vereinbarte Preisübersicht:", quote_text.strip()])
    if quote_text.strip() and price_confirmed:
        approve_lines.extend(
            [
                "",
                "Die Buchung ist damit verbindlich bestätigt. Ich freue mich auf die Veranstaltung.",
            ]
        )
    elif quote_text.strip() and not price_confirmed:
        approve_lines.extend(
            [
                "",
                "Bitte bestätigen Sie das Angebot kurz schriftlich. "
                "Direkt danach bestätige ich den Termin verbindlich.",
            ]
        )
    else:
        approve_lines.extend(
            [
                "",
                "Als nächsten Schritt erhalten Sie das konkrete Angebot. "
                "Nach Ihrer Angebotsbestätigung bestätige ich den Termin verbindlich.",
            ]
        )
    approve = "\n".join(approve_lines).strip()

    reject = (
        "Vielen Dank für Ihre Anfrage.\n\n"
        "Nach Prüfung kann ich den Auftrag unter den aktuell angefragten Rahmenbedingungen "
        "leider nicht verbindlich bestätigen.\n\n"
        "Wenn Sie möchten, sende ich Ihnen gern einen Alternativvorschlag."
    ).strip()

    counteroffer_lines = [
        "Vielen Dank für Ihre Anfrage.",
        "",
        "Ich kann Ihnen folgenden angepassten Vorschlag anbieten:",
    ]
    if base_event_line:
        counteroffer_lines.append(f"- Terminrahmen: {base_event_line}")
    if facts_duration > max_duration:
        counteroffer_lines.append(f"- Einsatzdauer: max. {max_duration:.1f} Stunden statt {facts_duration:.1f} Stunden")
    if quote_text.strip():
        counteroffer_lines.extend(["", "Preisübersicht (bei angepasster Anfrage):", quote_text.strip()])
    counteroffer_lines.extend(
        [
            "",
            "Wenn der Vorschlag für Sie passt, bestätigen Sie ihn bitte kurz schriftlich.",
        ]
    )
    counteroffer = "\n".join(counteroffer_lines).strip()

    if booking_decision == "auto_decline":
        approve = "\n".join(
            [
                f"{salutation},",
                "",
                "vielen Dank für Ihre Anfrage.",
                "Ich bestätige den Auftrag ausnahmsweise trotz Abweichung von den Standardregeln.",
                "",
                "Bitte betrachten Sie diese Zusage als individuelle Freigabe.",
            ]
        ).strip()
    elif booking_decision == "need_clarification":
        reject = (
            "Vielen Dank für Ihre Anfrage.\n\n"
            "Ohne die fehlenden Angaben kann ich den Auftrag leider nicht bestätigen."
        ).strip()

    return {
        "approve": approve,
        "reject": reject,
        "counteroffer": counteroffer,
        "default": draft_body.strip(),
    }


def _combine_start_end_iso(facts: Dict[str, Any]) -> tuple[str, str]:
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


def _early_booking_rule_decision(*, facts: Dict[str, Any], booking_rules: Dict[str, Any]) -> tuple[str, List[str]]:
    reasons: List[str] = []
    weekend_only = bool(booking_rules.get("weekend_only", True))
    date_iso = str(facts.get("event_date") or "").strip()
    if weekend_only and date_iso:
        is_weekend = _is_weekend_date(date_iso)
        if is_weekend is False:
            reasons.append("Buchungen sind nur am Wochenende möglich.")
            return "auto_decline", reasons

    try:
        duration = float(facts.get("duration_hours") or 0.0)
    except Exception:
        duration = 0.0
    try:
        max_duration = float(booking_rules.get("max_duration_hours") or 8.0)
    except Exception:
        max_duration = 8.0
    if duration > 0 and duration > max_duration:
        reasons.append(f"Anfrage überschreitet Maximaldauer ({duration:.1f}h > {max_duration:.1f}h).")
        return "human_review", reasons

    return "", reasons


def _find_existing_hold(
    *,
    state: Dict[str, Any],
    thread_id: str,
    start_iso: str,
    end_iso: str,
) -> Dict[str, Any] | None:
    if not thread_id:
        return None
    holds = state.get("holds") if isinstance(state.get("holds"), list) else []
    for item in reversed(holds):
        if not isinstance(item, dict):
            continue
        if str(item.get("thread_id") or "").strip() != thread_id:
            continue
        if str(item.get("start_iso") or "").strip() == start_iso and str(item.get("end_iso") or "").strip() == end_iso:
            return item
    return None


def _remember_hold(
    *,
    state: Dict[str, Any],
    thread_id: str,
    mail_id: str,
    start_iso: str,
    end_iso: str,
    event_id: str,
    html_link: str,
    max_items: int = 500,
) -> None:
    if not thread_id:
        return
    holds = state.get("holds")
    if not isinstance(holds, list):
        holds = []
    for item in holds:
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("thread_id") or "").strip() == thread_id
            and str(item.get("start_iso") or "").strip() == start_iso
            and str(item.get("end_iso") or "").strip() == end_iso
        ):
            item["event_id"] = event_id
            item["html_link"] = html_link
            item["mail_id"] = mail_id
            item["updated_at"] = _now_iso()
            state["holds"] = holds
            return
    holds.append(
        {
            "thread_id": thread_id,
            "mail_id": mail_id,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "event_id": event_id,
            "html_link": html_link,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
    )
    if len(holds) > max_items:
        holds = holds[-max_items:]
    state["holds"] = holds


def _get_thread_booking_context(*, state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    if not thread_id:
        return {}
    items = state.get("thread_booking_contexts")
    if not isinstance(items, list):
        return {}
    for entry in reversed(items):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("thread_id") or "").strip() != thread_id:
            continue
        facts = entry.get("facts")
        if isinstance(facts, dict):
            return dict(facts)
    return {}


def _remember_thread_booking_context(
    *,
    state: Dict[str, Any],
    thread_id: str,
    facts: Dict[str, Any],
    max_items: int = 500,
) -> None:
    if not thread_id:
        return
    keys = ("event_date", "start_time", "duration_hours", "location", "occasion", "client_name", "price_confirmed")
    compact: Dict[str, Any] = {}
    for key in keys:
        value = facts.get(key)
        if value in (None, "", 0, 0.0, False):
            continue
        compact[key] = value
    if not compact:
        return

    items = state.get("thread_booking_contexts")
    if not isinstance(items, list):
        items = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("thread_id") or "").strip() != thread_id:
            continue
        existing = entry.get("facts") if isinstance(entry.get("facts"), dict) else {}
        merged = dict(existing)
        merged.update(compact)
        entry["facts"] = merged
        entry["updated_at"] = _now_iso()
        state["thread_booking_contexts"] = items
        return

    items.append(
        {
            "thread_id": thread_id,
            "facts": compact,
            "updated_at": _now_iso(),
        }
    )
    if len(items) > max_items:
        items = items[-max_items:]
    state["thread_booking_contexts"] = items


def _precheck_calendar(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    facts: Dict[str, Any],
) -> Dict[str, Any]:
    date_iso = str(facts.get("event_date") or "").strip()
    if not date_iso:
        return {"checked": False, "reason": "missing_event_date"}

    start_raw = str(facts.get("start_time") or "").strip()
    duration_raw = float(facts.get("duration_hours") or 0.0)
    assumed_start = not bool(start_raw)
    assumed_duration = duration_raw <= 0.0

    start_iso, end_iso = _combine_start_end_iso(facts)
    if not start_iso or not end_iso:
        return {"checked": False, "reason": "invalid_window"}

    avail = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="calendar_check_availability",
        args={"start_iso": start_iso, "end_iso": end_iso, "calendar_id": "primary"},
    )
    if avail.get("_error"):
        return {"checked": False, "reason": str(avail.get("_error") or "calendar_precheck_failed")}

    is_available = bool(avail.get("is_available"))
    busy_count = len(avail.get("busy") or []) if isinstance(avail.get("busy"), list) else 0

    if is_available:
        note = "Vorab-Check Kalender: Der Termin ist im aktuell geprüften Zeitfenster frei."
    elif assumed_start or assumed_duration:
        note = (
            "Vorab-Check Kalender: Das vorläufig geprüfte Zeitfenster ist belegt. "
            "Für einen exakten Check werden Startzeit und Dauer benötigt."
        )
    else:
        note = "Vorab-Check Kalender: Das gewünschte Zeitfenster ist bereits belegt."

    return {
        "checked": True,
        "is_available": is_available,
        "busy_count": busy_count,
        "start_iso": str(avail.get("start_iso") or start_iso),
        "end_iso": str(avail.get("end_iso") or end_iso),
        "assumed_start": assumed_start,
        "assumed_duration": assumed_duration,
        "note": note,
        "text": str(avail.get("text") or "").strip(),
    }


def run_once(*, user_id: str, settings: Settings, api_key: str, req: BookingAssistantRunRequest) -> BookingAssistantRunResponse:
    trace_token = _TRACE_ENABLED.set(bool(req.trace_steps))
    step_token = _TRACE_STEP.set(0)
    run_id = f"run_{uuid4().hex[:12]}"
    started_at = _now_iso()

    _trace_log(
        "BOOKING ASSISTANT RUN",
        [
            f"user_id={user_id}",
            f"mailbox={req.mailbox}",
            f"limit={req.limit}",
            f"assistant_profile_name={req.assistant_profile_name}",
        ],
    )

    registry = build_registry(settings=settings, user_id=user_id)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_run_once")
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
        return BookingAssistantRunResponse(
            ok=True,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            lock_blocked=True,
            lock_reason=str(lock.get("reason") or "run_already_active"),
            processed_count=0,
            sent_count=0,
            review_count=0,
            skipped_count=0,
            items=[],
        )

    profile = _ensure_profile(registry=registry, ctx=ctx, req=req)
    profile_instructions = (
        [str(x).strip() for x in (profile.get("instructions") or []) if str(x).strip()]
        if isinstance(profile.get("instructions"), list)
        else []
    )
    rules = profile.get("rules") if isinstance(profile.get("rules"), dict) else {}
    booking_rules = rules.get("booking") if isinstance(rules.get("booking"), dict) else {}
    pricing_rules = rules.get("pricing") if isinstance(rules.get("pricing"), dict) else {}
    required_fields = rules.get("required_fields") if isinstance(rules.get("required_fields"), list) else []
    detail_required_fields = [str(x).strip() for x in required_fields if str(x).strip() and str(x).strip() != "price_confirmed"]
    if not detail_required_fields:
        detail_required_fields = ["event_date", "start_time", "duration_hours", "location", "occasion", "client_name"]

    run_items: List[BookingAssistantRunItem] = []
    sent_count = 0
    review_count = 0
    skipped_count = 0
    processed_count = 0

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
                BookingAssistantRunItem(
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
                    booking_decision="",
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
            mail_thread_id = str(mail_payload.get("thread_id") or "").strip()
        except Exception as exc:
            run_items.append(
                BookingAssistantRunItem(
                    mail_id=mail_id,
                    subject=subject,
                    from_email=from_email,
                    decision="failed",
                    reason=f"read_mail_failed:{exc}",
                )
            )
            append_activity(
                state,
                _activity_record(
                    mail_id=mail_id,
                    thread_id="",
                    decision="failed",
                    booking_decision="",
                    reason=f"read_mail_failed:{exc}",
                ),
            )
            continue

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
                BookingAssistantRunItem(
                    mail_id=mail_id,
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
                    thread_id=mail_thread_id,
                    decision="skipped",
                    booking_decision="",
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
        sources: List[str] = []
        reasons: List[str] = []
        action_templates: Dict[str, str] = {}
        quote_text = ""
        facts_for_review: Dict[str, Any] = {}
        missing: List[str] = []
        calendar_event_id = ""
        instruction_allowed = True
        instruction_reason = ""
        instruction_violations: List[str] = []
        instruction_risk = "low"
        repeated_weekend_rule = False
        priced_offer_in_thread = False

        if intent == "info":
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
        else:
            extraction_text = "\n".join(
                x
                for x in [
                    str(mail_payload.get("subject") or "").strip(),
                    str(mail_payload.get("body_text") or "").strip(),
                ]
                if x
            ).strip() or str(mail_payload.get("text") or "")
            facts_out = _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="booking_extract_facts",
                args={"text": extraction_text, "required_fields": required_fields},
            )
            facts = facts_out.get("facts") if isinstance(facts_out.get("facts"), dict) else {}
            facts_for_review = dict(facts)
            cached_thread_facts = _get_thread_booking_context(state=state, thread_id=mail_thread_id)
            if cached_thread_facts:
                for key in ("event_date", "start_time", "duration_hours", "location", "occasion", "client_name"):
                    if facts.get(key) in (None, "", 0, 0.0) and cached_thread_facts.get(key) not in (None, "", 0, 0.0):
                        facts[key] = cached_thread_facts.get(key)
                facts["price_confirmed"] = bool(facts.get("price_confirmed")) or bool(cached_thread_facts.get("price_confirmed"))
            thread_facts_text = str(mail_payload.get("thread_facts_text") or "").strip()
            if thread_facts_text:
                thread_out = _safe_tool_call(
                    registry=registry,
                    ctx=ctx,
                    tool="booking_extract_facts",
                    args={"text": thread_facts_text, "required_fields": required_fields},
                )
                thread_facts = thread_out.get("facts") if isinstance(thread_out.get("facts"), dict) else {}
                for key in ("event_date", "start_time", "duration_hours", "location", "occasion", "client_name"):
                    if facts.get(key) in (None, "", 0, 0.0) and thread_facts.get(key) not in (None, "", 0, 0.0):
                        facts[key] = thread_facts.get(key)
                facts["price_confirmed"] = bool(facts.get("price_confirmed")) or bool(thread_facts.get("price_confirmed"))
                repeated_weekend_rule = "buchungen sind nur am wochenende möglich" in thread_facts_text.lower()
            else:
                repeated_weekend_rule = "buchungen sind nur am wochenende möglich" in str(mail_payload.get("text") or "").lower()
            offer_gate = _llm_classify_offer_confirmation(
                latest_body_text=str(mail_payload.get("body_text") or ""),
                thread_text=str(mail_payload.get("text") or ""),
            )
            offer_gate_conf = float(offer_gate.get("confidence") or 0.0)
            priced_offer_in_thread = bool(offer_gate.get("offer_present_in_thread")) and offer_gate_conf >= 0.6
            offer_accepted_latest = bool(offer_gate.get("offer_accepted_by_latest_reply")) and offer_gate_conf >= 0.7
            if offer_accepted_latest:
                facts["price_confirmed"] = True
                _trace_log(
                    "OFFER ACCEPTANCE GATE",
                    [
                        f"mail_id={mail_id}",
                        f"offer_present={priced_offer_in_thread}",
                        f"accepted={offer_accepted_latest}",
                        f"confidence={offer_gate_conf:.2f}",
                        f"reason={offer_gate.get('reason')}",
                    ],
                )
            adjusted_facts, counteroffer_changed, counteroffer_note = _apply_counteroffer_acceptance(
                facts=facts,
                latest_body_text=str(mail_payload.get("body_text") or ""),
                thread_text=str(mail_payload.get("text") or ""),
            )
            if counteroffer_changed:
                facts = adjusted_facts
                _trace_log(
                    "COUNTEROFFER ACCEPTED",
                    [
                        f"mail_id={mail_id}",
                        counteroffer_note or "counteroffer_acceptance_applied",
                        f"duration_hours={facts.get('duration_hours')}",
                    ],
                )
            if bool(facts.get("price_confirmed")) and not (priced_offer_in_thread and offer_accepted_latest):
                facts["price_confirmed"] = False
                _trace_log(
                    "PRICE CONFIRMATION GATE",
                    [
                        f"mail_id={mail_id}",
                        "price_confirmed_ignored_without_llm_offer_acceptance=true",
                        f"offer_present={priced_offer_in_thread}",
                        f"accepted={offer_accepted_latest}",
                        f"confidence={offer_gate_conf:.2f}",
                        f"reason={offer_gate.get('reason')}",
                    ],
                )
            facts_for_review = dict(facts)
            _remember_thread_booking_context(state=state, thread_id=mail_thread_id, facts=facts)
            precheck = _precheck_calendar(registry=registry, ctx=ctx, facts=facts)
            if precheck.get("checked"):
                _trace_log(
                    "CALENDAR PRECHECK",
                    [
                        f"mail_id={mail_id}",
                        f"is_available={bool(precheck.get('is_available'))}",
                        f"assumed_start={bool(precheck.get('assumed_start'))}",
                        f"assumed_duration={bool(precheck.get('assumed_duration'))}",
                        f"window={precheck.get('start_iso')} -> {precheck.get('end_iso')}",
                    ],
                )
                facts["calendar_precheck"] = str(precheck.get("note") or "").strip()

            comp = _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="booking_validate_completeness",
                args={"facts": facts, "required_fields": detail_required_fields},
            )
            missing = comp.get("missing_fields") if isinstance(comp.get("missing_fields"), list) else []
            early_decision, early_reasons = _early_booking_rule_decision(facts=facts, booking_rules=booking_rules)

            if early_decision in {"auto_decline", "human_review"}:
                booking_decision = early_decision
                reasons = [str(x).strip() for x in early_reasons if str(x).strip()]
                if booking_decision == "auto_decline":
                    draft = _compose_decision_mail(
                        decision=booking_decision,
                        reasons=reasons,
                        quote_text="",
                        event_date=str(facts.get("event_date") or "").strip(),
                        repeated_same_rule=repeated_weekend_rule,
                    )
                else:
                    draft = (
                        "Vielen Dank für Ihre Anfrage. "
                        "Ich habe Ihr Anliegen intern zur persönlichen Prüfung weitergegeben "
                        "und melde mich zeitnah mit einer verbindlichen Rückmeldung."
                    ).strip()
            elif not bool(comp.get("complete")):
                clar = _safe_tool_call(
                    registry=registry,
                    ctx=ctx,
                    tool="mail_compose_clarification",
                    args={"missing_fields": missing, "known_facts": facts},
                )
                draft = str(clar.get("body") or "").strip()
                if precheck.get("checked"):
                    note = str(precheck.get("note") or "").strip()
                    if note:
                        draft = f"{draft}\n\n{note}".strip()
                    if not bool(precheck.get("is_available")) and not (
                        bool(precheck.get("assumed_start")) or bool(precheck.get("assumed_duration"))
                    ):
                        draft = (
                            f"{draft}\n\n"
                            "Bitte nennen Sie einen alternativen Termin oder eine alternative Startzeit."
                        ).strip()
                booking_decision = "need_clarification"
            else:
                if precheck.get("checked") and not bool(precheck.get("is_available")) and not (
                    bool(precheck.get("assumed_start")) or bool(precheck.get("assumed_duration"))
                ):
                    booking_decision = "need_clarification"
                    draft = (
                        "Vielen Dank für Ihre Anfrage. "
                        "Das gewünschte Zeitfenster ist im Kalender bereits belegt.\n\n"
                        "Bitte nennen Sie einen alternativen Termin oder eine alternative Startzeit."
                    ).strip()
                    missing = []
                else:
                    origin = str(booking_rules.get("base_address") or "Pforzheim, Deutschland")
                    destination = str(facts.get("location") or "").strip()
                    dist = _safe_tool_call(
                        registry=registry,
                        ctx=ctx,
                        tool="distance_check",
                        args={
                            "origin": origin,
                            "destination": destination,
                            "max_distance_km": float(booking_rules.get("max_distance_km") or 0.0),
                        },
                    )
                    distance_km = float(dist.get("distance_km") or 0.0)

                    quote = _safe_tool_call(
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
                    quote_text = str(quote.get("text") or "").strip()

                    # Phase 0: Harte Regeln vor jeder Kundennachricht prüfen.
                    # So landen kritische Fälle direkt im Human-Review, bevor ein Angebot rausgeht.
                    pre_dec = _safe_tool_call(
                        registry=registry,
                        ctx=ctx,
                        tool="booking_decision_engine",
                        args={
                            "facts": facts,
                            "profile_rules": booking_rules,
                            "completeness": comp,
                            "distance": dist,
                            "quote": quote,
                            "require_price_confirmation": False,
                        },
                    )
                    pre_decision = str(pre_dec.get("decision") or "human_review").strip().lower()
                    reasons = [str(x).strip() for x in (pre_dec.get("reasons") or []) if str(x).strip()]

                    if pre_decision in {"human_review", "auto_decline"}:
                        booking_decision = pre_decision
                        if booking_decision == "auto_decline":
                            draft = _compose_decision_mail(
                                decision=booking_decision,
                                reasons=reasons,
                                quote_text=str(quote.get("text") or ""),
                                event_date=str(facts.get("event_date") or "").strip(),
                                repeated_same_rule=repeated_weekend_rule,
                            )
                        else:
                            draft = (
                                "Vielen Dank für Ihre Anfrage. "
                                "Ich habe Ihr Anliegen intern zur persönlichen Prüfung weitergegeben "
                                "und melde mich zeitnah mit einer verbindlichen Rückmeldung."
                            ).strip()
                    # Phase 1: Sobald Event-Details vollständig sind, Preis senden.
                    # Phase 2: Erst nach Angebotsbestätigung reservieren/akzeptieren.
                    elif not bool(facts.get("price_confirmed")):
                        booking_decision = "need_clarification"
                        reason = "Angebotsbestätigung ausstehend."
                        if not priced_offer_in_thread:
                            reason = "Angebot wurde noch nicht bestätigt."
                        draft = _compose_quote_confirmation_mail(
                            quote_text=quote_text,
                            reasons=[reason],
                        )
                    else:
                        dec = _safe_tool_call(
                            registry=registry,
                            ctx=ctx,
                            tool="booking_decision_engine",
                            args={
                                "facts": facts,
                                "profile_rules": booking_rules,
                                "completeness": comp,
                                "distance": dist,
                                "quote": quote,
                                "require_price_confirmation": True,
                            },
                        )
                        booking_decision = str(dec.get("decision") or "human_review").strip().lower()
                        reasons = [str(x).strip() for x in (dec.get("reasons") or []) if str(x).strip()]

                    if booking_decision == "auto_accept":
                        start_iso, end_iso = _combine_start_end_iso(facts)
                        hold_link = ""
                        if start_iso and end_iso:
                            existing_hold = _find_existing_hold(
                                state=state,
                                thread_id=mail_thread_id,
                                start_iso=start_iso,
                                end_iso=end_iso,
                            )
                            if existing_hold is not None:
                                existing_event_id = str(existing_hold.get("event_id") or "").strip()
                                if existing_event_id:
                                    updated = _safe_tool_call(
                                        registry=registry,
                                        ctx=ctx,
                                        tool="calendar_update_event",
                                        args={
                                            "event_id": existing_event_id,
                                            "calendar_id": "primary",
                                            "summary": f"DJ Booking: {str(facts.get('occasion') or 'Event')}",
                                            "description": "Verbindlich bestätigte Buchung aus Booking Assistant",
                                            "location": str(facts.get("location") or ""),
                                        },
                                    )
                                    calendar_event_id = str(updated.get("event_id") or existing_event_id).strip()
                                    hold_link = str(updated.get("html_link") or "").strip() or str(existing_hold.get("html_link") or "").strip()
                                    _remember_hold(
                                        state=state,
                                        thread_id=mail_thread_id,
                                        mail_id=mail_id,
                                        start_iso=start_iso,
                                        end_iso=end_iso,
                                        event_id=calendar_event_id,
                                        html_link=hold_link,
                                    )
                                else:
                                    hold_link = str(existing_hold.get("html_link") or "").strip()
                                if not calendar_event_id:
                                    calendar_event_id = existing_event_id
                            else:
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
                                            "description": "Verbindlich bestätigte Buchung aus Booking Assistant",
                                            "attendees": [from_email] if from_email else [],
                                            "calendar_id": "primary",
                                        },
                                    )
                                    calendar_event_id = str(created.get("event_id") or "").strip()
                                    hold_link = str(created.get("html_link") or "").strip()
                                    _remember_hold(
                                        state=state,
                                        thread_id=mail_thread_id,
                                        mail_id=mail_id,
                                        start_iso=start_iso,
                                        end_iso=end_iso,
                                        event_id=calendar_event_id,
                                        html_link=hold_link,
                                    )
                                else:
                                    booking_decision = "need_clarification"
                                    reasons.append("Gewünschter Zeitraum ist bereits belegt.")
                        draft = _compose_decision_mail(
                            decision=booking_decision,
                            reasons=reasons,
                            quote_text=quote_text,
                            hold_link=hold_link,
                        )
                    elif booking_decision == "auto_decline":
                        draft = _compose_decision_mail(
                            decision=booking_decision,
                            reasons=reasons,
                            quote_text=quote_text,
                            event_date=str(facts.get("event_date") or "").strip(),
                            repeated_same_rule=repeated_weekend_rule,
                        )
                    elif booking_decision == "human_review":
                        # Interne Regelgründe nie automatisch an Kunden senden.
                        if not draft:
                            draft = (
                                "Vielen Dank für Ihre Anfrage. "
                                "Ich habe Ihr Anliegen intern zur persönlichen Prüfung weitergegeben "
                                "und melde mich zeitnah mit einer verbindlichen Rückmeldung."
                            ).strip()
                    elif booking_decision == "need_clarification":
                        if not draft:
                            clar = _safe_tool_call(
                                registry=registry,
                                ctx=ctx,
                                tool="mail_compose_clarification",
                                args={"missing_fields": missing or ["details"], "known_facts": facts},
                            )
                            draft = str(clar.get("body") or "").strip()
                    else:
                        booking_decision = "human_review"
                        draft = (
                            "Vielen Dank für Ihre Anfrage. "
                            "Ich habe Ihr Anliegen intern zur persönlichen Prüfung weitergegeben "
                            "und melde mich zeitnah mit einer verbindlichen Rückmeldung."
                        ).strip()

                    action_templates = _build_review_action_templates(
                        booking_decision=booking_decision,
                        draft_body=draft,
                        facts=facts_for_review,
                        booking_rules=booking_rules,
                        quote_text=quote_text,
                    )

        if intent != "info" and booking_decision and not action_templates:
            action_templates = _build_review_action_templates(
                booking_decision=booking_decision,
                draft_body=draft,
                facts=facts_for_review,
                booking_rules=booking_rules,
                quote_text=quote_text,
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
                facts=facts_for_review,
                required_fields=required_fields,
                missing_fields=missing if isinstance(missing, list) else [],
            )
        score_total = float(score.get("score_total") or 0.0)
        verdict = str(score.get("verdict") or "needs_review").strip().lower()

        if draft and profile_instructions:
            instr = _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="booking_instruction_check",
                args={
                    "instructions": profile_instructions,
                    "user_message": str(mail_payload.get("text") or "").strip(),
                    "draft_reply": draft,
                    "booking_decision": booking_decision,
                    "facts": facts_for_review,
                },
            )
            if not instr.get("_error"):
                instruction_allowed = bool(instr.get("allowed"))
                instruction_reason = str(instr.get("reason") or "").strip()
                instruction_risk = str(instr.get("risk_level") or "low").strip().lower()
                instruction_violations = [
                    str(x).strip()
                    for x in (instr.get("violations") or [])
                    if str(x).strip()
                ]
                _trace_log(
                    "INSTRUCTION CHECK",
                    [
                        f"mail_id={mail_id}",
                        f"allowed={instruction_allowed}",
                        f"risk={instruction_risk}",
                        f"reason={instruction_reason}",
                        f"violations={'; '.join(instruction_violations) if instruction_violations else '-'}",
                    ],
                )

        policy = _policy_check(registry=registry, ctx=ctx, text=draft, strict_mode=req.strict_policy)
        policy_allowed = bool(policy.get("allowed"))
        policy_risk = str(policy.get("risk_level") or "").strip().lower()

        force_human = booking_decision == "human_review" or not instruction_allowed
        if booking_decision in {"", "info_reply"}:
            # info-branch decision based on quality gates.
            can_auto_send = bool(
                draft and policy_allowed and score_total >= req.auto_send_threshold and verdict == "send" and policy_risk not in {"high", "critical"}
            )
        elif booking_decision in {"auto_decline", "need_clarification"}:
            can_auto_send = bool(draft and policy_allowed and policy_risk not in {"high", "critical"})
        elif booking_decision == "auto_accept":
            can_auto_send = bool(
                draft and policy_allowed and score_total >= req.auto_send_threshold and verdict == "send" and policy_risk not in {"high", "critical"}
            )
        else:
            can_auto_send = False

        if force_human:
            can_auto_send = False

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
                BookingAssistantRunItem(
                    mail_id=mail_id,
                    subject=subject,
                    from_email=from_email,
                    decision="auto_sent",
                    booking_decision=booking_decision,
                    sent=bool(send_out.get("sent")),
                    score_total=score_total,
                    reason=str(score.get("reason") or "auto_sent"),
                )
            )
            _trace_log("MAIL SENT", [f"mail_id={mail_id}", f"booking_decision={booking_decision or 'info_reply'}"])
            append_activity(
                state,
                _activity_record(
                    mail_id=mail_id,
                    thread_id=mail_thread_id,
                    decision="auto_sent",
                    booking_decision=booking_decision or "info_reply",
                    event_id=calendar_event_id,
                    review_id="",
                    reason=str(score.get("reason") or "auto_sent"),
                ),
            )
            continue

        if force_human:
            reason_parts = ["human_review_required"]
        else:
            reason_parts = [str(score.get("reason") or "needs_human_review")]
        if booking_decision:
            reason_parts.append(f"booking_decision={booking_decision}")
        if booking_decision == "human_review" and reasons:
            reason_parts.append("booking_reasons=" + "; ".join(reasons))
        if not instruction_allowed:
            reason_parts.append("instruction_blocked")
            if instruction_reason:
                reason_parts.append(f"instruction_reason={instruction_reason}")
            if instruction_violations:
                reason_parts.append("instruction_violations=" + "; ".join(instruction_violations))
        if not policy_allowed:
            reason_parts.append("policy_blocked")
        reason_text = " | ".join(x for x in reason_parts if x)

        ticket_id = ""
        if registry.get_tool("customer_support_review_ticket_create") is not None:
            ticket = _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="customer_support_review_ticket_create",
                args={
                    "title": f"Booking review: {subject or mail_id}",
                    "user_message": str(mail_payload.get("text") or "").strip(),
                    "draft_reply": draft,
                    "score": score_total,
                    "reasons": [reason_text],
                    "priority": "high" if force_human else "medium",
                    "metadata": {
                        "mail_id": mail_id,
                        "mailbox": req.mailbox,
                        "from_email": from_email,
                        "intent": intent,
                        "booking_decision": booking_decision,
                    },
                },
            )
            ticket_id = str(ticket.get("ticket_id") or "").strip()

        review = _build_review(
            mail_id=mail_id,
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
            ticket_id=ticket_id,
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
                thread_id=mail_thread_id,
                decision="needs_human",
                booking_decision=booking_decision,
                event_id=calendar_event_id,
                review_id=review_id,
                reason=reason_text,
            ),
        )
        run_items.append(
            BookingAssistantRunItem(
                mail_id=mail_id,
                subject=subject,
                from_email=from_email,
                decision="needs_human",
                booking_decision=booking_decision,
                review_id=review_id,
                sent=False,
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
    result = BookingAssistantRunResponse(
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
    _trace_log(
        "RUN SUMMARY",
        [
            f"processed_count={result.processed_count}",
            f"sent_count={result.sent_count}",
            f"review_count={result.review_count}",
            f"skipped_count={result.skipped_count}",
        ],
    )

    _TRACE_ENABLED.reset(trace_token)
    _TRACE_STEP.reset(step_token)
    return result


def get_reviews(*, user_id: str, settings: Settings, status: str = "pending") -> BookingAssistantReviewsResponse:
    state = load_state(settings, user_id)
    filter_status = str(status or "").strip().lower()
    if filter_status in {"", "all", "*"}:
        filter_status = ""
    reviews = [BookingAssistantReviewItem(**r) for r in list_reviews(state, status=filter_status)]
    reviews.sort(key=lambda x: x.created_at, reverse=True)
    return BookingAssistantReviewsResponse(ok=True, reviews=reviews)


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
    req: BookingAssistantApproveRequest,
) -> BookingAssistantReviewActionResponse:
    state = load_state(settings, user_id)
    review = find_review(state, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if str(review.get("status") or "").strip().lower() != "pending":
        raise HTTPException(status_code=409, detail="Review is not pending")

    has_manual_body = bool(str(req.edited_body or "").strip())
    body = str(req.edited_body or "").strip() or _review_template_body(
        review,
        "approve",
        fallback=str(review.get("draft_body") or ""),
    )
    if (
        str(review.get("booking_decision") or "").strip().lower() == "human_review"
        and body
        and not has_manual_body
        and (
            body.strip() == str(review.get("draft_body") or "").strip()
            or "finalen Abschluss" in body
            or "Preisbestätigung" in body
            or "Angebotsbestätigung" in body
            or "persönlichen Prüfung weitergegeben" in body
            or ("verbindlich bestätigt" in body.lower() and "preisübersicht" not in body.lower())
        )
    ):
        body = (
            "Guten Tag,\n\n"
            "vielen Dank für Ihre Anfrage. Ich kann Ihnen den angefragten Termin grundsätzlich zusagen.\n\n"
            "Als nächsten Schritt erhalten Sie das konkrete Angebot. "
            "Nach Ihrer Angebotsbestätigung bestätige ich den Termin verbindlich."
        ).strip()
    if not body:
        raise HTTPException(status_code=422, detail="No draft body available for approval")

    registry = build_registry(settings=settings, user_id=user_id)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_review_approve")

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
            thread_id=str(send_result.get("thread_id") or ""),
            decision="operator_approve",
            booking_decision=str(review.get("booking_decision") or ""),
            event_id="",
            review_id=str(review.get("id") or review_id),
            reason="review_approved",
        ),
    )
    save_state(settings, user_id, state)

    return BookingAssistantReviewActionResponse(
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
    req: BookingAssistantRejectRequest,
) -> BookingAssistantReviewActionResponse:
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
        ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_review_reject")
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
    review["selected_action"] = "reject"
    review["draft_body"] = body
    append_activity(
        state,
        _activity_record(
            mail_id=str(review.get("mail_id") or ""),
            thread_id=str(send_result.get("thread_id") or ""),
            decision="operator_reject",
            booking_decision=str(review.get("booking_decision") or ""),
            event_id="",
            review_id=str(review.get("id") or review_id),
            reason=reason or "review_rejected",
        ),
    )
    save_state(settings, user_id, state)

    return BookingAssistantReviewActionResponse(
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
    req: BookingAssistantCounterofferRequest,
) -> BookingAssistantReviewActionResponse:
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
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_review_counteroffer")

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
            thread_id=str(send_result.get("thread_id") or ""),
            decision="operator_counteroffer",
            booking_decision=str(review.get("booking_decision") or ""),
            event_id="",
            review_id=str(review.get("id") or review_id),
            reason="review_counteroffered",
        ),
    )
    save_state(settings, user_id, state)

    return BookingAssistantReviewActionResponse(
        ok=True,
        review_id=str(review.get("id") or review_id),
        status=str(review.get("status") or "counteroffered"),
        sent=bool(review.get("sent")),
        reason=str(review.get("reason") or ""),
    )


def _review_summary_item(review: Dict[str, Any]) -> Dict[str, Any]:
    templates = review.get("action_templates") if isinstance(review.get("action_templates"), dict) else {}
    return {
        "id": str(review.get("id") or "").strip(),
        "mail_id": str(review.get("mail_id") or "").strip(),
        "mailbox": str(review.get("mailbox") or "INBOX").strip(),
        "from_email": str(review.get("from_email") or "").strip(),
        "subject": str(review.get("subject") or "").strip(),
        "status": str(review.get("status") or "").strip(),
        "booking_decision": str(review.get("booking_decision") or "").strip(),
        "reason": str(review.get("reason") or "").strip(),
        "created_at": str(review.get("created_at") or "").strip(),
        "score_total": float(review.get("score_total") or 0.0),
        "ticket_id": str(review.get("ticket_id") or "").strip(),
        "options": {
            "approve": str(templates.get("approve") or "").strip(),
            "reject": str(templates.get("reject") or "").strip(),
            "counteroffer": str(templates.get("counteroffer") or "").strip(),
        },
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
        recommendations.append("Offene Pendings mit /pending/next nacheinander triagieren.")
    if not recommendations:
        recommendations.append("Aktueller Ablauf ist stabil; keine akuten Prozess-Blocker erkannt.")
    return recommendations[:5]


def get_status_meeting(
    *,
    user_id: str,
    settings: Settings,
    since: str = "",
) -> BookingAssistantStatusMeetingResponse:
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

    return BookingAssistantStatusMeetingResponse(
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


def get_pending_next(*, user_id: str, settings: Settings) -> BookingAssistantPendingNextResponse:
    state = load_state(settings, user_id)
    pending = [r for r in list_reviews(state, status="pending") if isinstance(r, dict)]
    pending.sort(key=lambda x: str(x.get("created_at") or ""))
    if not pending:
        return BookingAssistantPendingNextResponse(
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
    return BookingAssistantPendingNextResponse(
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
    req: BookingAssistantPendingApplyRequest,
) -> BookingAssistantReviewActionResponse:
    action = str(req.action or "").strip().lower()
    if action == "approve":
        return approve_review(
            user_id=user_id,
            settings=settings,
            api_key=api_key,
            review_id=review_id,
            req=BookingAssistantApproveRequest(
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
            req=BookingAssistantRejectRequest(
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
            req=BookingAssistantCounterofferRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
            ),
        )
    raise HTTPException(status_code=422, detail=f"Unsupported action: {action}")


def _heuristic_operator_parse(message: str, pending_ids: List[str]) -> Dict[str, Any]:
    text = str(message or "").strip()
    low = text.lower()
    review_match = re.search(r"\b[a-f0-9]{32}\b", low)
    review_id = review_match.group(0) if review_match else ""

    action = "none"
    if "counteroffer" in low or "gegenangebot" in low:
        action = "counteroffer"
    elif "approve" in low or "freig" in low or re.search(r"best[aä]tig", low):
        action = "approve"
    elif "reject" in low or "ablehn" in low:
        action = "reject"

    intent = "status_meeting"
    if action != "none":
        intent = "apply_action"
    elif "pending" in low and ("next" in low or "naechst" in low or re.search(r"n[aä]chst", low)):
        intent = "triage_pending"
    elif "pending" in low:
        intent = "list_pending"
    elif "profil" in low or "profile" in low or "regel" in low or "instruction" in low:
        intent = "profile_update"
    elif "status" in low or "meeting" in low or "zusammenfassung" in low:
        intent = "status_meeting"

    since = ""
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if date_match:
        since = f"{date_match.group(0)}T00:00:00Z"

    if not review_id and len(pending_ids) == 1 and action != "none":
        review_id = pending_ids[0]

    return {
        "intent": intent,
        "action": action,
        "review_id": review_id,
        "since": since,
        "instructions_add": [],
        "rules_patch": {},
        "reply_body": "",
        "reason": "heuristic_fallback",
    }


def _llm_operator_parse(*, message: str, pending_ids: List[str]) -> Dict[str, Any]:
    client = IonosLLM()
    if not client.enabled():
        return _heuristic_operator_parse(message, pending_ids)

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_operator_chat_intent",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "status_meeting",
                            "profile_update",
                            "list_pending",
                            "triage_pending",
                            "apply_action",
                        ],
                    },
                    "action": {"type": "string", "enum": ["approve", "reject", "counteroffer", "none"]},
                    "review_id": {"type": "string"},
                    "since": {"type": "string"},
                    "instructions_add": {"type": "array", "items": {"type": "string"}},
                    "rules_patch": {"type": "object", "additionalProperties": True},
                    "reply_body": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "intent",
                    "action",
                    "review_id",
                    "since",
                    "instructions_add",
                    "rules_patch",
                    "reply_body",
                    "reason",
                ],
            },
            "strict": True,
        },
    }
    pending_hint = ", ".join(pending_ids[:15]) if pending_ids else "-"
    prompt = (
        "Mappe die Operator-Nachricht auf eine der Intent-Klassen:\n"
        "- status_meeting: Statusbericht anfordern\n"
        "- profile_update: Profilregeln/-instruktionen anpassen\n"
        "- list_pending: nur Pending-Liste sehen\n"
        "- triage_pending: naechsten Pending-Fall aufrufen\n"
        "- apply_action: konkrete Aktion auf Pending anwenden\n\n"
        "Bei apply_action falls vorhanden action/review_id extrahieren.\n"
        "Bei profile_update relevante instructions_add und rules_patch extrahieren.\n"
        "since nur als ISO setzen, falls klar erkennbar, sonst leer.\n"
        "reply_body nur setzen, wenn ein expliziter Entwurfstext gewuenscht ist.\n\n"
        f"Bekannte Pending IDs: {pending_hint}\n\n"
        f"Operator-Nachricht:\n{message.strip()}"
    )
    try:
        completion = client.chat_completions(
            messages=[
                {"role": "system", "content": "Du bist ein Intent-Parser. Antworte strikt im JSON-Schema."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            max_tokens=260,
            temperature=0.0,
            top_p=0.1,
        )
        raw = IonosLLM.extract_text(completion)
        parsed = json.loads(raw) if raw else {}
        out = {
            "intent": str(parsed.get("intent") or "").strip(),
            "action": str(parsed.get("action") or "none").strip(),
            "review_id": str(parsed.get("review_id") or "").strip(),
            "since": str(parsed.get("since") or "").strip(),
            "instructions_add": [
                str(x).strip()
                for x in (parsed.get("instructions_add") or [])
                if str(x).strip()
            ],
            "rules_patch": parsed.get("rules_patch") if isinstance(parsed.get("rules_patch"), dict) else {},
            "reply_body": str(parsed.get("reply_body") or "").strip(),
            "reason": str(parsed.get("reason") or "").strip(),
        }
        if out["intent"] not in {"status_meeting", "profile_update", "list_pending", "triage_pending", "apply_action"}:
            return _heuristic_operator_parse(message, pending_ids)
        if out["action"] not in {"approve", "reject", "counteroffer", "none"}:
            out["action"] = "none"
        return out
    except Exception:
        return _heuristic_operator_parse(message, pending_ids)


def operator_chat(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    req: BookingAssistantOperatorChatRequest,
) -> BookingAssistantOperatorChatResponse:
    state = load_state(settings, user_id)
    pending_reviews = [r for r in list_reviews(state, status="pending") if isinstance(r, dict)]
    pending_reviews.sort(key=lambda x: str(x.get("created_at") or ""))
    pending_ids = [str(r.get("id") or "").strip() for r in pending_reviews if str(r.get("id") or "").strip()]
    parsed = _llm_operator_parse(message=req.message, pending_ids=pending_ids)
    intent = str(parsed.get("intent") or "status_meeting").strip()

    if intent == "status_meeting":
        since = str(parsed.get("since") or "").strip()
        status = get_status_meeting(user_id=user_id, settings=settings, since=since)
        return BookingAssistantOperatorChatResponse(
            ok=True,
            intent=intent,
            action_taken="status_meeting",
            data=status.model_dump(),
            text=status.text,
        )

    if intent == "list_pending":
        items = [_review_summary_item(r) for r in pending_reviews[:20]]
        return BookingAssistantOperatorChatResponse(
            ok=True,
            intent=intent,
            action_taken="list_pending",
            data={"pending_count": len(pending_reviews), "pending": items},
            text=f"Offene Pendings: {len(pending_reviews)}",
        )

    if intent == "triage_pending":
        nxt = get_pending_next(user_id=user_id, settings=settings)
        return BookingAssistantOperatorChatResponse(
            ok=True,
            intent=intent,
            action_taken="triage_pending",
            data=nxt.model_dump(),
            text=nxt.text,
        )

    if intent == "apply_action":
        action = str(parsed.get("action") or "none").strip().lower()
        review_id = str(parsed.get("review_id") or "").strip()
        if not review_id and len(pending_ids) == 1:
            review_id = pending_ids[0]
        if not review_id:
            return BookingAssistantOperatorChatResponse(
                ok=True,
                intent=intent,
                action_taken="apply_action_missing_review_id",
                data={"pending_ids": pending_ids[:10]},
                text="Bitte geben Sie eine review_id an oder lassen Sie sich den naechsten Pending-Fall zeigen.",
            )
        if action not in {"approve", "reject", "counteroffer"}:
            return BookingAssistantOperatorChatResponse(
                ok=True,
                intent=intent,
                action_taken="apply_action_missing_action",
                data={"review_id": review_id},
                text="Bitte Aktion angeben: approve, reject oder counteroffer.",
            )
        apply_result = apply_pending_action(
            user_id=user_id,
            settings=settings,
            api_key=api_key,
            review_id=review_id,
            req=BookingAssistantPendingApplyRequest(
                provider=req.provider,
                action=action,
                edited_body=str(parsed.get("reply_body") or "").strip(),
                subject="",
                reason="",
                send_to_customer=True,
            ),
        )
        next_pending = get_pending_next(user_id=user_id, settings=settings)
        return BookingAssistantOperatorChatResponse(
            ok=True,
            intent=intent,
            action_taken=f"apply_action:{action}",
            data={"result": apply_result.model_dump(), "next_pending": next_pending.model_dump()},
            text=f"Aktion {action} fuer Review {review_id} ausgefuehrt.",
        )

    if intent == "profile_update":
        instructions_add = [str(x).strip() for x in (parsed.get("instructions_add") or []) if str(x).strip()]
        rules_patch = parsed.get("rules_patch") if isinstance(parsed.get("rules_patch"), dict) else {}
        if not instructions_add and not rules_patch:
            instructions_add = [str(req.message or "").strip()]

        registry = build_registry(settings=settings, user_id=user_id)
        ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="booking_assistant_operator_chat")
        profile_name = str(req.assistant_profile_name or "booking_default").strip() or "booking_default"

        get_out = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="assistent_profile_get",
            args={"assistent_profile_name": profile_name},
        )
        if get_out.get("_error"):
            base = _default_profile(profile_name)
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

        update_out = _tool_call(
            registry=registry,
            ctx=ctx,
            tool="assistent_profile_update",
            args={
                "assistent_profile_name": profile_name,
                "instructions_add": instructions_add,
                "rules_patch": rules_patch,
            },
        )
        current = _tool_call(
            registry=registry,
            ctx=ctx,
            tool="assistent_profile_get",
            args={"assistent_profile_name": profile_name},
        )
        return BookingAssistantOperatorChatResponse(
            ok=True,
            intent=intent,
            action_taken="profile_update",
            data={
                "profile_name": profile_name,
                "instructions_add": instructions_add,
                "rules_patch": rules_patch,
                "update": update_out,
                "profile": current.get("profile") if isinstance(current, dict) else {},
            },
            text=f"Profil {profile_name} wurde aktualisiert.",
        )

    status = get_status_meeting(user_id=user_id, settings=settings, since="")
    return BookingAssistantOperatorChatResponse(
        ok=True,
        intent="status_meeting",
        action_taken="fallback_status_meeting",
        data=status.model_dump(),
        text=status.text,
    )
