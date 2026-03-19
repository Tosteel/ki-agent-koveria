from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import HTTPException

from server.agent.langchain_runtime import dispatch_tool_chain
from server.agent.tool_registry import ToolContext, ToolRegistry
from server.core.settings import Settings
from server.services.agent_service import build_registry

from .models import (
    BookingAssistantApproveRequest,
    BookingAssistantRejectRequest,
    BookingAssistantReviewActionResponse,
    BookingAssistantReviewItem,
    BookingAssistantReviewsResponse,
    BookingAssistantRunItem,
    BookingAssistantRunRequest,
    BookingAssistantRunResponse,
)
from .store import add_review, find_review, has_processed, list_reviews, load_state, mark_processed, save_state

_TRACE_ENABLED: ContextVar[bool] = ContextVar("booking_assistant_trace_enabled", default=False)
_TRACE_STEP: ContextVar[int] = ContextVar("booking_assistant_trace_step", default=0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
            "Kommuniziere Preise transparent und verlange eine Preisbestätigung.",
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
        for item in msgs:
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


def _score_reply(*, registry: ToolRegistry, ctx: ToolContext, user_message: str, draft: str, sources: List[str]) -> Dict[str, Any]:
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


def _compose_decision_mail(*, decision: str, reasons: List[str], quote_text: str, hold_link: str = "") -> str:
    rs = "\n".join(f"- {r}" for r in reasons if r)
    if decision == "auto_decline":
        return (
            "Vielen Dank für Ihre Anfrage. Leider kann ich den Termin auf Basis der aktuellen Rahmenbedingungen nicht zusagen.\n\n"
            f"Gründe:\n{rs}\n\n"
            "Wenn Sie möchten, können wir eine alternative Anfrage mit angepassten Rahmenbedingungen prüfen."
        ).strip()
    if decision == "auto_accept":
        link_line = f"\nKalender-Hold: {hold_link}" if hold_link else ""
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
    return (
        "Vielen Dank, alle wichtigen Veranstaltungsdaten liegen vor.\n\n"
        f"{reason_block}"
        "Hier ist das konkrete Angebot:\n"
        f"{quote_text}\n\n"
        "Bitte bestätigen Sie den Preis kurz schriftlich. "
        "Erst danach reserviere ich den Termin verbindlich im Kalender."
    ).strip()


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

    profile = _ensure_profile(registry=registry, ctx=ctx, req=req)
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
                args={"text": extraction_text},
            )
            facts = facts_out.get("facts") if isinstance(facts_out.get("facts"), dict) else {}
            thread_facts_text = str(mail_payload.get("thread_facts_text") or "").strip()
            if thread_facts_text:
                thread_out = _safe_tool_call(
                    registry=registry,
                    ctx=ctx,
                    tool="booking_extract_facts",
                    args={"text": thread_facts_text},
                )
                thread_facts = thread_out.get("facts") if isinstance(thread_out.get("facts"), dict) else {}
                for key in ("event_date", "start_time", "duration_hours", "location", "occasion", "client_name"):
                    if facts.get(key) in (None, "", 0, 0.0) and thread_facts.get(key) not in (None, "", 0, 0.0):
                        facts[key] = thread_facts.get(key)
                facts["price_confirmed"] = bool(facts.get("price_confirmed")) or bool(thread_facts.get("price_confirmed"))
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

            if not bool(comp.get("complete")):
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

                    # Phase 1: Sobald Event-Details vollständig sind, Preis senden.
                    # Phase 2: Erst nach Preisbestätigung reservieren/akzeptieren.
                    if not bool(facts.get("price_confirmed")):
                        booking_decision = "need_clarification"
                        draft = _compose_quote_confirmation_mail(
                            quote_text=str(quote.get("text") or ""),
                            reasons=["Preisbestätigung ausstehend."],
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
                                hold_link = str(existing_hold.get("html_link") or "").strip()
                            else:
                                avail = _safe_tool_call(
                                    registry=registry,
                                    ctx=ctx,
                                    tool="calendar_check_availability",
                                    args={"start_iso": start_iso, "end_iso": end_iso, "calendar_id": "primary"},
                                )
                                if bool(avail.get("is_available")):
                                    hold = _safe_tool_call(
                                        registry=registry,
                                        ctx=ctx,
                                        tool="calender_hold_event",
                                        args={
                                            "summary": f"DJ Booking: {str(facts.get('occasion') or 'Event')}",
                                            "start_iso": start_iso,
                                            "end_iso": end_iso,
                                            "location": str(facts.get("location") or ""),
                                            "description": "Vorläufige Reservierung aus Booking Assistant",
                                            "attendees": [from_email] if from_email else [],
                                            "calendar_id": "primary",
                                        },
                                    )
                                    hold_link = str(hold.get("html_link") or "").strip()
                                    _remember_hold(
                                        state=state,
                                        thread_id=mail_thread_id,
                                        mail_id=mail_id,
                                        start_iso=start_iso,
                                        end_iso=end_iso,
                                        event_id=str(hold.get("event_id") or "").strip(),
                                        html_link=hold_link,
                                    )
                                else:
                                    booking_decision = "need_clarification"
                                    reasons.append("Gewünschter Zeitraum ist bereits belegt.")
                        draft = _compose_decision_mail(
                            decision=booking_decision,
                            reasons=reasons,
                            quote_text=str(quote.get("text") or ""),
                            hold_link=hold_link,
                        )
                    elif booking_decision == "auto_decline":
                        draft = _compose_decision_mail(
                            decision=booking_decision,
                            reasons=reasons,
                            quote_text=str(quote.get("text") or ""),
                        )
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
                        draft = _compose_decision_mail(
                            decision="need_clarification",
                            reasons=reasons or ["Bitte intern prüfen."],
                            quote_text=str(quote.get("text") or ""),
                        )

        score = _score_reply(
            registry=registry,
            ctx=ctx,
            user_message=str(mail_payload.get("text") or "").strip(),
            draft=draft,
            sources=sources,
        )
        score_total = float(score.get("score_total") or 0.0)
        verdict = str(score.get("verdict") or "needs_review").strip().lower()

        policy = _policy_check(registry=registry, ctx=ctx, text=draft, strict_mode=req.strict_policy)
        policy_allowed = bool(policy.get("allowed"))
        policy_risk = str(policy.get("risk_level") or "").strip().lower()

        force_human = booking_decision == "human_review"
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
            continue

        reason_parts = [str(score.get("reason") or "needs_human_review")]
        if booking_decision:
            reason_parts.append(f"booking_decision={booking_decision}")
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
            ticket_id=ticket_id,
        )
        add_review(state, review)
        mark_processed(state, mail_id)
        processed_count += 1
        review_count += 1
        run_items.append(
            BookingAssistantRunItem(
                mail_id=mail_id,
                subject=subject,
                from_email=from_email,
                decision="needs_human",
                booking_decision=booking_decision,
                review_id=str(review.get("id") or ""),
                sent=False,
                score_total=score_total,
                reason=reason_text,
            )
        )

    save_state(settings, user_id, state)
    result = BookingAssistantRunResponse(
        ok=True,
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

    body = str(req.edited_body or "").strip() or str(review.get("draft_body") or "").strip()
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
    review["status"] = "rejected"
    review["updated_at"] = _now_iso()
    if reason:
        review["reason"] = reason
    review["sent"] = False
    save_state(settings, user_id, state)

    return BookingAssistantReviewActionResponse(
        ok=True,
        review_id=str(review.get("id") or review_id),
        status=str(review.get("status") or "rejected"),
        sent=False,
        reason=str(review.get("reason") or ""),
    )
