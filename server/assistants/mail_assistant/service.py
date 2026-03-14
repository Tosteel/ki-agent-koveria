from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import HTTPException

from server.agent.langchain_runtime import dispatch_tool_chain
from server.agent.tool_registry import ToolContext, ToolRegistry
from server.core.settings import Settings
from server.services.agent_service import build_registry

from .models import (
    MailAssistantApproveRequest,
    MailAssistantRejectRequest,
    MailAssistantReviewActionResponse,
    MailAssistantReviewItem,
    MailAssistantReviewsResponse,
    MailAssistantRunItem,
    MailAssistantRunRequest,
    MailAssistantRunResponse,
)
from .store import add_review, find_review, has_processed, list_reviews, load_state, mark_processed, save_state

_TRACE_ENABLED: ContextVar[bool] = ContextVar("mail_assistant_trace_enabled", default=False)
_TRACE_STEP: ContextVar[int] = ContextVar("mail_assistant_trace_step", default=0)


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
                f"args={json.dumps(args, ensure_ascii=False)[:2000]}",
            ],
        )
    out = dispatch_tool_chain(registry=registry, tool_name=tool, ctx=ctx, args=args)
    if isinstance(out, dict):
        if _TRACE_ENABLED.get():
            _trace_log(
                f"STEP OUTPUT { _TRACE_STEP.get() }",
                [
                    f"tool={tool}",
                    f"payload={json.dumps(out, ensure_ascii=False)[:3000]}",
                ],
            )
        return out
    wrapped = {"value": out}
    if _TRACE_ENABLED.get():
        _trace_log(
            f"STEP OUTPUT { _TRACE_STEP.get() }",
            [
                f"tool={tool}",
                f"payload={json.dumps(wrapped, ensure_ascii=False)[:3000]}",
            ],
        )
    return wrapped


def _safe_tool_call(*, registry: ToolRegistry, ctx: ToolContext, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if registry.get_tool(tool) is None:
        return {}
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


def _mail_query(mail_payload: Dict[str, Any]) -> str:
    subject = str(mail_payload.get("subject") or "").strip()
    body = str(mail_payload.get("body_text") or "").strip()
    if len(body) > 1600:
        body = body[:1600].rstrip() + "…"
    return "\n".join(x for x in [subject, body] if x).strip()


def _extract_sources(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if not isinstance(payload, dict):
        return out

    for key in ("final_url", "url", "source", "link", "href", "document", "file"):
        val = str(payload.get(key) or "").strip()
        if val:
            out.append(val)

    hits = payload.get("hits") if isinstance(payload.get("hits"), list) else []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        for key in ("source", "url", "link", "href", "document", "file"):
            val = str(hit.get(key) or "").strip()
            if val:
                out.append(val)
                break

    matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
    for match in matches:
        if not isinstance(match, dict):
            continue
        href = str(match.get("href") or "").strip()
        if href:
            out.append(href)

    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if url:
            out.append(url)

    visited = payload.get("visited_urls") if isinstance(payload.get("visited_urls"), list) else []
    for v in visited:
        val = str(v or "").strip()
        if val:
            out.append(val)

    return _dedupe_str_list(out)


def _collect_mail_context(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    mailbox: str,
    mail_id: str,
    include_thread: bool,
    include_attachments: bool,
) -> Dict[str, Any]:
    base = _tool_call(
        registry=registry,
        ctx=ctx,
        tool="read_mail",
        args={"mail_id": mail_id, "mailbox": mailbox, "max_chars": 20000},
    )
    parts: List[str] = [str(base.get("text") or "").strip()]

    if include_thread:
        thread = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="read_mail_thread",
            args={"mail_id": mail_id, "mailbox": mailbox, "max_messages": 20, "max_chars": 8000},
        )
        t = str(thread.get("text") or "").strip()
        if t:
            parts.append("THREAD:\n" + t)

    if include_attachments:
        atts = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="read_mail_attachments",
            args={
                "mail_id": mail_id,
                "mailbox": mailbox,
                "include_content": True,
                "max_attachment_chars": 4000,
                "extract_text_pdf": True,
            },
        )
        a = str(atts.get("text") or "").strip()
        if a:
            parts.append("ATTACHMENTS:\n" + a)

    merged = "\n\n".join(p for p in parts if p).strip()
    out = dict(base)
    out["text"] = merged or str(base.get("text") or "").strip()
    return out


def _try_customer_context(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    from_email: str,
    query: str,
) -> Dict[str, Any]:
    if registry.get_tool("customer_context_lookup") is None:
        return {}

    attempts = [
        {"from_email": from_email, "query": query},
        {"customer_id": from_email, "query": query},
        {"email": from_email, "query": query},
    ]
    for args in attempts:
        out = _safe_tool_call(registry=registry, ctx=ctx, tool="customer_context_lookup", args=args)
        if out and not out.get("_error"):
            return out
    return {}


def _classify_mail_intent(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    mail_payload: Dict[str, Any],
) -> Dict[str, Any]:
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="classify_mail",
        args={
            "text": str(mail_payload.get("text") or "").strip(),
            "subject": str(mail_payload.get("subject") or "").strip(),
            "body_text": str(mail_payload.get("body_text") or "").strip(),
            "from_email": str(mail_payload.get("from_email") or "").strip(),
        },
    )
    if out and not out.get("_error"):
        intent = str(out.get("intent") or "info").strip().lower()
        if intent not in {"info", "beschwerde", "angebot", "termin", "eskalation", "newsletter"}:
            intent = "info"
        confidence = float(out.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        reason = str(out.get("reason") or "").strip()
        return {"intent": intent, "confidence": confidence, "reason": reason, "raw": out}
    return {"intent": "info", "confidence": 0.0, "reason": "classify_mail_unavailable", "raw": {}}


def _intent_policy(intent: str) -> Dict[str, Any]:
    i = str(intent or "info").strip().lower()
    if i == "newsletter":
        return {
            "force_skip": True,
            "force_human_review": False,
            "require_actionable": False,
            "rag_top_k_boost": 0,
            "draft_hint": "Keine Antwort senden; als Newsletter/Systemmail behandeln.",
        }
    if i in {"eskalation", "beschwerde"}:
        return {
            "force_skip": False,
            "force_human_review": True,
            "require_actionable": True,
            "rag_top_k_boost": 2,
            "draft_hint": "Sei deeskalierend, empathisch und biete klare nächste Schritte an.",
        }
    if i == "termin":
        return {
            "force_skip": False,
            "force_human_review": False,
            "require_actionable": True,
            "rag_top_k_boost": 1,
            "draft_hint": "Gib konkrete Terminoptionen oder frage gezielt nach fehlenden Zeitangaben.",
        }
    if i == "angebot":
        return {
            "force_skip": False,
            "force_human_review": False,
            "require_actionable": True,
            "rag_top_k_boost": 1,
            "draft_hint": "Antworte strukturiert und nenne klare Angebots-/Nächste-Schritte-Optionen.",
        }
    return {
        "force_skip": False,
        "force_human_review": False,
        "require_actionable": False,
        "rag_top_k_boost": 0,
        "draft_hint": "Antworte sachlich und präzise.",
    }


def _retrieve_context(
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
    trace: List[str] = []

    rag = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="rag_knowledgebase",
        args={"query": query, "top_k": rag_top_k},
    )
    if rag and not rag.get("_error"):
        txt = str(rag.get("text") or "").strip()
        if txt:
            parts.append("RAG:\n" + txt)
        sources.extend(_extract_sources(rag))
        trace.append("rag_knowledgebase:ok")
    elif rag.get("_error"):
        trace.append("rag_knowledgebase:error")

    for url in web_sources:
        u = str(url or "").strip()
        if not u:
            continue
        out = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="web_crawl_site_whitelist",
            args={
                "url": u,
                "query": query,
                "allowed_domains": web_whitelist_domains,
                "max_pages": 3,
                "max_matches": 8,
            },
        )
        if out and not out.get("_error"):
            txt = str(out.get("text") or "").strip()
            if txt:
                parts.append(f"WEB({u}):\n{txt}")
            sources.extend(_extract_sources(out))
            trace.append(f"web_crawl_site_whitelist:{u}:ok")
            continue
        if out.get("_error"):
            trace.append(f"web_crawl_site_whitelist:{u}:error")

        fallback = _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="web_search_page",
            args={"url": u, "query": query},
        )
        if fallback and not fallback.get("_error"):
            txt = str(fallback.get("text") or "").strip()
            if txt:
                parts.append(f"WEB({u}):\n{txt}")
            sources.extend(_extract_sources(fallback))
            trace.append(f"web_search_page:{u}:ok")

    merged = "\n\n".join(parts).strip()
    if len(merged) < 220 or len(sources) == 0:
        for fallback_tool, args in (
            ("websearch_table", {"user_prompt": query}),
            ("langsearch", {"query": query, "count": 5}),
        ):
            out = _safe_tool_call(registry=registry, ctx=ctx, tool=fallback_tool, args=args)
            if out and not out.get("_error"):
                txt = str(out.get("text") or "").strip()
                if txt:
                    parts.append(f"{fallback_tool}:\n{txt}")
                sources.extend(_extract_sources(out))
                trace.append(f"{fallback_tool}:ok")
                break
            if out.get("_error"):
                trace.append(f"{fallback_tool}:error")

    merged = "\n\n".join(parts).strip()
    if len(merged) > max_context_chars:
        merged = merged[:max_context_chars].rstrip() + "…"
    return {
        "context_text": merged,
        "sources": _dedupe_str_list(sources),
        "trace": trace,
    }


def _draft_reply(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    mail_payload: Dict[str, Any],
    context_text: str,
    customer_context: str,
    sources: List[str],
    intent: str,
    intent_hint: str,
) -> str:
    source_lines = "\n".join(f"- {s}" for s in sources[:20])
    compose_input = (
        "Eingehende Mail:\n"
        f"{str(mail_payload.get('text') or '').strip()}\n\n"
        "Kundenkontext:\n"
        f"{customer_context or 'Kein spezifischer Kundenkontext gefunden.'}\n\n"
        "Recherchierter Kontext:\n"
        f"{context_text or 'Kein verwertbarer Kontext gefunden.'}\n\n"
        "Quellen:\n"
        f"{source_lines or '- keine'}"
    )
    instruction = (
        "Erstelle eine präzise, höfliche Antwortmail auf Deutsch. "
        "Nutze nur belastbare Informationen aus dem Kontext. "
        "Wenn Daten fehlen, stelle eine kurze Rückfrage statt zu raten. "
        f"Intent={intent}. {intent_hint}"
    )
    out = _tool_call(
        registry=registry,
        ctx=ctx,
        tool="llm_text_compose",
        args={"text": compose_input, "instruction": instruction, "max_chars": 2500},
    )
    return str(out.get("text") or "").strip()


def _score_reply(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    user_message: str,
    draft: str,
    sources: List[str],
    require_actionable: bool,
) -> Dict[str, Any]:
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="score_reply",
        args={
            "user_message": user_message,
            "draft_reply": draft,
            "knowledge_evidence": sources[:20],
            "require_actionable": bool(require_actionable),
        },
    )
    if out and not out.get("_error"):
        total = float(out.get("total_score") or 0.0)
        total = max(0.0, min(1.0, total))
        verdict = str(out.get("verdict") or "needs_review").strip().lower()
        return {
            "score_total": total,
            "verdict": verdict,
            "reason": "; ".join(str(x) for x in (out.get("reasons") or [])[:3] if str(x).strip()),
            "raw": out,
        }

    # Minimal fallback if scoring tool is unavailable.
    text = (draft or "").strip()
    score = 0.45
    if len(text) >= 120:
        score += 0.2
    if len(sources) >= 1:
        score += 0.2
    if len(sources) >= 3:
        score += 0.1
    score = max(0.0, min(1.0, score))
    verdict = "send" if score >= 0.8 else "needs_review"
    return {
        "score_total": score,
        "verdict": verdict,
        "reason": "score_reply_unavailable_fallback",
        "raw": {},
    }


def _policy_check(
    *,
    registry: ToolRegistry,
    ctx: ToolContext,
    text: str,
    strict_mode: bool,
) -> Dict[str, Any]:
    out = _safe_tool_call(
        registry=registry,
        ctx=ctx,
        tool="policy_check",
        args={"text": text, "policy_profile": "default", "strict_mode": strict_mode},
    )
    if out and not out.get("_error"):
        return out
    return {
        "allowed": True,
        "risk_level": "unknown",
        "violations": [],
        "warnings": ["policy_check_unavailable"],
        "text": "policy_check unavailable",
    }


def _build_review(
    *,
    mail_id: str,
    mailbox: str,
    from_email: str,
    subject: str,
    mail_text: str,
    draft_body: str,
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
        "score_total": float(score.get("score_total") or 0.0),
        "score": dict(score),
        "sources": list(sources),
        "ticket_id": ticket_id,
        "reason": reason,
        "sent": False,
        "send_result": {},
    }


def run_once(*, user_id: str, settings: Settings, api_key: str, req: MailAssistantRunRequest) -> MailAssistantRunResponse:
    trace_token = _TRACE_ENABLED.set(bool(req.trace_steps))
    step_token = _TRACE_STEP.set(0)
    _trace_log(
        "MAIL ASSISTANT RUN",
        [
            f"user_id={user_id}",
            f"mailbox={req.mailbox}",
            f"limit={req.limit}",
            f"auto_send_threshold={req.auto_send_threshold}",
            f"trace_steps={req.trace_steps}",
        ],
    )

    registry = build_registry(settings=settings, user_id=user_id)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="mail_assistant_run_once")
    state = load_state(settings, user_id)

    run_items: List[MailAssistantRunItem] = []
    sent_count = 0
    review_count = 0
    skipped_count = 0
    processed_count = 0

    inbox = _tool_call(
        registry=registry,
        ctx=ctx,
        tool="fetch_unanswered_mails",
        args={"mailbox": req.mailbox, "limit": req.limit},
    )
    emails = inbox.get("emails") if isinstance(inbox.get("emails"), list) else []
    _trace_log("INBOX RESULT", [f"emails_found={len(emails)}"])

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
                MailAssistantRunItem(
                    mail_id=mail_id,
                    subject=subject,
                    from_email=from_email,
                    decision="skipped",
                    reason="already_processed",
                )
            )
            _trace_log("MAIL SKIPPED", [f"mail_id={mail_id}", "reason=already_processed"])
            continue

        try:
            mail_payload = _collect_mail_context(
                registry=registry,
                ctx=ctx,
                mailbox=req.mailbox,
                mail_id=mail_id,
                include_thread=req.include_thread,
                include_attachments=req.include_attachments,
            )
        except Exception as exc:
            run_items.append(
                MailAssistantRunItem(
                    mail_id=mail_id,
                    subject=subject,
                    from_email=from_email,
                    decision="failed",
                    reason=f"read_mail_failed:{exc}",
                )
            )
            _trace_log("MAIL FAILED", [f"mail_id={mail_id}", f"reason=read_mail_failed:{exc}"])
            continue

        query = _mail_query(mail_payload) or subject or from_email or f"mail {mail_id}"
        intent_info = _classify_mail_intent(registry=registry, ctx=ctx, mail_payload=mail_payload)
        intent = str(intent_info.get("intent") or "info")
        intent_conf = float(intent_info.get("confidence") or 0.0)
        intent_reason = str(intent_info.get("reason") or "")
        ip = _intent_policy(intent)
        _trace_log(
            "INTENT",
            [
                f"mail_id={mail_id}",
                f"intent={intent}",
                f"confidence={intent_conf:.2f}",
                f"reason={intent_reason}",
            ],
        )
        if bool(ip.get("force_skip")):
            skipped_count += 1
            processed_count += 1
            mark_processed(state, mail_id)
            run_items.append(
                MailAssistantRunItem(
                    mail_id=mail_id,
                    subject=subject,
                    from_email=from_email,
                    decision="skipped",
                    reason=f"intent={intent}",
                )
            )
            _trace_log("MAIL SKIPPED", [f"mail_id={mail_id}", f"reason=intent={intent}"])
            continue

        customer_ctx_out = _try_customer_context(
            registry=registry,
            ctx=ctx,
            from_email=from_email,
            query=query,
        )
        customer_text = str(customer_ctx_out.get("text") or "").strip()
        _trace_log(
            "CONTEXT PREP",
            [
                f"mail_id={mail_id}",
                f"query_len={len(query)}",
                f"customer_context={'yes' if customer_text else 'no'}",
            ],
        )

        context = _retrieve_context(
            registry=registry,
            ctx=ctx,
            query=query,
            rag_top_k=max(1, min(20, int(req.rag_top_k) + int(ip.get("rag_top_k_boost") or 0))),
            web_sources=req.web_sources,
            web_whitelist_domains=req.web_whitelist_domains,
            max_context_chars=req.max_context_chars,
        )
        context_text = str(context.get("context_text") or "").strip()
        sources = [str(x).strip() for x in (context.get("sources") or []) if str(x).strip()]
        _trace_log(
            "RESEARCH RESULT",
            [
                f"mail_id={mail_id}",
                f"context_chars={len(context_text)}",
                f"sources={len(sources)}",
            ],
        )

        try:
            draft = _draft_reply(
                registry=registry,
                ctx=ctx,
                mail_payload=mail_payload,
                context_text=context_text,
                customer_context=customer_text,
                sources=sources,
                intent=intent,
                intent_hint=str(ip.get("draft_hint") or ""),
            )
        except Exception as exc:
            run_items.append(
                MailAssistantRunItem(
                    mail_id=mail_id,
                    subject=subject,
                    from_email=from_email,
                    decision="failed",
                    reason=f"draft_failed:{exc}",
                )
            )
            _trace_log("MAIL FAILED", [f"mail_id={mail_id}", f"reason=draft_failed:{exc}"])
            continue

        score = _score_reply(
            registry=registry,
            ctx=ctx,
            user_message=str(mail_payload.get("text") or "").strip(),
            draft=draft,
            sources=sources,
            require_actionable=bool(ip.get("require_actionable")),
        )
        score_total = float(score.get("score_total") or 0.0)
        verdict = str(score.get("verdict") or "needs_review").strip().lower()
        decision_reason = str(score.get("reason") or "").strip()
        _trace_log(
            "SCORING",
            [
                f"mail_id={mail_id}",
                f"score_total={score_total:.3f}",
                f"verdict={verdict}",
                f"reason={decision_reason}",
            ],
        )

        policy = _policy_check(registry=registry, ctx=ctx, text=draft, strict_mode=req.strict_policy)
        policy_allowed = bool(policy.get("allowed"))
        policy_risk = str(policy.get("risk_level") or "").strip().lower()
        violations = policy.get("violations") if isinstance(policy.get("violations"), list) else []
        _trace_log(
            "POLICY",
            [
                f"mail_id={mail_id}",
                f"allowed={policy_allowed}",
                f"risk={policy_risk}",
                f"violations={len(violations)}",
            ],
        )

        can_auto_send = bool(
            draft
            and policy_allowed
            and score_total >= req.auto_send_threshold
            and verdict == "send"
            and policy_risk not in {"high", "critical"}
            and not bool(ip.get("force_human_review"))
        )

        if can_auto_send:
            try:
                _tool_call(
                    registry=registry,
                    ctx=ctx,
                    tool="answer_mail",
                    args={"mail_id": mail_id, "mailbox": req.mailbox, "body": draft},
                )
                sent_count += 1
                processed_count += 1
                mark_processed(state, mail_id)
                run_items.append(
                    MailAssistantRunItem(
                        mail_id=mail_id,
                        subject=subject,
                        from_email=from_email,
                        decision="auto_sent",
                        score_total=score_total,
                        risk=policy_risk or "low",
                        sent=True,
                        reason=decision_reason or "score_and_policy_ok",
                    )
                )
                _trace_log("MAIL SENT", [f"mail_id={mail_id}", "decision=auto_sent"])
                continue
            except Exception as exc:
                decision_reason = f"auto_send_failed:{exc}"
                _trace_log("AUTO SEND FAILED", [f"mail_id={mail_id}", f"reason={decision_reason}"])

        reason_parts: List[str] = [decision_reason or "needs_human_review"]
        if not policy_allowed:
            reason_parts.append("policy_blocked")
        if violations:
            reason_parts.append("violations=" + ", ".join(str(v) for v in violations[:3]))
        reason_parts.append(f"intent={intent}")
        reason_text = " | ".join(x for x in reason_parts if x)

        ticket_id = ""
        if registry.get_tool("create_review_ticket") is not None:
            ticket = _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="create_review_ticket",
                args={
                    "title": f"Mail review: {subject or mail_id}",
                    "user_message": str(mail_payload.get("text") or "").strip(),
                    "draft_reply": draft,
                    "score": score_total,
                    "reasons": [reason_text],
                    "priority": "high" if not policy_allowed else "medium",
                    "metadata": {
                        "mail_id": mail_id,
                        "mailbox": req.mailbox,
                        "from_email": from_email,
                        "intent": intent,
                    },
                },
            )
            ticket_id = str(ticket.get("ticket_id") or "").strip()
            _trace_log("REVIEW TICKET", [f"mail_id={mail_id}", f"ticket_id={ticket_id or '-'}"])

        review = _build_review(
            mail_id=mail_id,
            mailbox=req.mailbox,
            from_email=from_email,
            subject=subject,
            mail_text=str(mail_payload.get("text") or "").strip(),
            draft_body=draft,
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
            MailAssistantRunItem(
                mail_id=mail_id,
                subject=subject,
                from_email=from_email,
                decision="needs_human",
                score_total=score_total,
                risk=policy_risk or "medium",
                review_id=str(review.get("id") or ""),
                reason=reason_text,
            )
        )
        _trace_log("MAIL TO REVIEW", [f"mail_id={mail_id}", f"review_id={review.get('id')}", f"ticket_id={ticket_id or '-'}"])

    save_state(settings, user_id, state)
    result = MailAssistantRunResponse(
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


def get_reviews(*, user_id: str, settings: Settings, status: str = "pending") -> MailAssistantReviewsResponse:
    state = load_state(settings, user_id)
    filter_status = str(status or "").strip().lower()
    if filter_status in {"", "all", "*"}:
        filter_status = ""
    reviews = [MailAssistantReviewItem(**r) for r in list_reviews(state, status=filter_status)]
    reviews.sort(key=lambda x: x.created_at, reverse=True)
    return MailAssistantReviewsResponse(ok=True, reviews=reviews)


def approve_review(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    review_id: str,
    req: MailAssistantApproveRequest,
) -> MailAssistantReviewActionResponse:
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
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal="mail_assistant_review_approve")

    policy = _policy_check(registry=registry, ctx=ctx, text=body, strict_mode=True)
    if not bool(policy.get("allowed")):
        reason = "manual_approve_blocked_by_policy"
        review["reason"] = reason
        review["updated_at"] = _now_iso()
        ticket_id = str(review.get("ticket_id") or "").strip()
        if ticket_id and registry.get_tool("update_review_ticket") is not None:
            _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="update_review_ticket",
                args={"ticket_id": ticket_id, "status": "rejected", "reviewer_note": reason},
            )
        save_state(settings, user_id, state)
        raise HTTPException(status_code=422, detail=reason)

    args: Dict[str, Any] = {
        "mail_id": str(review.get("mail_id") or ""),
        "mailbox": str(review.get("mailbox") or "INBOX"),
        "body": body,
    }
    subject = str(req.subject or "").strip()
    if subject:
        args["subject"] = subject

    send_result = _tool_call(registry=registry, ctx=ctx, tool="answer_mail", args=args)
    review["status"] = "approved"
    review["updated_at"] = _now_iso()
    review["sent"] = bool(send_result.get("sent"))
    review["send_result"] = send_result
    review["draft_body"] = body

    ticket_id = str(review.get("ticket_id") or "").strip()
    if ticket_id and registry.get_tool("update_review_ticket") is not None:
        _safe_tool_call(
            registry=registry,
            ctx=ctx,
            tool="update_review_ticket",
            args={"ticket_id": ticket_id, "status": "approved", "reviewer_note": "approved_and_sent"},
        )

    save_state(settings, user_id, state)
    return MailAssistantReviewActionResponse(
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
    req: MailAssistantRejectRequest,
) -> MailAssistantReviewActionResponse:
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

    ticket_id = str(review.get("ticket_id") or "").strip()
    if ticket_id:
        registry = build_registry(settings=settings, user_id=user_id)
        ctx = ToolContext(user_id=user_id, settings=settings, api_key="", goal="mail_assistant_review_reject")
        if registry.get_tool("update_review_ticket") is not None:
            _safe_tool_call(
                registry=registry,
                ctx=ctx,
                tool="update_review_ticket",
                args={"ticket_id": ticket_id, "status": "rejected", "reviewer_note": reason or "rejected"},
            )

    save_state(settings, user_id, state)
    return MailAssistantReviewActionResponse(
        ok=True,
        review_id=str(review.get("id") or review_id),
        status=str(review.get("status") or "rejected"),
        sent=False,
        reason=str(review.get("reason") or ""),
    )


def cleanup_state(*, user_id: str, settings: Settings, keep_reviews: int = 300) -> None:
    state = load_state(settings, user_id)
    reviews = state.get("reviews") if isinstance(state.get("reviews"), list) else []
    if len(reviews) > keep_reviews:
        state["reviews"] = reviews[-keep_reviews:]
    processed = state.get("processed_mail_ids") if isinstance(state.get("processed_mail_ids"), list) else []
    if len(processed) > 10000:
        state["processed_mail_ids"] = processed[-10000:]
    save_state(settings, user_id, state)
