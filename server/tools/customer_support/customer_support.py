from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import HTTPException


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_priority(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in {"low", "medium", "high", "urgent"}:
        return raw
    return "medium"


def _tickets_file(user_dir: Path) -> Path:
    base = user_dir / "customer_support"
    base.mkdir(parents=True, exist_ok=True)
    return base / "review_tickets.json"


def _load_tickets(user_dir: Path) -> Dict[str, Any]:
    path = _tickets_file(user_dir)
    if not path.exists():
        return {"tickets": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"tickets": []}
    if not isinstance(data, dict):
        return {"tickets": []}
    tickets = data.get("tickets")
    if not isinstance(tickets, list):
        data["tickets"] = []
    return data


def _save_tickets(user_dir: Path, data: Dict[str, Any]) -> None:
    path = _tickets_file(user_dir)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text or ""))


def score_reply(
    *,
    user_message: str,
    draft_reply: str,
    knowledge_evidence: List[str] | None = None,
    require_actionable: bool = True,
) -> Dict[str, Any]:
    user_message = (user_message or "").strip()
    draft_reply = (draft_reply or "").strip()
    if not draft_reply:
        raise HTTPException(status_code=422, detail="draft_reply is required")

    evidence = [str(x).strip() for x in (knowledge_evidence or []) if str(x).strip()]
    reply_l = draft_reply.lower()
    user_l = user_message.lower()

    relevance = 0.7
    if user_l:
        overlaps = 0
        for token in re.findall(r"[A-Za-z0-9_]{4,}", user_l):
            if token in reply_l:
                overlaps += 1
        relevance = min(1.0, 0.45 + overlaps * 0.08)

    clarity = 0.45
    wc = _word_count(draft_reply)
    if wc >= 30:
        clarity = 0.75
    if wc >= 60:
        clarity = 0.9
    if wc < 8:
        clarity = 0.2

    groundedness = 0.55
    if evidence:
        groundedness = 0.8
        if any(str(e).lower() in reply_l for e in evidence):
            groundedness = 0.9

    tone = 0.8
    if any(bad in reply_l for bad in ("idiot", "dumm", "blod", "stupid", "nonsense")):
        tone = 0.2

    actionable = 0.65
    has_action_terms = any(
        t in reply_l
        for t in ("naechste", "next", "schritt", "option", "vorschlag", "bitte", "empfehle", "ich kann")
    )
    if require_actionable:
        actionable = 0.85 if has_action_terms else 0.35

    dimensions = {
        "relevance": round(relevance, 3),
        "clarity": round(clarity, 3),
        "groundedness": round(groundedness, 3),
        "tone": round(tone, 3),
        "actionable": round(actionable, 3),
    }

    total = (
        dimensions["relevance"] * 0.3
        + dimensions["clarity"] * 0.2
        + dimensions["groundedness"] * 0.25
        + dimensions["tone"] * 0.1
        + dimensions["actionable"] * 0.15
    )
    total = round(max(0.0, min(1.0, total)), 3)

    reasons: List[str] = []
    improvements: List[str] = []
    if dimensions["groundedness"] < 0.65:
        reasons.append("Antwort ist zu wenig mit Evidenz belegt.")
        improvements.append("Mehr konkrete Fakten aus RAG/Web zitieren.")
    if dimensions["relevance"] < 0.6:
        reasons.append("Antwort trifft die Nutzerfrage nicht praezise genug.")
        improvements.append("Direkt auf die Kernfrage eingehen.")
    if dimensions["clarity"] < 0.6:
        reasons.append("Antwort ist zu kurz oder unstrukturiert.")
        improvements.append("In 3-6 klaren Saetzen strukturieren.")
    if dimensions["tone"] < 0.6:
        reasons.append("Tonfall ist nicht kundengeeignet.")
        improvements.append("Neutralen und professionellen Ton nutzen.")
    if dimensions["actionable"] < 0.6:
        reasons.append("Antwort hat zu wenig konkrete Handlungsoptionen.")
        improvements.append("Naechste Schritte explizit nennen.")

    if total >= 0.8:
        verdict = "send"
    elif total >= 0.6:
        verdict = "needs_review"
    else:
        verdict = "reject"

    summary = f"Score={total:.2f}, verdict={verdict}"
    return {
        "total_score": total,
        "verdict": verdict,
        "dimensions": dimensions,
        "reasons": reasons,
        "improvements": improvements,
        "text": summary,
    }


def create_review_ticket(
    *,
    user_dir: Path,
    title: str,
    user_message: str = "",
    draft_reply: str = "",
    score: float = 0.0,
    reasons: List[str] | None = None,
    priority: str = "medium",
    metadata: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    title = (title or "").strip()
    if len(title) < 3:
        raise HTTPException(status_code=422, detail="title must be at least 3 characters")
    now = _now_iso()
    tid = f"rtk_{uuid4().hex[:12]}"
    item = {
        "ticket_id": tid,
        "status": "open",
        "created_at": now,
        "updated_at": now,
        "title": title,
        "priority": _normalize_priority(priority),
        "score": round(max(0.0, min(1.0, float(score or 0.0))), 3),
        "user_message": (user_message or "").strip(),
        "draft_reply": (draft_reply or "").strip(),
        "reasons": [str(x).strip() for x in (reasons or []) if str(x).strip()],
        "metadata": {str(k): str(v) for k, v in (metadata or {}).items()},
        "assignee": "",
        "reviewer_notes": [],
        "resolution": "",
    }
    data = _load_tickets(user_dir)
    data.setdefault("tickets", []).append(item)
    _save_tickets(user_dir, data)
    return {
        "ticket_id": item["ticket_id"],
        "status": item["status"],
        "created_at": item["created_at"],
        "updated_at": item["updated_at"],
        "title": item["title"],
        "score": item["score"],
        "priority": item["priority"],
        "text": f"Review-Ticket erstellt: {item['ticket_id']}",
    }


def update_review_ticket(
    *,
    user_dir: Path,
    ticket_id: str,
    status: str = "",
    reviewer_note: str = "",
    assignee: str = "",
    draft_reply: str = "",
    score: float | None = None,
    resolution: str = "",
) -> Dict[str, Any]:
    ticket_id = (ticket_id or "").strip()
    if not ticket_id:
        raise HTTPException(status_code=422, detail="ticket_id is required")

    data = _load_tickets(user_dir)
    tickets = data.get("tickets")
    if not isinstance(tickets, list):
        tickets = []
        data["tickets"] = tickets

    target: Dict[str, Any] | None = None
    for item in tickets:
        if isinstance(item, dict) and str(item.get("ticket_id") or "") == ticket_id:
            target = item
            break
    if target is None:
        raise HTTPException(status_code=404, detail=f"review ticket not found: {ticket_id}")

    if status.strip():
        target["status"] = status.strip().lower()
    if assignee.strip():
        target["assignee"] = assignee.strip()
    if draft_reply.strip():
        target["draft_reply"] = draft_reply.strip()
    if score is not None:
        target["score"] = round(max(0.0, min(1.0, float(score))), 3)
    if resolution.strip():
        target["resolution"] = resolution.strip()
    if reviewer_note.strip():
        notes = target.get("reviewer_notes")
        if not isinstance(notes, list):
            notes = []
        notes.append({"at": _now_iso(), "note": reviewer_note.strip()})
        target["reviewer_notes"] = notes

    target["updated_at"] = _now_iso()
    _save_tickets(user_dir, data)

    notes_count = len(target.get("reviewer_notes") or [])
    return {
        "ticket_id": ticket_id,
        "status": str(target.get("status") or ""),
        "updated_at": str(target.get("updated_at") or ""),
        "assignee": str(target.get("assignee") or ""),
        "notes_count": notes_count,
        "text": f"Review-Ticket aktualisiert: {ticket_id} (status={target.get('status')})",
    }


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:(?:\+|00)\d{1,3}[\s\-]?)?(?:\d[\s\-]?){7,}\d")
_IBAN_RE = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b")


def _redact_sensitive(text: str) -> str:
    out = _EMAIL_RE.sub("[redacted-email]", text)
    out = _PHONE_RE.sub("[redacted-phone]", out)
    out = _IBAN_RE.sub("[redacted-iban]", out)
    return out


def policy_check(*, text: str, policy_profile: str = "default", strict_mode: bool = True) -> Dict[str, Any]:
    body = (text or "").strip()
    if not body:
        raise HTTPException(status_code=422, detail="text is required")
    body_l = body.lower()

    violations: List[str] = []
    warnings: List[str] = []

    has_email = bool(_EMAIL_RE.search(body))
    has_phone = bool(_PHONE_RE.search(body))
    has_iban = bool(_IBAN_RE.search(body))
    if has_email or has_phone or has_iban:
        violations.append("PII detected (email/phone/bank data).")

    forbidden = [
        ("100% garantie", "Unzulaessige Garantiezusage."),
        ("garantiert", "Unzulaessige Garantiezusage."),
        ("rechtsverbindlich", "Potenziell rechtlich kritische Formulierung."),
        ("wir haften nicht", "Haftungsausschluss ohne Pruefung."),
    ]
    for needle, reason in forbidden:
        if needle in body_l:
            violations.append(reason)

    if any(x in body_l for x in ("idiot", "dumm", "bloede", "stupid")):
        violations.append("Unprofessioneller Ton.")

    if len(body) < 40:
        warnings.append("Antwort sehr kurz; Qualitaetsrisiko.")

    profile = (policy_profile or "default").strip().lower()
    if profile == "strict_finance" and not has_iban and "konto" in body_l:
        warnings.append("Finanz-Kontext erkannt, aber keine Kontodaten geprueft.")

    if violations:
        risk_level = "high" if strict_mode else "medium"
        allowed = False if strict_mode else len(violations) <= 1
    else:
        risk_level = "low" if not warnings else "medium"
        allowed = True

    redacted = _redact_sensitive(body)
    return {
        "allowed": allowed,
        "risk_level": risk_level,
        "violations": violations,
        "warnings": warnings,
        "redacted_text": redacted,
        "text": f"Policy-Check: allowed={str(allowed).lower()}, risk={risk_level}",
    }

