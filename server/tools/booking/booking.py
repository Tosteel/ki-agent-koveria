from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List

from fastapi import HTTPException
from server.services.llm_ionos import IonosLLM


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _focus_booking_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    focused = raw
    for marker in ("\nTHREAD:", "\nATTACHMENTS:", "\nKundenkontext:", "\nRecherchierter Kontext:"):
        if marker in focused:
            focused = focused.split(marker, 1)[0]

    lines: List[str] = []
    for line in focused.splitlines():
        s = line.strip()
        l = s.lower()
        if not s:
            lines.append("")
            continue
        if s.startswith("________________________________"):
            break
        if s.startswith(">"):
            continue
        if re.match(r"^am\s.+\sschrieb\s", l):
            break
        if " schrieb am " in l:
            break
        if l.startswith(
            (
                "mailbox:",
                "mail id:",
                "from:",
                "subject:",
                "date:",
                "sent:",
                "to:",
                "cc:",
                "message-id:",
                "message id:",
                "in-reply-to:",
                "references:",
                "thread for mail_id",
                "messages:",
            )
        ):
            continue
        if re.match(r"^\[\d+\]\s", s):
            continue
        if l.startswith(("from=", "date=")) or " date=" in l:
            continue
        lines.append(s)

    out = "\n".join(lines).strip()
    return out or raw


def _extract_date(text: str) -> str:
    raw = str(text or "")
    m_iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", raw)
    if m_iso:
        return m_iso.group(1)

    m_de = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b", raw)
    if m_de:
        d = int(m_de.group(1))
        mo = int(m_de.group(2))
        y = int(m_de.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    return ""


def _extract_time(text: str) -> str:
    raw = str(text or "")
    m2 = re.search(r"\b(?:ab|start|beginn)\s*([01]?\d|2[0-3])\s*uhr\b", raw.lower())
    if m2:
        h = int(m2.group(1))
        return f"{h:02d}:00"

    m3 = re.search(
        r"\b(?:ab|start|beginn|beginnend|von)\s*([01]?\d|2[0-3])[:.]([0-5]\d)\b",
        raw.lower(),
    )
    if m3:
        h = int(m3.group(1))
        mi = int(m3.group(2))
        return f"{h:02d}:{mi:02d}"

    # Allow dotted time notation only with explicit "Uhr" marker, e.g. "20.30 Uhr".
    m4 = re.search(r"\b([01]?\d|2[0-3])\.([0-5]\d)\s*uhr\b", raw.lower())
    if m4:
        h = int(m4.group(1))
        mi = int(m4.group(2))
        return f"{h:02d}:{mi:02d}"

    # Fallback: generic time with ":" only (not ".") to avoid interpreting dates like 04.04 as 04:04.
    # Also ignore typical mail headers like "Date:"/"Sent:" lines.
    body_lines = []
    for line in raw.splitlines():
        l = line.strip().lower()
        if l.startswith(("date:", "sent:", "from:", "subject:", "to:", "cc:", "bcc:", "mailbox:", "mail id:")):
            continue
        if l.startswith(("date=", "from=", "thread for mail_id", "messages:")) or " date=" in l:
            continue
        if " schrieb am " in l:
            continue
        body_lines.append(line)
    body = "\n".join(body_lines)
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", body)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2))
        return f"{h:02d}:{mi:02d}"
    return ""


def _extract_duration_hours(text: str) -> float | None:
    raw = str(text or "")
    m = re.search(r"\b(\d{1,2})(?:[,.](\d))?\s*(?:stunden|stunde|h)\b", raw.lower())
    if m:
        base = int(m.group(1))
        dec = m.group(2)
        if dec is not None:
            return float(f"{base}.{dec}")
        return float(base)
    return None


def _extract_time_window(text: str) -> tuple[str, float | None]:
    raw = str(text or "").lower()
    m = re.search(
        r"\b(?:von|ab)?\s*([01]?\d|2[0-3])(?:[:.]([0-5]\d))?\s*(?:uhr)?\s*"
        r"(?:bis|to|-|–|—)\s*([01]?\d|2[0-3])(?:[:.]([0-5]\d))?\s*(?:uhr)?\b",
        raw,
    )
    if not m:
        return "", None
    sh = int(m.group(1))
    sm = int(m.group(2) or 0)
    eh = int(m.group(3))
    em = int(m.group(4) or 0)
    start_minutes = sh * 60 + sm
    end_minutes = eh * 60 + em
    if end_minutes <= start_minutes:
        end_minutes += 24 * 60
    duration_hours = (end_minutes - start_minutes) / 60.0
    if duration_hours <= 0:
        return "", None
    return f"{sh:02d}:{sm:02d}", round(duration_hours, 2)


def _extract_location(text: str) -> str:
    raw = str(text or "")
    patterns = [
        # Explicit location markers, with some typo tolerance ("veranstaltungsport").
        r"\b(?:ort|location|veranstaltungsort|veranstaltungsport)\s*(?:ist|=|:|-)\s*([^\n\r]+?)(?=[.;!?]|\n|$)",
        r"\b(?:der\s+)?(?:veranstaltungsort|veranstaltungsport)\s+([^\n\r.,;!?]{2,100})",
        r"\b(?:in|bei)\s+(\d{5}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-\s]{1,80}?)(?=[,.;!?]|\n|$)",
        r"\b(?:in|bei)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-\s]{2,80}?)(?=[,.;!?]|\n|$)",
    ]
    for p in patterns:
        m = re.search(p, raw, flags=re.IGNORECASE)
        if m:
            loc = _normalize_text(m.group(1))
            loc = loc.strip(" ,.;:")
            if len(loc) >= 4:
                return loc
    return ""


def _extract_client_name(text: str) -> Dict[str, object]:
    client = IonosLLM()
    if not client.enabled():
        return {
            "client_name": "",
            "confidence": 0.0,
            "reason": "llm_unavailable",
            "fallback_used": True,
            "model": "",
        }

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_client_name_extract",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "client_name": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["client_name", "confidence", "reason"],
            },
            "strict": True,
        },
    }

    completion = client.chat_completions(
        messages=[
            {
                "role": "system",
                "content": (
                    "Extrahiere den Namen des anfragenden Kunden aus einer Booking-Nachricht.\n"
                    "Gib nur den Personennamen zurück (z. B. 'Dietmar Maier').\n"
                    "Wenn kein verlässlicher Name enthalten ist, gib einen leeren String zurück.\n"
                    "Ignoriere quoted Verlaufstexte, Header und Signaturrauschen.\n"
                    "Antworte ausschließlich als JSON laut Schema."
                ),
            },
            {"role": "user", "content": f"Nachricht:\n{text}"},
        ],
        response_format=schema,
        temperature=0.0,
        top_p=0.1,
        max_tokens=120,
    )
    parsed = _parse_json_obj(client.extract_text(completion))
    if not parsed:
        return {
            "client_name": "",
            "confidence": 0.0,
            "reason": "llm_parse_failed",
            "fallback_used": True,
            "model": getattr(client.cfg, "model", ""),
        }
    return {
        "client_name": _normalize_text(str(parsed.get("client_name") or "")),
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence") or 0.0))),
        "reason": str(parsed.get("reason") or "").strip(),
        "fallback_used": False,
        "model": getattr(client.cfg, "model", ""),
    }


def _extract_occasion(text: str) -> str:
    raw = str(text or "").lower()
    mapping = {
        "hochzeit": "Hochzeit",
        "geburtstag": "Geburtstag",
        "firmen": "Firmenevent",
        "messe": "Messe",
        "party": "Party",
        "feier": "Feier",
        "jubil": "Jubiläum",
    }
    for key, value in mapping.items():
        if key in raw:
            return value
    return ""


def _parse_json_obj(text: str) -> Dict[str, object]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _fallback_booking_reply_score(
    *,
    user_message: str,
    draft_reply: str,
    booking_decision: str,
    missing_fields: List[str] | None,
    require_actionable: bool,
) -> Dict[str, Any]:
    user_l = str(user_message or "").lower()
    draft_l = str(draft_reply or "").lower()
    overlaps = 0
    for tok in re.findall(r"[A-Za-z0-9_]{4,}", user_l):
        if tok in draft_l:
            overlaps += 1
    booking_fit = min(1.0, 0.4 + overlaps * 0.06)

    process_progress = 0.6
    md = [str(x).strip().lower() for x in (missing_fields or []) if str(x).strip()]
    decision_l = str(booking_decision or "").strip().lower()
    if decision_l == "need_clarification":
        asks_question = "?" in draft_reply or "bitte" in draft_l
        process_progress = 0.8 if asks_question else 0.45
        if md and any(x in draft_l for x in md):
            process_progress = min(1.0, process_progress + 0.1)
    elif decision_l == "auto_accept":
        has_confirm = any(x in draft_l for x in ("bestaetigt", "reserviert", "zugesagt", "verbindlich"))
        process_progress = 0.85 if has_confirm else 0.55
    elif decision_l == "auto_decline":
        has_reason = any(x in draft_l for x in ("grund", "leider", "nicht zusagen", "nicht moeglich"))
        process_progress = 0.8 if has_reason else 0.5
    elif decision_l == "human_review":
        process_progress = 0.75 if "pruefung" in draft_l or "persoenlich" in draft_l else 0.5

    wc = len(re.findall(r"[A-Za-z0-9_]+", draft_reply or ""))
    clarity = 0.9 if wc >= 30 else 0.7 if wc >= 15 else 0.45
    groundedness = 0.7
    tone = 0.8
    if any(bad in draft_l for bad in ("idiot", "dumm", "stupid", "nonsense")):
        tone = 0.2
    actionable = 0.65
    if require_actionable:
        actionable_terms = ("naechste", "schritt", "bitte", "bestaetigen", "alternativ", "rueckmeldung", "vorschlag")
        actionable = 0.85 if any(t in draft_l for t in actionable_terms) else 0.35

    dimensions = {
        "booking_fit": round(_clamp01(booking_fit), 3),
        "process_progress": round(_clamp01(process_progress), 3),
        "groundedness": round(_clamp01(groundedness), 3),
        "clarity": round(_clamp01(clarity), 3),
        "tone": round(_clamp01(tone), 3),
        "actionable": round(_clamp01(actionable), 3),
    }
    total = round(
        dimensions["booking_fit"] * 0.25
        + dimensions["process_progress"] * 0.25
        + dimensions["groundedness"] * 0.15
        + dimensions["clarity"] * 0.15
        + dimensions["tone"] * 0.10
        + dimensions["actionable"] * 0.10,
        3,
    )
    if total >= 0.8:
        verdict = "send"
    elif total >= 0.6:
        verdict = "needs_review"
    else:
        verdict = "reject"

    reasons: List[str] = []
    improvements: List[str] = []
    if dimensions["process_progress"] < 0.6:
        reasons.append("Antwort bringt den Booking-Prozess nicht klar voran.")
        improvements.append("Naechsten Prozessschritt explizit formulieren.")
    if dimensions["actionable"] < 0.6:
        reasons.append("Antwort enthaelt zu wenig konkrete Handlungsaufforderung.")
        improvements.append("Klar um eine bestimmte Rueckmeldung bitten.")
    if dimensions["booking_fit"] < 0.6:
        reasons.append("Antwort passt nicht praezise zur Buchungsanfrage.")
        improvements.append("Direkt auf Anfragekontext und Regeln eingehen.")

    return {
        "total_score": total,
        "verdict": verdict,
        "dimensions": dimensions,
        "reasons": reasons,
        "improvements": improvements,
        "next_step": "Manuelle Pruefung empfohlen." if verdict != "send" else "Kann versendet werden.",
        "text": f"Score={total:.2f}, verdict={verdict}",
        "model": "",
        "fallback_used": True,
    }


def booking_reply_score(
    *,
    user_message: str,
    draft_reply: str,
    booking_decision: str = "",
    facts: Dict[str, Any] | None = None,
    required_fields: List[str] | None = None,
    missing_fields: List[str] | None = None,
    knowledge_evidence: List[str] | None = None,
    require_actionable: bool = True,
) -> Dict[str, Any]:
    draft = str(draft_reply or "").strip()
    if not draft:
        raise HTTPException(status_code=422, detail="draft_reply is required")

    msg = str(user_message or "").strip()
    decision = str(booking_decision or "").strip().lower()
    ff = dict(facts or {})
    req_fields = [str(x).strip() for x in (required_fields or []) if str(x).strip()]
    miss_fields = [str(x).strip() for x in (missing_fields or []) if str(x).strip()]
    evidence = [str(x).strip() for x in (knowledge_evidence or []) if str(x).strip()]

    client = IonosLLM()
    if not client.enabled():
        return _fallback_booking_reply_score(
            user_message=msg,
            draft_reply=draft,
            booking_decision=decision,
            missing_fields=miss_fields,
            require_actionable=bool(require_actionable),
        )

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_reply_score",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "total_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "verdict": {"type": "string", "enum": ["send", "needs_review", "reject"]},
                    "dimensions": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "booking_fit": {"type": "number", "minimum": 0, "maximum": 1},
                            "process_progress": {"type": "number", "minimum": 0, "maximum": 1},
                            "groundedness": {"type": "number", "minimum": 0, "maximum": 1},
                            "clarity": {"type": "number", "minimum": 0, "maximum": 1},
                            "tone": {"type": "number", "minimum": 0, "maximum": 1},
                            "actionable": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": [
                            "booking_fit",
                            "process_progress",
                            "groundedness",
                            "clarity",
                            "tone",
                            "actionable",
                        ],
                    },
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "improvements": {"type": "array", "items": {"type": "string"}},
                    "next_step": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": [
                    "total_score",
                    "verdict",
                    "dimensions",
                    "reasons",
                    "improvements",
                    "next_step",
                    "text",
                ],
            },
            "strict": True,
        },
    }

    completion = client.chat_completions(
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein strenger QA-Reviewer fuer Booking-E-Mail-Antworten.\n"
                    "Bewerte, ob der Draft zur Anfrage passt UND den Prozess sinnvoll voranbringt.\n"
                    "Prozessfortschritt bedeutet z. B.:\n"
                    "- fehlende Pflichtdaten gezielt abfragen,\n"
                    "- ein konkretes Angebot kommunizieren,\n"
                    "- Angebotsbestaetigung einholen,\n"
                    "- verbindlich bestaetigen oder sauber ablehnen.\n"
                    "Keine Halluzinationen. Nutze nur den gegebenen Kontext.\n"
                    "Wenn unklar: konservativ bewerten.\n"
                    "Antworte ausschliesslich als JSON laut Schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"booking_decision={decision or '(none)'}\n"
                    f"require_actionable={bool(require_actionable)}\n"
                    f"required_fields={json.dumps(req_fields, ensure_ascii=False)}\n"
                    f"missing_fields={json.dumps(miss_fields, ensure_ascii=False)}\n"
                    f"facts={json.dumps(ff, ensure_ascii=False)}\n"
                    f"knowledge_evidence={json.dumps(evidence[:20], ensure_ascii=False)}\n\n"
                    f"User message:\n{msg}\n\n"
                    f"Draft reply:\n{draft}"
                ),
            },
        ],
        response_format=schema,
        temperature=0.0,
        top_p=0.1,
        max_tokens=420,
    )

    parsed = _parse_json_obj(client.extract_text(completion))
    if not parsed:
        return _fallback_booking_reply_score(
            user_message=msg,
            draft_reply=draft,
            booking_decision=decision,
            missing_fields=miss_fields,
            require_actionable=bool(require_actionable),
        ) | {"model": getattr(client.cfg, "model", "")}

    dims_raw = parsed.get("dimensions") if isinstance(parsed.get("dimensions"), dict) else {}
    dimensions = {
        "booking_fit": round(_clamp01(dims_raw.get("booking_fit")), 3),
        "process_progress": round(_clamp01(dims_raw.get("process_progress")), 3),
        "groundedness": round(_clamp01(dims_raw.get("groundedness")), 3),
        "clarity": round(_clamp01(dims_raw.get("clarity")), 3),
        "tone": round(_clamp01(dims_raw.get("tone")), 3),
        "actionable": round(_clamp01(dims_raw.get("actionable")), 3),
    }
    total = round(_clamp01(parsed.get("total_score")), 3)
    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in {"send", "needs_review", "reject"}:
        verdict = "send" if total >= 0.8 else "needs_review" if total >= 0.6 else "reject"

    reasons = [str(x).strip() for x in (parsed.get("reasons") or []) if str(x).strip()][:5]
    improvements = [str(x).strip() for x in (parsed.get("improvements") or []) if str(x).strip()][:5]
    next_step = str(parsed.get("next_step") or "").strip()
    text = str(parsed.get("text") or "").strip() or f"Score={total:.2f}, verdict={verdict}"

    return {
        "total_score": total,
        "verdict": verdict,
        "dimensions": dimensions,
        "reasons": reasons,
        "improvements": improvements,
        "next_step": next_step,
        "text": text,
        "model": getattr(client.cfg, "model", ""),
        "fallback_used": False,
    }


def booking_instruction_check(
    *,
    instructions: List[str] | None,
    user_message: str,
    draft_reply: str,
    booking_decision: str = "",
    facts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    draft = str(draft_reply or "").strip()
    if not draft:
        raise HTTPException(status_code=422, detail="draft_reply is required")

    ins = [str(x).strip() for x in (instructions or []) if str(x).strip()]
    if not ins:
        return {
            "allowed": True,
            "confidence": 1.0,
            "risk_level": "low",
            "violations": [],
            "suggestions": [],
            "reason": "no_instructions",
            "text": "Keine Instruktionen hinterlegt.",
            "model": "",
            "fallback_used": True,
        }

    client = IonosLLM()
    if not client.enabled():
        return {
            "allowed": True,
            "confidence": 0.4,
            "risk_level": "medium",
            "violations": [],
            "suggestions": ["LLM nicht verfügbar: Instruktionsprüfung konnte nicht verlässlich durchgeführt werden."],
            "reason": "llm_unavailable",
            "text": "Instruktionsprüfung übersprungen (LLM nicht verfügbar).",
            "model": "",
            "fallback_used": True,
        }

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_instruction_check",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "allowed": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                    "violations": {"type": "array", "items": {"type": "string"}},
                    "suggestions": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["allowed", "confidence", "risk_level", "violations", "suggestions", "reason", "text"],
            },
            "strict": True,
        },
    }

    completion = client.chat_completions(
        messages=[
            {
                "role": "system",
                "content": (
                    "Du prüfst, ob eine geplante Booking-Antwort mit Freitext-Instruktionen vereinbar ist.\n"
                    "Bewerte NUR klare, operative Regeln aus den Instruktionen.\n"
                    "Beispiele für operative Regeln: keine Termine nach 16 Uhr, keine Montage, Öffnungszeiten nie automatisch beantworten.\n"
                    "Wenn ein klarer Widerspruch vorliegt, setze allowed=false und risk_level=high.\n"
                    "Wenn unklar oder mehrdeutig, setze allowed=true, aber risk_level=medium und gib Verbesserungsvorschläge.\n"
                    "Antworte ausschließlich als JSON laut Schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instructions:\n{json.dumps(ins, ensure_ascii=False)}\n\n"
                    f"booking_decision={str(booking_decision or '').strip().lower()}\n"
                    f"facts={json.dumps(dict(facts or {}), ensure_ascii=False)}\n\n"
                    f"User message:\n{str(user_message or '').strip()}\n\n"
                    f"Draft reply:\n{draft}"
                ),
            },
        ],
        response_format=schema,
        temperature=0.0,
        top_p=0.1,
        max_tokens=300,
    )

    parsed = _parse_json_obj(client.extract_text(completion))
    if not parsed:
        return {
            "allowed": True,
            "confidence": 0.4,
            "risk_level": "medium",
            "violations": [],
            "suggestions": ["LLM-Antwort konnte nicht geparst werden; bitte manuell prüfen."],
            "reason": "llm_parse_failed",
            "text": "Instruktionsprüfung unvollständig (Parse-Fehler).",
            "model": getattr(client.cfg, "model", ""),
            "fallback_used": True,
        }

    risk_level = str(parsed.get("risk_level") or "medium").strip().lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "medium"

    return {
        "allowed": bool(parsed.get("allowed")),
        "confidence": _clamp01(parsed.get("confidence")),
        "risk_level": risk_level,
        "violations": [str(x).strip() for x in (parsed.get("violations") or []) if str(x).strip()][:8],
        "suggestions": [str(x).strip() for x in (parsed.get("suggestions") or []) if str(x).strip()][:8],
        "reason": str(parsed.get("reason") or "").strip(),
        "text": str(parsed.get("text") or "").strip(),
        "model": getattr(client.cfg, "model", ""),
        "fallback_used": False,
    }


def _extract_price_confirmation(text: str) -> Dict[str, object]:
    client = IonosLLM()
    if not client.enabled():
        return {
            "confirmed": False,
            "confidence": 0.0,
            "reason": "llm_unavailable",
            "fallback_used": True,
            "model": "",
        }

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_price_confirmation",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "confirmed": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["confirmed", "confidence", "reason"],
            },
            "strict": True,
        },
    }
    completion = client.chat_completions(
        messages=[
            {
                "role": "system",
                "content": (
                    "Du entscheidest, ob der Kunde den zuvor kommunizierten Preis verbindlich bestätigt.\n"
                    "Gib confirmed=true nur bei klarer, expliziter Zustimmung zur Preis-/Angebotsannahme.\n"
                    "Bei Unklarheit oder fehlender Preiszusage: confirmed=false.\n"
                    "Antworte ausschließlich als JSON laut Schema."
                ),
            },
            {"role": "user", "content": f"Nachricht:\n{text}"},
        ],
        response_format=schema,
        temperature=0.0,
        top_p=0.1,
        max_tokens=120,
    )
    parsed = _parse_json_obj(client.extract_text(completion))
    if not parsed:
        return {
            "confirmed": False,
            "confidence": 0.0,
            "reason": "llm_parse_failed",
            "fallback_used": True,
            "model": getattr(client.cfg, "model", ""),
        }
    return {
        "confirmed": bool(parsed.get("confirmed")),
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence") or 0.0))),
        "reason": str(parsed.get("reason") or "").strip(),
        "fallback_used": False,
        "model": getattr(client.cfg, "model", ""),
    }


def _extract_dynamic_required_fields(text: str, required_fields: List[str]) -> Dict[str, object]:
    targets = [str(x).strip() for x in (required_fields or []) if str(x).strip()]
    if not targets:
        return {"values": {}, "meta": {}, "model": "", "fallback_used": True}

    client = IonosLLM()
    if not client.enabled():
        return {"values": {}, "meta": {}, "model": "", "fallback_used": True}

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "booking_dynamic_required_fields",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": ["string", "number", "boolean", "null"]},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "reason": {"type": "string"},
                            },
                            "required": ["name", "value", "confidence", "reason"],
                        },
                    }
                },
                "required": ["fields"],
            },
            "strict": True,
        },
    }
    completion = client.chat_completions(
        messages=[
            {
                "role": "system",
                "content": (
                    "Extrahiere die angefragten Pflichtfelder aus einer Booking-Nachricht.\n"
                    "Verwende nur die Felder aus der Liste.\n"
                    "Wenn ein Feld nicht sicher enthalten ist, setze value auf null.\n"
                    "Antworte ausschließlich als JSON laut Schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Felder: {', '.join(targets)}\n\n"
                    f"Nachricht:\n{text}"
                ),
            },
        ],
        response_format=schema,
        temperature=0.0,
        top_p=0.1,
        max_tokens=300,
    )
    parsed = _parse_json_obj(client.extract_text(completion))
    fields = parsed.get("fields") if isinstance(parsed.get("fields"), list) else []
    values: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    target_set = {x.lower(): x for x in targets}
    for item in fields:
        if not isinstance(item, dict):
            continue
        name_raw = str(item.get("name") or "").strip()
        if not name_raw:
            continue
        k = target_set.get(name_raw.lower())
        if not k:
            continue
        value = item.get("value")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                value = None
        if value is None:
            continue
        values[k] = value
        meta[k] = {
            "confidence": max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
            "reason": str(item.get("reason") or "").strip(),
        }
    return {
        "values": values,
        "meta": meta,
        "model": getattr(client.cfg, "model", ""),
        "fallback_used": False,
    }


def _is_missing_value(name: str, value: Any) -> bool:
    if name == "price_confirmed":
        return bool(value) is not True
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def booking_extract_facts(*, text: str, timezone_name: str = "Europe/Berlin", required_fields: List[str] | None = None) -> Dict[str, Any]:
    merged = str(text or "").strip()
    if not merged:
        raise HTTPException(status_code=422, detail="text is required")

    focused = _focus_booking_text(merged)

    event_date = _extract_date(focused) or _extract_date(merged)
    range_start, range_duration = _extract_time_window(focused)
    start_time = range_start or _extract_time(focused)
    duration_hours = range_duration if range_duration is not None else _extract_duration_hours(focused)
    location = _extract_location(focused)
    occasion = _extract_occasion(focused)
    client_eval = _extract_client_name(focused)
    client_name = str(client_eval.get("client_name") or "").strip()
    price_eval = _extract_price_confirmation(focused)
    price_confidence = float(price_eval.get("confidence") or 0.0)
    # Conservative gating to avoid false positives on short/ambiguous replies.
    price_confirmed = bool(price_eval.get("confirmed")) and price_confidence >= 0.75

    facts: Dict[str, Any] = {
        "event_date": event_date,
        "start_time": start_time,
        "duration_hours": duration_hours,
        "location": location,
        "occasion": occasion,
        "client_name": client_name,
        "client_name_confidence": float(client_eval.get("confidence") or 0.0),
        "client_name_reason": str(client_eval.get("reason") or "").strip(),
        "client_name_model": str(client_eval.get("model") or "").strip(),
        "price_confirmed": price_confirmed,
        "price_confirmed_confidence": price_confidence,
        "price_confirmed_reason": str(price_eval.get("reason") or "").strip(),
        "price_confirmed_model": str(price_eval.get("model") or "").strip(),
        "timezone": str(timezone_name or "Europe/Berlin"),
    }

    requested_required = [str(x).strip() for x in (required_fields or []) if str(x).strip()]
    static_fields = {
        "event_date",
        "start_time",
        "duration_hours",
        "location",
        "occasion",
        "client_name",
        "price_confirmed",
        "timezone",
    }
    dynamic_targets = [x for x in requested_required if x not in static_fields]
    dyn = _extract_dynamic_required_fields(focused, dynamic_targets)
    dyn_values = dyn.get("values") if isinstance(dyn.get("values"), dict) else {}
    dyn_meta = dyn.get("meta") if isinstance(dyn.get("meta"), dict) else {}
    for key, val in dyn_values.items():
        if key not in facts or _is_missing_value(key, facts.get(key)):
            facts[key] = val
        info = dyn_meta.get(key) if isinstance(dyn_meta.get(key), dict) else {}
        if info:
            facts[f"{key}_confidence"] = float(info.get("confidence") or 0.0)
            facts[f"{key}_reason"] = str(info.get("reason") or "").strip()
            facts[f"{key}_model"] = str(dyn.get("model") or "")

    required_eval = requested_required or [
        "event_date",
        "start_time",
        "duration_hours",
        "location",
        "occasion",
        "client_name",
        "price_confirmed",
    ]
    missing_candidates = [key for key in required_eval if _is_missing_value(key, facts.get(key))]
    total = max(1, len(required_eval))
    known = total - len(missing_candidates)
    confidence = max(0.0, min(1.0, known / total))
    return {
        "facts": facts,
        "confidence": confidence,
        "missing_candidates": missing_candidates,
        "text": f"Booking-Facts extrahiert (confidence={confidence:.2f})",
    }


def booking_validate_completeness(*, facts: Dict[str, Any], required_fields: List[str] | None = None) -> Dict[str, Any]:
    ff = dict(facts or {})
    required = [
        str(x).strip()
        for x in (required_fields or [
            "event_date",
            "start_time",
            "duration_hours",
            "location",
            "occasion",
            "client_name",
            "price_confirmed",
        ])
        if str(x).strip()
    ]

    missing: List[str] = []
    present: List[str] = []
    for name in required:
        val = ff.get(name)
        if name == "price_confirmed":
            if bool(val) is True:
                present.append(name)
            else:
                missing.append(name)
            continue
        if val is None:
            missing.append(name)
            continue
        if isinstance(val, str) and not val.strip():
            missing.append(name)
            continue
        present.append(name)

    complete = len(missing) == 0
    return {
        "complete": complete,
        "missing_fields": missing,
        "present_fields": present,
        "text": "Alle Pflichtfelder vorhanden." if complete else f"Es fehlen Angaben: {', '.join(missing)}",
    }


def _is_weekend(date_iso: str) -> bool | None:
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


def _hour_from_time(time_value: Any) -> int | None:
    raw = str(time_value or "").strip()
    if not raw:
        return None
    m = re.match(r"^([01]?\d|2[0-3])[:.]([0-5]\d)$", raw)
    if not m:
        return None
    return int(m.group(1))


def booking_decision_engine(
    *,
    facts: Dict[str, Any],
    profile_rules: Dict[str, Any] | None = None,
    completeness: Dict[str, Any] | None = None,
    distance: Dict[str, Any] | None = None,
    quote: Dict[str, Any] | None = None,
    require_price_confirmation: bool = True,
) -> Dict[str, Any]:
    ff = dict(facts or {})
    rules = dict(profile_rules or {})
    comp = dict(completeness or {})
    dist = dict(distance or {})
    q = dict(quote or {})

    reasons: List[str] = []
    flags: Dict[str, Any] = {}

    if not bool(comp.get("complete")):
        missing = [str(x).strip() for x in (comp.get("missing_fields") or []) if str(x).strip()]
        reasons.append("Pflichtangaben unvollständig.")
        if missing:
            reasons.append("Fehlend: " + ", ".join(missing))
        return {
            "decision": "need_clarification",
            "reasons": reasons,
            "flags": {"missing_fields": missing},
            "text": "Entscheidung: need_clarification",
        }

    if require_price_confirmation and not bool(ff.get("price_confirmed")):
        reasons.append("Preis ist noch nicht explizit bestätigt.")
        return {
            "decision": "need_clarification",
            "reasons": reasons,
            "flags": {},
            "text": "Entscheidung: need_clarification",
        }

    weekend_only = bool(rules.get("weekend_only", True))
    is_weekend = _is_weekend(str(ff.get("event_date") or ""))
    if weekend_only and is_weekend is False:
        reasons.append("Buchungen sind nur am Wochenende möglich.")
        return {
            "decision": "auto_decline",
            "reasons": reasons,
            "flags": {"weekend_only": True},
            "text": "Entscheidung: auto_decline",
        }

    duration = _to_float(ff.get("duration_hours"), 0.0)
    max_duration = _to_float(rules.get("max_duration_hours"), 8.0)
    if duration > max_duration:
        reasons.append(f"Anfrage überschreitet Maximaldauer ({duration:.1f}h > {max_duration:.1f}h).")
        return {
            "decision": "human_review",
            "reasons": reasons,
            "flags": {"duration_exceeded": True},
            "text": "Entscheidung: human_review",
        }

    distance_km = _to_float(dist.get("distance_km"), -1.0)
    max_distance = _to_float(rules.get("max_distance_km"), 200.0)
    if distance_km >= 0 and distance_km > max_distance:
        reasons.append(f"Entfernung zu groß ({distance_km:.1f} km > {max_distance:.1f} km).")
        return {
            "decision": "auto_decline",
            "reasons": reasons,
            "flags": {"distance_exceeded": True},
            "text": "Entscheidung: auto_decline",
        }

    overnight_distance = _to_float(rules.get("overnight_distance_km"), 60.0)
    overnight_after_hour = int(_to_float(rules.get("overnight_after_hour"), 22.0))
    start_hour = _hour_from_time(ff.get("start_time"))
    end_hour = None
    if start_hour is not None and duration > 0:
        end_hour = int((start_hour + duration) % 24)

    needs_overnight = bool(distance_km > overnight_distance and end_hour is not None and end_hour >= overnight_after_hour)
    flags["needs_overnight"] = needs_overnight
    if needs_overnight:
        overnight_included = bool(q.get("overnight_included", False))
        overnight_confirmed = bool(ff.get("overnight_confirmed", False))
        if not overnight_included or not overnight_confirmed:
            reasons.append("Übernachtungspauschale nötig, aber noch nicht bestätigt.")
            return {
                "decision": "need_clarification",
                "reasons": reasons,
                "flags": flags,
                "text": "Entscheidung: need_clarification",
            }

    reasons.append("Alle Regeln erfüllt, Anfrage kann angenommen werden.")
    return {
        "decision": "auto_accept",
        "reasons": reasons,
        "flags": flags,
        "text": "Entscheidung: auto_accept",
    }
