from __future__ import annotations

import email
import email.utils
import html as html_lib
import imaplib
import io
import json
import mimetypes
import os
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List

from fastapi import HTTPException
from server.services.llm_ionos import IonosLLM


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _smtp_config() -> Dict[str, object]:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    if not smtp_host:
        raise HTTPException(status_code=500, detail="SMTP is not configured: SMTP_HOST is missing")

    smtp_port_raw = os.getenv("SMTP_PORT", "587").strip()
    try:
        smtp_port = int(smtp_port_raw)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="SMTP_PORT must be a number") from exc

    smtp_user = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", "").strip() or smtp_user
    if not smtp_from:
        raise HTTPException(status_code=500, detail="SMTP sender is missing (SMTP_FROM or SMTP_USERNAME)")

    use_ssl = _env_bool("SMTP_USE_SSL", False)
    use_starttls = _env_bool("SMTP_STARTTLS", True) and not use_ssl
    timeout_seconds = float(os.getenv("SMTP_TIMEOUT_SECONDS", "15").strip() or "15")
    return {
        "host": smtp_host,
        "port": smtp_port,
        "user": smtp_user,
        "password": smtp_password,
        "from": smtp_from,
        "use_ssl": use_ssl,
        "use_starttls": use_starttls,
        "timeout": timeout_seconds,
    }


def _send_smtp_message(msg: EmailMessage, *, from_addr: str, recipients: List[str], cfg: Dict[str, object]) -> None:
    try:
        if bool(cfg["use_ssl"]):
            server = smtplib.SMTP_SSL(str(cfg["host"]), int(cfg["port"]), timeout=float(cfg["timeout"]))
        else:
            server = smtplib.SMTP(str(cfg["host"]), int(cfg["port"]), timeout=float(cfg["timeout"]))

        with server:
            if not bool(cfg["use_ssl"]) and bool(cfg["use_starttls"]):
                server.starttls()
            if str(cfg["user"]):
                server.login(str(cfg["user"]), str(cfg["password"]))
            server.send_message(msg, from_addr=from_addr, to_addrs=recipients)
    except smtplib.SMTPException as exc:
        raise HTTPException(status_code=502, detail=f"SMTP send failed: {exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"SMTP connection failed: {exc}") from exc


def _decode_header_value(raw: str | None) -> str:
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    decoded: List[str] = []
    for value, enc in parts:
        if isinstance(value, bytes):
            try:
                decoded.append(value.decode(enc or "utf-8", errors="replace"))
            except Exception:
                decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    merged = "".join(decoded)
    # Prevent header injection / invalid headers caused by folded CRLF values.
    merged = merged.replace("\r", " ").replace("\n", " ")
    return " ".join(merged.split()).strip()


def _extract_text_snippet(msg: email.message.Message, max_len: int = 180) -> str:
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            if ctype == "text/plain":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                except Exception:
                    text = payload.decode("utf-8", errors="replace")
                if text.strip():
                    break
    else:
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except Exception:
            text = payload.decode("utf-8", errors="replace")

    out = " ".join((text or "").split()).strip()
    if len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "…"
    return out


def _extract_addresses(raw_header: str | None) -> List[str]:
    pairs = email.utils.getaddresses([raw_header or ""])
    out: List[str] = []
    for _, addr in pairs:
        a = (addr or "").strip()
        if a and a not in out:
            out.append(a)
    return out


def _decode_message_part(part: email.message.Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return str(raw or "")

    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except Exception:
        return payload.decode("utf-8", errors="replace")


def _normalize_text(value: str, *, max_chars: int) -> str:
    txt = str(value or "").strip()
    if max_chars > 0 and len(txt) > max_chars:
        return txt[: max_chars - 1].rstrip() + "…"
    return txt


def _html_to_text(html_raw: str) -> str:
    txt = str(html_raw or "")
    txt = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", txt)
    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
    txt = html_lib.unescape(txt)
    return " ".join(txt.split()).strip()


def _extract_mail_bodies(msg: email.message.Message, *, max_chars: int) -> tuple[str, str]:
    plain_parts: List[str] = []
    html_parts: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            ctype = (part.get_content_type() or "").lower()
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            if ctype == "text/plain":
                text = _decode_message_part(part)
                if text.strip():
                    plain_parts.append(text)
            elif ctype == "text/html":
                html_body = _decode_message_part(part)
                if html_body.strip():
                    html_parts.append(html_body)
    else:
        ctype = (msg.get_content_type() or "").lower()
        if ctype == "text/html":
            html_body = _decode_message_part(msg)
            if html_body.strip():
                html_parts.append(html_body)
        else:
            text = _decode_message_part(msg)
            if text.strip():
                plain_parts.append(text)

    body_text = _normalize_text("\n\n".join(plain_parts), max_chars=max_chars)
    body_html = _normalize_text("\n\n".join(html_parts), max_chars=max_chars)
    if not body_text and body_html:
        body_text = _normalize_text(_html_to_text(body_html), max_chars=max_chars)
    return body_text, body_html


def _attachment_info(msg: email.message.Message) -> tuple[bool, List[str]]:
    has_attachments = False
    names: List[str] = []
    if not msg.is_multipart():
        return has_attachments, names

    for part in msg.walk():
        if part.is_multipart():
            continue
        disp = (part.get("Content-Disposition") or "").lower()
        filename = _decode_header_value(part.get_filename())
        if "attachment" not in disp and not filename:
            continue
        has_attachments = True
        if filename and filename not in names:
            names.append(filename)
    return has_attachments, names


def _imap_fetch_message(*, mail_id: str, mailbox: str = "INBOX", readonly: bool = True) -> email.message.Message:
    mail_id_clean = (mail_id or "").strip()
    if not mail_id_clean:
        raise HTTPException(status_code=422, detail="mail_id is required")

    imap_host = os.getenv("IMAP_HOST", "").strip() or os.getenv("SMTP_HOST", "").strip()
    if not imap_host:
        raise HTTPException(status_code=500, detail="IMAP is not configured: IMAP_HOST is missing")

    imap_port_raw = os.getenv("IMAP_PORT", "").strip()
    if imap_port_raw:
        try:
            imap_port = int(imap_port_raw)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="IMAP_PORT must be a number") from exc
    else:
        imap_port = 993

    imap_user = os.getenv("IMAP_USERNAME", "").strip() or os.getenv("SMTP_USERNAME", "").strip()
    imap_password = os.getenv("IMAP_PASSWORD", "").strip() or os.getenv("SMTP_PASSWORD", "").strip()
    if not imap_user:
        raise HTTPException(status_code=500, detail="IMAP username is missing (IMAP_USERNAME/SMTP_USERNAME)")
    if not imap_password:
        raise HTTPException(status_code=500, detail="IMAP password is missing (IMAP_PASSWORD/SMTP_PASSWORD)")

    use_ssl = _env_bool("IMAP_USE_SSL", True)
    timeout_s = float(os.getenv("IMAP_TIMEOUT_SECONDS", "15").strip() or "15")
    mailbox_name = (mailbox or "INBOX").strip() or "INBOX"

    try:
        if use_ssl:
            client = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=timeout_s)
        else:
            client = imaplib.IMAP4(imap_host, imap_port, timeout=timeout_s)

        with client:
            client.login(imap_user, imap_password)
            status, _ = client.select(mailbox_name, readonly=readonly)
            if status != "OK":
                raise HTTPException(status_code=502, detail=f"IMAP mailbox not selectable: {mailbox_name}")

            status, msg_data = client.uid("FETCH", mail_id_clean, "(RFC822)")
            if status != "OK":
                raise HTTPException(status_code=502, detail=f"IMAP fetch failed for mail_id={mail_id_clean}")

            raw_msg = None
            for part in (msg_data or []):
                if isinstance(part, tuple) and len(part) >= 2:
                    raw_msg = part[1]
                    break
            if not raw_msg:
                raise HTTPException(status_code=404, detail=f"Mail not found: {mail_id_clean}")
            return email.message_from_bytes(raw_msg)

    except HTTPException:
        raise
    except (imaplib.IMAP4.error, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"IMAP read failed: {exc}") from exc


def _mail_payload_from_message(
    *,
    msg: email.message.Message,
    mailbox_name: str,
    mail_id_clean: str,
    include_html: bool,
    max_chars: int,
) -> Dict[str, object]:
    sender = _decode_header_value(msg.get("From"))
    subject = _decode_header_value(msg.get("Subject"))
    date_raw = _decode_header_value(msg.get("Date"))
    parsed_date = ""
    try:
        dt = email.utils.parsedate_to_datetime(date_raw)
        if dt is not None:
            parsed_date = dt.isoformat()
    except Exception:
        parsed_date = date_raw

    body_text, body_html = _extract_mail_bodies(msg, max_chars=max_chars)
    has_attachments, attachment_names = _attachment_info(msg)
    to_addrs = _extract_addresses(msg.get("To"))
    cc_addrs = _extract_addresses(msg.get("Cc"))

    lines = [
        f"Mailbox: {mailbox_name}",
        f"Mail ID: {mail_id_clean}",
        f"From: {sender}",
        f"Subject: {subject}",
        f"Date: {parsed_date or date_raw}",
        "",
    ]
    if body_text:
        lines.append(body_text)
    else:
        lines.append(_extract_text_snippet(msg, max_len=min(max_chars, 1000)))

    return {
        "mailbox": mailbox_name,
        "mail_id": mail_id_clean,
        "from_email": sender,
        "to": to_addrs,
        "cc": cc_addrs,
        "subject": subject,
        "date": parsed_date or date_raw,
        "message_id": _decode_header_value(msg.get("Message-Id")),
        "in_reply_to": _decode_header_value(msg.get("In-Reply-To")),
        "references": _decode_header_value(msg.get("References")),
        "has_attachments": has_attachments,
        "attachment_names": attachment_names,
        "body_text": body_text,
        "body_html": body_html if include_html else "",
        "text": "\n".join(lines).strip(),
    }


def _extract_ref_tokens(*headers: str) -> List[str]:
    out: List[str] = []
    for header in headers:
        raw = str(header or "").strip()
        if not raw:
            continue
        for token in re.findall(r"<[^>]+>", raw):
            t = token.strip()
            if t and t not in out:
                out.append(t)
    return out


def _mail_classification_heuristic(text: str) -> Dict[str, object]:
    content = str(text or "").lower()
    if any(
        marker in content
        for marker in (
            "mailings@",
            "noreply@",
            "no-reply@",
            "newsletter",
            "automatisch versendete nachricht",
            "antwort auf diese e-mail ist nicht möglich",
            "antwort ist nicht möglich",
        )
    ):
        return {
            "intent": "newsletter",
            "confidence": 0.92,
            "reason": "newsletter_sender_pattern",
            "fallback_used": True,
            "model": "",
        }
    buckets = {
        "newsletter": [
            "newsletter",
            "automatisch versendete nachricht",
            "antwort auf diese e-mail ist nicht möglich",
            "antwort ist nicht möglich",
            "noreply",
            "no-reply",
            "mailings@",
            "abbestellen",
            "unsubscribe",
            "impressum",
        ],
        "eskalation": [
            "eskal",
            "anwalt",
            "frist",
            "kündig",
            "schadensersatz",
            "compliance",
            "vorstand",
            "management",
            "sofort",
        ],
        "beschwerde": [
            "beschwerde",
            "reklamation",
            "unzufrieden",
            "problem",
            "fehler",
            "defekt",
            "enttäuscht",
            "ärger",
            "frustriert",
        ],
        "angebot": [
            "angebot",
            "preis",
            "kosten",
            "konditionen",
            "rabatt",
            "quote",
            "offerte",
            "budget",
        ],
        "termin": [
            "termin",
            "meeting",
            "besprechung",
            "call",
            "kalender",
            "uhr",
            "datum",
            "verfügbar",
            "verschieben",
        ],
        "info": ["info", "frage", "bitte", "anfrage", "auskunft", "details"],
    }

    best_intent = "info"
    best_hits = 0
    for intent, terms in buckets.items():
        hits = sum(1 for t in terms if t in content)
        if hits > best_hits:
            best_hits = hits
            best_intent = intent

    if best_hits >= 3:
        confidence = 0.9
    elif best_hits == 2:
        confidence = 0.78
    elif best_hits == 1:
        confidence = 0.64
    else:
        confidence = 0.5

    reason = "keyword_match" if best_hits > 0 else "default_info"
    return {
        "intent": best_intent,
        "confidence": confidence,
        "reason": reason,
        "fallback_used": True,
        "model": "",
    }


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


def _classify_mail_with_llm(text: str) -> Dict[str, object]:
    client = IonosLLM()
    if not client.enabled():
        return {}
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "mail_intent_classification",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["info", "beschwerde", "angebot", "termin", "eskalation", "newsletter"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["intent", "confidence", "reason"],
            },
            "strict": True,
        },
    }
    completion = client.chat_completions(
        messages=[
            {
                "role": "system",
                "content": (
                    "Du klassifizierst E-Mails in genau eine Absicht: "
                    "info, beschwerde, angebot, termin, eskalation, newsletter.\n"
                    "Nutze newsletter für Massenmails, Marketing-Mails, No-Reply-/Mailing-Absender "
                    "oder wenn im Text steht, dass Antworten nicht möglich sind.\n"
                    "Bewerte konservativ und gib nur JSON gemäß Schema zurück."
                ),
            },
            {"role": "user", "content": f"E-Mail-Inhalt:\n{text}"},
        ],
        response_format=schema,
        temperature=0.0,
        top_p=0.1,
        max_tokens=180,
    )
    parsed = _parse_json_obj(client.extract_text(completion))
    if not parsed:
        return {}
    return {
        "intent": str(parsed.get("intent") or "info").strip().lower(),
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence") or 0.0))),
        "reason": str(parsed.get("reason") or "").strip(),
        "fallback_used": False,
        "model": getattr(client.cfg, "model", ""),
    }


def classify_mail(
    *,
    text: str = "",
    subject: str = "",
    body_text: str = "",
    from_email: str = "",
) -> Dict[str, object]:
    merged = "\n".join(
        x
        for x in [
            f"From: {str(from_email or '').strip()}",
            f"Subject: {str(subject or '').strip()}",
            str(body_text or "").strip(),
            str(text or "").strip(),
        ]
        if x and str(x).strip()
    ).strip()
    if not merged:
        raise HTTPException(status_code=422, detail="text or subject/body_text is required")

    llm_out: Dict[str, object] = {}
    try:
        llm_out = _classify_mail_with_llm(merged)
    except Exception:
        llm_out = {}

    if not llm_out:
        llm_out = _mail_classification_heuristic(merged)

    intent = str(llm_out.get("intent") or "info").strip().lower()
    if intent not in {"info", "beschwerde", "angebot", "termin", "eskalation", "newsletter"}:
        intent = "info"
    confidence = float(llm_out.get("confidence") or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    reason = str(llm_out.get("reason") or "").strip()
    fallback_used = bool(llm_out.get("fallback_used"))
    model = str(llm_out.get("model") or "").strip()
    return {
        "intent": intent,
        "confidence": confidence,
        "reason": reason,
        "fallback_used": fallback_used,
        "model": model,
        "text": f"Absicht: {intent} (confidence={confidence:.2f})",
    }


def _mark_answered_flag(*, mail_id: str, mailbox: str) -> bool:
    imap_host = os.getenv("IMAP_HOST", "").strip() or os.getenv("SMTP_HOST", "").strip()
    if not imap_host:
        raise HTTPException(status_code=500, detail="IMAP is not configured: IMAP_HOST is missing")
    try:
        imap_port = int((os.getenv("IMAP_PORT", "").strip() or "993"))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="IMAP_PORT must be a number") from exc

    imap_user = os.getenv("IMAP_USERNAME", "").strip() or os.getenv("SMTP_USERNAME", "").strip()
    imap_password = os.getenv("IMAP_PASSWORD", "").strip() or os.getenv("SMTP_PASSWORD", "").strip()
    if not imap_user or not imap_password:
        raise HTTPException(status_code=500, detail="IMAP credentials are missing")

    use_ssl = _env_bool("IMAP_USE_SSL", True)
    timeout_s = float(os.getenv("IMAP_TIMEOUT_SECONDS", "15").strip() or "15")
    mailbox_name = (mailbox or "INBOX").strip() or "INBOX"

    try:
        if use_ssl:
            client = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=timeout_s)
        else:
            client = imaplib.IMAP4(imap_host, imap_port, timeout=timeout_s)
        with client:
            client.login(imap_user, imap_password)
            status, _ = client.select(mailbox_name, readonly=False)
            if status != "OK":
                raise HTTPException(status_code=502, detail=f"IMAP mailbox not selectable: {mailbox_name}")
            status, _ = client.uid("STORE", mail_id, "+FLAGS", r"(\Answered \Seen)")
            if status != "OK":
                raise HTTPException(status_code=502, detail=f"IMAP STORE failed for mail_id={mail_id}")
    except HTTPException:
        raise
    except (imaplib.IMAP4.error, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"IMAP flag update failed: {exc}") from exc
    return True


def send_mail(
    *,
    to: List[str],
    subject: str,
    body: str,
    attachments: List[str] | None = None,
    work_dir: Path | None = None,
    cc: List[str] | None = None,
    bcc: List[str] | None = None,
    from_email: str | None = None,
    reply_to: str | None = None,
    is_html: bool = False,
) -> Dict[str, object]:
    recipients = [x.strip() for x in (to or []) if isinstance(x, str) and x.strip()]
    cc_clean = [x.strip() for x in (cc or []) if isinstance(x, str) and x.strip()]
    bcc_clean = [x.strip() for x in (bcc or []) if isinstance(x, str) and x.strip()]

    if not recipients:
        raise HTTPException(status_code=422, detail="At least one recipient is required")

    cfg = _smtp_config()
    smtp_from = (from_email or "").strip() or str(cfg["from"])
    if not smtp_from:
        raise HTTPException(status_code=500, detail="SMTP sender is missing (SMTP_FROM or SMTP_USERNAME)")

    msg = EmailMessage()
    msg["From"] = smtp_from
    msg["To"] = ", ".join(recipients)
    if cc_clean:
        msg["Cc"] = ", ".join(cc_clean)
    if reply_to and reply_to.strip():
        msg["Reply-To"] = reply_to.strip()
    msg["Subject"] = subject.strip()
    if is_html:
        msg.add_alternative(body or "", subtype="html")
    else:
        msg.set_content(body or "")

    attachment_names: List[str] = []
    for rel_path in (attachments or []):
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue
        if work_dir is None:
            raise HTTPException(status_code=400, detail="Attachments require a user work directory")

        rel = Path(rel_path.strip().lstrip("/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise HTTPException(status_code=400, detail=f"Invalid attachment path: {rel_path}")

        base = work_dir.resolve()
        file_path = (base / rel).resolve()
        if base not in file_path.parents and file_path != base:
            raise HTTPException(status_code=400, detail=f"Invalid attachment path: {rel_path}")
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"Attachment not found: {rel_path}")

        data = file_path.read_bytes()
        ctype, _ = mimetypes.guess_type(file_path.name)
        if ctype:
            maintype, subtype = ctype.split("/", 1)
        else:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=file_path.name)
        attachment_names.append(file_path.name)

    all_recipients = recipients + cc_clean + bcc_clean

    _send_smtp_message(msg, from_addr=smtp_from, recipients=all_recipients, cfg=cfg)

    message_id = (msg.get("Message-Id") or "").strip()
    return {
        "sent": True,
        "message_id": message_id,
        "to": all_recipients,
        "subject": msg["Subject"],
        "attachments": attachment_names,
    }


def answer_mail(
    *,
    mail_id: str,
    body: str,
    mailbox: str = "INBOX",
    subject: str = "",
    reply_to_all: bool = False,
    is_html: bool = False,
) -> Dict[str, object]:
    mail_id_clean = (mail_id or "").strip()
    if not mail_id_clean:
        raise HTTPException(status_code=422, detail="mail_id is required")
    body_text = (body or "").strip()
    if not body_text:
        raise HTTPException(status_code=422, detail="body is required")

    imap_host = os.getenv("IMAP_HOST", "").strip() or os.getenv("SMTP_HOST", "").strip()
    if not imap_host:
        raise HTTPException(status_code=500, detail="IMAP is not configured: IMAP_HOST is missing")
    imap_port = int((os.getenv("IMAP_PORT", "").strip() or "993"))
    imap_user = os.getenv("IMAP_USERNAME", "").strip() or os.getenv("SMTP_USERNAME", "").strip()
    imap_password = os.getenv("IMAP_PASSWORD", "").strip() or os.getenv("SMTP_PASSWORD", "").strip()
    if not imap_user or not imap_password:
        raise HTTPException(status_code=500, detail="IMAP credentials are missing")
    use_ssl = _env_bool("IMAP_USE_SSL", True)
    timeout_s = float(os.getenv("IMAP_TIMEOUT_SECONDS", "15").strip() or "15")

    original = None
    mailbox_name = (mailbox or "INBOX").strip() or "INBOX"
    try:
        if use_ssl:
            client = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=timeout_s)
        else:
            client = imaplib.IMAP4(imap_host, imap_port, timeout=timeout_s)
        with client:
            client.login(imap_user, imap_password)
            status, _ = client.select(mailbox_name, readonly=True)
            if status != "OK":
                raise HTTPException(status_code=502, detail=f"IMAP mailbox not selectable: {mailbox_name}")
            status, msg_data = client.uid("FETCH", mail_id_clean, "(RFC822)")
            if status != "OK" or not msg_data:
                raise HTTPException(status_code=404, detail=f"Mail not found: {mail_id_clean}")
            raw_msg = None
            for part in msg_data:
                if isinstance(part, tuple) and len(part) >= 2:
                    raw_msg = part[1]
                    break
            if not raw_msg:
                raise HTTPException(status_code=404, detail=f"Mail not found: {mail_id_clean}")
            original = email.message_from_bytes(raw_msg)
    except HTTPException:
        raise
    except (imaplib.IMAP4.error, OSError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"IMAP fetch failed: {exc}") from exc

    from_addr = str(_smtp_config()["from"])
    to_list = _extract_addresses(original.get("Reply-To")) or _extract_addresses(original.get("From"))
    if not to_list:
        raise HTTPException(status_code=422, detail="Original mail has no reply address")
    recipients = list(to_list)
    if reply_to_all:
        cc_from_orig = _extract_addresses(original.get("Cc"))
        for addr in cc_from_orig:
            if addr not in recipients and addr.lower() != from_addr.lower():
                recipients.append(addr)

    orig_subject = _decode_header_value(original.get("Subject"))
    final_subject = (subject or "").strip()
    if not final_subject:
        final_subject = orig_subject or "(ohne Betreff)"
        if not final_subject.lower().startswith("re:"):
            final_subject = f"Re: {final_subject}"

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_list)
    if reply_to_all:
        cc_addrs = [a for a in recipients if a not in to_list]
        if cc_addrs:
            msg["Cc"] = ", ".join(cc_addrs)
    msg["Subject"] = final_subject

    orig_mid = _decode_header_value(original.get("Message-Id"))
    orig_refs = _decode_header_value(original.get("References"))
    if orig_mid:
        msg["In-Reply-To"] = orig_mid
        refs = f"{orig_refs} {orig_mid}".strip() if orig_refs else orig_mid
        msg["References"] = refs

    if is_html:
        msg.add_alternative(body_text, subtype="html")
    else:
        msg.set_content(body_text)

    cfg = _smtp_config()
    _send_smtp_message(msg, from_addr=from_addr, recipients=recipients, cfg=cfg)
    marked_answered = _mark_answered_flag(mail_id=mail_id_clean, mailbox=mailbox_name)

    return {
        "sent": True,
        "message_id": (msg.get("Message-Id") or "").strip(),
        "in_reply_to": orig_mid,
        "to": recipients,
        "subject": final_subject,
        "marked_answered": marked_answered,
    }


def fetch_inbox_mails(
    *,
    limit: int = 10,
    mailbox: str = "INBOX",
    unread_only: bool = False,
) -> Dict[str, object]:
    query = "UNSEEN" if unread_only else "ALL"
    return _fetch_mails_by_query(limit=limit, mailbox=mailbox, query=query)


def fetch_unanswered_mails(
    *,
    limit: int = 10,
    mailbox: str = "INBOX",
) -> Dict[str, object]:
    return _fetch_mails_by_query(limit=limit, mailbox=mailbox, query="UNANSWERED")


def read_mail(
    *,
    mail_id: str,
    mailbox: str = "INBOX",
    include_html: bool = False,
    max_chars: int = 20000,
) -> Dict[str, object]:
    mail_id_clean = (mail_id or "").strip()
    if not mail_id_clean:
        raise HTTPException(status_code=422, detail="mail_id is required")
    mailbox_name = (mailbox or "INBOX").strip() or "INBOX"
    max_chars = max(500, min(int(max_chars), 200000))
    msg = _imap_fetch_message(mail_id=mail_id_clean, mailbox=mailbox_name, readonly=True)
    return _mail_payload_from_message(
        msg=msg,
        mailbox_name=mailbox_name,
        mail_id_clean=mail_id_clean,
        include_html=include_html,
        max_chars=max_chars,
    )


def read_mail_thread(
    *,
    mail_id: str,
    mailbox: str = "INBOX",
    max_messages: int = 20,
    include_html: bool = False,
    max_chars: int = 8000,
) -> Dict[str, object]:
    mail_id_clean = (mail_id or "").strip()
    if not mail_id_clean:
        raise HTTPException(status_code=422, detail="mail_id is required")

    mailbox_name = (mailbox or "INBOX").strip() or "INBOX"
    max_messages = max(1, min(int(max_messages), 100))
    max_chars = max(500, min(int(max_chars), 200000))

    root = read_mail(
        mail_id=mail_id_clean,
        mailbox=mailbox_name,
        include_html=include_html,
        max_chars=max_chars,
    )

    tokens = _extract_ref_tokens(
        str(root.get("message_id") or ""),
        str(root.get("in_reply_to") or ""),
        str(root.get("references") or ""),
    )

    thread_ids: List[str] = [mail_id_clean]
    imap_host = os.getenv("IMAP_HOST", "").strip() or os.getenv("SMTP_HOST", "").strip()
    if imap_host and tokens:
        try:
            imap_port = int((os.getenv("IMAP_PORT", "").strip() or "993"))
        except ValueError:
            imap_port = 993
        imap_user = os.getenv("IMAP_USERNAME", "").strip() or os.getenv("SMTP_USERNAME", "").strip()
        imap_password = os.getenv("IMAP_PASSWORD", "").strip() or os.getenv("SMTP_PASSWORD", "").strip()
        use_ssl = _env_bool("IMAP_USE_SSL", True)
        timeout_s = float(os.getenv("IMAP_TIMEOUT_SECONDS", "15").strip() or "15")
        if imap_user and imap_password:
            try:
                if use_ssl:
                    client = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=timeout_s)
                else:
                    client = imaplib.IMAP4(imap_host, imap_port, timeout=timeout_s)
                with client:
                    client.login(imap_user, imap_password)
                    status, _ = client.select(mailbox_name, readonly=True)
                    if status == "OK":
                        for token in tokens[:30]:
                            for header_name in ("MESSAGE-ID", "IN-REPLY-TO", "REFERENCES"):
                                status, data = client.uid("SEARCH", None, "HEADER", header_name, token)
                                if status != "OK" or not data or not data[0]:
                                    continue
                                for raw_uid in data[0].split():
                                    uid = raw_uid.decode(errors="ignore").strip()
                                    if uid and uid not in thread_ids:
                                        thread_ids.append(uid)
            except Exception:
                pass

    messages: List[Dict[str, object]] = []
    for uid in thread_ids[:max_messages]:
        try:
            msg_payload = read_mail(
                mail_id=uid,
                mailbox=mailbox_name,
                include_html=include_html,
                max_chars=max_chars,
            )
            messages.append(msg_payload)
        except Exception:
            continue

    def _sort_key(item: Dict[str, object]) -> str:
        return str(item.get("date") or "")

    messages.sort(key=_sort_key)
    lines = [f"Mailbox: {mailbox_name}", f"Thread for mail_id={mail_id_clean}", f"Messages: {len(messages)}", ""]
    for i, item in enumerate(messages, start=1):
        lines.append(f"[{i}] {str(item.get('subject') or '(ohne Betreff)')}")
        lines.append(f"from={str(item.get('from_email') or '')} date={str(item.get('date') or '')}")
        snippet = str(item.get("body_text") or "").strip()
        if snippet:
            lines.append(snippet[:300] + ("…" if len(snippet) > 300 else ""))
        lines.append("")

    return {
        "mailbox": mailbox_name,
        "mail_id": mail_id_clean,
        "count": len(messages),
        "messages": messages,
        "text": "\n".join(lines).strip(),
    }


def read_mail_attachments(
    *,
    mail_id: str,
    mailbox: str = "INBOX",
    include_content: bool = False,
    max_attachment_chars: int = 4000,
    extract_text_pdf: bool = True,
) -> Dict[str, object]:
    mail_id_clean = (mail_id or "").strip()
    if not mail_id_clean:
        raise HTTPException(status_code=422, detail="mail_id is required")
    mailbox_name = (mailbox or "INBOX").strip() or "INBOX"
    max_attachment_chars = max(200, min(int(max_attachment_chars), 50000))

    msg = _imap_fetch_message(mail_id=mail_id_clean, mailbox=mailbox_name, readonly=True)
    attachments: List[Dict[str, object]] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        disp = (part.get("Content-Disposition") or "").lower()
        filename = _decode_header_value(part.get_filename())
        if "attachment" not in disp and not filename:
            continue
        payload = part.get_payload(decode=True) or b""
        ctype = str(part.get_content_type() or "application/octet-stream")
        text_content = ""
        if include_content:
            if ctype.startswith("text/"):
                text_content = _normalize_text(_decode_message_part(part), max_chars=max_attachment_chars)
            elif ctype == "application/pdf" and extract_text_pdf:
                try:
                    from pypdf import PdfReader  # type: ignore

                    reader = PdfReader(io.BytesIO(payload))
                    pages: List[str] = []
                    for p in reader.pages:
                        pages.append(str(p.extract_text() or ""))
                    text_content = _normalize_text("\n".join(pages).strip(), max_chars=max_attachment_chars)
                except Exception:
                    text_content = ""

        attachments.append(
            {
                "filename": filename,
                "content_type": ctype,
                "size_bytes": len(payload),
                "text": text_content,
            }
        )

    lines = [
        f"Mailbox: {mailbox_name}",
        f"Mail ID: {mail_id_clean}",
        f"Attachments: {len(attachments)}",
        "",
    ]
    for i, att in enumerate(attachments, start=1):
        lines.append(
            f"[{i}] {str(att.get('filename') or '(ohne Name)')} "
            f"type={str(att.get('content_type') or '')} bytes={int(att.get('size_bytes') or 0)}"
        )
        txt = str(att.get("text") or "").strip()
        if txt:
            lines.append(txt[:300] + ("…" if len(txt) > 300 else ""))
        lines.append("")

    return {
        "mailbox": mailbox_name,
        "mail_id": mail_id_clean,
        "count": len(attachments),
        "has_attachments": len(attachments) > 0,
        "attachments": attachments,
        "text": "\n".join(lines).strip(),
    }


def _fetch_mails_by_query(
    *,
    limit: int,
    mailbox: str,
    query: str,
) -> Dict[str, object]:
    imap_host = os.getenv("IMAP_HOST", "").strip() or os.getenv("SMTP_HOST", "").strip()
    if not imap_host:
        raise HTTPException(status_code=500, detail="IMAP is not configured: IMAP_HOST is missing")

    imap_port_raw = os.getenv("IMAP_PORT", "").strip()
    if imap_port_raw:
        try:
            imap_port = int(imap_port_raw)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="IMAP_PORT must be a number") from exc
    else:
        imap_port = 993

    imap_user = os.getenv("IMAP_USERNAME", "").strip() or os.getenv("SMTP_USERNAME", "").strip()
    imap_password = os.getenv("IMAP_PASSWORD", "").strip() or os.getenv("SMTP_PASSWORD", "").strip()
    if not imap_user:
        raise HTTPException(status_code=500, detail="IMAP username is missing (IMAP_USERNAME/SMTP_USERNAME)")
    if not imap_password:
        raise HTTPException(status_code=500, detail="IMAP password is missing (IMAP_PASSWORD/SMTP_PASSWORD)")

    use_ssl = _env_bool("IMAP_USE_SSL", True)
    timeout_s = float(os.getenv("IMAP_TIMEOUT_SECONDS", "15").strip() or "15")
    limit = max(1, min(int(limit), 50))
    mailbox_name = (mailbox or "INBOX").strip() or "INBOX"

    try:
        if use_ssl:
            client = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=timeout_s)
        else:
            client = imaplib.IMAP4(imap_host, imap_port, timeout=timeout_s)

        with client:
            client.login(imap_user, imap_password)
            status, _ = client.select(mailbox_name, readonly=True)
            if status != "OK":
                raise HTTPException(status_code=502, detail=f"IMAP mailbox not selectable: {mailbox_name}")

            status, data = client.uid("SEARCH", None, query)
            if status != "OK":
                raise HTTPException(status_code=502, detail="IMAP search failed")

            ids = data[0].split() if data and data[0] else []
            last_ids = ids[-limit:]
            emails: List[Dict[str, str]] = []

            for msg_id in reversed(last_ids):
                status, msg_data = client.uid("FETCH", msg_id, "(RFC822)")
                if status != "OK" or not msg_data:
                    continue
                raw_msg = None
                for part in msg_data:
                    if isinstance(part, tuple) and len(part) >= 2:
                        raw_msg = part[1]
                        break
                if not raw_msg:
                    continue

                msg = email.message_from_bytes(raw_msg)
                sender = _decode_header_value(msg.get("From"))
                subject = _decode_header_value(msg.get("Subject"))
                date_raw = _decode_header_value(msg.get("Date"))
                snippet = _extract_text_snippet(msg)

                parsed_date = ""
                try:
                    dt = email.utils.parsedate_to_datetime(date_raw)
                    if dt is not None:
                        parsed_date = dt.isoformat()
                except Exception:
                    parsed_date = date_raw

                emails.append(
                    {
                        "uid": msg_id.decode(errors="ignore"),
                        "from_email": sender,
                        "subject": subject,
                        "date": parsed_date or date_raw,
                        "snippet": snippet,
                    }
                )

    except HTTPException:
        raise
    except (imaplib.IMAP4.error, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"IMAP fetch failed: {exc}") from exc

    lines = [f"Mailbox: {mailbox_name}", f"Query: {query}", f"E-Mails: {len(emails)}", ""]
    for i, item in enumerate(emails, start=1):
        lines.append(f"[{i}] {item.get('subject') or '(ohne Betreff)'}")
        lines.append(f"von={item.get('from_email', '')} datum={item.get('date', '')}")
        if item.get("snippet"):
            lines.append(item["snippet"])
        lines.append("")

    return {
        "mailbox": mailbox_name,
        "count": len(emails),
        "mail_id": (emails[0].get("uid", "") if emails else ""),
        "emails": emails,
        "text": "\n".join(lines).strip(),
    }
