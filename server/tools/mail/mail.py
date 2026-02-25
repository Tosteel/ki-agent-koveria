from __future__ import annotations

import email
import email.utils
import imaplib
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List

from fastapi import HTTPException


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
