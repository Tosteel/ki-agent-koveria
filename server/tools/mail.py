from __future__ import annotations

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


def send_mail(
    *,
    to: List[str],
    subject: str,
    body: str,
    attachment_paths: List[str] | None = None,
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
    smtp_from = (from_email or os.getenv("SMTP_FROM", "")).strip()
    if not smtp_from:
        smtp_from = smtp_user
    if not smtp_from:
        raise HTTPException(status_code=500, detail="SMTP sender is missing (SMTP_FROM or SMTP_USERNAME)")

    use_ssl = _env_bool("SMTP_USE_SSL", False)
    use_starttls = _env_bool("SMTP_STARTTLS", True) and not use_ssl
    timeout_seconds = float(os.getenv("SMTP_TIMEOUT_SECONDS", "15").strip() or "15")

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
    for rel_path in (attachment_paths or []):
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

    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout_seconds)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=timeout_seconds)

        with server:
            if not use_ssl and use_starttls:
                server.starttls()
            if smtp_user:
                server.login(smtp_user, smtp_password)
            server.send_message(msg, from_addr=smtp_from, to_addrs=all_recipients)
    except smtplib.SMTPException as exc:
        raise HTTPException(status_code=502, detail=f"SMTP send failed: {exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"SMTP connection failed: {exc}") from exc

    message_id = (msg.get("Message-Id") or "").strip()
    return {
        "sent": True,
        "message_id": message_id,
        "recipients": all_recipients,
        "subject": msg["Subject"],
        "attachments": attachment_names,
    }
