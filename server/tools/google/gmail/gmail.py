from __future__ import annotations

import base64
import os
from email.message import EmailMessage
from email.utils import getaddresses
from typing import Any, Dict, List, Tuple

import requests
from fastapi import HTTPException

_TOKEN_CACHE: Dict[str, str] = {"access_token": ""}


def _normalize_text(value: str, *, max_chars: int) -> str:
    txt = str(value or "").strip()
    if max_chars > 0 and len(txt) > max_chars:
        return txt[: max_chars - 1].rstrip() + "..."
    return txt


def _extract_addresses(raw_header: str | None) -> List[str]:
    pairs = getaddresses([raw_header or ""])
    out: List[str] = []
    for _, addr in pairs:
        a = (addr or "").strip()
        if a and a not in out:
            out.append(a)
    return out


def _refresh_access_token() -> str:
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    refresh_token = os.getenv("GOOGLE_OAUTH_REFRESH_TOKEN", "").strip()
    if not client_id or not client_secret or not refresh_token:
        return ""

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    try:
        resp = requests.post("https://oauth2.googleapis.com/token", data=payload, timeout=20)
    except requests.RequestException:
        return ""
    if resp.status_code >= 400:
        return ""
    try:
        data = resp.json()
    except Exception:
        return ""
    token = str(data.get("access_token") or "").strip()
    if token:
        _TOKEN_CACHE["access_token"] = token
    return token


def _resolve_access_token() -> str:
    cached = str(_TOKEN_CACHE.get("access_token") or "").strip()
    if cached:
        return cached
    direct = os.getenv("GOOGLE_ACCESS_TOKEN", "").strip()
    if direct:
        _TOKEN_CACHE["access_token"] = direct
        return direct
    return _refresh_access_token()


def _gmail_request(
    *,
    method: str,
    path: str,
    params: Dict[str, Any] | None = None,
    json_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    token = _resolve_access_token()
    if not token:
        raise HTTPException(
            status_code=500,
            detail=(
                "Google OAuth token is missing. "
                "Set GOOGLE_ACCESS_TOKEN or configure refresh via "
                "GOOGLE_OAUTH_CLIENT_ID/GOOGLE_OAUTH_CLIENT_SECRET/GOOGLE_OAUTH_REFRESH_TOKEN."
            ),
        )
    base_url = "https://gmail.googleapis.com/gmail/v1/users/me"
    url = f"{base_url}{path}"

    def _do_request(token_value: str) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {token_value}",
            "Content-Type": "application/json",
        }
        return requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params or None,
            json=json_payload or None,
            timeout=30,
        )

    try:
        resp = _do_request(token)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Gmail request failed: {exc}") from exc

    if resp.status_code == 401:
        refreshed = _refresh_access_token()
        if refreshed:
            try:
                resp = _do_request(refreshed)
            except requests.RequestException as exc:
                raise HTTPException(status_code=502, detail=f"Gmail request failed: {exc}") from exc

    if resp.status_code >= 400:
        detail = ""
        try:
            detail = str(resp.json())
        except Exception:
            detail = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"Gmail API error: {detail}")

    try:
        data = resp.json()
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def _mark_message_read(mail_id: str) -> None:
    mid = str(mail_id or "").strip()
    if not mid:
        return
    try:
        _gmail_request(
            method="POST",
            path=f"/messages/{mid}/modify",
            json_payload={"removeLabelIds": ["UNREAD"]},
        )
    except Exception:
        # Non-fatal: sending a reply succeeded even if label update fails.
        return


def _headers_map(msg_payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    headers = msg_payload.get("headers") if isinstance(msg_payload.get("headers"), list) else []
    for item in headers:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        val = str(item.get("value") or "").strip()
        if name:
            out[name.lower()] = val
    return out


def _decode_b64url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    padding = "=" * ((4 - len(raw) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(raw + padding)
        return decoded.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_bodies(part: Dict[str, Any]) -> Tuple[str, str]:
    mime = str(part.get("mimeType") or "").lower()
    body = part.get("body") if isinstance(part.get("body"), dict) else {}
    data = str(body.get("data") or "")

    plain = ""
    html = ""
    if mime == "text/plain":
        plain = _decode_b64url(data)
    elif mime == "text/html":
        html = _decode_b64url(data)

    parts = part.get("parts") if isinstance(part.get("parts"), list) else []
    for child in parts:
        if not isinstance(child, dict):
            continue
        p, h = _extract_bodies(child)
        if p:
            plain = (plain + "\n\n" + p).strip() if plain else p
        if h:
            html = (html + "\n\n" + h).strip() if html else h
    return plain, html


def _extract_attachments(part: Dict[str, Any], out: List[str]) -> None:
    filename = str(part.get("filename") or "").strip()
    body = part.get("body") if isinstance(part.get("body"), dict) else {}
    attachment_id = str(body.get("attachmentId") or "").strip()
    if filename and attachment_id and filename not in out:
        out.append(filename)
    parts = part.get("parts") if isinstance(part.get("parts"), list) else []
    for child in parts:
        if isinstance(child, dict):
            _extract_attachments(child, out)


def _mail_payload_to_response(
    *,
    message: Dict[str, Any],
    mailbox: str,
    include_html: bool,
    max_chars: int,
) -> Dict[str, Any]:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    headers = _headers_map(payload)

    body_text, body_html = _extract_bodies(payload)
    body_text = _normalize_text(body_text, max_chars=max_chars)
    body_html = _normalize_text(body_html, max_chars=max_chars)

    attachments: List[str] = []
    _extract_attachments(payload, attachments)
    has_attachments = len(attachments) > 0

    from_email = headers.get("from", "")
    to = _extract_addresses(headers.get("to"))
    cc = _extract_addresses(headers.get("cc"))
    subject = headers.get("subject", "")
    date = headers.get("date", "")
    message_id = headers.get("message-id", "")
    in_reply_to = headers.get("in-reply-to", "")
    references = headers.get("references", "")

    lines = [
        f"Mailbox: {mailbox}",
        f"Mail ID: {str(message.get('id') or '')}",
        f"From: {from_email}",
        f"Subject: {subject}",
        f"Date: {date}",
        "",
        body_text,
    ]
    text = "\n".join(x for x in lines if x is not None).strip()

    return {
        "mailbox": mailbox,
        "mail_id": str(message.get("id") or ""),
        "thread_id": str(message.get("threadId") or ""),
        "from_email": from_email,
        "to": to,
        "cc": cc,
        "subject": subject,
        "date": date,
        "message_id": message_id,
        "in_reply_to": in_reply_to,
        "references": references,
        "has_attachments": has_attachments,
        "attachment_names": attachments,
        "body_text": body_text,
        "body_html": body_html if include_html else "",
        "text": text,
    }


def _gmail_search_query(*, mailbox: str, unread_only: bool, unanswered_mode: bool) -> str:
    in_box = f"in:{(mailbox or 'INBOX').lower()}"
    filters = [in_box]
    if unread_only or unanswered_mode:
        filters.append("is:unread")
    if unanswered_mode:
        filters.append("-from:me")
    return " ".join(filters).strip()


def fetch_inbox_mails(*, limit: int = 10, mailbox: str = "INBOX", unread_only: bool = False) -> Dict[str, Any]:
    q = _gmail_search_query(mailbox=mailbox, unread_only=unread_only, unanswered_mode=False)
    data = _gmail_request(
        method="GET",
        path="/messages",
        params={"q": q, "maxResults": max(1, min(int(limit), 50))},
    )
    msgs = data.get("messages") if isinstance(data.get("messages"), list) else []
    out_items: List[Dict[str, str]] = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        mid = str(msg.get("id") or "").strip()
        if not mid:
            continue
        detail = _gmail_request(method="GET", path=f"/messages/{mid}", params={"format": "metadata"})
        payload = detail.get("payload") if isinstance(detail.get("payload"), dict) else {}
        headers = _headers_map(payload)
        out_items.append(
            {
                "uid": mid,
                "from_email": headers.get("from", ""),
                "subject": headers.get("subject", ""),
                "date": headers.get("date", ""),
                "snippet": str(detail.get("snippet") or ""),
            }
        )

    lines = [f"Mailbox: {mailbox}", f"Count: {len(out_items)}", ""]
    for i, item in enumerate(out_items, start=1):
        lines.append(f"[{i}] {item['subject']} | from={item['from_email']}")
    return {
        "mailbox": mailbox,
        "count": len(out_items),
        "mail_id": "",
        "emails": out_items,
        "text": "\n".join(lines).strip(),
    }


def fetch_unanswered_mails(*, limit: int = 10, mailbox: str = "INBOX") -> Dict[str, Any]:
    q = _gmail_search_query(mailbox=mailbox, unread_only=True, unanswered_mode=True)
    data = _gmail_request(
        method="GET",
        path="/messages",
        params={"q": q, "maxResults": max(1, min(int(limit), 50))},
    )
    msgs = data.get("messages") if isinstance(data.get("messages"), list) else []
    out_items: List[Dict[str, str]] = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        mid = str(msg.get("id") or "").strip()
        if not mid:
            continue
        detail = _gmail_request(method="GET", path=f"/messages/{mid}", params={"format": "metadata"})
        payload = detail.get("payload") if isinstance(detail.get("payload"), dict) else {}
        headers = _headers_map(payload)
        out_items.append(
            {
                "uid": mid,
                "from_email": headers.get("from", ""),
                "subject": headers.get("subject", ""),
                "date": headers.get("date", ""),
                "snippet": str(detail.get("snippet") or ""),
            }
        )
    lines = [f"Mailbox: {mailbox}", f"Unanswered Count: {len(out_items)}", ""]
    for i, item in enumerate(out_items, start=1):
        lines.append(f"[{i}] {item['subject']} | from={item['from_email']}")
    return {
        "mailbox": mailbox,
        "count": len(out_items),
        "mail_id": "",
        "emails": out_items,
        "text": "\n".join(lines).strip(),
    }


def read_mail(*, mail_id: str, mailbox: str = "INBOX", include_html: bool = False, max_chars: int = 20000) -> Dict[str, Any]:
    mid = str(mail_id or "").strip()
    if not mid:
        raise HTTPException(status_code=422, detail="mail_id is required")
    detail = _gmail_request(method="GET", path=f"/messages/{mid}", params={"format": "full"})
    return _mail_payload_to_response(
        message=detail,
        mailbox=mailbox,
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
) -> Dict[str, Any]:
    root = read_mail(mail_id=mail_id, mailbox=mailbox, include_html=include_html, max_chars=max_chars)
    thread_id = str(root.get("thread_id") or "").strip()
    if not thread_id:
        return {
            "mailbox": mailbox,
            "mail_id": mail_id,
            "thread_id": "",
            "count": 1,
            "messages": [
                {
                    "mail_id": str(root.get("mail_id") or ""),
                    "thread_id": "",
                    "from_email": str(root.get("from_email") or ""),
                    "to": list(root.get("to") or []),
                    "cc": list(root.get("cc") or []),
                    "subject": str(root.get("subject") or ""),
                    "date": str(root.get("date") or ""),
                    "message_id": str(root.get("message_id") or ""),
                    "in_reply_to": str(root.get("in_reply_to") or ""),
                    "references": str(root.get("references") or ""),
                    "body_text": str(root.get("body_text") or ""),
                    "body_html": str(root.get("body_html") or ""),
                    "text": str(root.get("text") or ""),
                }
            ],
            "text": f"Mailbox: {mailbox}\nThread for mail_id={mail_id}\nMessages: 1",
        }

    data = _gmail_request(method="GET", path=f"/threads/{thread_id}", params={"format": "full"})
    msgs = data.get("messages") if isinstance(data.get("messages"), list) else []
    out_msgs: List[Dict[str, Any]] = []
    for m in msgs[: max(1, min(int(max_messages), 100))]:
        if not isinstance(m, dict):
            continue
        item = _mail_payload_to_response(message=m, mailbox=mailbox, include_html=include_html, max_chars=max_chars)
        out_msgs.append(
            {
                "mail_id": str(item.get("mail_id") or ""),
                "thread_id": str(item.get("thread_id") or ""),
                "from_email": str(item.get("from_email") or ""),
                "to": list(item.get("to") or []),
                "cc": list(item.get("cc") or []),
                "subject": str(item.get("subject") or ""),
                "date": str(item.get("date") or ""),
                "message_id": str(item.get("message_id") or ""),
                "in_reply_to": str(item.get("in_reply_to") or ""),
                "references": str(item.get("references") or ""),
                "body_text": str(item.get("body_text") or ""),
                "body_html": str(item.get("body_html") or ""),
                "text": str(item.get("text") or ""),
            }
        )

    lines = [f"Mailbox: {mailbox}", f"Thread for mail_id={mail_id}", f"Messages: {len(out_msgs)}", ""]
    for i, item in enumerate(out_msgs, start=1):
        lines.append(f"[{i}] {item['subject']}")
        lines.append(f"from={item['from_email']} date={item['date']}")
        lines.append(_normalize_text(item["body_text"], max_chars=220))
        lines.append("")
    return {
        "mailbox": mailbox,
        "mail_id": mail_id,
        "thread_id": thread_id,
        "count": len(out_msgs),
        "messages": out_msgs,
        "text": "\n".join(lines).strip(),
    }


def _build_raw_message(
    *,
    to: List[str],
    subject: str,
    body: str,
    cc: List[str] | None = None,
    bcc: List[str] | None = None,
    reply_to: str = "",
    from_email: str = "",
    is_html: bool = False,
    in_reply_to: str = "",
    references: str = "",
) -> str:
    msg = EmailMessage()
    if from_email:
        msg["From"] = from_email
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    if bcc:
        msg["Bcc"] = ", ".join(bcc)
    if reply_to:
        msg["Reply-To"] = reply_to
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references
    msg["Subject"] = subject
    if is_html:
        msg.add_alternative(body, subtype="html")
    else:
        msg.set_content(body)

    raw_bytes = msg.as_bytes()
    return base64.urlsafe_b64encode(raw_bytes).decode("utf-8")


def send_mail(
    *,
    to: List[str],
    subject: str,
    body: str,
    cc: List[str] | None = None,
    bcc: List[str] | None = None,
    reply_to: str = "",
    from_email: str = "",
    is_html: bool = False,
) -> Dict[str, Any]:
    rec = [x.strip() for x in (to or []) if isinstance(x, str) and x.strip()]
    if not rec:
        raise HTTPException(status_code=422, detail="At least one recipient is required")
    raw = _build_raw_message(
        to=rec,
        subject=str(subject or "").strip(),
        body=str(body or ""),
        cc=[x.strip() for x in (cc or []) if x.strip()],
        bcc=[x.strip() for x in (bcc or []) if x.strip()],
        reply_to=str(reply_to or "").strip(),
        from_email=str(from_email or "").strip(),
        is_html=bool(is_html),
    )
    data = _gmail_request(method="POST", path="/messages/send", json_payload={"raw": raw})
    return {
        "sent": True,
        "message_id": str(data.get("id") or ""),
        "thread_id": str(data.get("threadId") or ""),
        "to": rec,
        "subject": str(subject or "").strip(),
    }


def answer_mail(
    *,
    mail_id: str,
    body: str,
    mailbox: str = "INBOX",
    subject: str = "",
    reply_to_all: bool = False,
    is_html: bool = False,
) -> Dict[str, Any]:
    original = read_mail(mail_id=mail_id, mailbox=mailbox, include_html=False, max_chars=20000)
    to_primary = _extract_addresses(str(original.get("from_email") or ""))
    if not to_primary:
        raise HTTPException(status_code=422, detail="Original mail has no reply address")
    recipients = list(to_primary)
    cc = []
    if reply_to_all:
        cc = list(original.get("cc") or [])
        for addr in list(original.get("to") or []):
            if addr not in recipients:
                recipients.append(addr)

    orig_subject = str(original.get("subject") or "").strip()
    final_subject = str(subject or "").strip()
    if not final_subject:
        final_subject = orig_subject or "(ohne Betreff)"
        if not final_subject.lower().startswith("re:"):
            final_subject = f"Re: {final_subject}"

    in_reply_to = str(original.get("message_id") or "").strip()
    references = str(original.get("references") or "").strip()
    if in_reply_to and in_reply_to not in references:
        references = (references + " " + in_reply_to).strip() if references else in_reply_to

    raw = _build_raw_message(
        to=recipients,
        subject=final_subject,
        body=str(body or "").strip(),
        cc=cc,
        is_html=bool(is_html),
        in_reply_to=in_reply_to,
        references=references,
    )

    payload = {"raw": raw}
    thread_id = str(original.get("thread_id") or "").strip()
    if thread_id:
        payload["threadId"] = thread_id

    data = _gmail_request(method="POST", path="/messages/send", json_payload=payload)
    _mark_message_read(mail_id)
    return {
        "sent": True,
        "message_id": str(data.get("id") or ""),
        "thread_id": str(data.get("threadId") or thread_id),
        "to": recipients,
        "subject": final_subject,
    }
