from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class MailSendRequest(BaseModel):
    to: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Pflichtfeld fuer neue E-Mail: mindestens eine Empfaenger-Adresse "
            "(z. B. ['name@example.com'])."
        ),
    )
    subject: str = Field(..., min_length=1, description="Betreff der neuen E-Mail.")
    body: str = Field(..., min_length=1, description="Inhalt der neuen E-Mail.")
    attachments: List[str] = Field(
        default_factory=list,
        description="Optionale Dateipfade fuer Anhaenge bei neuer E-Mail.",
    )
    cc: List[str] = Field(default_factory=list, description="Optionale CC-Empfaenger.")
    bcc: List[str] = Field(default_factory=list, description="Optionale BCC-Empfaenger.")
    from_email: str = Field(
        default="",
        description="Optionaler Absender. Leer = Standard-Absender aus Konfiguration.",
    )
    reply_to: str = Field(default="", description="Optionales Reply-To fuer neue E-Mail.")
    is_html: bool = Field(default=False, description="True wenn body HTML ist, sonst Plain-Text.")


class MailSendResponse(BaseModel):
    sent: bool = True
    message_id: str = ""
    to: List[str] = Field(default_factory=list)
    subject: str = ""
    attachments: List[str] = Field(default_factory=list)


class MailAnswerRequest(BaseModel):
    mail_id: str = Field(
        ...,
        min_length=1,
        description=(
            "UID einer bereits vorhandenen E-Mail aus dem Posteingang. "
            "Nur fuer Antworten auf existierende Mails verwenden."
        ),
    )
    body: str = Field(..., min_length=1, description="Antworttext auf die vorhandene E-Mail.")
    mailbox: str = Field(default="INBOX", description="Mailbox der Original-Mail, standardmaessig INBOX.")
    subject: str = Field(
        default="",
        description="Optionaler Antwort-Betreff. Leer = Betreff der Original-Mail weiterverwenden.",
    )
    reply_to_all: bool = Field(
        default=False,
        description="True: an alle Reply-Empfaenger antworten, False: nur an den Absender.",
    )
    is_html: bool = Field(default=False, description="True wenn body HTML ist, sonst Plain-Text.")


class MailAnswerResponse(BaseModel):
    sent: bool = True
    message_id: str = ""
    in_reply_to: str = ""
    to: List[str] = Field(default_factory=list)
    subject: str = ""
    marked_answered: bool = False


class MailInboxFetchRequest(BaseModel):
    limit: int = Field(10, ge=1, le=50, description="Maximale Anzahl abzurufender E-Mails.")
    mailbox: str = Field(default="INBOX", description="Zu lesende Mailbox.")
    unread_only: bool = Field(default=False, description="Nur ungelesene E-Mails abrufen.")


class MailUnansweredFetchRequest(BaseModel):
    limit: int = Field(10, ge=1, le=50, description="Maximale Anzahl unbeantworteter E-Mails.")
    mailbox: str = Field(default="INBOX", description="Mailbox fuer die Suche nach unbeantworteten E-Mails.")


class MailReadRequest(BaseModel):
    mail_id: str = Field(..., min_length=1, description="UID der zu lesenden E-Mail.")
    mailbox: str = Field(default="INBOX", description="Mailbox der E-Mail.")
    include_html: bool = Field(default=False, description="Wenn True, wird HTML-Inhalt ebenfalls zurueckgegeben.")
    max_chars: int = Field(
        default=20000,
        ge=500,
        le=200000,
        description="Maximale Zeichenlaenge fuer body_text/body_html.",
    )


class MailReadThreadRequest(BaseModel):
    mail_id: str = Field(..., min_length=1, description="UID einer Mail im Thread.")
    mailbox: str = Field(default="INBOX", description="Mailbox der Thread-Mail.")
    max_messages: int = Field(default=20, ge=1, le=100, description="Maximale Anzahl Thread-Mails.")
    include_html: bool = Field(default=False, description="Wenn True, wird HTML pro Thread-Mail zurückgegeben.")
    max_chars: int = Field(default=8000, ge=500, le=200000, description="Maximale Länge der Textfelder.")


class MailReadAttachmentsRequest(BaseModel):
    mail_id: str = Field(..., min_length=1, description="UID der E-Mail.")
    mailbox: str = Field(default="INBOX", description="Mailbox der E-Mail.")
    include_content: bool = Field(
        default=False,
        description="Wenn True, wird soweit möglich Textinhalt aus Anhängen extrahiert.",
    )
    max_attachment_chars: int = Field(
        default=4000,
        ge=200,
        le=50000,
        description="Maximale Textlänge pro Anhang bei include_content.",
    )
    extract_text_pdf: bool = Field(
        default=True,
        description="Wenn True und pypdf verfügbar, wird aus PDFs Text extrahiert.",
    )


class MailClassifyRequest(BaseModel):
    text: str = Field(default="", description="Gesamter Mailtext (optional, falls subject/body_text separat vorhanden).")
    subject: str = Field(default="", description="Mail-Betreff (optional).")
    body_text: str = Field(default="", description="Mail-Inhalt als Klartext (optional).")
    from_email: str = Field(default="", description="Absender (optional).")


class MailInboxItem(BaseModel):
    uid: str = ""
    from_email: str = ""
    subject: str = ""
    date: str = ""
    snippet: str = ""


class MailInboxFetchResponse(BaseModel):
    mailbox: str = "INBOX"
    count: int = 0
    mail_id: str = ""
    emails: List[MailInboxItem] = Field(default_factory=list)
    text: str = ""


class MailReadResponse(BaseModel):
    mailbox: str = "INBOX"
    mail_id: str = ""
    from_email: str = ""
    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    subject: str = ""
    date: str = ""
    message_id: str = ""
    in_reply_to: str = ""
    references: str = ""
    has_attachments: bool = False
    attachment_names: List[str] = Field(default_factory=list)
    body_text: str = ""
    body_html: str = ""
    text: str = ""


class MailThreadItem(BaseModel):
    mail_id: str = ""
    from_email: str = ""
    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    subject: str = ""
    date: str = ""
    message_id: str = ""
    in_reply_to: str = ""
    references: str = ""
    body_text: str = ""
    body_html: str = ""
    text: str = ""


class MailReadThreadResponse(BaseModel):
    mailbox: str = "INBOX"
    mail_id: str = ""
    count: int = 0
    messages: List[MailThreadItem] = Field(default_factory=list)
    text: str = ""


class MailAttachmentItem(BaseModel):
    filename: str = ""
    content_type: str = ""
    size_bytes: int = 0
    text: str = ""


class MailReadAttachmentsResponse(BaseModel):
    mailbox: str = "INBOX"
    mail_id: str = ""
    count: int = 0
    has_attachments: bool = False
    attachments: List[MailAttachmentItem] = Field(default_factory=list)
    text: str = ""


class MailClassifyResponse(BaseModel):
    intent: str = "info"
    confidence: float = 0.0
    reason: str = ""
    fallback_used: bool = False
    model: str = ""
    text: str = ""


class MailComposeClarificationRequest(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)
    known_facts: dict = Field(default_factory=dict)
    salutation: str = Field(default="Guten Tag")
    closing: str = Field(default="Mit freundlichen Grüßen")


class MailComposeClarificationResponse(BaseModel):
    body: str = ""
    questions: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    text: str = ""


__all__ = [
    "MailSendRequest",
    "MailSendResponse",
    "MailAnswerRequest",
    "MailAnswerResponse",
    "MailInboxFetchRequest",
    "MailUnansweredFetchRequest",
    "MailInboxFetchResponse",
    "MailReadRequest",
    "MailReadResponse",
    "MailReadThreadRequest",
    "MailReadThreadResponse",
    "MailReadAttachmentsRequest",
    "MailReadAttachmentsResponse",
    "MailClassifyRequest",
    "MailClassifyResponse",
    "MailComposeClarificationRequest",
    "MailComposeClarificationResponse",
]
